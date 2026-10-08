"""Data-parallel dispatch across GPUs.

Evaluators construct a flat list of work items (one per prompt, or one per
(class, prompt) pair) and call :func:`dispatch`. Items
are split round-robin and each shard is processed by a fresh subprocess that
pins ``CUDA_VISIBLE_DEVICES`` to a single GPU. Results are collected back
in the original item order.

Worker functions must be importable by qualified name (``module:func``) so
the spawn context can import them. Each worker receives a
list of (index, item) pairs plus the common kwargs dict, and must return a
list of (index, value) pairs.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable


WorkerResult = list[tuple[int, Any]]
WorkerFn = Callable[[list[tuple[int, Any]], dict], WorkerResult]


def partition(items: list[Any], n: int) -> list[list[tuple[int, Any]]]:
    """Round-robin partition ``items`` into ``n`` shards, preserving indices."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    shards: list[list[tuple[int, Any]]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append((i, item))
    return shards


def _resolve(qualname: str) -> WorkerFn:
    module_name, _, attr = qualname.partition(":")
    if not attr:
        raise ValueError(f"worker qualname must be 'module:func', got {qualname!r}")
    return getattr(importlib.import_module(module_name), attr)


def _worker_entry(
    rank: int,
    *,
    qualname: str,
    device: str,
    shard_path: str,
    common_path: str,
    out_path: str,
) -> None:
    """Subprocess entry — runs in a fresh interpreter after spawn."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        with open(shard_path, "rb") as f:
            shard: list[tuple[int, Any]] = pickle.load(f)
        with open(common_path, "rb") as f:
            common: dict = pickle.load(f)
        fn = _resolve(qualname)
        result = fn(shard, common)
        with open(out_path, "wb") as f:
            pickle.dump({"ok": True, "result": result}, f)
    except Exception as e:  # noqa: BLE001  (surface to parent)
        with open(out_path, "wb") as f:
            pickle.dump(
                {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()},
                f,
            )
        raise


def _resolve_devices(devices: list[int]) -> list[str]:
    """Map visible ordinals to CUDA mask tokens without querying CUDA."""
    devices = devices or [0]
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in devices):
        raise ValueError("devices must be non-negative integer indices")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must not contain duplicate indices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(d) for d in devices]
    tokens = [token.strip() for token in visible.split(",")]
    # CUDA stops enumerating at an invalid ordinal. Do not silently turn a
    # disabled/invalid mask into access to GPUs outside the inherited mask.
    for token in tokens:
        if not token or (not token.startswith(("GPU-", "MIG-")) and
                         (not token.isdecimal())):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain GPU ordinals or UUIDs")
    if any(d >= len(tokens) for d in devices):
        raise ValueError(f"devices {devices} exceed CUDA_VISIBLE_DEVICES={visible!r}")
    return [tokens[d] for d in devices]


def dispatch(
    worker_qualname: str,
    items: list[Any],
    common: dict,
    devices: list[int],
    *,
    tmp_root: str | Path | None = None,
) -> list[Any]:
    """Distribute ``items`` across ``devices`` and merge results.

    ``worker_qualname`` is ``"oneig_bench.evaluators.text:_run_shard"`` etc.
    Device indices are ordinals in the inherited CUDA_VISIBLE_DEVICES list;
    when unset they are CUDA physical ordinals. Every worker sees only its
    selected GPU as cuda:0. The parent environment is never modified.
    The worker must accept ``(shard, common) -> [(idx, value), ...]``.
    Returns a list aligned with ``items``; entries for items that the
    worker skipped (didn't emit) are ``None``.
    """
    if not items:
        return []
    # Resolve against the parent's original mask before trimming idle workers.
    resolved_devices = _resolve_devices(devices)
    n = min(len(resolved_devices), len(items))
    devices = resolved_devices[:n]

    # Even one device uses spawn: changing the parent's mask cannot retarget
    # CUDA after initialization and would leak into subsequent dispatch calls.
    shards = partition(items, n)
    tmp_dir = Path(tempfile.mkdtemp(prefix="oib_dispatch_", dir=str(tmp_root) if tmp_root else None))
    common_path = tmp_dir / "common.pkl"
    with open(common_path, "wb") as f:
        pickle.dump(common, f)

    shard_paths: list[Path] = []
    out_paths: list[Path] = []
    for rank, shard in enumerate(shards):
        sp = tmp_dir / f"shard_{rank}.pkl"
        op = tmp_dir / f"out_{rank}.pkl"
        with open(sp, "wb") as f:
            pickle.dump(shard, f)
        shard_paths.append(sp)
        out_paths.append(op)

    procs = []
    ctx = mp.get_context("spawn")
    for rank, device in enumerate(devices):
        p = ctx.Process(
            target=_worker_entry,
            kwargs=dict(
                rank=rank,
                qualname=worker_qualname,
                device=device,
                shard_path=str(shard_paths[rank]),
                common_path=str(common_path),
                out_path=str(out_paths[rank]),
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    merged: list[Any] = [None] * len(items)
    failures: list[str] = []
    for rank, op in enumerate(out_paths):
        if not op.exists():
            failures.append(f"rank {rank} (device {devices[rank]}): no output file (worker crashed?)")
            continue
        with open(op, "rb") as f:
            payload = pickle.load(f)
        if not payload.get("ok"):
            failures.append(
                f"rank {rank} (device {devices[rank]}): {payload.get('error')}\n{payload.get('tb', '')}"
            )
            continue
        for idx, value in payload["result"]:
            merged[idx] = value

    # Best-effort cleanup; on failure, leave tmp dir so the user can debug.
    if not failures:
        for p in [*shard_paths, *out_paths, common_path]:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    else:
        sys.stderr.write("\n".join(failures) + "\n")
        sys.stderr.write(f"(left worker artefacts under {tmp_dir} for inspection)\n")
        raise RuntimeError(f"{len(failures)} worker(s) failed; see stderr.")

    return merged
