"""Data-parallel dispatch across GPUs.

Evaluators construct a flat list of work items (one per prompt, or one per
(class, prompt) pair) and call :func:`dispatch`. With a single device the
caller's worker function runs in-process; with multiple devices the items
are split round-robin and each shard is processed by a subprocess that
pins ``CUDA_VISIBLE_DEVICES`` to a single GPU. Results are collected back
in the original item order.

Worker functions must be importable by qualified name (``module:func``) so
``torch.multiprocessing.spawn`` can pickle them. Each worker receives a
list of (index, item) pairs plus the common kwargs dict, and must return a
list of (index, value) pairs.
"""
from __future__ import annotations

import importlib
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
    device: int,
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
    The worker must accept ``(shard, common) -> [(idx, value), ...]``.
    Returns a list aligned with ``items``; entries for items that the
    worker skipped (didn't emit) are ``None``.
    """
    if not items:
        return []
    if not devices:
        devices = [0]

    n = min(len(devices), len(items))
    devices = devices[:n]

    # Single-device fast path: run in this process.
    if n == 1:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(devices[0]))
        fn = _resolve(worker_qualname)
        shard = [(i, item) for i, item in enumerate(items)]
        flat = fn(shard, common)
        merged: list[Any] = [None] * len(items)
        for idx, value in flat:
            merged[idx] = value
        return merged

    # Multi-device path: torch.multiprocessing.spawn.
    import torch.multiprocessing as mp

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
