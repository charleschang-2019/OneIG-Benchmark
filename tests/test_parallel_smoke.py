"""Multi-process dispatch smoke test.

Verifies that :func:`oneig_bench.parallel.dispatch` spawns ``N`` subprocesses,
sets ``CUDA_VISIBLE_DEVICES`` per worker, and merges results in original
order. The fake worker doesn't actually touch CUDA so this runs anywhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneig_bench.parallel import dispatch, partition


def slow_double(shard, common):
    import time

    time.sleep(0.05)
    out = []
    for idx, item in shard:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        out.append((idx, {"value": item * 2, "cuda_visible": cvd, "pid": os.getpid()}))
    return out


def main() -> None:
    # Round-robin partition unit test
    parts = partition(list(range(7)), 3)
    assert [len(p) for p in parts] == [3, 2, 2], parts
    print("partition OK:", parts)

    items = list(range(8))

    print("\n-- single device path (devices=[0]) --")
    res = dispatch("tests.test_parallel_smoke:slow_double", items, {}, devices=[0])
    print(res)
    assert [r["value"] for r in res] == [x * 2 for x in items]

    print("\n-- multi device path (devices=[0,1,2,3], no real GPU) --")
    res = dispatch(
        "tests.test_parallel_smoke:slow_double", items, {}, devices=[0, 1, 2, 3]
    )
    for r in res:
        print(r)
    assert [r["value"] for r in res] == [x * 2 for x in items]
    seen_devices = {r["cuda_visible"] for r in res}
    seen_pids = {r["pid"] for r in res}
    print(f"\nseen CUDA_VISIBLE_DEVICES: {seen_devices}")
    print(f"distinct worker PIDs: {len(seen_pids)}")
    assert seen_devices == {"0", "1", "2", "3"}, seen_devices
    assert len(seen_pids) == 4, f"expected 4 distinct workers, got {seen_pids}"

    print("\nOK: dispatch routes items round-robin and merges in order.")


if __name__ == "__main__":
    main()
