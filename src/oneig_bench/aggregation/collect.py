"""Find evaluator outputs on disk so we can aggregate them."""
from __future__ import annotations

import re
from pathlib import Path

# Filenames follow `<metric>_<kind>_<MODE>_<timestamp>.csv` — see
# ``io.results_filename``.
_PATTERN = re.compile(
    r"^(?P<metric>[a-zA-Z_]+?)_(?P<kind>score|prompt_score|style_score)_(?P<mode>EN|ZH)_(?P<ts>[0-9_\-]+)\.csv$"
)


def list_runs(results_dir: str | Path, mode: str | None = None) -> list[str]:
    """Return all timestamps present in ``results_dir`` (newest first)."""
    seen: set[str] = set()
    for p in Path(results_dir).glob("*.csv"):
        m = _PATTERN.match(p.name)
        if not m:
            continue
        if mode and m.group("mode") != mode:
            continue
        seen.add(m.group("ts"))
    return sorted(seen, reverse=True)


def latest_timestamp(results_dir: str | Path, mode: str) -> str | None:
    runs = list_runs(results_dir, mode=mode)
    return runs[0] if runs else None


def find_files(results_dir: str | Path, mode: str, timestamp: str) -> dict[str, dict[str, Path]]:
    """Return ``{metric: {kind: path}}`` for a given run timestamp."""
    out: dict[str, dict[str, Path]] = {}
    for p in Path(results_dir).glob(f"*_{mode}_{timestamp}.csv"):
        m = _PATTERN.match(p.name)
        if not m or m.group("mode") != mode or m.group("ts") != timestamp:
            continue
        out.setdefault(m.group("metric"), {})[m.group("kind")] = p
    return out
