"""Shared IO helpers used by all evaluators and the aggregator."""
from __future__ import annotations

import datetime
import os
import stat
from pathlib import Path
from typing import Iterable

import pandas as pd


def timestamp() -> str:
    """Return a filesystem-safe ``YYYY-MM-DD_HH-MM-SS`` timestamp."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, csv_path: str | Path, *, index: bool = True) -> Path:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=index)
    print(f"Results saved to {csv_path}")
    return csv_path


def on_rm_error(func, path, exc_info):  # noqa: ARG001  (shutil.rmtree onerror signature)
    os.chmod(path, stat.S_IWRITE)
    func(path)


def results_filename(metric: str, kind: str, mode: str, ts: str) -> str:
    """Build the legacy result filename so old tooling keeps working.

    ``kind`` is one of: ``score``, ``prompt_score``, ``style_score`` (used
    only for the ``style`` metric).
    """
    return f"{metric}_{kind}_{mode}_{ts}.csv"


def iter_existing(paths: Iterable[Path | str]) -> list[Path]:
    return [Path(p) for p in paths if Path(p).exists()]
