"""Evaluator base class + result container."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from oneig_bench.config import EvalConfig
from oneig_bench.io import ensure_dir, results_filename, save_csv


@dataclass
class EvalResult:
    """A single evaluator's outputs.

    ``main_score`` is the per-model summary (one row per model).
    ``prompt_score`` is the per-prompt detail (one row per prompt id, one
    column per model). ``extra`` holds optional auxiliary tables — e.g.
    style emits a per-style breakdown there.
    """

    metric: str
    main_score: pd.DataFrame
    prompt_score: pd.DataFrame
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)

    def save(self, cfg: EvalConfig) -> dict[str, Path]:
        """Persist the result using legacy filenames (mode + timestamp)."""
        out_dir = ensure_dir(cfg.results_dir)
        ts = cfg.timestamp
        paths: dict[str, Path] = {}
        paths["main"] = save_csv(
            self.main_score, out_dir / results_filename(self.metric, "score", cfg.mode, ts)
        )
        paths["prompt"] = save_csv(
            self.prompt_score,
            out_dir / results_filename(self.metric, "prompt_score", cfg.mode, ts),
        )
        for kind, df in self.extra.items():
            paths[kind] = save_csv(
                df, out_dir / results_filename(self.metric, kind, cfg.mode, ts)
            )
        return paths


class Evaluator(ABC):
    name: str = ""

    @abstractmethod
    def run(self, cfg: EvalConfig) -> EvalResult:
        """Compute and return scores. Must not write files itself."""
