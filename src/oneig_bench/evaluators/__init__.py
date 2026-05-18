"""Evaluator registry."""
from __future__ import annotations

from oneig_bench.evaluators.base import Evaluator, EvalResult

# Lazy registry: classes are imported on demand so a `--dimensions text`
# run does not pull in DreamSim, LLM2CLIP, etc.
_FACTORY = {
    "alignment": ("oneig_bench.evaluators.alignment", "AlignmentEvaluator"),
    "diversity": ("oneig_bench.evaluators.diversity", "DiversityEvaluator"),
    "reasoning": ("oneig_bench.evaluators.reasoning", "ReasoningEvaluator"),
    "style": ("oneig_bench.evaluators.style", "StyleEvaluator"),
    "text": ("oneig_bench.evaluators.text", "TextEvaluator"),
}


def get_evaluator(name: str) -> Evaluator:
    if name not in _FACTORY:
        raise KeyError(f"Unknown evaluator: {name}. Available: {sorted(_FACTORY)}")
    module_name, cls_name = _FACTORY[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, cls_name)()


def available_evaluators() -> list[str]:
    return sorted(_FACTORY)


__all__ = ["Evaluator", "EvalResult", "get_evaluator", "available_evaluators"]
