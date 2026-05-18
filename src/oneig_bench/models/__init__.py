"""Lazy-loaded heavy model wrappers.

Models are *not* loaded at import time. Each ``get_*`` accessor materialises
its singleton on the first call so an evaluator that only needs one model
never pays for the others.
"""
from __future__ import annotations

from typing import Optional

_QWEN: Optional["Qwen2_5VLBatchInferencer"] = None  # type: ignore[name-defined]
_LLM2CLIP: Optional["LLM2CLIP"] = None  # type: ignore[name-defined]
_CSD: Optional["CSDStyleEmbedding"] = None  # type: ignore[name-defined]
_SE: Optional["SEStyleEmbedding"] = None  # type: ignore[name-defined]
_DREAMSIM = None


def get_qwen(model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    global _QWEN
    if _QWEN is None:
        from oneig_bench.models.qwen_vl import Qwen2_5VLBatchInferencer

        _QWEN = Qwen2_5VLBatchInferencer(model_path)
    return _QWEN


def get_llm2clip():
    global _LLM2CLIP
    if _LLM2CLIP is None:
        from oneig_bench.models.llm2clip import LLM2CLIP

        _LLM2CLIP = LLM2CLIP()
    return _LLM2CLIP


def get_csd_encoder(model_path: str | None = None):
    global _CSD
    if _CSD is None:
        from oneig_bench.data.paths import STYLE_CSD_CHECKPOINT
        from oneig_bench.models.style_encoders import CSDStyleEmbedding

        _CSD = CSDStyleEmbedding(model_path=str(model_path or STYLE_CSD_CHECKPOINT))
    return _CSD


def get_se_encoder(pretrained_path: str = "xingpng/OneIG-StyleEncoder"):
    global _SE
    if _SE is None:
        from oneig_bench.models.style_encoders import SEStyleEmbedding

        _SE = SEStyleEmbedding(pretrained_path=pretrained_path)
    return _SE


def get_dreamsim():
    """Return ``(model, preprocess)`` from the DreamSim package."""
    global _DREAMSIM
    if _DREAMSIM is None:
        from oneig_bench.models.dreamsim_wrap import load_dreamsim

        _DREAMSIM = load_dreamsim()
    return _DREAMSIM
