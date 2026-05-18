"""Locations of static data assets shipped with the repository.

Code under ``src/oneig_bench`` should reference data via these constants
rather than hard-coded relative paths so the package keeps working
regardless of the caller's current working directory.
"""
from __future__ import annotations

from pathlib import Path

# Repo root is two levels above this file: src/oneig_bench/data/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

SCRIPTS_DIR = REPO_ROOT / "scripts"

ALIGNMENT_QD_DIR = SCRIPTS_DIR / "alignment" / "Q_D"
REASONING_ANSWER_EN = SCRIPTS_DIR / "reasoning" / "gt_answer.json"
REASONING_ANSWER_ZH = SCRIPTS_DIR / "reasoning" / "gt_answer_zh.json"

STYLE_DIR = SCRIPTS_DIR / "style"
STYLE_CSV = STYLE_DIR / "style.csv"
STYLE_CSD_EMBED = STYLE_DIR / "CSD_embed.pt"
STYLE_SE_EMBED = STYLE_DIR / "SE_embed.pt"
STYLE_CSD_CHECKPOINT = STYLE_DIR / "models" / "checkpoint.pth"

TEXT_CONTENT_EN = SCRIPTS_DIR / "text" / "text_content.csv"
TEXT_CONTENT_ZH = SCRIPTS_DIR / "text" / "text_content_zh.csv"

BENCH_CSV_EN = REPO_ROOT / "OneIG-Bench.csv"
BENCH_CSV_ZH = REPO_ROOT / "OneIG-Bench-ZH.csv"
