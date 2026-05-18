"""Fine-grained slicing of per-prompt scores.

Replaces ``fine_grained_analysis.py`` and accepts results either from a
single run timestamp or from manually-curated per-model folders (legacy
layout). Returns DataFrames the summary writer assembles into sheets.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from oneig_bench.data.paths import BENCH_CSV_EN, BENCH_CSV_ZH

SOURCE_ID = {
    "anime": "Anime_Stylization",
    "human": "Portrait",
    "object": "General_Object",
    "text": "Text_Rendering",
    "reasoning": "Knowledge_Reasoning",
    "multilingualism": "Multilingualism",
}

REASONING_SUBJECTS = [
    "geography", "computer science", "biology", "mathematics",
    "physics", "chemistry", "common sense",
]

STYLE_GROUPS = {
    "traditional": [
        "abstract_expressionism", "art_nouveau", "baroque", "chinese_ink_painting",
        "cubism", "fauvism", "impressionism", "line_art", "minimalism", "pointillism",
        "pop_art", "rococo", "ukiyo-e",
    ],
    "media": [
        "clay", "crayon", "graffiti", "lego", "comic", "pencil_sketch",
        "stone_sculpture", "watercolor",
    ],
    "anime": [
        "celluloid", "chibi", "cyberpunk", "ghibli", "impasto", "pixar",
        "pixel_art", "3d_rendering",
    ],
}

TEXT_METRICS = ["ED", "CR", "WAC"]
LENGTH_BUCKETS = ["short", "middle", "long"]
TYPE_BUCKETS = ["NP", "T, P"]


def _bench_df(mode: str) -> pd.DataFrame:
    path = BENCH_CSV_ZH if mode == "ZH" else BENCH_CSV_EN
    return pd.read_csv(path, dtype=str)


def _split_category_id(row_key: str) -> tuple[str, str]:
    """``anime_001`` -> (``Anime_Stylization``, ``001``)."""
    cls, _, id_ = row_key.partition("_")
    return SOURCE_ID.get(cls, cls), id_


def _filter_avg(scores: list[float]) -> float | None:
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else None


def fine_grained_prompt_based(
    prompt_score_path: str | Path,
    *,
    mode: str,
    metric_label: str,
) -> pd.DataFrame:
    """Slice an ``alignment`` / ``diversity`` prompt score by length & type."""
    bench = _bench_df(mode)
    data = pd.read_csv(prompt_score_path)
    model_names = [c for c in data.columns if c != "Unnamed: 0"]
    if "Unnamed: 0" not in data.columns:
        # When CSVs are saved without an index name, pandas auto-names the
        # first column. Be defensive.
        data = data.rename(columns={data.columns[0]: "Unnamed: 0"})

    out = pd.DataFrame(index=model_names)
    for tag in LENGTH_BUCKETS + TYPE_BUCKETS:
        col_name = f"{metric_label}_{tag}"
        for model in model_names:
            scores: list[float] = []
            for _, row in data.iterrows():
                cat, id_ = _split_category_id(str(row["Unnamed: 0"]))
                lookup = bench[(bench["category"] == cat) & (bench["id"] == id_)]
                if lookup.empty:
                    continue
                if tag in LENGTH_BUCKETS:
                    if lookup["prompt_length"].iloc[0] != tag:
                        continue
                else:
                    if lookup["type"].iloc[0] != tag:
                        continue
                try:
                    scores.append(float(row[model]))
                except (TypeError, ValueError):
                    continue
            out.loc[model, col_name] = _filter_avg(scores)
    return out


def fine_grained_reasoning(prompt_score_path: str | Path, *, mode: str) -> pd.DataFrame:
    bench = _bench_df(mode)
    data = pd.read_csv(prompt_score_path)
    if "Unnamed: 0" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Unnamed: 0"})
    model_names = [c for c in data.columns if c != "Unnamed: 0"]
    out = pd.DataFrame(index=model_names)
    for subject in REASONING_SUBJECTS:
        col = f"reasoning_{subject.replace(' ', '_')}"
        for model in model_names:
            scores: list[float] = []
            for _, row in data.iterrows():
                lookup = bench[
                    (bench["category"] == SOURCE_ID["reasoning"])
                    & (bench["id"] == str(row["Unnamed: 0"]))
                ]
                if lookup.empty or lookup["class"].iloc[0] != subject:
                    continue
                try:
                    scores.append(float(row[model]))
                except (TypeError, ValueError):
                    continue
            out.loc[model, col] = _filter_avg(scores)
    return out


def fine_grained_style(style_score_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(style_score_path)
    if "Unnamed: 0" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Unnamed: 0"})
    out = pd.DataFrame(index=data["Unnamed: 0"].tolist())
    for group, styles in STYLE_GROUPS.items():
        present = [s for s in styles if s in data.columns]
        if not present:
            continue
        means = data[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        out[f"style_{group}"] = means.values
    return out


def fine_grained_text(prompt_score_path: str | Path, *, mode: str) -> pd.DataFrame:
    bench = _bench_df(mode)
    data = pd.read_csv(prompt_score_path)
    if "Unnamed: 0" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Unnamed: 0"})
    model_names = [c for c in data.columns if c != "Unnamed: 0"]
    out = pd.DataFrame(index=model_names)
    for length in LENGTH_BUCKETS:
        for model in model_names:
            triples: list[list[float]] = []
            for _, row in data.iterrows():
                lookup = bench[
                    (bench["category"] == SOURCE_ID["text"])
                    & (bench["id"] == str(row["Unnamed: 0"]))
                ]
                if lookup.empty or lookup["prompt_length"].iloc[0] != length:
                    continue
                cell = row[model]
                if pd.isna(cell):
                    continue
                try:
                    parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                    triples.append([float(x) for x in parsed])
            if not triples:
                for m in TEXT_METRICS:
                    out.loc[model, f"text_{length}_{m}"] = None
                continue
            avg = [sum(col) / len(col) for col in zip(*triples)]
            for m, v in zip(TEXT_METRICS, avg):
                out.loc[model, f"text_{length}_{m}"] = v
    return out
