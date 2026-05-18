"""Cross-dimension summary table builder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from oneig_bench.aggregation.collect import find_files, latest_timestamp
from oneig_bench.aggregation.fine_grained import (
    fine_grained_prompt_based,
    fine_grained_reasoning,
    fine_grained_style,
    fine_grained_text,
)
from oneig_bench.io import ensure_dir, save_csv


def _load_score_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.lower().startswith("unnamed") or first == "":
        df = df.rename(columns={first: "model"}).set_index("model")
    else:
        df = df.set_index(first)
    return df


def build_overall(files: dict[str, dict[str, Path]]) -> pd.DataFrame:
    """Wide table: rows = models, cols = main metric values per dimension."""
    parts: list[pd.DataFrame] = []
    for metric, kinds in files.items():
        score_path = kinds.get("score")
        if score_path is None:
            continue
        df = _load_score_df(score_path)
        if metric == "diversity":
            # Drop class-specific columns, keep only the rollup.
            keep = [c for c in df.columns if c == "total average"]
            if keep:
                df = df[keep].rename(columns={"total average": "diversity"})
        elif metric == "text":
            # Keep the headline ``text score`` and the components.
            cols = [c for c in ["text score", "ED", "CR", "WAC"] if c in df.columns]
            df = df[cols].rename(columns={"text score": "text_score"})
        else:
            # ``alignment``, ``reasoning``, ``style`` all expose a single
            # column already named after the metric.
            if metric in df.columns:
                df = df[[metric]]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).sort_index()


def build_fine_grained(
    files: dict[str, dict[str, Path]], mode: str
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    if "alignment" in files and "prompt_score" in files["alignment"]:
        sheets["fine_alignment"] = fine_grained_prompt_based(
            files["alignment"]["prompt_score"], mode=mode, metric_label="alignment"
        )
    if "diversity" in files and "prompt_score" in files["diversity"]:
        sheets["fine_diversity"] = fine_grained_prompt_based(
            files["diversity"]["prompt_score"], mode=mode, metric_label="diversity"
        )
    if "reasoning" in files and "prompt_score" in files["reasoning"]:
        sheets["fine_reasoning"] = fine_grained_reasoning(
            files["reasoning"]["prompt_score"], mode=mode
        )
    if "style" in files and "style_score" in files["style"]:
        sheets["fine_style"] = fine_grained_style(files["style"]["style_score"])
    if "text" in files and "prompt_score" in files["text"]:
        sheets["fine_text"] = fine_grained_text(files["text"]["prompt_score"], mode=mode)
    return sheets


def write_summary(
    results_dir: str | Path,
    mode: str,
    timestamp: str,
    *,
    emit_csv: bool = True,
    emit_xlsx: bool = True,
    include_fine_grained: bool = True,
) -> dict[str, Path]:
    files = find_files(results_dir, mode, timestamp)
    if not files:
        raise FileNotFoundError(
            f"No evaluator outputs found in {results_dir} for mode={mode}, ts={timestamp}"
        )

    overall = build_overall(files)
    sheets: dict[str, pd.DataFrame] = {"overall": overall}
    if include_fine_grained:
        sheets.update(build_fine_grained(files, mode))

    # Merge fine-grained columns into the flat summary so a single CSV
    # carries everything when users prefer CSV over xlsx.
    flat = overall.copy()
    for name, df in sheets.items():
        if name == "overall" or df is None or df.empty:
            continue
        flat = flat.join(df, how="outer")

    out_dir = ensure_dir(results_dir)
    out_paths: dict[str, Path] = {}
    if emit_csv:
        out_paths["csv"] = save_csv(
            flat, out_dir / f"summary_{mode}_{timestamp}.csv"
        )
    if emit_xlsx:
        xlsx_path = out_dir / f"summary_{mode}_{timestamp}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                if df is None or df.empty:
                    continue
                df.to_excel(writer, sheet_name=name[:31])
        print(f"Results saved to {xlsx_path}")
        out_paths["xlsx"] = xlsx_path
    return out_paths


def aggregate(
    results_dir: str | Path,
    mode: str,
    timestamp: str | None = None,
    *,
    emit_csv: bool = True,
    emit_xlsx: bool = True,
    include_fine_grained: bool = True,
) -> dict[str, Path]:
    """Top-level entry: pick the latest run if no timestamp was provided."""
    ts = timestamp or latest_timestamp(results_dir, mode)
    if ts is None:
        raise FileNotFoundError(
            f"No evaluator outputs found under {results_dir} for mode={mode}"
        )
    return write_summary(
        results_dir,
        mode,
        ts,
        emit_csv=emit_csv,
        emit_xlsx=emit_xlsx,
        include_fine_grained=include_fine_grained,
    )
