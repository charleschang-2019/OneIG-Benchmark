"""Mock-data smoke test for the aggregation pipeline.

Synthesises per-dimension CSVs that resemble the real evaluator outputs and
verifies that ``oneig_bench aggregate`` produces a coherent summary table.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from oneig_bench.aggregation import aggregate  # noqa: E402

MODE = "EN"
TS = "2099-01-01_00-00-00"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="oib_agg_smoke_"))
    try:
        # Real bench CSV is required for fine-grained slicing — copy a slim
        # subset from the repo root so the test is self-contained.
        bench = pd.read_csv(ROOT / "OneIG-Bench.csv", dtype=str)
        anime_ids = bench[bench["category"] == "Anime_Stylization"]["id"].tolist()[:3]
        human_ids = bench[bench["category"] == "Portrait"]["id"].tolist()[:3]
        text_ids = bench[bench["category"] == "Text_Rendering"]["id"].tolist()[:3]
        reasoning_ids = bench[bench["category"] == "Knowledge_Reasoning"]["id"].tolist()[:3]

        models = ["modelA", "modelB"]

        # alignment_score / alignment_prompt_score
        write_csv(
            pd.DataFrame({"alignment": [0.75, 0.82]}, index=models),
            workdir / f"alignment_score_{MODE}_{TS}.csv",
        )
        rows = {}
        for cls, ids in [("anime", anime_ids), ("human", human_ids)]:
            for id_ in ids:
                rows[f"{cls}_{id_}"] = [0.8, 0.85]
        write_csv(
            pd.DataFrame.from_dict(rows, orient="index", columns=models),
            workdir / f"alignment_prompt_score_{MODE}_{TS}.csv",
        )

        # diversity
        write_csv(
            pd.DataFrame(
                {"anime": [0.4, 0.5], "human": [0.45, 0.55], "total average": [0.42, 0.52]},
                index=models,
            ),
            workdir / f"diversity_score_{MODE}_{TS}.csv",
        )
        rows = {f"anime_{i}": [0.4, 0.5] for i in anime_ids}
        rows.update({f"human_{i}": [0.45, 0.55] for i in human_ids})
        write_csv(
            pd.DataFrame.from_dict(rows, orient="index", columns=models),
            workdir / f"diversity_prompt_score_{MODE}_{TS}.csv",
        )

        # reasoning
        write_csv(
            pd.DataFrame({"reasoning": [0.6, 0.7]}, index=models),
            workdir / f"reasoning_score_{MODE}_{TS}.csv",
        )
        write_csv(
            pd.DataFrame.from_dict(
                {i: [0.6, 0.7] for i in reasoning_ids}, orient="index", columns=models
            ),
            workdir / f"reasoning_prompt_score_{MODE}_{TS}.csv",
        )

        # style
        write_csv(
            pd.DataFrame({"style": [0.55, 0.65]}, index=models),
            workdir / f"style_score_{MODE}_{TS}.csv",
        )
        # style_style_score expects per-style means with model rows
        styles = ["abstract_expressionism", "art_nouveau", "clay", "comic", "celluloid", "chibi"]
        style_data = {s: [0.5, 0.6] for s in styles}
        write_csv(
            pd.DataFrame(style_data, index=models),
            workdir / f"style_style_score_{MODE}_{TS}.csv",
        )
        write_csv(
            pd.DataFrame.from_dict(
                {i: [0.5, 0.6] for i in anime_ids}, orient="index", columns=models
            ),
            workdir / f"style_prompt_score_{MODE}_{TS}.csv",
        )

        # text
        write_csv(
            pd.DataFrame(
                {
                    "ED": [10.0, 8.0],
                    "CR": [0.4, 0.5],
                    "WAC": [0.7, 0.8],
                    "text score": [0.85, 0.9],
                },
                index=models,
            ),
            workdir / f"text_score_{MODE}_{TS}.csv",
        )
        rows = {i: [str([12.0, 0.4, 0.7]), str([9.0, 0.55, 0.8])] for i in text_ids}
        write_csv(
            pd.DataFrame.from_dict(rows, orient="index", columns=models),
            workdir / f"text_prompt_score_{MODE}_{TS}.csv",
        )

        out_paths = aggregate(
            workdir,
            MODE,
            TS,
            emit_csv=True,
            emit_xlsx=True,
            include_fine_grained=True,
        )

        print("\nProduced files:")
        for k, v in out_paths.items():
            print(f"  {k}: {v}")

        summary = pd.read_csv(out_paths["csv"], index_col=0)
        print("\nFlat summary table:")
        print(summary)

        # Sanity-check expected columns appear.
        expected = ["alignment", "diversity", "reasoning", "style", "text_score", "ED", "CR", "WAC"]
        missing = [c for c in expected if c not in summary.columns]
        assert not missing, f"Missing main columns: {missing}"
        assert "modelA" in summary.index and "modelB" in summary.index

        print("\nOK — aggregate produced expected overall columns and rows.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
