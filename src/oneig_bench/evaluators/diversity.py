"""Diversity evaluator (DreamSim pairwise distance across seeds).

Per-(model, class, image) item is shipped to a worker that loads DreamSim
on the assigned GPU; pairwise distances are averaged per prompt.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import megfile
import pandas as pd

from oneig_bench.config import EvalConfig
from oneig_bench.evaluators.base import Evaluator, EvalResult
from oneig_bench.image_utils import split_mxn_grid
from oneig_bench.io import ensure_dir, on_rm_error
from oneig_bench.parallel import dispatch


# ---------------------------------------------------------------- worker side


def _img_similar_score(model, preprocess, p1, p2, device: str = "cuda"):
    from PIL import Image

    img1 = preprocess(Image.open(p1)).to(device)
    img2 = preprocess(Image.open(p2)).to(device)
    return model(img1, img2).item()


def _score_one(img_path: str, img_grid, cache_dir):
    from oneig_bench.models import get_dreamsim

    split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
    if len(split_img_list) <= 1:
        return None
    model, preprocess = get_dreamsim()
    scores: list[float] = []
    for i in range(len(split_img_list)):
        for j in range(i + 1, len(split_img_list)):
            scores.append(
                _img_similar_score(model, preprocess, split_img_list[i], split_img_list[j])
            )
    return sum(scores) / len(scores)


def _run_shard(shard, common):
    cache_dir = Path(common["cache_dir"]) / f"rank_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[int, object]] = []
    for idx, item in shard:
        result = _score_one(item["img_path"], item["img_grid"], cache_dir)
        out.append((idx, result))
    return out


# ---------------------------------------------------------------- evaluator


class DiversityEvaluator(Evaluator):
    name = "diversity"

    def run(self, cfg: EvalConfig) -> EvalResult:
        cache_dir = Path(f"tmp_diversity_{cfg.timestamp}")
        ensure_dir(cache_dir)

        class_items = cfg.class_items_for("diversity")
        columns = class_items + ["total average"]
        score_csv = pd.DataFrame(index=cfg.model_names(), columns=columns, dtype=float)
        prompt_csv = pd.DataFrame(columns=cfg.model_names())

        # Collect all work + the row label each item maps to.
        items: list[dict] = []
        row_keys: list[tuple[str, str]] = []  # (prompt_row_label, model_name)
        bucket_keys: list[tuple[str, str]] = []  # (model_name, class_item)
        for entry in cfg.models:
            for class_item in class_items:
                image_dir = f"{cfg.image_dirname}/{class_item}/{entry.name}"
                img_list = sorted(megfile.smart_glob(image_dir + "/*"))
                for img_path in img_list:
                    items.append(dict(img_path=img_path, img_grid=entry.grid_tuple))
                    row_keys.append(
                        (f"{class_item}_{Path(img_path).name[:3]}", entry.name)
                    )
                    bucket_keys.append((entry.name, class_item))

        print(f"[diversity] dispatching {len(items)} items across devices {cfg.devices}")
        results = dispatch(
            "oneig_bench.evaluators.diversity:_run_shard",
            items,
            common={"cache_dir": str(cache_dir)},
            devices=cfg.devices,
            tmp_root=cache_dir,
        )

        buckets: dict[tuple[str, str], list[float]] = {}
        for (row, model_name), bucket, value in zip(row_keys, bucket_keys, results):
            if value is None:
                continue
            prompt_csv.loc[row, model_name] = value
            buckets.setdefault(bucket, []).append(value)

        for entry in cfg.models:
            for class_item in class_items:
                vals = buckets.get((entry.name, class_item), [])
                score_csv.loc[entry.name, class_item] = (
                    sum(vals) / len(vals) if vals else None
                )

        score_csv["total average"] = prompt_csv.mean().values
        prompt_csv = prompt_csv.sort_index()

        if cache_dir.exists():
            shutil.rmtree(cache_dir, onerror=on_rm_error)

        return EvalResult(self.name, score_csv, prompt_csv)
