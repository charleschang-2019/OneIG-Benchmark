"""Alignment evaluator (Qwen2.5-VL semantic Q/A with dependency filtering).

The per-prompt work is sharded across GPUs by :mod:`oneig_bench.parallel`.
Each work item is one (model, class, prompt_id) triple — the worker loads
its own Qwen-VL on the assigned GPU and emits one score per item.
"""
from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import megfile
import pandas as pd

from oneig_bench.config import EvalConfig
from oneig_bench.data.paths import ALIGNMENT_QD_DIR
from oneig_bench.evaluators.base import Evaluator, EvalResult
from oneig_bench.image_utils import split_mxn_grid
from oneig_bench.io import ensure_dir, on_rm_error
from oneig_bench.parallel import dispatch


# ---------------------------------------------------------------- worker side


def _score_one(img_path, questions, dependencies, img_grid, cache_dir):
    """Compute the alignment score for one prompt. Returns ``None`` on miss."""
    from oneig_bench.models import get_qwen

    if len(img_path) != 1:
        return None
    split_img_list = split_mxn_grid(img_path[0], img_grid, cache_dir)
    if not split_img_list:
        return None

    inferencer = get_qwen()
    score = {}
    for qid, question in questions.items():
        batch_answer = inferencer.infer_semantic(split_img_list, question)
        score[qid] = [float(ans == "Yes") for ans in batch_answer]

    filtered = deepcopy(score)
    for img_idx in range(len(split_img_list)):
        for qid, parent_ids in dependencies.items():
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                try:
                    if score[parent_id][img_idx] == 0:
                        filtered[qid][img_idx] = 0
                        break
                except KeyError:
                    print("The score is not a number.")

    sums = [0.0] * len(split_img_list)
    for qid in filtered:
        for img_idx in range(len(split_img_list)):
            sums[img_idx] += filtered[qid][img_idx]
    per_img = [s / len(filtered) for s in sums]
    return sum(per_img) / len(per_img)


def _run_shard(shard, common):
    """Subprocess entry: process a list of (idx, item) work items."""
    cache_dir = Path(common["cache_dir"]) / f"rank_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[int, object]] = []
    for idx, item in shard:
        result = _score_one(
            item["img_path"],
            item["questions"],
            item["dependencies"],
            item["img_grid"],
            cache_dir,
        )
        out.append((idx, result))
    return out


# ---------------------------------------------------------------- evaluator


class AlignmentEvaluator(Evaluator):
    name = "alignment"

    def run(self, cfg: EvalConfig) -> EvalResult:
        cache_dir = Path(f"tmp_alignment_{cfg.timestamp}")
        ensure_dir(cache_dir)

        score_csv = pd.DataFrame(index=cfg.model_names(), columns=["alignment"])
        prompt_csv = pd.DataFrame(columns=cfg.model_names())

        class_items = cfg.class_items_for("alignment")

        # Flatten everything to one work-item list so the dispatcher can
        # balance load across GPUs evenly.
        items: list[dict] = []
        row_keys: list[tuple[str, str]] = []  # (row_label, model_name)
        for class_item in class_items:
            suffix = "_zh" if cfg.mode == "ZH" else ""
            qd_path = ALIGNMENT_QD_DIR / f"{class_item}{suffix}.json"
            with open(qd_path, "r", encoding="utf-8") as f:
                qd = json.load(f)

            for key, item in qd.items():
                if isinstance(item["question"], str):
                    item["question"] = {int(k): v for k, v in json.loads(item["question"]).items()}
                if isinstance(item["dependency"], str):
                    item["dependency"] = {
                        int(k): v for k, v in json.loads(item["dependency"]).items()
                    }

                for model in cfg.models:
                    img_paths = megfile.smart_glob(
                        f"{cfg.image_dirname}/{class_item}/{model.name}/{key}*"
                    )
                    items.append(
                        dict(
                            img_path=img_paths,
                            questions=item["question"],
                            dependencies=item["dependency"],
                            img_grid=model.grid_tuple,
                        )
                    )
                    row_keys.append((f"{class_item}_{key}", model.name))

        print(f"[alignment] dispatching {len(items)} items across devices {cfg.devices}")
        results = dispatch(
            "oneig_bench.evaluators.alignment:_run_shard",
            items,
            common={"cache_dir": str(cache_dir)},
            devices=cfg.devices,
            tmp_root=cache_dir,
        )

        for (row, model_name), value in zip(row_keys, results):
            prompt_csv.loc[row, model_name] = value

        score_csv["alignment"] = prompt_csv.mean().values
        prompt_csv = prompt_csv.sort_index()

        if cache_dir.exists():
            shutil.rmtree(cache_dir, onerror=on_rm_error)

        return EvalResult(self.name, score_csv, prompt_csv)
