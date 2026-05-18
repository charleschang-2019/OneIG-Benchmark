"""Reasoning evaluator (LLM2CLIP text/image similarity vs. GT answers)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import megfile
import pandas as pd

from oneig_bench.config import EvalConfig
from oneig_bench.data.paths import REASONING_ANSWER_EN, REASONING_ANSWER_ZH
from oneig_bench.evaluators.base import Evaluator, EvalResult
from oneig_bench.image_utils import split_mxn_grid
from oneig_bench.io import ensure_dir, on_rm_error
from oneig_bench.parallel import dispatch


# ---------------------------------------------------------------- worker side


def _score_one(img_path: str, img_grid, answer_text: str, cache_dir):
    from oneig_bench.models import get_llm2clip

    split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
    if not split_img_list:
        return None
    llm2clip = get_llm2clip()
    score = llm2clip.text_img_similarity_score(split_img_list, answer_text)
    if not score:
        return None
    score = [x for x in score if x is not None]
    return sum(score) / len(score) if score else None


def _run_shard(shard, common):
    cache_dir = Path(common["cache_dir"]) / f"rank_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[int, object]] = []
    for idx, item in shard:
        result = _score_one(
            item["img_path"], item["img_grid"], item["answer_text"], cache_dir
        )
        out.append((idx, result))
    return out


# ---------------------------------------------------------------- evaluator


class ReasoningEvaluator(Evaluator):
    name = "reasoning"

    def run(self, cfg: EvalConfig) -> EvalResult:
        cache_dir = Path(f"tmp_reasoning_{cfg.timestamp}")
        ensure_dir(cache_dir)

        answer_path = REASONING_ANSWER_EN if cfg.mode == "EN" else REASONING_ANSWER_ZH
        with open(answer_path, "r", encoding="utf-8") as f:
            answer_gt = json.load(f)

        score_csv = pd.DataFrame(index=cfg.model_names(), columns=["reasoning"])
        prompt_csv = pd.DataFrame(columns=cfg.model_names())

        items: list[dict] = []
        row_keys: list[tuple[str, str]] = []
        for entry in cfg.models:
            image_dir = f"{cfg.image_dirname}/{entry.name}"
            img_list = sorted(megfile.smart_glob(image_dir + "/*"))
            for img_path in img_list:
                img_id = Path(img_path).name[:3]
                items.append(
                    dict(
                        img_path=img_path,
                        img_grid=entry.grid_tuple,
                        answer_text=answer_gt[img_id],
                    )
                )
                row_keys.append((img_id, entry.name))

        print(f"[reasoning] dispatching {len(items)} items across devices {cfg.devices}")
        results = dispatch(
            "oneig_bench.evaluators.reasoning:_run_shard",
            items,
            common={"cache_dir": str(cache_dir)},
            devices=cfg.devices,
            tmp_root=cache_dir,
        )

        for (row, model_name), value in zip(row_keys, results):
            prompt_csv.loc[row, model_name] = value

        score_csv["reasoning"] = prompt_csv.mean().values
        prompt_csv = prompt_csv.sort_index()

        if cache_dir.exists():
            shutil.rmtree(cache_dir, onerror=on_rm_error)

        return EvalResult(self.name, score_csv, prompt_csv)
