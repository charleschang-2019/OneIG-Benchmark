"""Text rendering evaluator (Qwen2.5-VL OCR + edit distance metrics)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import megfile
import pandas as pd

from oneig_bench.config import EvalConfig
from oneig_bench.data.paths import TEXT_CONTENT_EN, TEXT_CONTENT_ZH
from oneig_bench.evaluators.base import Evaluator, EvalResult
from oneig_bench.image_utils import split_mxn_grid
from oneig_bench.io import ensure_dir, on_rm_error
from oneig_bench.parallel import dispatch
from scripts.text.text_utils import (
    calculate_char_match_ratio,
    clean_and_remove_hallucinations,
    levenshtein_distance,
    preprocess_string,
)


# ---------------------------------------------------------------- worker side


def _score_one(id_: str, text_gt: str, img_grid, image_dir_model: str, cache_dir):
    """Return (prompt_cell, ed_list, cr_list, gt_word_count_list).

    ``prompt_cell`` is the list ``[ED_avg, CR_avg, WAC_avg]`` written into
    the per-prompt CSV. The remaining lists carry the per-tile raw values
    so the parent can recompute the global ED/CR/WAC.
    """
    from oneig_bench.models import get_qwen

    word_count = len(text_gt.split())
    max_new_tokens = 256 if word_count > 60 else 128
    text_gt_pre = preprocess_string(text_gt)

    img_paths = megfile.smart_glob(f"{image_dir_model}/{id_}*")
    if len(img_paths) != 1:
        return None
    split_img_list = split_mxn_grid(img_paths[0], img_grid, cache_dir)
    if not split_img_list:
        return None

    inferencer = get_qwen()
    ocr_results = inferencer.infer_ocr(split_img_list, max_new_tokens)
    text_ocr_list = clean_and_remove_hallucinations(ocr_results)

    ed_score, cr_score, wac_score = [], [], []
    edit_distances: list[float] = []
    completion_ratios: list[int] = []
    match_word_counts: list[int] = []
    gt_word_counts: list[int] = []

    for text_ocr in text_ocr_list:
        ocr_pre = preprocess_string(text_ocr)
        ed = levenshtein_distance(ocr_pre, text_gt_pre)
        cr = 1 if ed == 0 else 0
        match_word_count, wac, gt_word_count = calculate_char_match_ratio(text_gt_pre, ocr_pre)
        edit_distances.append(float(ed))
        completion_ratios.append(int(cr))
        match_word_counts.append(int(match_word_count))
        gt_word_counts.append(int(gt_word_count))
        ed_score.append(float(ed))
        cr_score.append(int(cr))
        wac_score.append(float(wac))

    prompt_cell = [
        sum(ed_score) / len(ed_score),
        sum(cr_score) / len(cr_score),
        sum(wac_score) / len(wac_score),
    ]
    return {
        "prompt_cell": prompt_cell,
        "edit_distances": edit_distances,
        "completion_ratios": completion_ratios,
        "match_word_counts": match_word_counts,
        "gt_word_counts": gt_word_counts,
    }


def _run_shard(shard, common):
    cache_dir = Path(common["cache_dir"]) / f"rank_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[int, object]] = []
    for idx, item in shard:
        result = _score_one(
            item["id"], item["text_gt"], item["img_grid"], item["image_dir_model"], cache_dir
        )
        out.append((idx, result))
    return out


# ---------------------------------------------------------------- evaluator


class TextEvaluator(Evaluator):
    name = "text"

    def run(self, cfg: EvalConfig) -> EvalResult:
        cache_dir = Path(f"tmp_text_{cfg.timestamp}")
        ensure_dir(cache_dir)

        if cfg.mode == "EN":
            text_csv_path = TEXT_CONTENT_EN
            max_edit_distance = 100
        else:
            text_csv_path = TEXT_CONTENT_ZH
            max_edit_distance = 50
        text_df = pd.read_csv(text_csv_path, dtype=str)

        score_csv = pd.DataFrame(
            index=cfg.model_names(), columns=["ED", "CR", "WAC", "text score"]
        )
        prompt_csv = pd.DataFrame(columns=cfg.model_names())

        text_dir = cfg.image_dirname
        if not text_dir.rstrip("/").endswith("/text") and "/text" not in text_dir:
            text_dir = f"{cfg.image_dirname}/text"

        items: list[dict] = []
        row_keys: list[tuple[str, str]] = []
        for entry in cfg.models:
            image_dir_model = f"{text_dir}/{entry.name}"
            for id_, text_gt in zip(text_df["id"], text_df["text_content"]):
                items.append(
                    dict(
                        id=id_,
                        text_gt=text_gt,
                        img_grid=entry.grid_tuple,
                        image_dir_model=image_dir_model,
                    )
                )
                row_keys.append((id_, entry.name))

        print(f"[text] dispatching {len(items)} items across devices {cfg.devices}")
        results = dispatch(
            "oneig_bench.evaluators.text:_run_shard",
            items,
            common={"cache_dir": str(cache_dir)},
            devices=cfg.devices,
            tmp_root=cache_dir,
        )

        per_model: dict[str, dict[str, list]] = {
            entry.name: {"ed": [], "cr": [], "mwc": [], "gwc": []} for entry in cfg.models
        }
        for (id_, model_name), value in zip(row_keys, results):
            if value is None:
                prompt_csv.loc[id_, model_name] = None
                continue
            prompt_csv.loc[id_, model_name] = value["prompt_cell"]
            per_model[model_name]["ed"].extend(value["edit_distances"])
            per_model[model_name]["cr"].extend(value["completion_ratios"])
            per_model[model_name]["mwc"].extend(value["match_word_counts"])
            per_model[model_name]["gwc"].extend(value["gt_word_counts"])

        for model_name, bucket in per_model.items():
            if not bucket["ed"]:
                continue
            ed = sum(bucket["ed"]) / len(bucket["ed"])
            cr = sum(bucket["cr"]) / len(bucket["cr"])
            wac = sum(bucket["mwc"]) / sum(bucket["gwc"]) if sum(bucket["gwc"]) else 0.0
            score_csv.loc[model_name, "ED"] = ed
            score_csv.loc[model_name, "CR"] = cr
            score_csv.loc[model_name, "WAC"] = wac
            score_csv.loc[model_name, "text score"] = (
                1 - min(max_edit_distance, ed) * (1 - cr) * (1 - wac) / max_edit_distance
            )

        if cache_dir.exists():
            shutil.rmtree(cache_dir, onerror=on_rm_error)

        return EvalResult(self.name, score_csv, prompt_csv)
