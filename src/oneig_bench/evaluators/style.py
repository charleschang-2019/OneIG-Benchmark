"""Style evaluator (CSD + Style Encoder reference matching)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import megfile
import pandas as pd

from oneig_bench.config import EvalConfig
from oneig_bench.data.paths import STYLE_CSD_EMBED, STYLE_CSV, STYLE_SE_EMBED
from oneig_bench.evaluators.base import Evaluator, EvalResult
from oneig_bench.image_utils import split_mxn_grid
from oneig_bench.io import ensure_dir, on_rm_error
from oneig_bench.parallel import dispatch

STYLE_LIST = [
    "abstract_expressionism", "art_nouveau", "baroque", "chinese_ink_painting", "cubism",
    "fauvism", "impressionism", "line_art", "minimalism", "pointillism", "pop_art", "rococo",
    "ukiyo-e", "clay", "crayon", "graffiti", "lego", "comic", "pencil_sketch",
    "stone_sculpture", "watercolor", "celluloid", "chibi", "cyberpunk", "ghibli", "impasto",
    "pixar", "pixel_art", "3d_rendering",
]


# ---------------------------------------------------------------- worker side


def _score_one(img_path: str, img_grid, image_style: str, common, cache_dir):
    import torch
    from oneig_bench.models import get_csd_encoder, get_se_encoder

    split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
    if not split_img_list:
        return None
    csd_encoder = get_csd_encoder()
    se_encoder = get_se_encoder()
    csd_ref_embeds = common["csd_ref"][image_style]
    se_ref_embeds = common["se_ref"][image_style]

    scores: list[float] = []
    for sp in split_img_list:
        csd_embed = csd_encoder.get_style_embedding(sp)
        se_embed = se_encoder.get_style_embedding(sp)
        csd_max = max(torch.max(csd_embed @ csd_ref_embeds.to(csd_embed.device).T).item(), 0)
        se_max = max(torch.max(se_embed @ se_ref_embeds.to(se_embed.device).T).item(), 0)
        scores.append((csd_max + se_max) / 2)
    return sum(scores) / len(scores) if scores else None


def _run_shard(shard, common):
    cache_dir = Path(common["cache_dir"]) / f"rank_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[int, object]] = []
    for idx, item in shard:
        result = _score_one(
            item["img_path"], item["img_grid"], item["image_style"], common, cache_dir
        )
        out.append((idx, result))
    return out


# ---------------------------------------------------------------- evaluator


class StyleEvaluator(Evaluator):
    name = "style"

    def run(self, cfg: EvalConfig) -> EvalResult:
        import torch

        cache_dir = Path(f"tmp_style_{cfg.timestamp}")
        ensure_dir(cache_dir)

        torch.cuda.empty_cache()
        df = pd.read_csv(STYLE_CSV, dtype=str)
        csd_ref = torch.load(str(STYLE_CSD_EMBED), weights_only=False, map_location="cpu")
        se_ref = torch.load(str(STYLE_SE_EMBED), map_location="cpu")

        score_csv = pd.DataFrame(index=cfg.model_names(), columns=["style"])
        style_csv = pd.DataFrame(index=cfg.model_names(), columns=STYLE_LIST)
        prompt_csv = pd.DataFrame(columns=cfg.model_names())

        anime_dir = cfg.image_dirname
        if not anime_dir.rstrip("/").endswith("/anime") and "/anime" not in anime_dir:
            anime_dir = f"{cfg.image_dirname}/anime"

        items: list[dict] = []
        row_keys: list[tuple[str, str, str]] = []  # (img_id, model_name, image_style)
        for entry in cfg.models:
            image_dir = f"{anime_dir}/{entry.name}"
            img_list = sorted(megfile.smart_glob(image_dir + "/*"))
            for img_path in img_list:
                img_id = Path(img_path).name[:3]
                image_style = str(df.loc[df["id"] == img_id, "class"].values[0])
                if image_style[:3] == "nan":
                    continue
                image_style = image_style.lower().replace(" ", "_")
                items.append(
                    dict(
                        img_path=img_path,
                        img_grid=entry.grid_tuple,
                        image_style=image_style,
                    )
                )
                row_keys.append((img_id, entry.name, image_style))

        print(f"[style] dispatching {len(items)} items across devices {cfg.devices}")
        results = dispatch(
            "oneig_bench.evaluators.style:_run_shard",
            items,
            common={"cache_dir": str(cache_dir), "csd_ref": csd_ref, "se_ref": se_ref},
            devices=cfg.devices,
            tmp_root=cache_dir,
        )

        # Per-style bucket so we can compute the style_style_score table.
        style_dict: dict[str, dict[str, list[float]]] = {
            entry.name: {s: [] for s in STYLE_LIST} for entry in cfg.models
        }
        for (img_id, model_name, image_style), value in zip(row_keys, results):
            if value is None:
                continue
            prompt_csv.loc[img_id, model_name] = value
            if image_style in style_dict[model_name]:
                style_dict[model_name][image_style].append(value)

        for entry in cfg.models:
            for style in STYLE_LIST:
                vals = style_dict[entry.name][style]
                if vals:
                    style_csv.loc[entry.name, style] = sum(vals) / len(vals)

        score_csv["style"] = prompt_csv.mean().values
        prompt_csv = prompt_csv.sort_index()

        if cache_dir.exists():
            shutil.rmtree(cache_dir, onerror=on_rm_error)

        return EvalResult(self.name, score_csv, prompt_csv, extra={"style_score": style_csv})
