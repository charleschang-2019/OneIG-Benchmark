"""Evaluation run configuration (YAML loader + CLI overrides)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALL_DIMENSIONS = ("alignment", "diversity", "reasoning", "style", "text")

# Per-dimension default ``class_items``. ZH mode appends ``multilingualism``
# where applicable.
DEFAULT_CLASS_ITEMS: dict[str, dict[str, list[str]]] = {
    "alignment": {
        "EN": ["anime", "human", "object"],
        "ZH": ["anime", "human", "object", "multilingualism"],
    },
    "diversity": {
        "EN": ["anime", "human", "object", "text", "reasoning"],
        "ZH": ["anime", "human", "object", "text", "reasoning", "multilingualism"],
    },
}


@dataclass
class ModelEntry:
    name: str
    image_grid: str = "2,2"

    @property
    def grid_tuple(self) -> tuple[int, int]:
        a, _, b = self.image_grid.partition(",")
        if not b:  # accept "2" as shorthand for "2,2"
            b = a
        return int(a), int(b)


@dataclass
class AggregationConfig:
    emit_csv: bool = True
    emit_xlsx: bool = True
    include_fine_grained: bool = True


@dataclass
class EvalConfig:
    mode: str = "EN"
    image_dirname: str = ""
    results_dir: str = "results"
    models: list[ModelEntry] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=lambda: list(ALL_DIMENSIONS))
    class_items: dict[str, list[str]] = field(default_factory=dict)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    # Set lazily by ``cli.run`` so every evaluator in one invocation writes
    # files that share a timestamp suffix (needed by the aggregator).
    timestamp: str = ""
    # GPU device indices to use. ``[0]`` = single GPU; ``[0,1,2,3]`` = data
    # parallel across 4 GPUs. Empty list defaults to ``[0]`` at validation
    # time. Strings like ``"auto"`` are expanded to ``list(range(N))``
    # against ``torch.cuda.device_count()`` if available.
    devices: list[int] = field(default_factory=list)

    def class_items_for(self, metric: str) -> list[str]:
        if metric in self.class_items and self.class_items[metric]:
            return list(self.class_items[metric])
        return list(DEFAULT_CLASS_ITEMS.get(metric, {}).get(self.mode, []))

    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def image_grids(self) -> list[str]:
        return [m.image_grid for m in self.models]


def _coerce_devices(raw: Any) -> list[int]:
    """Normalise a devices spec into a list of int GPU indices.

    Accepts ``None``, ``"auto"``, an int, a comma-separated string,
    a list of ints/strings, etc. ``"auto"`` expands to ``list(range(N))``
    via ``torch.cuda.device_count()`` if CUDA is available, else ``[0]``.
    """
    if raw is None or raw == "" or raw == []:
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() == "auto":
            try:
                import torch

                n = torch.cuda.device_count()
            except Exception:
                n = 1
            return list(range(max(n, 1)))
        if "," in s:
            return [int(x) for x in s.split(",") if x.strip()]
        return [int(s)]
    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for item in raw:
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                out.extend(_coerce_devices(item))
            else:
                raise ValueError(f"Bad device entry: {item!r}")
        return out
    raise ValueError(f"Bad devices value: {raw!r}")


def _coerce_models(raw: Any) -> list[ModelEntry]:
    if raw is None:
        return []
    out: list[ModelEntry] = []
    for item in raw:
        if isinstance(item, str):
            out.append(ModelEntry(name=item))
        elif isinstance(item, dict):
            out.append(ModelEntry(name=item["name"], image_grid=str(item.get("image_grid", "2,2"))))
        else:
            raise ValueError(f"Bad model entry: {item!r}")
    return out


def load_config(path: str | Path | None) -> EvalConfig:
    """Load a YAML config file. ``None`` returns an empty config."""
    if path is None:
        return EvalConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}

    cfg = EvalConfig(
        mode=data.get("mode", "EN"),
        image_dirname=data.get("image_dirname", ""),
        results_dir=data.get("results_dir", "results"),
        models=_coerce_models(data.get("models", [])),
        dimensions=list(data.get("dimensions", ALL_DIMENSIONS)),
        class_items={k: list(v) for k, v in (data.get("class_items") or {}).items()},
        devices=_coerce_devices(data.get("devices")),
    )
    # Allow per-dimension class_items at the top level too, e.g.
    #   alignment: { class_items: [...] }
    for metric in ALL_DIMENSIONS:
        block = data.get(metric)
        if isinstance(block, dict) and "class_items" in block:
            cfg.class_items[metric] = list(block["class_items"])

    agg = data.get("aggregation") or {}
    cfg.aggregation = AggregationConfig(
        emit_csv=bool(agg.get("emit_csv", True)),
        emit_xlsx=bool(agg.get("emit_xlsx", True)),
        include_fine_grained=bool(agg.get("include_fine_grained", True)),
    )
    return cfg


def apply_cli_overrides(
    cfg: EvalConfig,
    *,
    mode: str | None = None,
    image_dirname: str | None = None,
    results_dir: str | None = None,
    model_names: list[str] | None = None,
    image_grids: list[str] | None = None,
    dimensions: list[str] | None = None,
    class_items: list[str] | None = None,
    class_items_metric: str | None = None,
    devices: Any = None,
) -> EvalConfig:
    """Apply CLI flags on top of the YAML-loaded config.

    Empty/None inputs are ignored. ``class_items`` only overrides the
    specified metric (``class_items_metric``); when no metric is given it
    overrides all metrics that consume ``class_items``.
    """
    if mode:
        cfg.mode = mode
    if image_dirname is not None and image_dirname != "":
        cfg.image_dirname = image_dirname
    if results_dir:
        cfg.results_dir = results_dir
    if model_names:
        if image_grids and len(image_grids) == len(model_names):
            cfg.models = [ModelEntry(n, g) for n, g in zip(model_names, image_grids)]
        elif image_grids and len(image_grids) == 1:
            cfg.models = [ModelEntry(n, image_grids[0]) for n in model_names]
        else:
            cfg.models = [ModelEntry(n) for n in model_names]
    elif image_grids and cfg.models:
        if len(image_grids) == len(cfg.models):
            for m, g in zip(cfg.models, image_grids):
                m.image_grid = g
        elif len(image_grids) == 1:
            for m in cfg.models:
                m.image_grid = image_grids[0]
    if dimensions:
        cfg.dimensions = list(dimensions)
    if class_items:
        if class_items_metric:
            cfg.class_items[class_items_metric] = list(class_items)
        else:
            for metric in DEFAULT_CLASS_ITEMS:
                cfg.class_items[metric] = list(class_items)
    if devices is not None:
        coerced = _coerce_devices(devices)
        if coerced:
            cfg.devices = coerced
    return cfg


def validate(cfg: EvalConfig) -> None:
    if cfg.mode not in ("EN", "ZH"):
        raise ValueError(f"mode must be EN or ZH, got {cfg.mode!r}")
    if not cfg.models:
        raise ValueError("No models configured (set --models or models: in YAML)")
    if not cfg.image_dirname:
        raise ValueError("image_dirname is empty (set --image-dirname or image_dirname: in YAML)")
    bad = [d for d in cfg.dimensions if d not in ALL_DIMENSIONS]
    if bad:
        raise ValueError(f"Unknown dimensions: {bad}. Valid: {list(ALL_DIMENSIONS)}")
    if not cfg.devices:
        cfg.devices = [0]
