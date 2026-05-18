"""Unified CLI: ``python -m oneig_bench {run, evaluate, aggregate, analyze}``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oneig_bench.aggregation import aggregate
from oneig_bench.config import (
    ALL_DIMENSIONS,
    apply_cli_overrides,
    EvalConfig,
    load_config,
    validate,
)
from oneig_bench.evaluators import available_evaluators, get_evaluator
from oneig_bench.io import timestamp as make_timestamp


# ------------------------------------------------------------------ helpers


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--mode", choices=["EN", "ZH"], default=None)
    # Both ``--image-dirname`` and the legacy ``--image_dirname`` work.
    p.add_argument(
        "--image-dirname", "--image_dirname", dest="image_dirname", default=None,
        help="Root dir containing <class>/<model>/<id>.webp",
    )
    p.add_argument("--results-dir", "--results_dir", dest="results_dir", default=None)
    p.add_argument(
        "--models", "--model_names", dest="model_names", nargs="+", default=None,
        help="Model name list. Accepts 'name' or 'name:grid' (e.g. gpt-4o:2,2).",
    )
    p.add_argument(
        "--image-grids", "--image_grid", "--image_grids", dest="image_grids",
        nargs="+", default=None, help="Per-model grid, e.g. 2,2 1,4",
    )
    p.add_argument(
        "--devices", dest="devices", nargs="+", default=None,
        help=(
            "GPU device indices for data-parallel evaluation. "
            "Examples: '--devices 0' (single GPU), '--devices 0 1 2 3' "
            "(4-way data parallel), '--devices 0,1,2,3' (same), "
            "'--devices auto' (use every visible CUDA device)."
        ),
    )


def _split_model_grid(models_arg: list[str] | None, image_grids: list[str] | None):
    """Allow ``name:grid`` shorthand in --models."""
    if not models_arg:
        return models_arg, image_grids
    names, inline = [], []
    for tok in models_arg:
        if ":" in tok:
            name, grid = tok.split(":", 1)
            names.append(name)
            inline.append(grid)
        else:
            names.append(tok)
            inline.append(None)
    if any(g is not None for g in inline):
        # Fill missing slots from image_grids or default to 2,2
        fallback = image_grids or []
        merged: list[str] = []
        for i, g in enumerate(inline):
            if g is not None:
                merged.append(g)
            elif i < len(fallback):
                merged.append(fallback[i])
            else:
                merged.append("2,2")
        return names, merged
    return names, image_grids


def _build_cfg(args, *, default_dimensions: list[str] | None = None) -> EvalConfig:
    cfg = load_config(args.config) if args.config else EvalConfig()
    models, grids = _split_model_grid(args.model_names, args.image_grids)
    cfg = apply_cli_overrides(
        cfg,
        mode=args.mode,
        image_dirname=args.image_dirname,
        results_dir=args.results_dir,
        model_names=models,
        image_grids=grids,
        dimensions=getattr(args, "dimensions", None) or default_dimensions,
        class_items=getattr(args, "class_items", None),
        class_items_metric=getattr(args, "class_items_metric", None),
        devices=getattr(args, "devices", None),
    )
    return cfg


# ------------------------------------------------------------------ subcommands


def cmd_run(args) -> int:
    cfg = _build_cfg(args)
    if not cfg.timestamp:
        cfg.timestamp = make_timestamp()
    validate(cfg)
    print(
        f"[oneig-bench] run mode={cfg.mode} models={cfg.model_names()} "
        f"dimensions={cfg.dimensions} devices={cfg.devices} timestamp={cfg.timestamp}"
    )
    for dim in cfg.dimensions:
        print(f"\n=== {dim} ===")
        evaluator = get_evaluator(dim)
        result = evaluator.run(cfg)
        result.save(cfg)
    print("\n=== aggregate ===")
    aggregate(
        cfg.results_dir,
        cfg.mode,
        cfg.timestamp,
        emit_csv=cfg.aggregation.emit_csv,
        emit_xlsx=cfg.aggregation.emit_xlsx,
        include_fine_grained=cfg.aggregation.include_fine_grained,
    )
    return 0


def cmd_evaluate(args) -> int:
    metric = args.metric
    if metric not in ALL_DIMENSIONS:
        print(f"Unknown metric: {metric}. Available: {list(ALL_DIMENSIONS)}", file=sys.stderr)
        return 2
    cfg = _build_cfg(args, default_dimensions=[metric])
    cfg.dimensions = [metric]
    if not cfg.timestamp:
        cfg.timestamp = make_timestamp()
    if args.class_items:
        cfg.class_items[metric] = list(args.class_items)
    validate(cfg)
    evaluator = get_evaluator(metric)
    result = evaluator.run(cfg)
    result.save(cfg)
    return 0


def cmd_aggregate(args) -> int:
    results_dir = args.results_dir or "results"
    mode = args.mode or "EN"
    aggregate(
        results_dir,
        mode,
        args.timestamp,
        emit_csv=not args.no_csv,
        emit_xlsx=not args.no_xlsx,
        include_fine_grained=not args.no_fine_grained,
    )
    return 0


def cmd_analyze(args) -> int:
    """Legacy alias of ``aggregate`` for fine_grained_analysis.py users."""
    return cmd_aggregate(args)


# ------------------------------------------------------------------ entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oneig-bench", description="OneIG-Bench evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # run: all dimensions + aggregate
    p_run = sub.add_parser("run", help="Run all configured dimensions and aggregate")
    _add_common_flags(p_run)
    p_run.add_argument("--dimensions", nargs="+", default=None,
                       help=f"Subset of {list(ALL_DIMENSIONS)} (default: all in config)")
    p_run.add_argument("--class-items", "--class_items", dest="class_items", nargs="+",
                       default=None)
    p_run.set_defaults(class_items_metric=None, func=cmd_run)

    # evaluate <metric>: single dimension
    p_eval = sub.add_parser("evaluate", help="Run a single evaluator")
    p_eval.add_argument("metric", choices=available_evaluators())
    _add_common_flags(p_eval)
    p_eval.add_argument("--class-items", "--class_items", dest="class_items", nargs="+",
                        default=None)
    p_eval.set_defaults(class_items_metric=None, func=cmd_evaluate)

    # aggregate: combine existing per-dimension CSVs into one summary table
    p_agg = sub.add_parser("aggregate", help="Aggregate per-dimension CSVs into a summary")
    p_agg.add_argument("--config", type=str, default=None)
    p_agg.add_argument("--mode", choices=["EN", "ZH"], default=None)
    p_agg.add_argument("--results-dir", "--results_dir", dest="results_dir", default=None)
    p_agg.add_argument("--timestamp", type=str, default=None,
                       help="Run timestamp suffix (default: latest in results_dir)")
    p_agg.add_argument("--no-csv", action="store_true")
    p_agg.add_argument("--no-xlsx", action="store_true")
    p_agg.add_argument("--no-fine-grained", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    # analyze: alias of aggregate
    p_an = sub.add_parser("analyze", help="Alias of aggregate (fine-grained analysis)")
    p_an.add_argument("--config", type=str, default=None)
    p_an.add_argument("--mode", choices=["EN", "ZH"], default=None)
    p_an.add_argument("--results-dir", "--results_dir", dest="results_dir", default=None)
    p_an.add_argument("--timestamp", type=str, default=None)
    p_an.add_argument("--no-csv", action="store_true")
    p_an.add_argument("--no-xlsx", action="store_true")
    p_an.add_argument("--no-fine-grained", action="store_true")
    p_an.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Ensure ``src/`` is importable when invoked without an install.
    _ensure_src_on_path()
    args = parser.parse_args(argv)
    return args.func(args)


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_dir = here.parents[1]
    repo_root = src_dir.parent
    for p in (str(src_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


if __name__ == "__main__":
    raise SystemExit(main())
