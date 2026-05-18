"""Fine-grained analysis entry point (legacy shim).

The actual logic lives in ``oneig_bench.aggregation`` now. This file is kept
so older docs / muscle memory keep working:

    python fine_grained_analysis.py            # aggregate latest EN run
    python fine_grained_analysis.py --mode ZH  # aggregate latest ZH run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable without installation.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oneig_bench.aggregation import aggregate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["EN", "ZH"], default="EN")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--timestamp", default=None,
                        help="Run timestamp suffix (default: latest in results-dir)")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-xlsx", action="store_true")
    parser.add_argument("--no-fine-grained", action="store_true")
    args = parser.parse_args()

    aggregate(
        args.results_dir,
        args.mode,
        args.timestamp,
        emit_csv=not args.no_csv,
        emit_xlsx=not args.no_xlsx,
        include_fine_grained=not args.no_fine_grained,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
