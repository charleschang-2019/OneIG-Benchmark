#!/bin/bash
# Common environment for OneIG-Bench run_*.sh scripts.
#
# Source this with `. scripts/_common.sh` from a run script to get the
# `OIB_PY` env var pointing to the project root and to ensure the
# `oneig_bench` package on `src/` is importable without installation.

OIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$OIB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
OIB_PY="python -m oneig_bench"
export OIB_PY
