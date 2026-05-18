#!/bin/bash
# Run every configured evaluation dimension and emit a consolidated summary
# table. The Python entry point picks a single timestamp and reuses it
# across dimensions so the aggregator can match them up afterwards.

set -e
start_time=$(date +%s)

. "$(dirname "$0")/scripts/_common.sh"

# mode (EN/ZH)
MODE=${MODE:-EN}

# image_root_dir — the folder containing <class>/<model>/<id>.webp
IMAGE_DIR=${IMAGE_DIR:-""}

# model list and per-model image grids (1-to-1 with MODEL_NAMES).
MODEL_NAMES=(${MODEL_NAMES[@]:-"janus-pro"})
IMAGE_GRIDS=(${IMAGE_GRIDS[@]:-"2,2"})

# GPUs (one int per device). Examples:
#   DEVICES=(0)                  # single GPU
#   DEVICES=(0 1 2 3)            # 4-way data parallel
#   DEVICES=(auto)               # every visible CUDA device
DEVICES=(${DEVICES[@]:-0})

echo "Running all evaluation scripts"

if [ -n "$INSTALL_TRANSFORMERS" ]; then
    pip install transformers==4.50.0
fi

$OIB_PY run \
    --mode "$MODE" \
    --image-dirname "$IMAGE_DIR" \
    --models "${MODEL_NAMES[@]}" \
    --image-grids "${IMAGE_GRIDS[@]}" \
    --devices "${DEVICES[@]}"

rm -rf tmp_*

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "✅ All evaluations finished in $duration seconds."
