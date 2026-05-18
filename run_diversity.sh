#!/bin/bash
set -e
start_time=$(date +%s)
. "$(dirname "$0")/scripts/_common.sh"

MODE=${MODE:-EN}
IMAGE_DIR=${IMAGE_DIR:-""}
MODEL_NAMES=(${MODEL_NAMES[@]:-"gpt-4o"})
IMAGE_GRIDS=(${IMAGE_GRIDS[@]:-"2,2"})
DEVICES=(${DEVICES[@]:-0})

if [ -n "$INSTALL_TRANSFORMERS" ]; then
    pip install transformers==4.50.0
fi

echo "It's diversity time."
$OIB_PY evaluate diversity \
    --mode "$MODE" \
    --image-dirname "$IMAGE_DIR" \
    --models "${MODEL_NAMES[@]}" \
    --image-grids "${IMAGE_GRIDS[@]}" \
    --devices "${DEVICES[@]}" \
    --class-items anime human object text reasoning

rm -rf tmp_*
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "✅ Diversity finished in $duration seconds."
