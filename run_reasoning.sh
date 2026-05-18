#!/bin/bash
set -e
start_time=$(date +%s)
. "$(dirname "$0")/scripts/_common.sh"

MODE=${MODE:-EN}
IMAGE_DIR=${IMAGE_DIR:-""}
MODEL_NAMES=(${MODEL_NAMES[@]:-"gpt-4o" "imagen4"})
IMAGE_GRIDS=(${IMAGE_GRIDS[@]:-"2,2" "1,4"})
DEVICES=(${DEVICES[@]:-0})

if [ -n "$INSTALL_TRANSFORMERS" ]; then
    pip install transformers==4.50.0
fi

echo "It's reasoning time."
$OIB_PY evaluate reasoning \
    --mode "$MODE" \
    --image-dirname "${IMAGE_DIR}/reasoning" \
    --models "${MODEL_NAMES[@]}" \
    --image-grids "${IMAGE_GRIDS[@]}" \
    --devices "${DEVICES[@]}"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "✅ Reasoning finished in $duration seconds."
