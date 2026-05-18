#!/bin/bash
set -e
start_time=$(date +%s)
. "$(dirname "$0")/scripts/_common.sh"

MODE=${MODE:-EN}
IMAGE_DIR=${IMAGE_DIR:-""}
MODEL_NAMES=(${MODEL_NAMES[@]:-"doubao"})
IMAGE_GRIDS=(${IMAGE_GRIDS[@]:-"2,2"})
DEVICES=(${DEVICES[@]:-0})

if [ -n "$INSTALL_TRANSFORMERS" ]; then
    pip install transformers==4.50.0
fi

echo "It's style time."
$OIB_PY evaluate style \
    --mode "$MODE" \
    --image-dirname "$IMAGE_DIR/anime" \
    --models "${MODEL_NAMES[@]}" \
    --image-grids "${IMAGE_GRIDS[@]}" \
    --devices "${DEVICES[@]}"

rm -rf tmp_*
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "✅ Style finished in $duration seconds."
