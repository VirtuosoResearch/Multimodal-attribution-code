#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
TRAFFIC_ROOT="${PROJECT_ROOT}/Satellite-images-for-accident-analysis"
if [[ ! -d "${TRAFFIC_ROOT}/data" || ! -d "${TRAFFIC_ROOT}/embeddings" ]]; then
    echo "Traffic dataset path is invalid: ${TRAFFIC_ROOT}" >&2
    exit 1
fi

conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" python -m src.kernel_surrogate.lds_real_train_gnn \
    --datasets traffic_accident \
    --seeds 0 \
    --device cuda:0 \
    --feature-device cuda:0 \
    --num-groups 50 \
    --num-subset-samples 100 \
    --data-keep-rate 0.5 \
    --modality-keep-rate 0.5 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --traffic-state-name DE \
    --traffic-data-dir "${TRAFFIC_ROOT}/data" \
    --traffic-embeddings-dir "${TRAFFIC_ROOT}/embeddings" \
    --run-name traffic_de_grouped50_subset100_data50_mod50

conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" python -m src.kernel_surrogate.lds_real_train_gnn \
    --datasets traffic_accident \
    --seeds 0 \
    --device cuda:0 \
    --feature-device cuda:0 \
    --num-groups 50 \
    --num-subset-samples 1 \
    --data-keep-rate 1.0 \
    --modality-keep-rate 1.0 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --traffic-state-name DE \
    --traffic-data-dir "${TRAFFIC_ROOT}/data" \
    --traffic-embeddings-dir "${TRAFFIC_ROOT}/embeddings" \
    --run-name traffic_de_grouped50_full_data_train
