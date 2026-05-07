#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
TRAFFIC_ROOT="${PROJECT_ROOT}/Satellite-images-for-accident-analysis"
if [[ ! -d "${TRAFFIC_ROOT}/data" || ! -d "${TRAFFIC_ROOT}/embeddings" ]]; then
  TRAFFIC_ROOT="${REPO_ROOT}/Satellite-images-for-accident-analysis"
fi
if [[ ! -d "${TRAFFIC_ROOT}/data" || ! -d "${TRAFFIC_ROOT}/embeddings" ]]; then
  echo "Traffic dataset path is invalid: ${TRAFFIC_ROOT}" >&2
  exit 1
fi

# One-command LDS pipeline on DE:
# 1) train subset models / save LDS payload
# 2) run kernel attribution evaluation on saved LDS payload
#
# Usage:
#   bash src/kernel_surrogate/scripts/run_full_lds_DE.sh [RUN_NAME]

RUN_NAME="${1:-traffic_de_gnnmlp_vit_grid_v2_group15_subset100_data060_mod060}"
DATASET="${DATASET:-traffic_accident}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
FEATURE_DEVICE="${FEATURE_DEVICE:-cuda:0}"
TRAFFIC_STATE="${TRAFFIC_STATE:-DE}"

NUM_GROUPS="${NUM_GROUPS:-15}"
NUM_SUBSET_SAMPLES="${NUM_SUBSET_SAMPLES:-100}"
DATA_KEEP_RATE="${DATA_KEEP_RATE:-0.60}"
MODALITY_KEEP_RATE="${MODALITY_KEEP_RATE:-0.60}"
TRAIN_REPEAT_SEEDS="${TRAIN_REPEAT_SEEDS:-0}"
MAX_TEST_LOSS_SAMPLES="${MAX_TEST_LOSS_SAMPLES:-1000}"

TRAFFIC_VISION_FEATURE_TYPE="${TRAFFIC_VISION_FEATURE_TYPE:-vit_image}"
TRAFFIC_GNN_LAYERS="${TRAFFIC_GNN_LAYERS:-2}"
TRAFFIC_CLASSIFIER_LAYERS="${TRAFFIC_CLASSIFIER_LAYERS:-2}"

EVAL_SEARCH_HYPERPARAMS="${EVAL_SEARCH_HYPERPARAMS:-1}"
EVAL_SEARCH_STRATEGY="${EVAL_SEARCH_STRATEGY:-validation}"
EVAL_TRAIN_FRACTION="${EVAL_TRAIN_FRACTION:-0.7}"
EVAL_VALIDATION_FRACTION="${EVAL_VALIDATION_FRACTION:-0.2}"

RESULT_DIR="${REPO_ROOT}/src/kernel_surrogate/subset_results/lds_real_train/${RUN_NAME}"
OUTPUT_JSON="${REPO_ROOT}/src/kernel_surrogate/results/kernel_attribution_lds_gnn_${RUN_NAME}.json"
OUTPUT_MD="${REPO_ROOT}/src/kernel_surrogate/results/kernel_attribution_lds_gnn_${RUN_NAME}.md"

read -r -a TRAIN_REPEAT_SEEDS_ARR <<< "${TRAIN_REPEAT_SEEDS}"
EVAL_EXTRA_ARGS=()
if [[ "${EVAL_SEARCH_HYPERPARAMS}" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--search-hyperparams --search-strategy "${EVAL_SEARCH_STRATEGY}")
  if [[ "${EVAL_SEARCH_STRATEGY}" == "validation" ]]; then
    EVAL_EXTRA_ARGS+=(--train-fraction "${EVAL_TRAIN_FRACTION}" --validation-fraction "${EVAL_VALIDATION_FRACTION}")
  fi
fi

echo "[1/2] Training LDS subset models on DE..."
conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
python -m src.kernel_surrogate.lds_real_train_gnn \
  --model traffic_pair_gnn_mlp \
  --datasets "${DATASET}" \
  --seeds "${SEED}" \
  --device "${DEVICE}" \
  --feature-device "${FEATURE_DEVICE}" \
  --num-groups "${NUM_GROUPS}" \
  --num-subset-samples "${NUM_SUBSET_SAMPLES}" \
  --data-keep-rate "${DATA_KEEP_RATE}" \
  --modality-keep-rate "${MODALITY_KEEP_RATE}" \
  --train-repeat-seeds "${TRAIN_REPEAT_SEEDS_ARR[@]}" \
  --max-test-loss-samples "${MAX_TEST_LOSS_SAMPLES}" \
  --traffic-state-name "${TRAFFIC_STATE}" \
  --traffic-data-dir "${TRAFFIC_ROOT}/data" \
  --traffic-embeddings-dir "${TRAFFIC_ROOT}/embeddings" \
  --traffic-vision-feature-type "${TRAFFIC_VISION_FEATURE_TYPE}" \
  --traffic-gnn-layers "${TRAFFIC_GNN_LAYERS}" \
  --traffic-classifier-layers "${TRAFFIC_CLASSIFIER_LAYERS}" \
  --run-name "${RUN_NAME}"

if [[ ! -d "${RESULT_DIR}" ]]; then
  echo "Missing LDS result directory after training: ${RESULT_DIR}" >&2
  exit 1
fi

echo "[2/2] Running LDS attribution evaluation..."
conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
python -m src.kernel_surrogate.evaluate_kernel_attribution_gnn \
  --result-dir "${RESULT_DIR}" \
  --datasets "${DATASET}" \
  --seeds "${SEED}" \
  --input-unit group \
  --target-metric test_loss \
  "${EVAL_EXTRA_ARGS[@]}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}"

echo "Done."
echo "LDS payload dir: ${RESULT_DIR}"
echo "Attribution JSON: ${OUTPUT_JSON}"
