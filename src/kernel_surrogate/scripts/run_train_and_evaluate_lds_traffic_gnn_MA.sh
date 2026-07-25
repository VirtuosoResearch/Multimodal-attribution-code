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

# Usage:
#   bash src/kernel_surrogate/scripts/run_train_and_evaluate_lds_traffic_gnn_MA.sh [RUN_PREFIX]
# Optional env overrides:
#   NUM_GROUPS_START, NUM_GROUPS_END, NUM_GROUPS_STEP
#   DATA_KEEP_START, DATA_KEEP_END, DATA_KEEP_STEP
#   MODALITY_KEEP_START, MODALITY_KEEP_END, MODALITY_KEEP_STEP
#   NUM_SUBSET_SAMPLES, TRAIN_REPEAT_SEEDS
#   TRAIN_REPEAT_SEEDS, TRAFFIC_STATE, DEVICE, FEATURE_DEVICE
#   TRAFFIC_GNN_LAYERS, TRAFFIC_CLASSIFIER_LAYERS
#   EVAL_SEARCH_HYPERPARAMS (default: 1), EVAL_SEARCH_STRATEGY (default: validation)
#   EVAL_TRAIN_FRACTION (default: 0.7), EVAL_VALIDATION_FRACTION (default: 0.2)
#   SKIP_EXISTING (default: 1)
RUN_PREFIX="${1:-traffic_ma_gnnmlp_grid_v2}"
NUM_GROUPS_START="${NUM_GROUPS_START:-100}"
NUM_GROUPS_END="${NUM_GROUPS_END:-110}"
NUM_GROUPS_STEP="${NUM_GROUPS_STEP:-1}"
DATA_KEEP_START="${DATA_KEEP_START:-0.08}"
DATA_KEEP_END="${DATA_KEEP_END:-0.3}"
DATA_KEEP_STEP="${DATA_KEEP_STEP:-0.04}"
MODALITY_KEEP_START="${MODALITY_KEEP_START:-0.6}"
MODALITY_KEEP_END="${MODALITY_KEEP_END:-0.9}"
MODALITY_KEEP_STEP="${MODALITY_KEEP_STEP:-0.02}"
NUM_SUBSET_SAMPLES="${NUM_SUBSET_SAMPLES:-100}"
TRAIN_REPEAT_SEEDS="${TRAIN_REPEAT_SEEDS:-0}"
TRAFFIC_STATE="${TRAFFIC_STATE:-MA}"
DEVICE="${DEVICE:-cuda:0}"
FEATURE_DEVICE="${FEATURE_DEVICE:-cuda:0}"
TRAFFIC_GNN_LAYERS="${TRAFFIC_GNN_LAYERS:-2}"
TRAFFIC_CLASSIFIER_LAYERS="${TRAFFIC_CLASSIFIER_LAYERS:-2}"
EVAL_SEARCH_HYPERPARAMS="${EVAL_SEARCH_HYPERPARAMS:-1}"
EVAL_SEARCH_STRATEGY="${EVAL_SEARCH_STRATEGY:-validation}"
EVAL_TRAIN_FRACTION="${EVAL_TRAIN_FRACTION:-0.7}"
EVAL_VALIDATION_FRACTION="${EVAL_VALIDATION_FRACTION:-0.2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -r -a TRAIN_REPEAT_SEEDS_ARR <<< "${TRAIN_REPEAT_SEEDS}"

mapfile -t NUM_GROUP_VALUES < <(
python - "${NUM_GROUPS_START}" "${NUM_GROUPS_END}" "${NUM_GROUPS_STEP}" <<'PY'
import sys
start, end, step = map(int, sys.argv[1:4])
if step <= 0:
    raise SystemExit("NUM_GROUPS_STEP must be positive")
value = start
while value <= end:
    print(value)
    value += step
PY
)

mapfile -t DATA_KEEP_VALUES < <(
python - "${DATA_KEEP_START}" "${DATA_KEEP_END}" "${DATA_KEEP_STEP}" <<'PY'
import sys
start, end, step = map(float, sys.argv[1:4])
if step <= 0:
    raise SystemExit("DATA_KEEP_STEP must be positive")
value = start
while value <= end + 1e-12:
    print(f"{value:.2f}")
    value += step
PY
)

mapfile -t MODALITY_KEEP_VALUES < <(
python - "${MODALITY_KEEP_START}" "${MODALITY_KEEP_END}" "${MODALITY_KEEP_STEP}" <<'PY'
import sys
start, end, step = map(float, sys.argv[1:4])
if step <= 0:
    raise SystemExit("MODALITY_KEEP_STEP must be positive")
value = start
while value <= end + 1e-12:
    print(f"{value:.2f}")
    value += step
PY
)

if [[ "${#NUM_GROUP_VALUES[@]}" -eq 0 || "${#DATA_KEEP_VALUES[@]}" -eq 0 || "${#MODALITY_KEEP_VALUES[@]}" -eq 0 ]]; then
    echo "Grid is empty. Check start/end/step settings." >&2
    exit 1
fi

TOTAL_COMBINATIONS=$(( ${#NUM_GROUP_VALUES[@]} * ${#DATA_KEEP_VALUES[@]} * ${#MODALITY_KEEP_VALUES[@]} ))
echo "Grid search combinations: ${TOTAL_COMBINATIONS}"
echo "num_groups: ${NUM_GROUPS_START}-${NUM_GROUPS_END} step ${NUM_GROUPS_STEP}"
echo "data_keep_rate: ${DATA_KEEP_START}-${DATA_KEEP_END} step ${DATA_KEEP_STEP}"
echo "modality_keep_rate: ${MODALITY_KEEP_START}-${MODALITY_KEEP_END} step ${MODALITY_KEEP_STEP}"

EVAL_EXTRA_ARGS=()
if [[ "${EVAL_SEARCH_HYPERPARAMS}" == "1" ]]; then
    EVAL_EXTRA_ARGS+=(--search-hyperparams --search-strategy "${EVAL_SEARCH_STRATEGY}")
    if [[ "${EVAL_SEARCH_STRATEGY}" == "validation" ]]; then
        EVAL_EXTRA_ARGS+=(--train-fraction "${EVAL_TRAIN_FRACTION}" --validation-fraction "${EVAL_VALIDATION_FRACTION}")
    fi
fi

combo_index=0
for NUM_GROUPS in "${NUM_GROUP_VALUES[@]}"; do
    for DATA_KEEP_RATE in "${DATA_KEEP_VALUES[@]}"; do
        for MODALITY_KEEP_RATE in "${MODALITY_KEEP_VALUES[@]}"; do
            combo_index=$((combo_index + 1))
            DATA_TOKEN="$(printf '%.2f' "${DATA_KEEP_RATE}" | sed 's/\.//')"
            MOD_TOKEN="$(printf '%.2f' "${MODALITY_KEEP_RATE}" | sed 's/\.//')"
            RUN_NAME_SUBSET="${RUN_PREFIX}_group${NUM_GROUPS}_subset${NUM_SUBSET_SAMPLES}_data${DATA_TOKEN}_mod${MOD_TOKEN}_rebuttal"
            RESULT_DIR="${REPO_ROOT}/src/kernel_surrogate/subset_results/lds_real_train/${RUN_NAME_SUBSET}"
            OUTPUT_JSON="${REPO_ROOT}/src/kernel_surrogate/results/kernel_attribution_lds_gnn_${RUN_NAME_SUBSET}.json"
            OUTPUT_MD="${REPO_ROOT}/src/kernel_surrogate/results/kernel_attribution_lds_gnn_${RUN_NAME_SUBSET}.md"

            echo
            echo "[${combo_index}/${TOTAL_COMBINATIONS}] num_groups=${NUM_GROUPS}, data_keep=${DATA_KEEP_RATE}, modality_keep=${MODALITY_KEEP_RATE}"
            echo "run_name=${RUN_NAME_SUBSET}"

            if [[ "${SKIP_EXISTING}" == "1" && -f "${OUTPUT_JSON}" ]]; then
                echo "Skip existing output: ${OUTPUT_JSON}"
                continue
            fi

            echo "[1/2] Training LDS traffic subsets (v2: traffic_pair_gnn_mlp)..."
            conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
            python -m src.kernel_surrogate.lds_real_train_gnn \
                --model traffic_pair_gnn_mlp \
                --datasets traffic_accident \
                --seeds 0 \
                --device "${DEVICE}" \
                --feature-device "${FEATURE_DEVICE}" \
                --num-groups "${NUM_GROUPS}" \
                --num-subset-samples "${NUM_SUBSET_SAMPLES}" \
                --data-keep-rate "${DATA_KEEP_RATE}" \
                --modality-keep-rate "${MODALITY_KEEP_RATE}" \
                --train-repeat-seeds "${TRAIN_REPEAT_SEEDS_ARR[@]}" \
                --max-test-loss-samples 1000 \
                --traffic-state-name "${TRAFFIC_STATE}" \
                --traffic-data-dir "${TRAFFIC_ROOT}/data" \
                --traffic-embeddings-dir "${TRAFFIC_ROOT}/embeddings" \
                --traffic-gnn-layers "${TRAFFIC_GNN_LAYERS}" \
                --traffic-classifier-layers "${TRAFFIC_CLASSIFIER_LAYERS}" \
                --run-name "${RUN_NAME_SUBSET}"

            if [[ ! -d "${RESULT_DIR}" ]]; then
                echo "Missing LDS result directory after training: ${RESULT_DIR}" >&2
                exit 1
            fi

            echo "[2/2] Evaluating LDS kernel attribution (GNN) on traffic..."
            conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
            python -m src.kernel_surrogate.evaluate_kernel_attribution_gnn \
                --result-dir "${RESULT_DIR}" \
                --datasets traffic_accident \
                --seeds 0 \
                --input-unit group \
                --target-metric test_loss \
                "${EVAL_EXTRA_ARGS[@]}" \
                --output-json "${OUTPUT_JSON}" \
                --output-md "${OUTPUT_MD}"
        done
    done
done

echo
echo "Grid run completed."
