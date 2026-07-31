#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# One-command downstream evaluation on DE using an existing LDS run_name.
# Default method here is structured RBF graph selection:
# keep 30% graph groups, keep all vision data, retrain, then report test metrics.

RUN_NAME="${1:-traffic_de_gnnmlp_vit_grid_v2_group15_subset100_data060_mod060}"
DATASET="${DATASET:-traffic_accident}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_SEED="${TRAIN_SEED:-0}"

KEEP_GRAPH_GROUP_RATIO="${KEEP_GRAPH_GROUP_RATIO:-0.30}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
GAMMA_SCALE="${GAMMA_SCALE:-1.0}"
STRUCTURED_SHARED_WEIGHT="${STRUCTURED_SHARED_WEIGHT:-0.0}"
SURROGATE_TRAIN_FRACTION="${SURROGATE_TRAIN_FRACTION:-1.0}"
SURROGATE_SPLIT_SEED="${SURROGATE_SPLIT_SEED:-0}"
RUN_FULL_BASELINE="${RUN_FULL_BASELINE:-1}"

RESULT_DIR="${REPO_ROOT}/src/mmtrace/subset_results/selection"
RESULTS_OUT_DIR="${REPO_ROOT}/src/mmtrace/results"
mkdir -p "${RESULTS_OUT_DIR}"

if [[ ! -d "${RESULT_DIR}/${RUN_NAME}" ]]; then
  echo "Missing selection result run directory: ${RESULT_DIR}/${RUN_NAME}" >&2
  exit 1
fi

keep_tag="$(printf '%.2f' "${KEEP_GRAPH_GROUP_RATIO}" | sed 's/\.//')"
OUT_TAG="${RUN_NAME}_keepg${keep_tag}_s${SEED}_ts${TRAIN_SEED}"
OUTPUT_JSON="${RESULTS_OUT_DIR}/selection_${OUT_TAG}.json"
OUTPUT_MD="${RESULTS_OUT_DIR}/selection_${OUT_TAG}.md"

EXTRA_ARGS=()
if [[ "${RUN_FULL_BASELINE}" == "1" ]]; then
  EXTRA_ARGS+=(--run-full-baseline)
fi

conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
python -m src.mmtrace.selection \
  --result-dir "${RESULT_DIR}" \
  --run-name "${RUN_NAME}" \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --keep-graph-group-ratio "${KEEP_GRAPH_GROUP_RATIO}" \
  --ridge-lambda "${RIDGE_LAMBDA}" \
  --gamma-scale "${GAMMA_SCALE}" \
  --structured-shared-weight "${STRUCTURED_SHARED_WEIGHT}" \
  --surrogate-train-fraction "${SURROGATE_TRAIN_FRACTION}" \
  --surrogate-split-seed "${SURROGATE_SPLIT_SEED}" \
  --train-seed "${TRAIN_SEED}" \
  "${EXTRA_ARGS[@]}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}"

echo "Done."
echo "Selection JSON: ${OUTPUT_JSON}"
