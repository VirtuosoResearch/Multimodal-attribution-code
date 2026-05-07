#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_selection.sh avmnist

DATASET="${1:-avmnist}"

# Edit constants here.
RUN_NAME="${DATASET}"
LDS_SEED=0
SELECTION_SEED=1

BUDGET=0.2
MAIN_MODALITY="x"
SCORE_KEY="modality_group_scores.y"
DIRECTION="largest"

MODEL="concat_mlp"
DEVICE="cuda:0"
EPOCHS=30
BATCH_SIZE=256
LR=5e-4

SCORE_FILE="src/attribution/results/sks_scores/${DATASET}_seed${LDS_SEED}.json"
GROUP_PAYLOAD="src/attribution/results/lds_real_train/${RUN_NAME}/${DATASET}_seed${LDS_SEED}.json"
INDEX_OUTPUT="src/attribution/results/dataselection/sks/${DATASET}/seed${SELECTION_SEED}.json"
OUTPUT_DIR="src/MultiBench_selection/results/sks_budget"

python -m src.attribution.score_to_dataselection \
  --score-file "${SCORE_FILE}" \
  --group-payload "${GROUP_PAYLOAD}" \
  --score-key "${SCORE_KEY}" \
  --direction "${DIRECTION}" \
  --budget "${BUDGET}" \
  --seed "${SELECTION_SEED}" \
  --output "${INDEX_OUTPUT}"

python -m src.MultiBench_selection.experiment \
  --datasets "${DATASET}" \
  --seeds "${SELECTION_SEED}" \
  --model "${MODEL}" \
  --selection-method index \
  --index-file "${INDEX_OUTPUT}" \
  --budget "${BUDGET}" \
  --main-modality "${MAIN_MODALITY}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --output-dir "${OUTPUT_DIR}"

echo "Selection index: ${INDEX_OUTPUT}"
echo "Training output dir: ${OUTPUT_DIR}"
