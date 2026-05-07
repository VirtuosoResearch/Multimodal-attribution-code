#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_attribution.sh avmnist

DATASET="${1:-avmnist}"

# Edit constants here.
SEED=0
NUM_GROUPS=25
GROUP_METHOD="kmeans"
FULL_MODALITY="x"

RUN_NAME="${DATASET}"
LDS_SEED=0
RHO=0.5
RIDGE_LAMBDA=1e-3
GAMMA_SCALE=1.0
TRAIN_FRACTION=1.0
TOP_K=20

GROUP_OUTPUT="src/attribution/results/groups/${DATASET}_seed${SEED}.json"
LDS_INPUT="src/attribution/results/lds_real_train/${RUN_NAME}/${DATASET}_seed${LDS_SEED}.json"
SKS_OUTPUT="src/attribution/results/sks_scores/${DATASET}_seed${LDS_SEED}.json"

python -m src.attribution.sample_groups \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --num-groups "${NUM_GROUPS}" \
  --group-method "${GROUP_METHOD}" \
  --full-modality "${FULL_MODALITY}" \
  --output "${GROUP_OUTPUT}"

python -m src.attribution.sks_attribution \
  --input "${LDS_INPUT}" \
  --output "${SKS_OUTPUT}" \
  --rho "${RHO}" \
  --ridge-lambda "${RIDGE_LAMBDA}" \
  --gamma-scale "${GAMMA_SCALE}" \
  --train-fraction "${TRAIN_FRACTION}" \
  --seed "${SEED}" \
  --top-k "${TOP_K}"

echo "Attribution output: ${SKS_OUTPUT}"
