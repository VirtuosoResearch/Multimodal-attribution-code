#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASETS=(
  imdb
  humor
  sarcasm
  mosi
  mosei
  avmnist
)

DEVICE="${DEVICE:-cuda:0}"
FEATURE_DEVICE="${FEATURE_DEVICE:-${DEVICE}}"
BATCH_SIZE="${BATCH_SIZE:-4196}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
DATA_KEEP_RATE="${DATA_KEEP_RATE:-0.25}"
MODALITY_KEEP_RATE="${MODALITY_KEEP_RATE:-0.9}"
TRAIN_REPEAT_SEEDS="${TRAIN_REPEAT_SEEDS:-0 1 2}"
MAX_TEST_LOSS_SAMPLES="${MAX_TEST_LOSS_SAMPLES:-1000}"
read -r -a TRAIN_REPEAT_SEED_ARGS <<< "${TRAIN_REPEAT_SEEDS}"

conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
  --datasets "${DATASETS[@]}" \
  --device "${DEVICE}" \
  --feature-device "${FEATURE_DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --data-keep-rate "${DATA_KEEP_RATE}" \
  --modality-keep-rate "${MODALITY_KEEP_RATE}" \
  --train-repeat-seeds "${TRAIN_REPEAT_SEED_ARGS[@]}" \
  --max-test-loss-samples "${MAX_TEST_LOSS_SAMPLES}" \
  "$@"
