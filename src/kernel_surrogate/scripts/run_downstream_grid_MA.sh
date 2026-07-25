#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Usage:
#   bash src/kernel_surrogate/scripts/run_downstream_grid_MA.sh [RUN_NAME]
#
# Default MA run:
#   traffic_ma_gnnmlp_grid_v2_group100_subset100_data008_mod082
#
# This script is intentionally standalone (does NOT call run_downstream_grid.sh).
# You can still override every grid via env vars.

RUN_NAME="${1:-traffic_ma_gnnmlp_grid_v2_group100_subset100_data008_mod082}"
DATASET="${DATASET:-traffic_accident}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
RUN_FULL_BASELINE="${RUN_FULL_BASELINE:-1}"
GRID_PRESET="${GRID_PRESET:-light}"  # light | heavy
CHECK_RUN_NAME_MATCH="${CHECK_RUN_NAME_MATCH:-1}"  # 1 enforces MA run_name pattern

if [[ "${CHECK_RUN_NAME_MATCH}" == "1" ]]; then
  if [[ "${RUN_NAME}" != *"traffic_ma_"* ]]; then
    echo "[ERROR] RUN_NAME does not look like MA: ${RUN_NAME}" >&2
    echo "        set CHECK_RUN_NAME_MATCH=0 to override." >&2
    exit 1
  fi
fi

# Toggle methods.
RUN_STRUCTURED="${RUN_STRUCTURED:-1}"
RUN_TRACIN="${RUN_TRACIN:-1}"
RUN_TRAK="${RUN_TRAK:-1}"
RUN_IF="${RUN_IF:-1}"

if [[ "${GRID_PRESET}" == "heavy" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-0 1}"
  KEEP_VISION_GROUP_RATIOS="${KEEP_VISION_GROUP_RATIOS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7}"

  RIDGE_LAMBDAS="${RIDGE_LAMBDAS:-1e-4 1e-3 1e-2}"
  GAMMA_SCALES="${GAMMA_SCALES:-0.5 1.0 2.0}"
  STRUCTURED_SHARED_WEIGHTS="${STRUCTURED_SHARED_WEIGHTS:-0.0 0.5 1.0}"
  SURROGATE_TRAIN_FRACTIONS="${SURROGATE_TRAIN_FRACTIONS:-1.0}"
  SURROGATE_SPLIT_SEEDS="${SURROGATE_SPLIT_SEEDS:-0 1}"

  TRACIN_PARAM_SCOPES="${TRACIN_PARAM_SCOPES:-classifier fusion}"
  TRACIN_EVAL_BATCH_SIZES="${TRACIN_EVAL_BATCH_SIZES:-256}"
  TRACIN_MAX_TEST_SAMPLES_LIST="${TRACIN_MAX_TEST_SAMPLES_LIST:-4096}"
  TRACIN_MAX_GROUP_SAMPLES_LIST="${TRACIN_MAX_GROUP_SAMPLES_LIST:-1024 2048}"

  TRAK_PARAM_SCOPES="${TRAK_PARAM_SCOPES:-classifier fusion}"
  TRAK_EVAL_BATCH_SIZES="${TRAK_EVAL_BATCH_SIZES:-256}"
  TRAK_MAX_TEST_SAMPLES_LIST="${TRAK_MAX_TEST_SAMPLES_LIST:-4096}"
  TRAK_MAX_GROUP_SAMPLES_LIST="${TRAK_MAX_GROUP_SAMPLES_LIST:-1024 2048}"
  TRAK_PROJ_DIMS="${TRAK_PROJ_DIMS:-1024 2048}"
  TRAK_PROJ_SEEDS="${TRAK_PROJ_SEEDS:-0 1}"
  TRAK_RIDGE_LAMBDAS="${TRAK_RIDGE_LAMBDAS:-1e-2 1e-1}"

  IF_PARAM_SCOPES="${IF_PARAM_SCOPES:-classifier fusion}"
  IF_EVAL_BATCH_SIZES="${IF_EVAL_BATCH_SIZES:-256}"
  IF_MAX_TEST_SAMPLES_LIST="${IF_MAX_TEST_SAMPLES_LIST:-4096}"
  IF_MAX_GROUP_SAMPLES_LIST="${IF_MAX_GROUP_SAMPLES_LIST:-1024 2048}"
  IF_HESSIAN_MAX_SAMPLES_LIST="${IF_HESSIAN_MAX_SAMPLES_LIST:-2048 4096}"
  IF_HESSIAN_BATCH_SIZES="${IF_HESSIAN_BATCH_SIZES:-128}"
  IF_CG_MAX_ITERS="${IF_CG_MAX_ITERS:-20 40}"
  IF_CG_TOLS="${IF_CG_TOLS:-1e-5}"
  IF_HESSIAN_DAMPINGS="${IF_HESSIAN_DAMPINGS:-1e-3 1e-2}"
else
  # light preset
  TRAIN_SEEDS="${TRAIN_SEEDS:-0}"
  KEEP_VISION_GROUP_RATIOS="${KEEP_VISION_GROUP_RATIOS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7}"

  RIDGE_LAMBDAS="${RIDGE_LAMBDAS:-1e-4 1e-3}"
  GAMMA_SCALES="${GAMMA_SCALES:-0.5 1.0}"
  STRUCTURED_SHARED_WEIGHTS="${STRUCTURED_SHARED_WEIGHTS:-0.0 0.5}"
  SURROGATE_TRAIN_FRACTIONS="${SURROGATE_TRAIN_FRACTIONS:-1.0}"
  SURROGATE_SPLIT_SEEDS="${SURROGATE_SPLIT_SEEDS:-0}"

  TRACIN_PARAM_SCOPES="${TRACIN_PARAM_SCOPES:-classifier}"
  TRACIN_EVAL_BATCH_SIZES="${TRACIN_EVAL_BATCH_SIZES:-256}"
  TRACIN_MAX_TEST_SAMPLES_LIST="${TRACIN_MAX_TEST_SAMPLES_LIST:-4096}"
  TRACIN_MAX_GROUP_SAMPLES_LIST="${TRACIN_MAX_GROUP_SAMPLES_LIST:-1024 2048}"

  TRAK_PARAM_SCOPES="${TRAK_PARAM_SCOPES:-classifier}"
  TRAK_EVAL_BATCH_SIZES="${TRAK_EVAL_BATCH_SIZES:-256}"
  TRAK_MAX_TEST_SAMPLES_LIST="${TRAK_MAX_TEST_SAMPLES_LIST:-4096}"
  TRAK_MAX_GROUP_SAMPLES_LIST="${TRAK_MAX_GROUP_SAMPLES_LIST:-1024 2048}"
  TRAK_PROJ_DIMS="${TRAK_PROJ_DIMS:-1024 2048}"
  TRAK_PROJ_SEEDS="${TRAK_PROJ_SEEDS:-0}"
  TRAK_RIDGE_LAMBDAS="${TRAK_RIDGE_LAMBDAS:-1e-2}"

  IF_PARAM_SCOPES="${IF_PARAM_SCOPES:-classifier}"
  IF_EVAL_BATCH_SIZES="${IF_EVAL_BATCH_SIZES:-256}"
  IF_MAX_TEST_SAMPLES_LIST="${IF_MAX_TEST_SAMPLES_LIST:-4096}"
  IF_MAX_GROUP_SAMPLES_LIST="${IF_MAX_GROUP_SAMPLES_LIST:-1024 2048}"
  IF_HESSIAN_MAX_SAMPLES_LIST="${IF_HESSIAN_MAX_SAMPLES_LIST:-2048 4096}"
  IF_HESSIAN_BATCH_SIZES="${IF_HESSIAN_BATCH_SIZES:-128}"
  IF_CG_MAX_ITERS="${IF_CG_MAX_ITERS:-20 40}"
  IF_CG_TOLS="${IF_CG_TOLS:-1e-5}"
  IF_HESSIAN_DAMPINGS="${IF_HESSIAN_DAMPINGS:-1e-2}"
fi

RESULT_DIR="${REPO_ROOT}/src/kernel_surrogate/subset_results/lds_real_train"
RESULTS_DIR="${REPO_ROOT}/src/kernel_surrogate/results"
mkdir -p "${RESULTS_DIR}"

to_words() {
  local raw="$1"
  echo "${raw//,/ }"
}

read -r -a KEEP_LIST <<< "$(to_words "${KEEP_VISION_GROUP_RATIOS}")"
read -r -a TRAIN_SEED_LIST <<< "$(to_words "${TRAIN_SEEDS}")"
read -r -a RIDGE_LIST <<< "$(to_words "${RIDGE_LAMBDAS}")"
read -r -a GAMMA_LIST <<< "$(to_words "${GAMMA_SCALES}")"
read -r -a SHARED_LIST <<< "$(to_words "${STRUCTURED_SHARED_WEIGHTS}")"
read -r -a STRAIN_LIST <<< "$(to_words "${SURROGATE_TRAIN_FRACTIONS}")"
read -r -a SSPLIT_LIST <<< "$(to_words "${SURROGATE_SPLIT_SEEDS}")"

read -r -a TRACIN_SCOPE_LIST <<< "$(to_words "${TRACIN_PARAM_SCOPES}")"
read -r -a TRACIN_EBS_LIST <<< "$(to_words "${TRACIN_EVAL_BATCH_SIZES}")"
read -r -a TRACIN_MTS_LIST <<< "$(to_words "${TRACIN_MAX_TEST_SAMPLES_LIST}")"
read -r -a TRACIN_MGS_LIST <<< "$(to_words "${TRACIN_MAX_GROUP_SAMPLES_LIST}")"

read -r -a TRAK_SCOPE_LIST <<< "$(to_words "${TRAK_PARAM_SCOPES}")"
read -r -a TRAK_EBS_LIST <<< "$(to_words "${TRAK_EVAL_BATCH_SIZES}")"
read -r -a TRAK_MTS_LIST <<< "$(to_words "${TRAK_MAX_TEST_SAMPLES_LIST}")"
read -r -a TRAK_MGS_LIST <<< "$(to_words "${TRAK_MAX_GROUP_SAMPLES_LIST}")"
read -r -a TRAK_PROJ_DIM_LIST <<< "$(to_words "${TRAK_PROJ_DIMS}")"
read -r -a TRAK_PROJ_SEED_LIST <<< "$(to_words "${TRAK_PROJ_SEEDS}")"
read -r -a TRAK_RIDGE_LIST <<< "$(to_words "${TRAK_RIDGE_LAMBDAS}")"

read -r -a IF_SCOPE_LIST <<< "$(to_words "${IF_PARAM_SCOPES}")"
read -r -a IF_EBS_LIST <<< "$(to_words "${IF_EVAL_BATCH_SIZES}")"
read -r -a IF_MTS_LIST <<< "$(to_words "${IF_MAX_TEST_SAMPLES_LIST}")"
read -r -a IF_MGS_LIST <<< "$(to_words "${IF_MAX_GROUP_SAMPLES_LIST}")"
read -r -a IF_HMS_LIST <<< "$(to_words "${IF_HESSIAN_MAX_SAMPLES_LIST}")"
read -r -a IF_HBS_LIST <<< "$(to_words "${IF_HESSIAN_BATCH_SIZES}")"
read -r -a IF_CG_ITERS_LIST <<< "$(to_words "${IF_CG_MAX_ITERS}")"
read -r -a IF_CG_TOL_LIST <<< "$(to_words "${IF_CG_TOLS}")"
read -r -a IF_DAMP_LIST <<< "$(to_words "${IF_HESSIAN_DAMPINGS}")"

EXTRA_ARGS=()
if [[ "${RUN_FULL_BASELINE}" == "1" ]]; then
  EXTRA_ARGS+=(--run-full-baseline)
fi

structured_combo_count=$(( ${#KEEP_LIST[@]} * ${#TRAIN_SEED_LIST[@]} * ${#RIDGE_LIST[@]} * ${#GAMMA_LIST[@]} * ${#SHARED_LIST[@]} * ${#STRAIN_LIST[@]} * ${#SSPLIT_LIST[@]} ))
tracin_combo_count=$(( ${#KEEP_LIST[@]} * ${#TRAIN_SEED_LIST[@]} * ${#TRACIN_SCOPE_LIST[@]} * ${#TRACIN_EBS_LIST[@]} * ${#TRACIN_MTS_LIST[@]} * ${#TRACIN_MGS_LIST[@]} ))
trak_combo_count=$(( ${#KEEP_LIST[@]} * ${#TRAIN_SEED_LIST[@]} * ${#TRAK_SCOPE_LIST[@]} * ${#TRAK_EBS_LIST[@]} * ${#TRAK_MTS_LIST[@]} * ${#TRAK_MGS_LIST[@]} * ${#TRAK_PROJ_DIM_LIST[@]} * ${#TRAK_PROJ_SEED_LIST[@]} * ${#TRAK_RIDGE_LIST[@]} ))
if_combo_count=$(( ${#KEEP_LIST[@]} * ${#TRAIN_SEED_LIST[@]} * ${#IF_SCOPE_LIST[@]} * ${#IF_EBS_LIST[@]} * ${#IF_MTS_LIST[@]} * ${#IF_MGS_LIST[@]} * ${#IF_HMS_LIST[@]} * ${#IF_HBS_LIST[@]} * ${#IF_CG_ITERS_LIST[@]} * ${#IF_CG_TOL_LIST[@]} * ${#IF_DAMP_LIST[@]} ))

echo "[MA-grid] run_name=${RUN_NAME}"
echo "[MA-grid] dataset=${DATASET}, seed=${SEED}, device=${DEVICE}, preset=${GRID_PRESET}"
echo "[MA-grid] structured combos: ${structured_combo_count} (enabled=${RUN_STRUCTURED})"
echo "[MA-grid] tracin combos:    ${tracin_combo_count} (enabled=${RUN_TRACIN})"
echo "[MA-grid] trak combos:      ${trak_combo_count} (enabled=${RUN_TRAK})"
echo "[MA-grid] influence combos: ${if_combo_count} (enabled=${RUN_IF})"

tagify() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  echo "${v}"
}

set -x

for keep in "${KEEP_LIST[@]}"; do
  keep_tag="$(tagify "${keep}")"
  for train_seed in "${TRAIN_SEED_LIST[@]}"; do
    common_tag="${RUN_NAME}_keep${keep_tag}_s${SEED}_ts${train_seed}"

    if [[ "${RUN_STRUCTURED}" == "1" ]]; then
      for ridge in "${RIDGE_LIST[@]}"; do
        for gamma in "${GAMMA_LIST[@]}"; do
          for shared in "${SHARED_LIST[@]}"; do
            for sfrac in "${STRAIN_LIST[@]}"; do
              for ssplit in "${SSPLIT_LIST[@]}"; do
                out_tag="${common_tag}_r$(tagify "${ridge}")_g$(tagify "${gamma}")_sh$(tagify "${shared}")_sf$(tagify "${sfrac}")_ss${ssplit}"
                conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
                python -m src.kernel_surrogate.downstream_structured_rbf_gnn \
                  --result-dir "${RESULT_DIR}" \
                  --run-name "${RUN_NAME}" \
                  --dataset "${DATASET}" \
                  --seed "${SEED}" \
                  --device "${DEVICE}" \
                  --keep-vision-group-ratio "${keep}" \
                  --ridge-lambda "${ridge}" \
                  --gamma-scale "${gamma}" \
                  --structured-shared-weight "${shared}" \
                  --surrogate-train-fraction "${sfrac}" \
                  --surrogate-split-seed "${ssplit}" \
                  --train-seed "${train_seed}" \
                  "${EXTRA_ARGS[@]}" \
                  --output-json "${RESULTS_DIR}/downstream_structured_${out_tag}.json" \
                  --output-md "${RESULTS_DIR}/downstream_structured_${out_tag}.md"
              done
            done
          done
        done
      done
    fi

    if [[ "${RUN_TRACIN}" == "1" ]]; then
      for scope in "${TRACIN_SCOPE_LIST[@]}"; do
        for ebs in "${TRACIN_EBS_LIST[@]}"; do
          for mts in "${TRACIN_MTS_LIST[@]}"; do
            for mgs in "${TRACIN_MGS_LIST[@]}"; do
              out_tag="${common_tag}_sc${scope}_eb${ebs}_mts${mts}_mgs${mgs}"
              conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
              python -m src.kernel_surrogate.downstream_tracin \
                --result-dir "${RESULT_DIR}" \
                --run-name "${RUN_NAME}" \
                --dataset "${DATASET}" \
                --seed "${SEED}" \
                --device "${DEVICE}" \
                --keep-vision-group-ratio "${keep}" \
                --train-seed "${train_seed}" \
                --param-scope "${scope}" \
                --eval-batch-size "${ebs}" \
                --max-test-samples "${mts}" \
                --max-group-samples "${mgs}" \
                "${EXTRA_ARGS[@]}" \
                --output-json "${RESULTS_DIR}/downstream_tracin_${out_tag}.json" \
                --output-md "${RESULTS_DIR}/downstream_tracin_${out_tag}.md"
            done
          done
        done
      done
    fi

    if [[ "${RUN_TRAK}" == "1" ]]; then
      for scope in "${TRAK_SCOPE_LIST[@]}"; do
        for ebs in "${TRAK_EBS_LIST[@]}"; do
          for mts in "${TRAK_MTS_LIST[@]}"; do
            for mgs in "${TRAK_MGS_LIST[@]}"; do
              for pdim in "${TRAK_PROJ_DIM_LIST[@]}"; do
                for pseed in "${TRAK_PROJ_SEED_LIST[@]}"; do
                  for tridge in "${TRAK_RIDGE_LIST[@]}"; do
                    out_tag="${common_tag}_sc${scope}_eb${ebs}_mts${mts}_mgs${mgs}_pd${pdim}_ps${pseed}_tr$(tagify "${tridge}")"
                    conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
                    python -m src.kernel_surrogate.downstream_trak \
                      --result-dir "${RESULT_DIR}" \
                      --run-name "${RUN_NAME}" \
                      --dataset "${DATASET}" \
                      --seed "${SEED}" \
                      --device "${DEVICE}" \
                      --keep-vision-group-ratio "${keep}" \
                      --train-seed "${train_seed}" \
                      --param-scope "${scope}" \
                      --eval-batch-size "${ebs}" \
                      --max-test-samples "${mts}" \
                      --max-group-samples "${mgs}" \
                      --trak-proj-dim "${pdim}" \
                      --trak-proj-seed "${pseed}" \
                      --trak-ridge-lambda "${tridge}" \
                      "${EXTRA_ARGS[@]}" \
                      --output-json "${RESULTS_DIR}/downstream_trak_${out_tag}.json" \
                      --output-md "${RESULTS_DIR}/downstream_trak_${out_tag}.md"
                  done
                done
              done
            done
          done
        done
      done
    fi

    if [[ "${RUN_IF}" == "1" ]]; then
      for scope in "${IF_SCOPE_LIST[@]}"; do
        for ebs in "${IF_EBS_LIST[@]}"; do
          for mts in "${IF_MTS_LIST[@]}"; do
            for mgs in "${IF_MGS_LIST[@]}"; do
              for hms in "${IF_HMS_LIST[@]}"; do
                for hbs in "${IF_HBS_LIST[@]}"; do
                  for cgi in "${IF_CG_ITERS_LIST[@]}"; do
                    for cgt in "${IF_CG_TOL_LIST[@]}"; do
                      for damp in "${IF_DAMP_LIST[@]}"; do
                        out_tag="${common_tag}_sc${scope}_eb${ebs}_mts${mts}_mgs${mgs}_hms${hms}_hbs${hbs}_cgi${cgi}_cgt$(tagify "${cgt}")_d$(tagify "${damp}")"
                        conda run --live-stream -n vlm env PYTHONPATH="${REPO_ROOT}" \
                        python -m src.kernel_surrogate.downstream_influence \
                          --result-dir "${RESULT_DIR}" \
                          --run-name "${RUN_NAME}" \
                          --dataset "${DATASET}" \
                          --seed "${SEED}" \
                          --device "${DEVICE}" \
                          --keep-vision-group-ratio "${keep}" \
                          --train-seed "${train_seed}" \
                          --param-scope "${scope}" \
                          --eval-batch-size "${ebs}" \
                          --max-test-samples "${mts}" \
                          --max-group-samples "${mgs}" \
                          --hessian-max-samples "${hms}" \
                          --hessian-batch-size "${hbs}" \
                          --cg-max-iter "${cgi}" \
                          --cg-tol "${cgt}" \
                          --hessian-damping "${damp}" \
                          "${EXTRA_ARGS[@]}" \
                          --output-json "${RESULTS_DIR}/downstream_if_${out_tag}.json" \
                          --output-md "${RESULTS_DIR}/downstream_if_${out_tag}.md"
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    fi
  done
done

set +x
echo "[MA-grid] completed."
