#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.

# 1. Train aligned Gentle Push attribution subsets.
conda run -n vlm python -u -B -m src.robotic.gentle_push.lds_lf \
  --device cuda:1 \
  --seeds 0 \
  --intervention auxiliary_acquisition \
  --main-modality control \
  --aux-mask-granularity group \
  --group-method kmeans \
  --num-groups 25 \
  --partition-seed 0 \
  --data-keep-rate 0.4 \
  --modality-keep-rate 0.5 \
  --num-subset-samples 100 \
  --train-repeat-seeds 0 1 2 \
  --epochs 20 \
  --lr 1e-5 \
  --batch-size 32 \
  --eval-batch-size 32 \
  --parallel-workers 1 \
  --max-test-loss-samples 1000 \
  --run-name gentle_push_lf_group_unit_kmeans_auxacq40_mod50_group_main-control_g25_trainavg3

# 2. Compare structured RBF and raw-flat RBF LDS.
conda run -n vlm python -u -B -m src.kernel_surrogate.evaluate_kernel_attribution \
  --result-dir src/kernel_surrogate/subset_results/lds_real_train/gentle_push_lf_group_unit_kmeans_auxacq40_mod50_group_main-control_g25_trainavg3 \
  --datasets gentle_push_lf \
  --seeds 0 \
  --target-metric test_loss \
  --input-unit group \
  --methods kernel_surrogate_1_structured_rbf raw_flat_rbf \
  --train-fraction 0.6 \
  --validation-fraction 0.2 \
  --search-hyperparams \
  --search-strategy validation \
  --ridge-grid 1e-5 1e-4 1e-3 1e-2 \
  --gamma-scale-grid 0.25 0.5 1 2 4 \
  --modality-shared-grid 0.25 0.5 1 2 4 \
  --output-json src/kernel_surrogate/results/gentle_push_auxacq_lds.json \
  --output-md src/kernel_surrogate/results/gentle_push_auxacq_lds.md

# 3. Generate equal-budget structured, raw-flat, and random selection indices.
conda run -n vlm python -u -B -m src.kernel_surrogate.gentle_push_data_selection \
  --run-name gentle_push_lf_group_unit_kmeans_auxacq40_mod50_group_main-control_g25_trainavg3 \
  --surrogate-seed 0 \
  --selection-seed 1 \
  --score-group-fraction 0.2 \
  --selection-budget 0.2 \
  --top-k 5 \
  --num-score-samples 20000 \
  --prediction-batch-size 5000 \
  --search-hyperparams \
  --ridge-grid 1e-5 1e-4 1e-3 1e-2 \
  --gamma-scale-grid 0.25 0.5 1 2 4 \
  --modality-shared-grid 0.25 0.5 1 2 4 \
  --output-root src/kernel_surrogate/results/gentle_push_auxacq_data_selection_indices

# 4. Retrain the downstream data-selection arms.
conda run -n vlm python -u -B -m src.robotic.gentle_push.data_selection \
  --device cuda:1 \
  --seeds 1 2 3 \
  --arms full main_only structured_rbf raw_flat_rbf random_group \
  --index-root src/kernel_surrogate/results/gentle_push_auxacq_data_selection_indices \
  --index-seed 1 \
  --epochs 20 \
  --lr 1e-5 \
  --batch-size 32 \
  --eval-batch-size 32 \
  --output-dir src/data_selection/results/gentle_push_auxacq_group_selection
