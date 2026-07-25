#!/usr/bin/env bash

datasets=(
  imdb
  humor
  sarcasm
  mosi
  mosei
  avmnist
)

for dataset in "${datasets[@]}"; do
  conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
    --datasets "${dataset}" \
    --seeds 0 \
    --device cuda:1 \
    --feature-device cuda:1 \
    --num-groups 25 \
    --group-method random \
    --num-subset-samples 100 \
    --data-keep-rate 0.9 \
    --modality-keep-rate 0.9 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --run-name random_grouped25_subset100_data90_mod90

  conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
    --datasets "${dataset}" \
    --seeds 0 \
    --device cuda:1 \
    --feature-device cuda:1 \
    --num-groups 25 \
    --group-method random \
    --num-subset-samples 100 \
    --data-keep-rate 0.8 \
    --modality-keep-rate 0.8 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --run-name random_grouped25_subset100_data80_mod80

  conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
    --datasets "${dataset}" \
    --seeds 0 \
    --device cuda:1 \
    --feature-device cuda:1 \
    --num-groups 25 \
    --group-method kmeans \
    --num-subset-samples 100 \
    --data-keep-rate 0.9 \
    --modality-keep-rate 0.9 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --run-name kmeans_grouped25_subset100_data90_mod90

  conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
    --datasets "${dataset}" \
    --seeds 0 \
    --device cuda:1 \
    --feature-device cuda:1 \
    --num-groups 25 \
    --group-method kmeans \
    --num-subset-samples 100 \
    --data-keep-rate 0.8 \
    --modality-keep-rate 0.8 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --run-name kmeans_grouped25_subset100_data80_mod80
done
