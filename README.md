## Designing Kernel Surrogate Models for Multimodal Attribution

We provide the implementation of Designing Kernel Surrogate Models for Multimodal Attribution.

## Run experiments

The review code exposes the attribution/data-selection workflow through three
small entry points under `src/attribution`.

Two launch scripts are provided:

```bash
bash scripts/run_attribution.sh avmnist
bash scripts/run_selection.sh avmnist
```

The dataset is the first script argument. `RUN_NAME` defaults to the dataset
name; change it near the top of the script only if the LDS artifact directory
uses a different name.

### 1. K-means sample groups

```bash
python -m src.attribution.sample_groups \
  --dataset avmnist \
  --num-groups 25 \
  --group-method kmeans \
  --output src/attribution/results/groups/avmnist_seed0.json
```

This writes a JSON payload with `group_partition.group_indices`. The same
script also supports `--group-method random` for a random grouping baseline.

### 2. SKS surrogate attribution

Given a grouped LDS/retraining payload with `group_partition` and `runs`, fit
the SKS kernel surrogate and export attribution scores:

```bash
python -m src.attribution.sks_attribution \
  --input src/attribution/results/lds_real_train/<run_name>/avmnist_seed0.json \
  --output src/attribution/results/sks_scores/avmnist_seed0.json \
  --rho 0.5 \
  --ridge-lambda 1e-3 \
  --gamma-scale 1.0
```

The output includes pairwise cross-modality scores plus group-level scores at
`modality_group_scores.x`, `modality_group_scores.y`, and
`modality_group_scores.combined`.

### 3. Attribution-score downsampling and training

Convert group attribution scores into a downsampled selection index:

```bash
python -m src.attribution.score_to_dataselection \
  --score-file src/attribution/results/sks_scores/avmnist_seed0.json \
  --group-payload src/attribution/results/lds_real_train/<run_name>/avmnist_seed0.json \
  --score-key modality_group_scores.y \
  --direction largest \
  --budget 0.2 \
  --seed 1 \
  --output src/attribution/results/dataselection/sks_y/avmnist/seed1.json
```

Train with the generated index:

```bash
python -m src.MultiBench_selection.experiment \
  --datasets avmnist \
  --seeds 1 \
  --selection-method index \
  --index-file src/attribution/results/dataselection/sks_y/avmnist/seed1.json \
  --budget 0.2 \
  --main-modality x
```
