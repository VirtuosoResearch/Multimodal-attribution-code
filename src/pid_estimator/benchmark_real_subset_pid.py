from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fisher_geometry.data import load_dataset
from pid_estimator.subset_attribution import (
    RBFSubsetSurrogateConfig,
    StructuredSubsetSurrogateConfig,
    aggregate_cross_modal_pid_masses,
    build_cross_modal_candidate_edges,
    estimate_cross_modal_pid_from_scores,
    fit_linear_subset_surrogate,
    fit_rbf_subset_surrogate,
)
from pid_estimator.subset_attribution_experiment import (
    binary_auprc,
    binary_auroc,
    binary_recall_at_k,
    pearson_corr,
    r2_score,
    rmse,
    spearman_corr,
    summarize_numeric_dicts,
)

from .benchmark_real_subset_attribution import (
    SUPPORTED_DATASETS,
    choose_top_classes,
    label_similarity_to_anchor,
    l2_normalize,
    maybe_random_project_pair,
    remap_labels,
    select_anchor_indices,
    split_positive_negative_candidates,
    stratified_subsample,
    summarize_curve_runs,
    tensor_labels_to_numpy,
    tensor_to_numpy,
)


@dataclass(frozen=True)
class RealSubsetPIDConfig:
    datasets: tuple[str, ...] = SUPPORTED_DATASETS
    seeds: tuple[int, ...] = (0, 1)
    num_train_samples: int = 1200
    num_val_samples: int = 400
    num_focal_samples: int = 5
    donor_num_anchors: int = 4
    donor_candidates_per_anchor: int = 3
    donor_partner_purity: float = 1.0 / 3.0
    train_queries: int = 160
    holdout_queries: int = 256
    lds_num_subsets: int = 256
    ridge_lambda: float = 1e-3
    kernel_ridge_lambda: float = 1e-3
    score_metric: str = "accuracy"
    corruption_alpha: float = 1.0
    corruption_sweep: tuple[float, ...] = (0.0, 0.5, 1.0)
    max_classes_food101: int = 10
    max_modality_dim: int = 64
    use_dataset_presets: bool = True


@dataclass(frozen=True)
class DatasetRunConfig:
    dataset_name: str
    seed: int
    num_train_samples: int
    num_val_samples: int
    num_focal_samples: int
    donor_num_anchors: int
    donor_candidates_per_anchor: int
    donor_partner_purity: float
    train_queries: int
    holdout_queries: int
    lds_num_subsets: int
    ridge_lambda: float
    kernel_ridge_lambda: float
    score_metric: str
    corruption_alpha: float
    corruption_sweep: tuple[float, ...]
    max_classes_food101: int = 10
    max_modality_dim: int = 64


@dataclass
class RealSubsetPIDBenchmark:
    dataset_name: str
    train_x: np.ndarray
    train_y: np.ndarray
    train_targets: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_targets: np.ndarray
    focal_indices: np.ndarray
    corruption_alpha: float
    task: str
    ridge_lambda: float
    score_metric: str

    def __post_init__(self) -> None:
        self.train_x = np.asarray(self.train_x, dtype=np.float64)
        self.train_y = np.asarray(self.train_y, dtype=np.float64)
        self.train_targets = np.asarray(self.train_targets)
        self.val_x = np.asarray(self.val_x, dtype=np.float64)
        self.val_y = np.asarray(self.val_y, dtype=np.float64)
        self.val_targets = np.asarray(self.val_targets)
        self.focal_indices = np.asarray(self.focal_indices, dtype=np.int64).reshape(-1)
        self.num_focal_samples = int(self.focal_indices.shape[0])
        self.num_atoms = int(2 * self.num_focal_samples)
        self.atom_samples = np.repeat(np.arange(self.num_focal_samples, dtype=np.int64), 2)
        self.atom_modalities = np.tile(np.arange(2, dtype=np.int64), self.num_focal_samples)
        self.candidate_edges = build_cross_modal_candidate_edges(self.atom_modalities, self.atom_samples)
        self._cache: dict[bytes, float] = {}

    def utility(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.num_atoms:
            raise ValueError(f"expected mask with {self.num_atoms} atoms, got {mask.shape[0]}")
        key = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
        if key in self._cache:
            return self._cache[key]

        train_x_variant = self.train_x.copy()
        train_y_variant = self.train_y.copy()
        keep_scale = max(1.0 - float(self.corruption_alpha), 0.0)
        for local_index, train_index in enumerate(self.focal_indices):
            if not mask[2 * local_index]:
                train_x_variant[train_index] *= keep_scale
            if not mask[2 * local_index + 1]:
                train_y_variant[train_index] *= keep_scale

        train_features = np.concatenate([train_x_variant, train_y_variant], axis=1)
        val_features = np.concatenate([self.val_x, self.val_y], axis=1)
        value = ridge_probe_score(
            train_features=train_features,
            train_targets=self.train_targets,
            val_features=val_features,
            val_targets=self.val_targets,
            ridge_lambda=self.ridge_lambda,
            task=self.task,
            metric=self.score_metric,
        )
        self._cache[key] = float(value)
        return float(value)

    def utility_batch(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=bool)
        return np.asarray([self.utility(mask) for mask in masks], dtype=np.float64)


@dataclass
class LocalDonorRescuePIDGame:
    train_x: np.ndarray
    train_y: np.ndarray
    train_targets: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_targets: np.ndarray
    anchor_train_index: int
    donor_indices: np.ndarray
    donor_is_positive: np.ndarray
    corruption_alpha: float
    task: str
    ridge_lambda: float
    score_metric: str

    def __post_init__(self) -> None:
        self.train_x = np.asarray(self.train_x, dtype=np.float64)
        self.train_y = np.asarray(self.train_y, dtype=np.float64)
        self.train_targets = np.asarray(self.train_targets)
        self.val_x = np.asarray(self.val_x, dtype=np.float64)
        self.val_y = np.asarray(self.val_y, dtype=np.float64)
        self.val_targets = np.asarray(self.val_targets)
        self.anchor_train_index = int(self.anchor_train_index)
        self.donor_indices = np.asarray(self.donor_indices, dtype=np.int64).reshape(-1)
        self.donor_is_positive = np.asarray(self.donor_is_positive, dtype=bool).reshape(-1)
        self.num_atoms = int(1 + self.donor_indices.shape[0])
        self.atom_samples = np.concatenate(
            [
                np.asarray([self.anchor_train_index], dtype=np.int64),
                self.donor_indices.astype(np.int64),
            ]
        )
        self.atom_modalities = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.ones(self.donor_indices.shape[0], dtype=np.int64),
            ]
        )
        self._cache: dict[bytes, float] = {}

    def utility(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.num_atoms:
            raise ValueError(f"expected mask with {self.num_atoms} atoms, got {mask.shape[0]}")
        key = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
        if key in self._cache:
            return self._cache[key]

        keep_scale = max(1.0 - float(self.corruption_alpha), 0.0)
        train_x_variant = self.train_x.copy()
        train_y_variant = self.train_y.copy()
        if not mask[0]:
            train_x_variant[self.anchor_train_index] *= keep_scale
        train_y_variant[self.anchor_train_index] *= keep_scale
        selected = self.donor_indices[mask[1:]]
        if selected.size > 0:
            donor_mean = np.mean(self.train_y[selected], axis=0)
            train_y_variant[self.anchor_train_index] = (
                keep_scale * self.train_y[self.anchor_train_index]
                + float(self.corruption_alpha) * donor_mean
            )

        score = ridge_probe_score(
            train_features=np.concatenate([train_x_variant, train_y_variant], axis=1),
            train_targets=self.train_targets,
            val_features=np.concatenate([self.val_x, self.val_y], axis=1),
            val_targets=self.val_targets,
            ridge_lambda=self.ridge_lambda,
            task=self.task,
            metric=self.score_metric,
        )
        self._cache[key] = float(score)
        return float(score)

    def utility_batch(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=bool)
        return np.asarray([self.utility(mask) for mask in masks], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-data subset-to-score PID benchmarks across available datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--num-train-samples", type=int, default=1200)
    parser.add_argument("--num-val-samples", type=int, default=400)
    parser.add_argument("--num-focal-samples", type=int, default=5)
    parser.add_argument("--donor-num-anchors", type=int, default=4)
    parser.add_argument("--donor-candidates-per-anchor", type=int, default=3)
    parser.add_argument("--donor-partner-purity", type=float, default=1.0 / 3.0)
    parser.add_argument("--train-queries", type=int, default=160)
    parser.add_argument("--holdout-queries", type=int, default=256)
    parser.add_argument("--lds-num-subsets", type=int, default=256)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--kernel-ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--score-metric", choices=["accuracy", "neg_loss", "loss"], default="accuracy")
    parser.add_argument("--corruption-alpha", type=float, default=1.0)
    parser.add_argument("--corruption-sweep", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--max-classes-food101", type=int, default=10)
    parser.add_argument("--max-modality-dim", type=int, default=64)
    parser.add_argument("--no-dataset-presets", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/pid_estimator/results/real_subset_pid_experiment.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/real_subset_pid_report.md",
    )
    args = parser.parse_args()

    config = RealSubsetPIDConfig(
        datasets=tuple(args.datasets),
        seeds=tuple(args.seeds),
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        num_focal_samples=args.num_focal_samples,
        donor_num_anchors=args.donor_num_anchors,
        donor_candidates_per_anchor=args.donor_candidates_per_anchor,
        donor_partner_purity=args.donor_partner_purity,
        train_queries=args.train_queries,
        holdout_queries=args.holdout_queries,
        lds_num_subsets=args.lds_num_subsets,
        ridge_lambda=args.ridge_lambda,
        kernel_ridge_lambda=args.kernel_ridge_lambda,
        score_metric=args.score_metric,
        corruption_alpha=args.corruption_alpha,
        corruption_sweep=tuple(args.corruption_sweep),
        max_classes_food101=args.max_classes_food101,
        max_modality_dim=args.max_modality_dim,
        use_dataset_presets=not args.no_dataset_presets,
    )

    payload = run_real_subset_pid_benchmarks(config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload))


def run_real_subset_pid_benchmarks(config: RealSubsetPIDConfig) -> dict[str, object]:
    resolved_datasets, skipped_datasets = resolve_available_datasets(config.datasets)
    dataset_results: dict[str, object] = {}
    effective_configs: dict[str, object] = {}
    aggregate_method_rows: dict[str, list[dict[str, float]]] = {}
    aggregate_sweep_rows: list[dict[str, float]] = []
    aggregate_behavior_rows: dict[str, list[dict[str, float]]] = {}
    aggregate_donor_rows: dict[str, list[dict[str, float]]] = {}

    for dataset_name in resolved_datasets:
        print(f"[real-subset-pid] start dataset={dataset_name} metric={config.score_metric}", flush=True)
        runs = []
        configs = []
        for seed in config.seeds:
            run_config = resolve_dataset_run_config(config, dataset_name=dataset_name, seed=seed)
            configs.append(asdict(run_config))
            run = run_single_dataset(run_config)
            runs.append(run)
            for method_name, metrics in run["method_metrics"].items():
                aggregate_method_rows.setdefault(method_name, []).append(metrics)
                aggregate_behavior_rows.setdefault(method_name, []).append(run["behavior_metrics"][method_name])
                aggregate_donor_rows.setdefault(method_name, []).append(run["donor_alignment_metrics"][method_name])
            aggregate_sweep_rows.append(run["sweep_summary"])
        dataset_results[dataset_name] = summarize_dataset_runs(runs)
        effective_configs[dataset_name] = configs
        linear_summary = dataset_results[dataset_name]["method_summary"]["linear"]
        rbf_summary = dataset_results[dataset_name]["method_summary"]["rbf_kernel"]
        print(
            f"[real-subset-pid] done dataset={dataset_name} "
            f"linear_lds={linear_summary['subset_lds_spearman']['mean']:.4f} "
            f"rbf_lds={rbf_summary['subset_lds_spearman']['mean']:.4f}",
            flush=True,
        )

    return {
        "config": asdict(config),
        "resolved_datasets": resolved_datasets,
        "skipped_datasets": skipped_datasets,
        "effective_dataset_configs": effective_configs,
        "datasets": dataset_results,
        "aggregate": {
            "methods": {
                method_name: summarize_numeric_dicts(rows)
                for method_name, rows in aggregate_method_rows.items()
            },
            "behavior": {
                method_name: summarize_numeric_dicts(rows)
                for method_name, rows in aggregate_behavior_rows.items()
            },
            "donor_alignment": {
                method_name: summarize_numeric_dicts(rows)
                for method_name, rows in aggregate_donor_rows.items()
            },
            "sweeps": summarize_numeric_dicts(aggregate_sweep_rows) if aggregate_sweep_rows else {},
        },
    }


def resolve_dataset_run_config(
    config: RealSubsetPIDConfig,
    *,
    dataset_name: str,
    seed: int,
) -> DatasetRunConfig:
    num_train_samples = config.num_train_samples
    num_val_samples = config.num_val_samples

    if config.use_dataset_presets:
        if dataset_name == "avmnist":
            num_train_samples = 1200
            num_val_samples = 400
        elif dataset_name == "humor":
            num_train_samples = 1200
            num_val_samples = 400
        elif dataset_name == "sarcasm":
            num_train_samples = 300
            num_val_samples = 100
        elif dataset_name == "mosi":
            num_train_samples = 800
            num_val_samples = 200
        elif dataset_name == "mosei":
            num_train_samples = 1200
            num_val_samples = 400
        elif dataset_name == "imdb":
            num_train_samples = 1200
            num_val_samples = 400
        elif dataset_name == "food101":
            num_train_samples = 500
            num_val_samples = 300

    return DatasetRunConfig(
        dataset_name=dataset_name,
        seed=seed,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        num_focal_samples=config.num_focal_samples,
        donor_num_anchors=config.donor_num_anchors,
        donor_candidates_per_anchor=config.donor_candidates_per_anchor,
        donor_partner_purity=config.donor_partner_purity,
        train_queries=config.train_queries,
        holdout_queries=config.holdout_queries,
        lds_num_subsets=config.lds_num_subsets,
        ridge_lambda=config.ridge_lambda,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        score_metric=config.score_metric,
        corruption_alpha=config.corruption_alpha,
        corruption_sweep=config.corruption_sweep,
        max_classes_food101=config.max_classes_food101,
        max_modality_dim=config.max_modality_dim,
    )


def run_single_dataset(config: DatasetRunConfig) -> dict[str, object]:
    benchmark = build_real_subset_pid_benchmark(config)
    all_masks = enumerate_masks(benchmark.num_atoms)
    oracle_scores = benchmark.utility_batch(all_masks)

    split = sample_mask_splits(
        num_masks=all_masks.shape[0],
        train_queries=min(config.train_queries, all_masks.shape[0]),
        holdout_queries=min(config.holdout_queries, all_masks.shape[0]),
        lds_num_subsets=min(config.lds_num_subsets, all_masks.shape[0]),
        seed=config.seed,
    )

    linear = fit_linear_subset_surrogate(
        all_masks[split["train"]],
        oracle_scores[split["train"]],
        num_atoms=benchmark.num_atoms,
        candidate_edges=benchmark.candidate_edges,
        config=StructuredSubsetSurrogateConfig(
            feature_bernoulli_p=0.5,
            ridge_lambda=config.kernel_ridge_lambda,
            pair_probe_weight=0.0,
        ),
    )
    rbf = fit_rbf_subset_surrogate(
        all_masks[split["train"]],
        oracle_scores[split["train"]],
        num_atoms=benchmark.num_atoms,
        candidate_edges=benchmark.candidate_edges,
        config=RBFSubsetSurrogateConfig(ridge_lambda=config.kernel_ridge_lambda),
    )

    oracle_pid = estimate_cross_modal_pid_from_scores(
        oracle_scores,
        num_atoms=benchmark.num_atoms,
        atom_samples=benchmark.atom_samples,
        atom_modalities=benchmark.atom_modalities,
        modality_pair=(0, 1),
    )
    oracle_masses = aggregate_cross_modal_pid_masses(oracle_pid)
    oracle_pair_arrays = pair_component_arrays(oracle_pid)
    oracle_behavior = compute_pair_behavior_from_scores(
        oracle_scores,
        num_atoms=benchmark.num_atoms,
        atom_samples=benchmark.atom_samples,
        atom_modalities=benchmark.atom_modalities,
        modality_pair=(0, 1),
    )
    donor_alignment_oracle, donor_alignment_metrics = run_donor_alignment_benchmark(config)

    method_metrics = {
        "linear": evaluate_method(
            model=linear,
            all_masks=all_masks,
            oracle_scores=oracle_scores,
            split=split,
            benchmark=benchmark,
            oracle_masses=oracle_masses,
            oracle_pair_arrays=oracle_pair_arrays,
        ),
        "rbf_kernel": evaluate_method(
            model=rbf,
            all_masks=all_masks,
            oracle_scores=oracle_scores,
            split=split,
            benchmark=benchmark,
            oracle_masses=oracle_masses,
            oracle_pair_arrays=oracle_pair_arrays,
        ),
    }
    behavior_metrics = {
        "linear": evaluate_behavior(
            model=linear,
            all_masks=all_masks,
            benchmark=benchmark,
            oracle_behavior=oracle_behavior,
        ),
        "rbf_kernel": evaluate_behavior(
            model=rbf,
            all_masks=all_masks,
            benchmark=benchmark,
            oracle_behavior=oracle_behavior,
        ),
    }

    sweep_rows = []
    for alpha in config.corruption_sweep:
        sweep_config = DatasetRunConfig(**{**asdict(config), "corruption_alpha": float(alpha)})
        sweep_benchmark = build_real_subset_pid_benchmark(sweep_config, focal_indices=benchmark.focal_indices)
        sweep_scores = sweep_benchmark.utility_batch(all_masks)
        sweep_oracle_pid = estimate_cross_modal_pid_from_scores(
            sweep_scores,
            num_atoms=sweep_benchmark.num_atoms,
            atom_samples=sweep_benchmark.atom_samples,
            atom_modalities=sweep_benchmark.atom_modalities,
            modality_pair=(0, 1),
        )
        sweep_linear = fit_linear_subset_surrogate(
            all_masks[split["train"]],
            sweep_scores[split["train"]],
            num_atoms=sweep_benchmark.num_atoms,
            candidate_edges=sweep_benchmark.candidate_edges,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.kernel_ridge_lambda,
                pair_probe_weight=0.0,
            ),
        )
        sweep_rbf = fit_rbf_subset_surrogate(
            all_masks[split["train"]],
            sweep_scores[split["train"]],
            num_atoms=sweep_benchmark.num_atoms,
            candidate_edges=sweep_benchmark.candidate_edges,
            config=RBFSubsetSurrogateConfig(ridge_lambda=config.kernel_ridge_lambda),
        )
        sweep_linear_scores = np.asarray(sweep_linear.predict_batch(all_masks), dtype=np.float64)
        sweep_rbf_scores = np.asarray(sweep_rbf.predict_batch(all_masks), dtype=np.float64)
        sweep_linear_masses = aggregate_cross_modal_pid_masses(
            estimate_cross_modal_pid_from_scores(
                sweep_linear_scores,
                num_atoms=sweep_benchmark.num_atoms,
                atom_samples=sweep_benchmark.atom_samples,
                atom_modalities=sweep_benchmark.atom_modalities,
                modality_pair=(0, 1),
            )
        )
        sweep_rbf_masses = aggregate_cross_modal_pid_masses(
            estimate_cross_modal_pid_from_scores(
                sweep_rbf_scores,
                num_atoms=sweep_benchmark.num_atoms,
                atom_samples=sweep_benchmark.atom_samples,
                atom_modalities=sweep_benchmark.atom_modalities,
                modality_pair=(0, 1),
            )
        )
        sweep_oracle_masses = aggregate_cross_modal_pid_masses(sweep_oracle_pid)
        sweep_rows.append(
            {
                "corruption_alpha": float(alpha),
                "oracle_positive_synergy_mass": float(sweep_oracle_masses["positive_synergy_mass"]),
                "oracle_positive_redundancy_mass": float(sweep_oracle_masses["positive_redundancy_mass"]),
                "linear_positive_synergy_mass": float(sweep_linear_masses["positive_synergy_mass"]),
                "linear_positive_redundancy_mass": float(sweep_linear_masses["positive_redundancy_mass"]),
                "rbf_positive_synergy_mass": float(sweep_rbf_masses["positive_synergy_mass"]),
                "rbf_positive_redundancy_mass": float(sweep_rbf_masses["positive_redundancy_mass"]),
            }
        )

    sweep_summary = summarize_sweep_rows(sweep_rows)
    oracle_stats = {
        "score_min": float(np.min(oracle_scores)),
        "score_max": float(np.max(oracle_scores)),
        "score_mean": float(np.mean(oracle_scores)),
        "score_std": float(np.std(oracle_scores)),
        "positive_synergy_mass": float(oracle_masses["positive_synergy_mass"]),
        "positive_redundancy_mass": float(oracle_masses["positive_redundancy_mass"]),
        "mean_joint_over_best": float(np.mean(oracle_behavior["joint_over_best"])),
        "mean_empty_rescue_gain": float(np.mean(oracle_behavior["empty_joint_over_best"])),
        "mean_full_context_joint_over_best": float(np.mean(oracle_behavior["full_context_joint_over_best"])),
        "positive_joint_over_best_rate": float(np.mean(oracle_behavior["joint_over_best"] > 0.0)),
        "positive_empty_rescue_rate": float(np.mean(oracle_behavior["empty_joint_over_best"] > 0.0)),
        "donor_empirical_positive_rate": float(donor_alignment_oracle["empirical_positive_rate"]),
        "donor_heuristic_positive_rate": float(donor_alignment_oracle["heuristic_positive_rate"]),
    }

    return {
        "dataset": config.dataset_name,
        "seed": config.seed,
        "task": benchmark.task,
        "num_atoms": int(benchmark.num_atoms),
        "num_masks_total": int(all_masks.shape[0]),
        "num_candidate_edges": int(len(benchmark.candidate_edges)),
        "focal_indices": [int(index) for index in benchmark.focal_indices.tolist()],
        "oracle_stats": oracle_stats,
        "method_metrics": method_metrics,
        "behavior_metrics": behavior_metrics,
        "donor_alignment_metrics": donor_alignment_metrics,
        "sweep_rows": sweep_rows,
        "sweep_summary": sweep_summary,
    }


def evaluate_method(
    *,
    model: object,
    all_masks: np.ndarray,
    oracle_scores: np.ndarray,
    split: dict[str, np.ndarray],
    benchmark: RealSubsetPIDBenchmark,
    oracle_masses: dict[str, float],
    oracle_pair_arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    predicted_holdout = np.asarray(model.predict_batch(all_masks[split["holdout"]]), dtype=np.float64)
    predicted_lds = np.asarray(model.predict_batch(all_masks[split["lds"]]), dtype=np.float64)
    predicted_all = np.asarray(model.predict_batch(all_masks), dtype=np.float64)
    estimated_pid = estimate_cross_modal_pid_from_scores(
        predicted_all,
        num_atoms=benchmark.num_atoms,
        atom_samples=benchmark.atom_samples,
        atom_modalities=benchmark.atom_modalities,
        modality_pair=(0, 1),
    )
    estimated_masses = aggregate_cross_modal_pid_masses(estimated_pid)
    estimated_pair_arrays = pair_component_arrays(estimated_pid)
    return {
        "utility_r2": r2_score(oracle_scores[split["holdout"]], predicted_holdout),
        "utility_rmse": rmse(oracle_scores[split["holdout"]], predicted_holdout),
        "subset_lds_spearman": spearman_corr(oracle_scores[split["lds"]], predicted_lds),
        "subset_lds_pearson": pearson_corr(oracle_scores[split["lds"]], predicted_lds),
        "synergy_mass_abs_error": abs(
            float(estimated_masses["positive_synergy_mass"] - oracle_masses["positive_synergy_mass"])
        ),
        "redundancy_mass_abs_error": abs(
            float(estimated_masses["positive_redundancy_mass"] - oracle_masses["positive_redundancy_mass"])
        ),
        "pair_synergy_spearman": spearman_corr(oracle_pair_arrays["S"], estimated_pair_arrays["S"]),
        "pair_redundancy_spearman": spearman_corr(oracle_pair_arrays["R"], estimated_pair_arrays["R"]),
    }


def evaluate_behavior(
    *,
    model: object,
    all_masks: np.ndarray,
    benchmark: RealSubsetPIDBenchmark,
    oracle_behavior: dict[str, np.ndarray],
) -> dict[str, float]:
    predicted_all = np.asarray(model.predict_batch(all_masks), dtype=np.float64)
    estimated_pid = estimate_cross_modal_pid_from_scores(
        predicted_all,
        num_atoms=benchmark.num_atoms,
        atom_samples=benchmark.atom_samples,
        atom_modalities=benchmark.atom_modalities,
        modality_pair=(0, 1),
    )
    estimated_pair_arrays = pair_component_arrays(estimated_pid)
    k = max(1, min(5, oracle_behavior["joint_over_best"].shape[0] // 4))
    synergy_scores = estimated_pair_arrays["S"]
    topk_synergy = np.argsort(-synergy_scores)[:k]
    return {
        "synergy_vs_joint_over_best_spearman": spearman_corr(synergy_scores, oracle_behavior["joint_over_best"]),
        "synergy_vs_joint_over_best_pearson": pearson_corr(synergy_scores, oracle_behavior["joint_over_best"]),
        "synergy_vs_empty_rescue_spearman": spearman_corr(synergy_scores, oracle_behavior["empty_joint_over_best"]),
        "synergy_vs_empty_rescue_pearson": pearson_corr(synergy_scores, oracle_behavior["empty_joint_over_best"]),
        "synergy_vs_full_context_spearman": spearman_corr(
            synergy_scores, oracle_behavior["full_context_joint_over_best"]
        ),
        "synergy_vs_full_context_pearson": pearson_corr(
            synergy_scores, oracle_behavior["full_context_joint_over_best"]
        ),
        "topk_synergy_joint_over_best_mean": float(np.mean(oracle_behavior["joint_over_best"][topk_synergy])),
        "topk_synergy_joint_over_best_positive_rate": float(
            np.mean(oracle_behavior["joint_over_best"][topk_synergy] > 0.0)
        ),
        "topk_synergy_empty_rescue_mean": float(np.mean(oracle_behavior["empty_joint_over_best"][topk_synergy])),
        "topk_synergy_empty_rescue_positive_rate": float(
            np.mean(oracle_behavior["empty_joint_over_best"][topk_synergy] > 0.0)
        ),
        "topk_synergy_full_context_mean": float(
            np.mean(oracle_behavior["full_context_joint_over_best"][topk_synergy])
        ),
        "topk_synergy_full_context_positive_rate": float(
            np.mean(oracle_behavior["full_context_joint_over_best"][topk_synergy] > 0.0)
        ),
    }


def summarize_sweep_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    alphas = np.asarray([row["corruption_alpha"] for row in rows], dtype=np.float64)
    oracle_synergy = np.asarray([row["oracle_positive_synergy_mass"] for row in rows], dtype=np.float64)
    oracle_redundancy = np.asarray([row["oracle_positive_redundancy_mass"] for row in rows], dtype=np.float64)
    linear_synergy = np.asarray([row["linear_positive_synergy_mass"] for row in rows], dtype=np.float64)
    linear_redundancy = np.asarray([row["linear_positive_redundancy_mass"] for row in rows], dtype=np.float64)
    rbf_synergy = np.asarray([row["rbf_positive_synergy_mass"] for row in rows], dtype=np.float64)
    rbf_redundancy = np.asarray([row["rbf_positive_redundancy_mass"] for row in rows], dtype=np.float64)
    return {
        "corr_alpha_oracle_synergy": pearson_corr(alphas, oracle_synergy),
        "corr_alpha_linear_synergy": pearson_corr(alphas, linear_synergy),
        "corr_alpha_rbf_synergy": pearson_corr(alphas, rbf_synergy),
        "corr_oracle_linear_synergy": pearson_corr(oracle_synergy, linear_synergy),
        "corr_oracle_rbf_synergy": pearson_corr(oracle_synergy, rbf_synergy),
        "corr_alpha_oracle_redundancy": pearson_corr(alphas, oracle_redundancy),
        "corr_alpha_linear_redundancy": pearson_corr(alphas, linear_redundancy),
        "corr_alpha_rbf_redundancy": pearson_corr(alphas, rbf_redundancy),
        "corr_oracle_linear_redundancy": pearson_corr(oracle_redundancy, linear_redundancy),
        "corr_oracle_rbf_redundancy": pearson_corr(oracle_redundancy, rbf_redundancy),
    }


def compute_pair_behavior_from_scores(
    scores: np.ndarray,
    *,
    num_atoms: int,
    atom_samples: np.ndarray,
    atom_modalities: np.ndarray,
    modality_pair: tuple[int, int],
) -> dict[str, np.ndarray]:
    pair_keys = list_pair_keys(atom_samples=atom_samples, atom_modalities=atom_modalities, modality_pair=modality_pair)
    joint_over_best = []
    empty_joint_over_best = []
    empty_joint_gain = []
    full_context_joint_over_best = []
    beat_best_rate = []
    for left_sample, right_sample in pair_keys:
        left_atom = int(np.where((atom_samples == left_sample) & (atom_modalities == modality_pair[0]))[0][0])
        right_atom = int(np.where((atom_samples == right_sample) & (atom_modalities == modality_pair[1]))[0][0])
        left_mask = 1 << left_atom
        right_mask = 1 << right_atom
        local_joint_over_best = []
        local_beats = []
        for index in range(2**num_atoms):
            if (index & left_mask) or (index & right_mask):
                continue
            g00 = float(scores[index])
            g10 = float(scores[index | left_mask])
            g01 = float(scores[index | right_mask])
            g11 = float(scores[index | left_mask | right_mask])
            best_simple = max(g00, g10, g01)
            value = g11 - best_simple
            local_joint_over_best.append(value)
            local_beats.append(float(g11 > best_simple + 1e-12))
        base00 = float(scores[0])
        base10 = float(scores[left_mask])
        base01 = float(scores[right_mask])
        base11 = float(scores[left_mask | right_mask])
        full_background = ((1 << num_atoms) - 1) & (~left_mask) & (~right_mask)
        full00 = float(scores[full_background])
        full10 = float(scores[full_background | left_mask])
        full01 = float(scores[full_background | right_mask])
        full11 = float(scores[full_background | left_mask | right_mask])
        joint_over_best.append(float(np.mean(local_joint_over_best)))
        beat_best_rate.append(float(np.mean(local_beats)))
        empty_joint_over_best.append(float(base11 - max(base00, base10, base01)))
        empty_joint_gain.append(float(base11 - base00))
        full_context_joint_over_best.append(float(full11 - max(full00, full10, full01)))
    return {
        "pair_keys": np.asarray(pair_keys, dtype=object),
        "joint_over_best": np.asarray(joint_over_best, dtype=np.float64),
        "empty_joint_over_best": np.asarray(empty_joint_over_best, dtype=np.float64),
        "empty_joint_gain": np.asarray(empty_joint_gain, dtype=np.float64),
        "full_context_joint_over_best": np.asarray(full_context_joint_over_best, dtype=np.float64),
        "beat_best_rate": np.asarray(beat_best_rate, dtype=np.float64),
    }


def run_donor_alignment_benchmark(
    config: DatasetRunConfig,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    processed = load_processed_dataset(config)
    anchors = select_anchor_indices(
        processed["train_targets"],
        task=processed["task"],
        num_anchors=config.donor_num_anchors,
        candidates_per_anchor=config.donor_candidates_per_anchor,
        seed=70_000 + config.seed,
    )
    norm_x = l2_normalize(processed["train_x"])
    all_rows: list[dict[str, float]] = []
    for group_id, anchor_index in enumerate(anchors):
        x_similarities = norm_x @ norm_x[anchor_index]
        label_similarities = label_similarity_to_anchor(
            processed["train_targets"],
            anchor_index=int(anchor_index),
            task=processed["task"],
        )
        positive_pool, negative_pool = split_positive_negative_candidates(
            label_similarities,
            x_similarities=x_similarities,
            anchor_index=int(anchor_index),
        )
        num_positive = min(
            config.donor_candidates_per_anchor,
            max(0, int(round(config.donor_partner_purity * config.donor_candidates_per_anchor))),
        )
        if config.donor_partner_purity > 0 and num_positive == 0:
            num_positive = 1
        if config.donor_partner_purity < 1.0 and num_positive == config.donor_candidates_per_anchor:
            num_positive = config.donor_candidates_per_anchor - 1
        num_negative = config.donor_candidates_per_anchor - num_positive
        donor_indices = np.asarray(
            list(positive_pool[:num_positive]) + list(negative_pool[:num_negative]),
            dtype=np.int64,
        )
        donor_is_positive = np.asarray(
            [True] * num_positive + [False] * num_negative,
            dtype=bool,
        )
        if donor_indices.shape[0] != config.donor_candidates_per_anchor:
            continue
        local_game = LocalDonorRescuePIDGame(
            train_x=processed["train_x"],
            train_y=processed["train_y"],
            train_targets=processed["train_targets"],
            val_x=processed["val_x"],
            val_y=processed["val_y"],
            val_targets=processed["val_targets"],
            anchor_train_index=int(anchor_index),
            donor_indices=donor_indices,
            donor_is_positive=donor_is_positive,
            corruption_alpha=config.corruption_alpha,
            task=processed["task"],
            ridge_lambda=config.ridge_lambda,
            score_metric=config.score_metric,
        )
        local_rows = evaluate_local_donor_game(
            local_game,
            seed=80_000 + config.seed + int(anchor_index),
            group_id=group_id,
        )
        all_rows.extend(local_rows)

    if not all_rows:
        empty = {
            "empirical_positive_rate": 0.0,
            "heuristic_positive_rate": 0.0,
        }
        zero_metrics = {
            "linear": {
                "donor_synergy_vs_gain_spearman": 0.0,
                "donor_synergy_vs_gain_pearson": 0.0,
                "donor_empirical_auprc": 0.0,
                "donor_empirical_auroc": 0.0,
                "donor_empirical_recall_at_1": 0.0,
                "donor_heuristic_auprc": 0.0,
                "donor_heuristic_auroc": 0.0,
                "donor_heuristic_recall_at_1": 0.0,
                "top1_empirical_gain": 0.0,
                "anchor_best_hit_at_1": 0.0,
                "anchor_top1_gain_mean": 0.0,
                "anchor_gain_regret_mean": 0.0,
                "anchor_top1_heuristic_rate": 0.0,
            },
            "rbf_kernel": {
                "donor_synergy_vs_gain_spearman": 0.0,
                "donor_synergy_vs_gain_pearson": 0.0,
                "donor_empirical_auprc": 0.0,
                "donor_empirical_auroc": 0.0,
                "donor_empirical_recall_at_1": 0.0,
                "donor_heuristic_auprc": 0.0,
                "donor_heuristic_auroc": 0.0,
                "donor_heuristic_recall_at_1": 0.0,
                "top1_empirical_gain": 0.0,
                "anchor_best_hit_at_1": 0.0,
                "anchor_top1_gain_mean": 0.0,
                "anchor_gain_regret_mean": 0.0,
                "anchor_top1_heuristic_rate": 0.0,
            },
        }
        return empty, zero_metrics

    empirical_positive = np.asarray([row["empirical_positive"] for row in all_rows], dtype=bool)
    heuristic_positive = np.asarray([row["heuristic_positive"] for row in all_rows], dtype=bool)
    oracle_summary = {
        "empirical_positive_rate": float(np.mean(empirical_positive)),
        "heuristic_positive_rate": float(np.mean(heuristic_positive)),
    }
    metrics = {
        "linear": summarize_donor_rows(all_rows, score_key="linear_synergy"),
        "rbf_kernel": summarize_donor_rows(all_rows, score_key="rbf_synergy"),
    }
    return oracle_summary, metrics


def evaluate_local_donor_game(
    local_game: LocalDonorRescuePIDGame,
    *,
    seed: int,
    group_id: int,
) -> list[dict[str, float]]:
    all_masks = enumerate_masks(local_game.num_atoms)
    oracle_scores = local_game.utility_batch(all_masks)
    split = sample_mask_splits(
        num_masks=all_masks.shape[0],
        train_queries=min(8, all_masks.shape[0] - 2),
        holdout_queries=min(4, all_masks.shape[0]),
        lds_num_subsets=min(4, all_masks.shape[0]),
        seed=seed,
    )
    linear = fit_linear_subset_surrogate(
        all_masks[split["train"]],
        oracle_scores[split["train"]],
        num_atoms=local_game.num_atoms,
        candidate_edges=[],
        config=StructuredSubsetSurrogateConfig(feature_bernoulli_p=0.5, ridge_lambda=1e-3, pair_probe_weight=0.0),
    )
    rbf = fit_rbf_subset_surrogate(
        all_masks[split["train"]],
        oracle_scores[split["train"]],
        num_atoms=local_game.num_atoms,
        candidate_edges=[],
        config=RBFSubsetSurrogateConfig(ridge_lambda=1e-3),
    )
    oracle_pid = estimate_cross_modal_pid_from_scores(
        oracle_scores,
        num_atoms=local_game.num_atoms,
        atom_samples=local_game.atom_samples,
        atom_modalities=local_game.atom_modalities,
        modality_pair=(0, 1),
    )
    linear_scores = np.asarray(linear.predict_batch(all_masks), dtype=np.float64)
    rbf_scores = np.asarray(rbf.predict_batch(all_masks), dtype=np.float64)
    linear_pid = estimate_cross_modal_pid_from_scores(
        linear_scores,
        num_atoms=local_game.num_atoms,
        atom_samples=local_game.atom_samples,
        atom_modalities=local_game.atom_modalities,
        modality_pair=(0, 1),
    )
    rbf_pid = estimate_cross_modal_pid_from_scores(
        rbf_scores,
        num_atoms=local_game.num_atoms,
        atom_samples=local_game.atom_samples,
        atom_modalities=local_game.atom_modalities,
        modality_pair=(0, 1),
    )
    rows: list[dict[str, float]] = []
    for donor_offset, donor_index in enumerate(local_game.donor_indices, start=1):
        pair_key = (int(local_game.anchor_train_index), int(donor_index))
        left_mask = 1 << 0
        right_mask = 1 << donor_offset
        g00 = float(oracle_scores[0])
        g10 = float(oracle_scores[left_mask])
        g01 = float(oracle_scores[right_mask])
        g11 = float(oracle_scores[left_mask | right_mask])
        rows.append(
            {
                "group_id": float(group_id),
                "oracle_synergy": float(oracle_pid[pair_key]["S"]),
                "linear_synergy": float(linear_pid[pair_key]["S"]),
                "rbf_synergy": float(rbf_pid[pair_key]["S"]),
                "empirical_gain": float(g11 - g10),
                "joint_over_best": float(g11 - max(g00, g10, g01)),
                "empirical_positive": float((g11 - g10) > 1e-12),
                "heuristic_positive": float(local_game.donor_is_positive[donor_offset - 1]),
            }
        )
    return rows


def summarize_donor_rows(rows: list[dict[str, float]], *, score_key: str) -> dict[str, float]:
    scores = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    gains = np.asarray([row["empirical_gain"] for row in rows], dtype=np.float64)
    empirical_labels = np.asarray([row["empirical_positive"] for row in rows], dtype=bool)
    heuristic_labels = np.asarray([row["heuristic_positive"] for row in rows], dtype=bool)
    group_ids = np.asarray([row["group_id"] for row in rows], dtype=np.int64)
    top1 = int(np.argmax(scores)) if scores.size > 0 else 0
    anchor_best_hits = []
    anchor_top1_gains = []
    anchor_regrets = []
    anchor_top1_heuristic = []
    for group_id in np.unique(group_ids):
        group_mask = group_ids == int(group_id)
        group_scores = scores[group_mask]
        group_gains = gains[group_mask]
        group_heuristic = heuristic_labels[group_mask]
        if group_scores.size == 0:
            continue
        pred_index = int(np.argmax(group_scores))
        true_index = int(np.argmax(group_gains))
        anchor_best_hits.append(float(pred_index == true_index))
        anchor_top1_gains.append(float(group_gains[pred_index]))
        anchor_regrets.append(float(np.max(group_gains) - group_gains[pred_index]))
        anchor_top1_heuristic.append(float(group_heuristic[pred_index]))
    return {
        "donor_synergy_vs_gain_spearman": spearman_corr(scores, gains),
        "donor_synergy_vs_gain_pearson": pearson_corr(scores, gains),
        "donor_empirical_auprc": binary_auprc(empirical_labels, scores),
        "donor_empirical_auroc": binary_auroc(empirical_labels, scores),
        "donor_empirical_recall_at_1": binary_recall_at_k(empirical_labels, scores, k=1),
        "donor_heuristic_auprc": binary_auprc(heuristic_labels, scores),
        "donor_heuristic_auroc": binary_auroc(heuristic_labels, scores),
        "donor_heuristic_recall_at_1": binary_recall_at_k(heuristic_labels, scores, k=1),
        "top1_empirical_gain": float(gains[top1]) if gains.size > 0 else 0.0,
        "anchor_best_hit_at_1": float(np.mean(anchor_best_hits)) if anchor_best_hits else 0.0,
        "anchor_top1_gain_mean": float(np.mean(anchor_top1_gains)) if anchor_top1_gains else 0.0,
        "anchor_gain_regret_mean": float(np.mean(anchor_regrets)) if anchor_regrets else 0.0,
        "anchor_top1_heuristic_rate": float(np.mean(anchor_top1_heuristic)) if anchor_top1_heuristic else 0.0,
    }


def build_real_subset_pid_benchmark(
    config: DatasetRunConfig,
    *,
    focal_indices: np.ndarray | None = None,
) -> RealSubsetPIDBenchmark:
    processed = load_processed_dataset(config)
    if focal_indices is None:
        focal_indices = select_focal_indices(
            processed["train_targets"],
            task=processed["task"],
            num_focal_samples=config.num_focal_samples,
            seed=config.seed,
        )
    return RealSubsetPIDBenchmark(
        dataset_name=config.dataset_name,
        train_x=processed["train_x"],
        train_y=processed["train_y"],
        train_targets=processed["train_targets"],
        val_x=processed["val_x"],
        val_y=processed["val_y"],
        val_targets=processed["val_targets"],
        focal_indices=np.asarray(focal_indices, dtype=np.int64),
        corruption_alpha=config.corruption_alpha,
        task=processed["task"],
        ridge_lambda=config.ridge_lambda,
        score_metric=config.score_metric,
    )


def load_processed_dataset(config: DatasetRunConfig) -> dict[str, np.ndarray]:
    raw = load_dataset(config.dataset_name, seed=config.seed)
    return preprocess_real_dataset(
        raw,
        dataset_name=config.dataset_name,
        task=str(raw["task"]),
        seed=config.seed,
        num_train_samples=config.num_train_samples,
        num_val_samples=config.num_val_samples,
        max_classes_food101=config.max_classes_food101,
        max_modality_dim=config.max_modality_dim,
    )


def preprocess_real_dataset(
    data: dict[str, object],
    *,
    dataset_name: str,
    task: str,
    seed: int,
    num_train_samples: int,
    num_val_samples: int,
    max_classes_food101: int,
    max_modality_dim: int,
) -> dict[str, np.ndarray]:
    train_x = tensor_to_numpy(data["train_x"])
    train_y = tensor_to_numpy(data["train_y"])
    train_targets = tensor_labels_to_numpy(data["train_labels"], task=task)
    val_x = tensor_to_numpy(data["val_x"])
    val_y = tensor_to_numpy(data["val_y"])
    val_targets = tensor_labels_to_numpy(data["val_labels"], task=task)

    if dataset_name == "food101":
        selected_classes = choose_top_classes(train_targets, max_classes_food101)
        train_mask = np.isin(train_targets, selected_classes)
        val_mask = np.isin(val_targets, selected_classes)
        train_x = train_x[train_mask]
        train_y = train_y[train_mask]
        train_targets = remap_labels(train_targets[train_mask], selected_classes)
        val_x = val_x[val_mask]
        val_y = val_y[val_mask]
        val_targets = remap_labels(val_targets[val_mask], selected_classes)

    train_x, val_x = maybe_random_project_pair(train_x, val_x, max_dim=max_modality_dim, seed=1_000 + seed)
    train_y, val_y = maybe_random_project_pair(train_y, val_y, max_dim=max_modality_dim, seed=2_000 + seed)

    train_indices = subsample_indices(train_targets, num_train_samples, seed=seed, task=task)
    val_indices = subsample_indices(val_targets, num_val_samples, seed=10_000 + seed, task=task)
    return {
        "train_x": train_x[train_indices],
        "train_y": train_y[train_indices],
        "train_targets": train_targets[train_indices],
        "val_x": val_x[val_indices],
        "val_y": val_y[val_indices],
        "val_targets": val_targets[val_indices],
        "task": task,
    }


def ridge_probe_score(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    ridge_lambda: float,
    task: str,
    metric: str,
) -> float:
    train = np.asarray(train_features, dtype=np.float64)
    val = np.asarray(val_features, dtype=np.float64)

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train = (train - mean) / std
    val = (val - mean) / std

    if task == "multilabel":
        y_train = np.asarray(train_targets, dtype=np.float64)
        y_val = np.asarray(val_targets, dtype=np.float64)
        classes = None
        labels_val = None
    else:
        labels_train = np.asarray(train_targets, dtype=np.int64).reshape(-1)
        labels_val = np.asarray(val_targets, dtype=np.int64).reshape(-1)
        classes = np.unique(labels_train)
        class_to_index = {int(label): index for index, label in enumerate(classes)}
        y_train = np.zeros((labels_train.shape[0], classes.shape[0]), dtype=np.float64)
        for row, label in enumerate(labels_train):
            y_train[row, class_to_index[int(label)]] = 1.0
        y_val = None

    if train.shape[1] <= train.shape[0]:
        lhs = train.T @ train + ridge_lambda * np.eye(train.shape[1], dtype=np.float64)
        rhs = train.T @ y_train
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs
    else:
        kernel = train @ train.T + ridge_lambda * np.eye(train.shape[0], dtype=np.float64)
        try:
            dual = np.linalg.solve(kernel, y_train)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(kernel) @ y_train
        weights = train.T @ dual

    logits = val @ weights

    if metric == "accuracy":
        if task == "multilabel":
            predictions = (logits >= 0.5).astype(np.float64)
            return multilabel_micro_f1(y_val, predictions)
        predictions = classes[np.argmax(logits, axis=1)]
        return float(np.mean(predictions == labels_val))

    if task == "multilabel":
        loss = binary_cross_entropy_with_logits(logits, y_val)
    else:
        class_to_index = {int(label): index for index, label in enumerate(classes)}
        target_indices = np.asarray([class_to_index[int(label)] for label in labels_val], dtype=np.int64)
        loss = multiclass_cross_entropy(logits, target_indices)

    if metric == "loss":
        return float(loss)
    if metric == "neg_loss":
        return float(-loss)
    raise ValueError(f"unsupported metric: {metric}")


def multiclass_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - logsumexp
    return float(-np.mean(log_probs[np.arange(targets.shape[0]), targets]))


def binary_cross_entropy_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    positive = np.maximum(logits, 0.0)
    return float(np.mean(positive - logits * targets + np.log1p(np.exp(-np.abs(logits)))))


def multilabel_micro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(targets, dtype=np.float64) > 0.0
    pred = np.asarray(predictions, dtype=np.float64) > 0.0
    tp = float(np.sum(truth & pred))
    fp = float(np.sum((~truth) & pred))
    fn = float(np.sum(truth & (~pred)))
    denom = 2.0 * tp + fp + fn
    if denom <= 0.0:
        return 1.0
    return float((2.0 * tp) / denom)


def subsample_indices(targets: np.ndarray, max_samples: int, *, seed: int, task: str) -> np.ndarray:
    if task == "multilabel":
        count = int(targets.shape[0])
        if count <= max_samples:
            return np.arange(count, dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.permutation(count)[:max_samples]).astype(np.int64)
    return stratified_subsample(np.asarray(targets, dtype=np.int64).reshape(-1), max_samples, seed=seed)


def select_focal_indices(
    train_targets: np.ndarray,
    *,
    task: str,
    num_focal_samples: int,
    seed: int,
) -> np.ndarray:
    if task == "multilabel":
        rng = np.random.default_rng(seed)
        count = int(train_targets.shape[0])
        if count < num_focal_samples:
            raise ValueError(f"need at least {num_focal_samples} train samples, got {count}")
        return np.sort(rng.permutation(count)[:num_focal_samples]).astype(np.int64)
    return stratified_subsample(
        np.asarray(train_targets, dtype=np.int64).reshape(-1),
        num_focal_samples,
        seed=30_000 + seed,
    )


def enumerate_masks(num_atoms: int) -> np.ndarray:
    masks = np.zeros((2**num_atoms, num_atoms), dtype=bool)
    for index in range(2**num_atoms):
        for bit in range(num_atoms):
            masks[index, bit] = bool((index >> bit) & 1)
    return masks


def sample_mask_splits(
    *,
    num_masks: int,
    train_queries: int,
    holdout_queries: int,
    lds_num_subsets: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_masks)
    train = np.sort(permutation[:train_queries])
    holdout_start = train_queries
    holdout = np.sort(permutation[holdout_start: holdout_start + holdout_queries])
    lds_start = holdout_start + holdout_queries
    lds = np.sort(permutation[lds_start: lds_start + lds_num_subsets])
    if holdout.shape[0] == 0:
        holdout = train.copy()
    if lds.shape[0] == 0:
        lds = holdout.copy()
    return {"train": train, "holdout": holdout, "lds": lds}


def pair_component_arrays(pair_components: dict[tuple[int, int], dict[str, float]]) -> dict[str, np.ndarray]:
    pair_keys = sorted(pair_components)
    return {
        name: np.asarray([pair_components[key][name] for key in pair_keys], dtype=np.float64)
        for name in ["R", "U1", "U2", "S"]
    }


def list_pair_keys(
    *,
    atom_samples: np.ndarray,
    atom_modalities: np.ndarray,
    modality_pair: tuple[int, int],
) -> list[tuple[int, int]]:
    left_atoms = {
        int(sample): int(index)
        for index, (sample, modality) in enumerate(zip(atom_samples, atom_modalities, strict=True))
        if int(modality) == int(modality_pair[0])
    }
    right_atoms = {
        int(sample): int(index)
        for index, (sample, modality) in enumerate(zip(atom_samples, atom_modalities, strict=True))
        if int(modality) == int(modality_pair[1])
    }
    keys: list[tuple[int, int]] = []
    for left_sample in sorted(left_atoms):
        for right_sample in sorted(right_atoms):
            if left_sample == right_sample:
                continue
            keys.append((left_sample, right_sample))
    return keys


def summarize_dataset_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    method_rows: dict[str, list[dict[str, float]]] = {}
    behavior_rows: dict[str, list[dict[str, float]]] = {}
    donor_rows: dict[str, list[dict[str, float]]] = {}
    for run in runs:
        for method_name, metrics in run["method_metrics"].items():
            method_rows.setdefault(method_name, []).append(metrics)
        for method_name, metrics in run["behavior_metrics"].items():
            behavior_rows.setdefault(method_name, []).append(metrics)
        for method_name, metrics in run["donor_alignment_metrics"].items():
            donor_rows.setdefault(method_name, []).append(metrics)
    return {
        "runs": runs,
        "oracle_summary": summarize_numeric_dicts([run["oracle_stats"] for run in runs]),
        "method_summary": {
            method_name: summarize_numeric_dicts(rows) for method_name, rows in method_rows.items()
        },
        "behavior_summary": {
            method_name: summarize_numeric_dicts(rows) for method_name, rows in behavior_rows.items()
        },
        "donor_alignment_summary": {
            method_name: summarize_numeric_dicts(rows) for method_name, rows in donor_rows.items()
        },
        "sweep_curve": summarize_curve_runs(runs, row_key="sweep_rows", x_key="corruption_alpha"),
        "sweep_summary": summarize_numeric_dicts([run["sweep_summary"] for run in runs]),
    }


def resolve_available_datasets(requested_datasets: Iterable[str]) -> tuple[list[str], list[dict[str, str]]]:
    resolved: list[str] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    requested = list(requested_datasets)
    expanded = list(SUPPORTED_DATASETS) if "all" in requested else requested
    for dataset_name in expanded:
        if dataset_name in seen:
            continue
        seen.add(dataset_name)
        try:
            load_dataset(dataset_name, seed=0)
            resolved.append(dataset_name)
        except Exception as exc:
            skipped.append(
                {
                    "dataset": dataset_name,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return resolved, skipped


def to_markdown(payload: dict[str, object]) -> str:
    score_metric = payload.get("config", {}).get("score_metric", "accuracy")
    lines = [
        "# Real Subset PID Benchmark",
        "",
        "This report evaluates a real-data subset-to-score surrogate on all available datasets in the repo.",
        "",
        "## Algorithm",
        "",
        "1. Pick a small focal set of training samples. Each atom is one observed sample-modality pair `(i, m)` from that focal set.",
        "2. Define a subset mask `s in {0,1}^P` over those atoms. If `s_(i,m)=0`, that modality is corrupted for focal sample `i`; if `s_(i,m)=1`, the original frozen feature is restored.",
        f"3. Define the oracle scalar outcome `F(s)` as the validation `{score_metric}` of a ridge probe trained on the corrupted/restored training set.",
        "4. Sample fixed training / holdout / LDS subset masks, fit a mask-to-score surrogate `g(s)`, and compare `g(s)` to `F(s)` with heldout Spearman as in LDS.",
        "5. On the same real game, compute PID-compatible cross-sample cross-modal components from either the oracle score table `F` or the learned score table `g` via finite differences.",
        "",
    ]
    resolved = payload.get("resolved_datasets", [])
    skipped = payload.get("skipped_datasets", [])
    if resolved:
        lines.append("Resolved datasets: " + ", ".join(resolved))
        lines.append("")
    if skipped:
        lines.append("Skipped datasets:")
        for item in skipped:
            lines.append(f"- {item['dataset']}: {item['reason']}")
        lines.append("")

    for dataset_name, result in payload["datasets"].items():
        config_rows = payload.get("effective_dataset_configs", {}).get(dataset_name, [])
        config_line = ""
        if config_rows:
            first = config_rows[0]
            config_line = (
                f"Train {first['num_train_samples']}, Val {first['num_val_samples']}, "
                f"Focal samples {first['num_focal_samples']}, Atoms {2 * first['num_focal_samples']}, "
                f"Donor anchors {first['donor_num_anchors']}, Donors/anchor {first['donor_candidates_per_anchor']}, "
                f"Train queries {first['train_queries']}, Holdout {first['holdout_queries']}, LDS subsets {first['lds_num_subsets']}, "
                f"Projection dim <= {first['max_modality_dim']}, Outcome metric `{first['score_metric']}`"
            )
        oracle_summary = result["oracle_summary"]
        method_summary = result["method_summary"]
        behavior_summary = result["behavior_summary"]
        donor_alignment_summary = result["donor_alignment_summary"]
        sweep_summary = result["sweep_summary"]
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                config_line,
                "",
                "### Oracle Game",
                _metric_line("score mean", oracle_summary["score_mean"]),
                _metric_line("score std", oracle_summary["score_std"]),
                _metric_line("positive synergy mass", oracle_summary["positive_synergy_mass"]),
                _metric_line("positive redundancy mass", oracle_summary["positive_redundancy_mass"]),
                _metric_line("mean joint-over-best", oracle_summary["mean_joint_over_best"]),
                _metric_line("mean empty rescue gain", oracle_summary["mean_empty_rescue_gain"]),
                _metric_line("mean full-context joint-over-best", oracle_summary["mean_full_context_joint_over_best"]),
                "",
                "### Surrogate Comparison",
                _method_line("Linear", method_summary["linear"]),
                _method_line("Hamming-RBF", method_summary["rbf_kernel"]),
                "",
                "### Behavioral Validity",
                _behavior_line("Linear", behavior_summary["linear"]),
                _behavior_line("Hamming-RBF", behavior_summary["rbf_kernel"]),
                "",
                "### Donor-Style Rescue Alignment",
                _donor_line("Linear", donor_alignment_summary["linear"]),
                _donor_line("Hamming-RBF", donor_alignment_summary["rbf_kernel"]),
                "",
                "### Corruption Sweep",
                _metric_line("Corr(alpha, oracle synergy)", sweep_summary["corr_alpha_oracle_synergy"]),
                _metric_line("Corr(alpha, linear synergy)", sweep_summary["corr_alpha_linear_synergy"]),
                _metric_line("Corr(alpha, RBF synergy)", sweep_summary["corr_alpha_rbf_synergy"]),
                _metric_line("Corr(oracle, linear synergy)", sweep_summary["corr_oracle_linear_synergy"]),
                _metric_line("Corr(oracle, RBF synergy)", sweep_summary["corr_oracle_rbf_synergy"]),
                "",
                "### Raw Sweep Rows",
                "| alpha | oracle S+ mean | linear S+ mean | rbf S+ mean | oracle R+ mean | linear R+ mean | rbf R+ mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["sweep_curve"]:
            lines.append(
                f"| {row['corruption_alpha']:.2f} | "
                f"{row['oracle_positive_synergy_mass_mean']:.4f} | "
                f"{row['linear_positive_synergy_mass_mean']:.4f} | "
                f"{row['rbf_positive_synergy_mass_mean']:.4f} | "
                f"{row['oracle_positive_redundancy_mass_mean']:.4f} | "
                f"{row['linear_positive_redundancy_mass_mean']:.4f} | "
                f"{row['rbf_positive_redundancy_mass_mean']:.4f} |"
            )
        lines.extend(
            [
                "",
                "### Raw Runs",
            ]
        )
        for run in result["runs"]:
            lines.append(
                f"- seed {run['seed']}: focal={run['focal_indices']}, "
                f"oracle S+={run['oracle_stats']['positive_synergy_mass']:.4f}, "
                f"oracle R+={run['oracle_stats']['positive_redundancy_mass']:.4f}, "
                f"linear LDS={run['method_metrics']['linear']['subset_lds_spearman']:.4f}, "
                f"rbf LDS={run['method_metrics']['rbf_kernel']['subset_lds_spearman']:.4f}"
            )
        lines.append("")

    aggregate = payload.get("aggregate", {})
    if aggregate:
        methods = aggregate.get("methods", {})
        behavior = aggregate.get("behavior", {})
        donor_alignment = aggregate.get("donor_alignment", {})
        sweeps = aggregate.get("sweeps", {})
        lines.extend(
            [
                "## Aggregate",
                "",
                _method_line("Linear", methods["linear"]),
                _method_line("Hamming-RBF", methods["rbf_kernel"]),
                "",
                "### Aggregate Behavioral Validity",
                _behavior_line("Linear", behavior["linear"]),
                _behavior_line("Hamming-RBF", behavior["rbf_kernel"]),
                "",
                "### Aggregate Donor Alignment",
                _donor_line("Linear", donor_alignment["linear"]),
                _donor_line("Hamming-RBF", donor_alignment["rbf_kernel"]),
                "",
                "### Aggregate Sweep",
                _metric_line("Corr(alpha, oracle synergy)", sweeps["corr_alpha_oracle_synergy"]),
                _metric_line("Corr(alpha, linear synergy)", sweeps["corr_alpha_linear_synergy"]),
                _metric_line("Corr(alpha, RBF synergy)", sweeps["corr_alpha_rbf_synergy"]),
                _metric_line("Corr(oracle, linear synergy)", sweeps["corr_oracle_linear_synergy"]),
                _metric_line("Corr(oracle, RBF synergy)", sweeps["corr_oracle_rbf_synergy"]),
                "",
            ]
        )
    return "\n".join(lines)


def _metric_line(name: str, summary: dict[str, float]) -> str:
    return f"- {name}: {summary['mean']:.4f} ± {summary['std']:.4f}"


def _method_line(name: str, summary: dict[str, dict[str, float]]) -> str:
    return (
        f"- {name}: "
        f"LDS-Spearman {summary['subset_lds_spearman']['mean']:.4f} ± {summary['subset_lds_spearman']['std']:.4f}, "
        f"R2 {summary['utility_r2']['mean']:.4f} ± {summary['utility_r2']['std']:.4f}, "
        f"S+ mass error {summary['synergy_mass_abs_error']['mean']:.4f} ± {summary['synergy_mass_abs_error']['std']:.4f}, "
        f"R+ mass error {summary['redundancy_mass_abs_error']['mean']:.4f} ± {summary['redundancy_mass_abs_error']['std']:.4f}"
    )


def _behavior_line(name: str, summary: dict[str, dict[str, float]]) -> str:
    return (
        f"- {name}: "
        f"synergy vs joint-over-best Spearman {summary['synergy_vs_joint_over_best_spearman']['mean']:.4f} ± {summary['synergy_vs_joint_over_best_spearman']['std']:.4f}, "
        f"synergy vs full-context Spearman {summary['synergy_vs_full_context_spearman']['mean']:.4f} ± {summary['synergy_vs_full_context_spearman']['std']:.4f}, "
        f"top-k joint-over-best mean {summary['topk_synergy_joint_over_best_mean']['mean']:.4f} ± {summary['topk_synergy_joint_over_best_mean']['std']:.4f}, "
        f"top-k full-context positive rate {summary['topk_synergy_full_context_positive_rate']['mean']:.4f} ± {summary['topk_synergy_full_context_positive_rate']['std']:.4f}"
    )


def _donor_line(name: str, summary: dict[str, dict[str, float]]) -> str:
    return (
        f"- {name}: "
        f"synergy vs donor gain Spearman {summary['donor_synergy_vs_gain_spearman']['mean']:.4f} ± {summary['donor_synergy_vs_gain_spearman']['std']:.4f}, "
        f"anchor best hit@1 {summary['anchor_best_hit_at_1']['mean']:.4f} ± {summary['anchor_best_hit_at_1']['std']:.4f}, "
        f"anchor regret {summary['anchor_gain_regret_mean']['mean']:.4f} ± {summary['anchor_gain_regret_mean']['std']:.4f}, "
        f"heuristic-donor AUPRC {summary['donor_heuristic_auprc']['mean']:.4f} ± {summary['donor_heuristic_auprc']['std']:.4f}, "
        f"empirical-donor AUPRC {summary['donor_empirical_auprc']['mean']:.4f} ± {summary['donor_empirical_auprc']['std']:.4f}, "
        f"anchor top1 gain {summary['anchor_top1_gain_mean']['mean']:.4f} ± {summary['anchor_top1_gain_mean']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
