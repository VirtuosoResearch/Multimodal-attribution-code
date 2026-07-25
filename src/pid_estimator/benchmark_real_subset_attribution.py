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
    fit_linear_subset_surrogate,
    fit_rbf_subset_surrogate,
    fit_structured_subset_surrogate,
    sample_pair_probes,
)
from pid_estimator.subset_attribution_experiment import (
    binary_auprc,
    binary_auroc,
    binary_recall_at_k,
    monotonicity_score,
    pearson_corr,
    r2_score,
    rmse,
    spearman_corr,
    summarize_numeric_dicts,
)

SUPPORTED_DATASETS = (
    "food101",
    "oxford_pets",
    "imdb",
    "humor",
    "sarcasm",
    "mosi",
    "mosei",
    "avmnist",
)


@dataclass(frozen=True)
class RealSubsetAttributionConfig:
    datasets: tuple[str, ...] = SUPPORTED_DATASETS
    seeds: tuple[int, ...] = (0, 1, 2)
    num_train_samples: int = 2000
    num_val_samples: int = 1000
    num_anchors: int = 6
    candidates_per_anchor: int = 3
    global_queries: int = 48
    holdout_queries: int = 64
    lds_num_subsets: int = 64
    pair_probes_per_edge: int = 1
    ridge_lambda: float = 1e-3
    pair_probe_weight: float = 4.0
    corruption_alpha: float = 1.0
    partner_purity: float = 1.0 / 3.0
    corruption_sweep: tuple[float, ...] = (0.0, 0.5, 1.0)
    purity_sweep: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    max_classes_food101: int = 10
    max_modality_dim: int = 128
    use_dataset_presets: bool = True


@dataclass(frozen=True)
class DatasetRunConfig:
    dataset_name: str
    seed: int
    num_train_samples: int
    num_val_samples: int
    num_anchors: int
    candidates_per_anchor: int
    corruption_alpha: float
    partner_purity: float
    ridge_lambda: float
    pair_probe_weight: float
    max_classes_food101: int = 10
    max_modality_dim: int = 128


@dataclass
class RealRescueBenchmark:
    dataset_name: str
    train_x: np.ndarray
    train_y_clean: np.ndarray
    train_targets: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_targets: np.ndarray
    anchor_train_indices: np.ndarray
    atom_anchor_ids: np.ndarray
    atom_is_positive: np.ndarray
    atom_donor_y: np.ndarray
    candidate_edges: list[tuple[int, int]]
    corruption_alpha: float
    task: str

    def __post_init__(self) -> None:
        self.train_x = np.asarray(self.train_x, dtype=np.float64)
        self.train_y_clean = np.asarray(self.train_y_clean, dtype=np.float64)
        self.train_targets = np.asarray(self.train_targets)
        self.val_x = np.asarray(self.val_x, dtype=np.float64)
        self.val_y = np.asarray(self.val_y, dtype=np.float64)
        self.val_targets = np.asarray(self.val_targets)
        self.anchor_train_indices = np.asarray(self.anchor_train_indices, dtype=np.int64).reshape(-1)
        self.atom_anchor_ids = np.asarray(self.atom_anchor_ids, dtype=np.int64).reshape(-1)
        self.atom_is_positive = np.asarray(self.atom_is_positive, dtype=bool).reshape(-1)
        self.atom_donor_y = np.asarray(self.atom_donor_y, dtype=np.float64)
        self.num_atoms = int(self.atom_anchor_ids.shape[0])
        self.empty_mask = np.zeros(self.num_atoms, dtype=bool)
        self.base_missing_y = self.train_y_clean.copy()
        self.base_missing_y[self.anchor_train_indices] *= max(1.0 - float(self.corruption_alpha), 0.0)
        self._group_to_atom_indices = {
            int(anchor_id): np.where(self.atom_anchor_ids == anchor_id)[0]
            for anchor_id in np.unique(self.atom_anchor_ids)
        }
        self._cache: dict[bytes, float] = {}
        self.empty_accuracy = self._evaluate_accuracy(self.base_missing_y)
        self._cache[self._mask_key(self.empty_mask)] = 0.0

    def utility(self, mask: np.ndarray) -> float:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.shape[0] != self.num_atoms:
            raise ValueError(f"expected mask with {self.num_atoms} atoms, got {selected.shape[0]}")
        key = self._mask_key(selected)
        if key in self._cache:
            return self._cache[key]
        rescued_y = self.base_missing_y.copy()
        for anchor_id, atom_indices in self._group_to_atom_indices.items():
            chosen = atom_indices[selected[atom_indices]]
            if chosen.size == 0:
                continue
            donor_mean = np.mean(self.atom_donor_y[chosen], axis=0)
            anchor_train_index = int(self.anchor_train_indices[anchor_id])
            rescued_y[anchor_train_index] = (
                (1.0 - float(self.corruption_alpha)) * self.train_y_clean[anchor_train_index]
                + float(self.corruption_alpha) * donor_mean
            )
        value = self._evaluate_accuracy(rescued_y) - self.empty_accuracy
        self._cache[key] = float(value)
        return float(value)

    def utility_batch(self, masks: np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=bool)
        if values.ndim != 2 or values.shape[1] != self.num_atoms:
            raise ValueError(f"expected masks with shape [N, {self.num_atoms}]")
        return np.asarray([self.utility(row) for row in values], dtype=np.float64)

    def positive_mask(self) -> np.ndarray:
        return self.atom_is_positive.copy()

    def _mask_key(self, mask: np.ndarray) -> bytes:
        return np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()

    def _evaluate_accuracy(self, train_y_variant: np.ndarray) -> float:
        train_features = np.concatenate([self.train_x, train_y_variant], axis=1)
        val_features = np.concatenate([self.val_x, self.val_y], axis=1)
        return ridge_probe_score(
            train_features,
            self.train_targets,
            val_features,
            self.val_targets,
            ridge_lambda=1e-3,
            task=self.task,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data subset-attribution rescue benchmarks.")
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-train-samples", type=int, default=2000)
    parser.add_argument("--num-val-samples", type=int, default=1000)
    parser.add_argument("--num-anchors", type=int, default=6)
    parser.add_argument("--candidates-per-anchor", type=int, default=3)
    parser.add_argument("--global-queries", type=int, default=48)
    parser.add_argument("--holdout-queries", type=int, default=64)
    parser.add_argument("--lds-num-subsets", type=int, default=64)
    parser.add_argument("--pair-probes-per-edge", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--pair-probe-weight", type=float, default=4.0)
    parser.add_argument("--corruption-alpha", type=float, default=1.0)
    parser.add_argument("--partner-purity", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-classes-food101", type=int, default=10)
    parser.add_argument("--max-modality-dim", type=int, default=128)
    parser.add_argument("--no-dataset-presets", action="store_true")
    parser.add_argument("--output-json", type=str, default="src/pid_estimator/results/real_subset_attribution_experiment.json")
    parser.add_argument("--output-md", type=str, default="src/pid_estimator/results/real_subset_attribution_experiment.md")
    args = parser.parse_args()

    config = RealSubsetAttributionConfig(
        datasets=tuple(args.datasets),
        seeds=tuple(args.seeds),
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        num_anchors=args.num_anchors,
        candidates_per_anchor=args.candidates_per_anchor,
        global_queries=args.global_queries,
        holdout_queries=args.holdout_queries,
        lds_num_subsets=args.lds_num_subsets,
        pair_probes_per_edge=args.pair_probes_per_edge,
        ridge_lambda=args.ridge_lambda,
        pair_probe_weight=args.pair_probe_weight,
        corruption_alpha=args.corruption_alpha,
        partner_purity=args.partner_purity,
        max_classes_food101=args.max_classes_food101,
        max_modality_dim=args.max_modality_dim,
        use_dataset_presets=not args.no_dataset_presets,
    )
    payload = run_real_benchmarks(config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload))


def run_real_benchmarks(config: RealSubsetAttributionConfig) -> dict[str, object]:
    resolved_datasets, skipped_datasets = resolve_available_datasets(config.datasets)
    dataset_results: dict[str, object] = {}
    effective_configs: dict[str, object] = {}
    aggregate_method_rows: dict[str, list[dict[str, float]]] = {}
    aggregate_ablation_rows: list[dict[str, float]] = []
    aggregate_sweep_rows: list[dict[str, float]] = []

    for dataset_name in resolved_datasets:
        runs = []
        dataset_configs = []
        for seed in config.seeds:
            resolved = resolve_dataset_run_config(config, dataset_name=dataset_name, seed=seed)
            dataset_configs.append(
                {
                    "seed": int(seed),
                    "dataset_config": asdict(resolved["dataset_config"]),
                    "global_queries": int(resolved["global_queries"]),
                    "holdout_queries": int(resolved["holdout_queries"]),
                }
            )
            run = run_single_dataset(
                resolved["dataset_config"],
                global_queries=resolved["global_queries"],
                holdout_queries=resolved["holdout_queries"],
                lds_num_subsets=config.lds_num_subsets,
                pair_probes_per_edge=config.pair_probes_per_edge,
                corruption_sweep=config.corruption_sweep,
                purity_sweep=config.purity_sweep,
            )
            runs.append(run)
            for method, metrics in run["base_metrics"].items():
                aggregate_method_rows.setdefault(method, []).append(metrics)
            aggregate_ablation_rows.append(run["ablation"])
            aggregate_sweep_rows.append(run["sweeps"])
        dataset_results[dataset_name] = summarize_dataset_runs(runs)
        effective_configs[dataset_name] = dataset_configs

    return {
        "config": asdict(config),
        "resolved_datasets": resolved_datasets,
        "skipped_datasets": skipped_datasets,
        "effective_dataset_configs": effective_configs,
        "datasets": dataset_results,
        "aggregate": {
            "base_methods": {
                method: summarize_numeric_dicts(rows)
                for method, rows in aggregate_method_rows.items()
            },
            "pair_probe_ablation": summarize_numeric_dicts(aggregate_ablation_rows),
            "sweeps": summarize_numeric_dicts(aggregate_sweep_rows),
        },
    }


def resolve_dataset_run_config(
    config: RealSubsetAttributionConfig,
    *,
    dataset_name: str,
    seed: int,
) -> dict[str, object]:
    num_train_samples = config.num_train_samples
    num_val_samples = config.num_val_samples
    num_anchors = config.num_anchors
    global_queries = config.global_queries
    holdout_queries = config.holdout_queries

    if config.use_dataset_presets:
        if dataset_name == "avmnist":
            num_train_samples = 1200
            num_val_samples = 400
            num_anchors = 4
            global_queries = 32
            holdout_queries = 32
        elif dataset_name == "humor":
            num_train_samples = 1200
            num_val_samples = 400
            num_anchors = 4
            global_queries = 24
            holdout_queries = 24
        elif dataset_name == "sarcasm":
            num_train_samples = 300
            num_val_samples = 100
            num_anchors = 4
            global_queries = 24
            holdout_queries = 24
        elif dataset_name == "mosi":
            num_train_samples = 800
            num_val_samples = 200
            num_anchors = 4
            global_queries = 24
            holdout_queries = 24
        elif dataset_name == "mosei":
            num_train_samples = 1200
            num_val_samples = 400
            num_anchors = 4
            global_queries = 24
            holdout_queries = 24
        elif dataset_name == "imdb":
            num_train_samples = 1200
            num_val_samples = 400
            num_anchors = 4
            global_queries = 24
            holdout_queries = 24
        elif dataset_name == "food101":
            num_train_samples = 500
            num_val_samples = 300
            num_anchors = 8
            global_queries = 24
            holdout_queries = 24
        elif dataset_name == "oxford_pets":
            num_train_samples = 500
            num_val_samples = 300
            num_anchors = 8
            global_queries = 24
            holdout_queries = 24

    return {
        "dataset_config": DatasetRunConfig(
            dataset_name=dataset_name,
            seed=seed,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            num_anchors=num_anchors,
            candidates_per_anchor=config.candidates_per_anchor,
            corruption_alpha=config.corruption_alpha,
            partner_purity=config.partner_purity,
            ridge_lambda=config.ridge_lambda,
            pair_probe_weight=config.pair_probe_weight,
            max_classes_food101=config.max_classes_food101,
            max_modality_dim=config.max_modality_dim,
        ),
        "global_queries": global_queries,
        "holdout_queries": holdout_queries,
    }


def run_single_dataset(
    config: DatasetRunConfig,
    *,
    global_queries: int,
    holdout_queries: int,
    lds_num_subsets: int,
    pair_probes_per_edge: int,
    corruption_sweep: tuple[float, ...],
    purity_sweep: tuple[float, ...],
) -> dict[str, object]:
    benchmark = build_real_rescue_benchmark(config)
    subset_masks, subset_values = sample_global_queries_for_benchmark(benchmark, num_queries=global_queries, seed=config.seed)
    holdout_masks, holdout_values = sample_global_queries_for_benchmark(benchmark, num_queries=holdout_queries, seed=10_000 + config.seed)
    lds_masks, lds_values = sample_global_queries_for_benchmark(benchmark, num_queries=lds_num_subsets, seed=15_000 + config.seed)
    pair_probes = sample_pair_probes(
        benchmark,
        num_backgrounds_per_edge=pair_probes_per_edge,
        edges=benchmark.candidate_edges,
        seed=20_000 + config.seed,
    )

    methods = {
        "linear": fit_linear_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=benchmark.num_atoms,
            candidate_edges=benchmark.candidate_edges,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        ),
        "rbf": fit_rbf_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=benchmark.num_atoms,
            candidate_edges=benchmark.candidate_edges,
            config=RBFSubsetSurrogateConfig(ridge_lambda=config.ridge_lambda),
        ),
        "structured_no_pair_probes": fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=benchmark.num_atoms,
            candidate_edges=benchmark.candidate_edges,
            pair_probes=None,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        ),
        "structured_pair_probes": fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=benchmark.num_atoms,
            candidate_edges=benchmark.candidate_edges,
            pair_probes=pair_probes,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        ),
    }

    atom_truth = benchmark.positive_mask()
    base_metrics: dict[str, dict[str, float]] = {}
    for method_name, model in methods.items():
        predictions = np.asarray(model.predict_batch(holdout_masks), dtype=np.float64)
        lds_predictions = np.asarray(model.predict_batch(lds_masks), dtype=np.float64)
        atom_scores = singleton_scores(model, benchmark.num_atoms)
        positive_mass = float(np.sum(np.maximum(atom_scores, 0.0)))
        k = max(int(np.sum(atom_truth)), 1)
        base_metrics[method_name] = {
            "utility_r2": r2_score(holdout_values, predictions),
            "utility_rmse": rmse(holdout_values, predictions),
            "subset_lds_spearman": spearman_corr(lds_values, lds_predictions),
            "subset_lds_pearson": pearson_corr(lds_values, lds_predictions),
            "rescue_auroc": binary_auroc(atom_truth, atom_scores),
            "rescue_auprc": binary_auprc(atom_truth, atom_scores),
            "rescue_recall_at_k": binary_recall_at_k(atom_truth, atom_scores, k=k),
            "positive_singleton_mass": positive_mass,
        }

    ablation = {
        "utility_r2_gain_pair_probes": (
            base_metrics["structured_pair_probes"]["utility_r2"]
            - base_metrics["structured_no_pair_probes"]["utility_r2"]
        ),
        "subset_lds_spearman_gain_pair_probes": (
            base_metrics["structured_pair_probes"]["subset_lds_spearman"]
            - base_metrics["structured_no_pair_probes"]["subset_lds_spearman"]
        ),
        "rescue_auprc_gain_pair_probes": (
            base_metrics["structured_pair_probes"]["rescue_auprc"]
            - base_metrics["structured_no_pair_probes"]["rescue_auprc"]
        ),
        "rescue_recall_gain_pair_probes": (
            base_metrics["structured_pair_probes"]["rescue_recall_at_k"]
            - base_metrics["structured_no_pair_probes"]["rescue_recall_at_k"]
        ),
    }

    corruption_rows = []
    for index, alpha in enumerate(corruption_sweep):
        sweep_benchmark = build_real_rescue_benchmark(
            DatasetRunConfig(
                **{
                    **asdict(config),
                    "corruption_alpha": float(alpha),
                }
            )
        )
        subset_masks, subset_values = sample_global_queries_for_benchmark(
            sweep_benchmark,
            num_queries=max(global_queries // 2, 24),
            seed=30_000 + 100 * config.seed + index,
        )
        pair_probes = sample_pair_probes(
            sweep_benchmark,
            num_backgrounds_per_edge=pair_probes_per_edge,
            edges=sweep_benchmark.candidate_edges,
            seed=40_000 + 100 * config.seed + index,
        )
        no_pair = fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=sweep_benchmark.num_atoms,
            candidate_edges=sweep_benchmark.candidate_edges,
            pair_probes=None,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        )
        with_pair = fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=sweep_benchmark.num_atoms,
            candidate_edges=sweep_benchmark.candidate_edges,
            pair_probes=pair_probes,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        )
        corruption_rows.append(
            {
                "corruption_alpha": float(alpha),
                "structured_no_pair_probes_mass": positive_singleton_mass(no_pair, sweep_benchmark.num_atoms),
                "structured_pair_probes_mass": positive_singleton_mass(with_pair, sweep_benchmark.num_atoms),
            }
        )

    purity_rows = []
    for index, purity in enumerate(purity_sweep):
        sweep_benchmark = build_real_rescue_benchmark(
            DatasetRunConfig(
                **{
                    **asdict(config),
                    "partner_purity": float(purity),
                }
            )
        )
        subset_masks, subset_values = sample_global_queries_for_benchmark(
            sweep_benchmark,
            num_queries=max(global_queries // 2, 24),
            seed=50_000 + 100 * config.seed + index,
        )
        pair_probes = sample_pair_probes(
            sweep_benchmark,
            num_backgrounds_per_edge=pair_probes_per_edge,
            edges=sweep_benchmark.candidate_edges,
            seed=60_000 + 100 * config.seed + index,
        )
        no_pair = fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=sweep_benchmark.num_atoms,
            candidate_edges=sweep_benchmark.candidate_edges,
            pair_probes=None,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        )
        with_pair = fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=sweep_benchmark.num_atoms,
            candidate_edges=sweep_benchmark.candidate_edges,
            pair_probes=pair_probes,
            config=StructuredSubsetSurrogateConfig(
                feature_bernoulli_p=0.5,
                ridge_lambda=config.ridge_lambda,
                pair_probe_weight=config.pair_probe_weight,
            ),
        )
        purity_rows.append(
            {
                "partner_purity": float(purity),
                "structured_no_pair_probes_mass": positive_singleton_mass(no_pair, sweep_benchmark.num_atoms),
                "structured_pair_probes_mass": positive_singleton_mass(with_pair, sweep_benchmark.num_atoms),
            }
        )

    sweeps = {
        "corr_corruption_vs_mass_no_pair": pearson_corr(
            np.asarray([row["corruption_alpha"] for row in corruption_rows], dtype=np.float64),
            np.asarray([row["structured_no_pair_probes_mass"] for row in corruption_rows], dtype=np.float64),
        ),
        "corr_corruption_vs_mass_pair": pearson_corr(
            np.asarray([row["corruption_alpha"] for row in corruption_rows], dtype=np.float64),
            np.asarray([row["structured_pair_probes_mass"] for row in corruption_rows], dtype=np.float64),
        ),
        "mono_corruption_no_pair": monotonicity_score(
            np.asarray([row["structured_no_pair_probes_mass"] for row in corruption_rows], dtype=np.float64)
        ),
        "mono_corruption_pair": monotonicity_score(
            np.asarray([row["structured_pair_probes_mass"] for row in corruption_rows], dtype=np.float64)
        ),
        "corr_purity_vs_mass_no_pair": pearson_corr(
            np.asarray([row["partner_purity"] for row in purity_rows], dtype=np.float64),
            np.asarray([row["structured_no_pair_probes_mass"] for row in purity_rows], dtype=np.float64),
        ),
        "corr_purity_vs_mass_pair": pearson_corr(
            np.asarray([row["partner_purity"] for row in purity_rows], dtype=np.float64),
            np.asarray([row["structured_pair_probes_mass"] for row in purity_rows], dtype=np.float64),
        ),
        "mono_purity_no_pair": monotonicity_score(
            np.asarray([row["structured_no_pair_probes_mass"] for row in purity_rows], dtype=np.float64)
        ),
        "mono_purity_pair": monotonicity_score(
            np.asarray([row["structured_pair_probes_mass"] for row in purity_rows], dtype=np.float64)
        ),
    }

    return {
        "dataset": config.dataset_name,
        "seed": config.seed,
        "num_atoms": benchmark.num_atoms,
        "num_positive_atoms": int(np.sum(atom_truth)),
        "num_candidate_edges": len(benchmark.candidate_edges),
        "lds_num_subsets": int(lds_num_subsets),
        "base_metrics": base_metrics,
        "ablation": ablation,
        "sweeps": sweeps,
        "corruption_rows": corruption_rows,
        "purity_rows": purity_rows,
    }


def build_real_rescue_benchmark(config: DatasetRunConfig) -> RealRescueBenchmark:
    raw = load_dataset(config.dataset_name, seed=config.seed)
    processed = preprocess_real_dataset(
        raw,
        dataset_name=config.dataset_name,
        task=str(raw["task"]),
        seed=config.seed,
        num_train_samples=config.num_train_samples,
        num_val_samples=config.num_val_samples,
        max_classes_food101=config.max_classes_food101,
        max_modality_dim=config.max_modality_dim,
    )
    anchors = select_anchor_indices(
        processed["train_targets"],
        task=processed["task"],
        num_anchors=config.num_anchors,
        candidates_per_anchor=config.candidates_per_anchor,
        seed=config.seed,
    )
    atom_anchor_ids: list[int] = []
    atom_is_positive: list[bool] = []
    atom_donor_y: list[np.ndarray] = []
    candidate_edges: list[tuple[int, int]] = []

    norm_x = l2_normalize(processed["train_x"])
    atom_offset = 0
    for anchor_id, anchor_train_index in enumerate(anchors):
        x_similarities = norm_x @ norm_x[anchor_train_index]
        label_similarities = label_similarity_to_anchor(
            processed["train_targets"],
            anchor_index=anchor_train_index,
            task=processed["task"],
        )
        positive_pool, negative_pool = split_positive_negative_candidates(
            label_similarities,
            x_similarities=x_similarities,
            anchor_index=anchor_train_index,
        )

        num_positive = min(
            config.candidates_per_anchor,
            max(0, int(round(config.partner_purity * config.candidates_per_anchor))),
        )
        if config.partner_purity > 0 and num_positive == 0:
            num_positive = 1
        if config.partner_purity < 1.0 and num_positive == config.candidates_per_anchor:
            num_positive = config.candidates_per_anchor - 1
        num_negative = config.candidates_per_anchor - num_positive
        donors = list(positive_pool[:num_positive]) + list(negative_pool[:num_negative])
        positives = [True] * num_positive + [False] * num_negative
        if len(donors) != config.candidates_per_anchor:
            raise ValueError(
                f"dataset={config.dataset_name} seed={config.seed}: unable to build {config.candidates_per_anchor} donors "
                f"for anchor {anchor_train_index}"
            )
        for donor_index, is_positive in zip(donors, positives):
            atom_anchor_ids.append(anchor_id)
            atom_is_positive.append(is_positive)
            atom_donor_y.append(processed["train_y"][donor_index])
        group_atoms = list(range(atom_offset, atom_offset + config.candidates_per_anchor))
        for left in range(len(group_atoms)):
            for right in range(left + 1, len(group_atoms)):
                candidate_edges.append((group_atoms[left], group_atoms[right]))
        atom_offset += config.candidates_per_anchor

    return RealRescueBenchmark(
        dataset_name=config.dataset_name,
        train_x=processed["train_x"],
        train_y_clean=processed["train_y"],
        train_targets=processed["train_targets"],
        val_x=processed["val_x"],
        val_y=processed["val_y"],
        val_targets=processed["val_targets"],
        anchor_train_indices=np.asarray(anchors, dtype=np.int64),
        atom_anchor_ids=np.asarray(atom_anchor_ids, dtype=np.int64),
        atom_is_positive=np.asarray(atom_is_positive, dtype=bool),
        atom_donor_y=np.asarray(atom_donor_y, dtype=np.float64),
        candidate_edges=candidate_edges,
        corruption_alpha=config.corruption_alpha,
        task=processed["task"],
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


def sample_global_queries_for_benchmark(
    benchmark: RealRescueBenchmark,
    *,
    num_queries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    masks = np.zeros((num_queries, benchmark.num_atoms), dtype=bool)
    probs = np.asarray([0.3, 0.5, 0.7], dtype=np.float64)
    for index in range(num_queries):
        p = float(rng.choice(probs))
        masks[index] = rng.random(benchmark.num_atoms) < p
    return masks, benchmark.utility_batch(masks)


def singleton_scores(model: object, num_atoms: int) -> np.ndarray:
    empty = np.zeros(num_atoms, dtype=bool)
    baseline = float(model.predict(empty))
    scores = np.zeros(num_atoms, dtype=np.float64)
    for atom in range(num_atoms):
        mask = np.zeros(num_atoms, dtype=bool)
        mask[atom] = True
        scores[atom] = float(model.predict(mask)) - baseline
    return scores


def positive_singleton_mass(model: object, num_atoms: int) -> float:
    return float(np.sum(np.maximum(singleton_scores(model, num_atoms), 0.0)))


def summarize_dataset_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    base_by_method: dict[str, list[dict[str, float]]] = {}
    for run in runs:
        for method, metrics in run["base_metrics"].items():
            base_by_method.setdefault(method, []).append(metrics)
    corruption_curve = summarize_curve_runs(runs, row_key="corruption_rows", x_key="corruption_alpha")
    purity_curve = summarize_curve_runs(runs, row_key="purity_rows", x_key="partner_purity")
    return {
        "runs": runs,
        "base_summary": {
            method: summarize_numeric_dicts(rows)
            for method, rows in base_by_method.items()
        },
        "ablation_summary": summarize_numeric_dicts([run["ablation"] for run in runs]),
        "sweep_summary": summarize_numeric_dicts([run["sweeps"] for run in runs]),
        "corruption_curve": corruption_curve,
        "purity_curve": purity_curve,
    }


def summarize_curve_runs(
    runs: list[dict[str, object]],
    *,
    row_key: str,
    x_key: str,
) -> list[dict[str, float]]:
    buckets: dict[float, dict[str, list[float]]] = {}
    for run in runs:
        for row in run[row_key]:
            x = float(row[x_key])
            current = buckets.setdefault(x, {})
            for key, value in row.items():
                if key == x_key:
                    continue
                current.setdefault(key, []).append(float(value))
    curve = []
    for x, metric_values in sorted(buckets.items(), key=lambda item: item[0]):
        row = {x_key: x}
        for key, values in metric_values.items():
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        curve.append(row)
    return curve


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


def choose_top_classes(labels: np.ndarray, max_classes: int) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    return values[order[:max_classes]]


def remap_labels(labels: np.ndarray, selected_classes: np.ndarray) -> np.ndarray:
    mapping = {int(label): index for index, label in enumerate(selected_classes)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def subsample_indices(targets: np.ndarray, max_samples: int, *, seed: int, task: str) -> np.ndarray:
    if task == "multilabel":
        count = int(targets.shape[0])
        if count <= max_samples:
            return np.arange(count, dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.permutation(count)[:max_samples]).astype(np.int64)
    return stratified_subsample(np.asarray(targets, dtype=np.int64).reshape(-1), max_samples, seed=seed)


def stratified_subsample(labels: np.ndarray, max_samples: int, *, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] <= max_samples:
        return np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    per_class_indices = {int(label): np.where(labels == label)[0] for label in np.unique(labels)}
    chosen: list[int] = []
    classes = list(per_class_indices.keys())
    class_cursor = 0
    shuffled = {label: rng.permutation(indices) for label, indices in per_class_indices.items()}
    positions = {label: 0 for label in per_class_indices}
    while len(chosen) < max_samples:
        label = classes[class_cursor % len(classes)]
        class_cursor += 1
        indices = shuffled[label]
        pos = positions[label]
        if pos >= indices.shape[0]:
            if all(positions[c] >= shuffled[c].shape[0] for c in classes):
                break
            continue
        chosen.append(int(indices[pos]))
        positions[label] += 1
    return np.asarray(sorted(chosen), dtype=np.int64)


def select_anchor_indices(
    train_targets: np.ndarray,
    *,
    task: str,
    num_anchors: int,
    candidates_per_anchor: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    total = int(train_targets.shape[0])
    order = rng.permutation(total)
    anchors: list[int] = []
    for index in order:
        if task != "multilabel":
            label = int(np.asarray(train_targets).reshape(-1)[index])
            same_count = int(np.sum(np.asarray(train_targets).reshape(-1) == label)) - 1
            diff_count = int(np.sum(np.asarray(train_targets).reshape(-1) != label))
            if same_count < candidates_per_anchor or diff_count < candidates_per_anchor:
                continue
        elif total - 1 < 2 * candidates_per_anchor:
            continue
        anchors.append(int(index))
        if len(anchors) == num_anchors:
            return anchors
    raise ValueError(
        f"Unable to choose {num_anchors} anchors with {candidates_per_anchor} positive and negative donors"
    )


def tensor_to_numpy(tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)


def tensor_labels_to_numpy(tensor, *, task: str) -> np.ndarray:
    array = np.asarray(tensor.detach().cpu().numpy())
    if task == "multilabel":
        if array.ndim != 2:
            raise ValueError("Expected multilabel targets to have shape [N, C]")
        return (array > 0).astype(np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected single-label targets for task={task}, got shape={array.shape}")
    return array.astype(np.int64)


def label_similarity_to_anchor(
    targets: np.ndarray,
    *,
    anchor_index: int,
    task: str,
) -> np.ndarray:
    if task == "multilabel":
        target_array = np.asarray(targets, dtype=np.float64)
        anchor = target_array[anchor_index] > 0.0
        others = target_array > 0.0
        intersection = np.sum(others & anchor, axis=1, dtype=np.float64)
        union = np.sum(others | anchor, axis=1, dtype=np.float64)
        union = np.maximum(union, 1.0)
        return intersection / union
    flat = np.asarray(targets, dtype=np.int64).reshape(-1)
    return (flat == flat[anchor_index]).astype(np.float64)


def split_positive_negative_candidates(
    label_similarities: np.ndarray,
    *,
    x_similarities: np.ndarray,
    anchor_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(label_similarities.shape[0], dtype=np.int64)
    valid = all_indices != int(anchor_index)
    candidates = all_indices[valid]
    label_scores = label_similarities[valid]
    x_scores = x_similarities[valid]

    positive_order = np.lexsort((-x_scores, -label_scores))
    negative_order = np.lexsort((-x_scores, label_scores))
    return candidates[positive_order], candidates[negative_order]


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return values / norms


def maybe_random_project_pair(
    train_features: np.ndarray,
    val_features: np.ndarray,
    *,
    max_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_features.shape[1] <= max_dim:
        return train_features, val_features
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((train_features.shape[1], max_dim)).astype(np.float64)
    projection /= np.sqrt(train_features.shape[1])
    return train_features @ projection, val_features @ projection


def ridge_probe_score(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    *,
    ridge_lambda: float,
    task: str,
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
    else:
        labels_train = np.asarray(train_targets, dtype=np.int64).reshape(-1)
        labels_val = np.asarray(val_targets, dtype=np.int64).reshape(-1)
        classes = np.unique(labels_train)
        class_to_index = {int(label): index for index, label in enumerate(classes)}
        y_train = np.zeros((labels_train.shape[0], classes.shape[0]), dtype=np.float64)
        for row, label in enumerate(labels_train):
            y_train[row, class_to_index[int(label)]] = 1.0

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
    if task == "multilabel":
        predictions = (logits >= 0.5).astype(np.float64)
        return multilabel_micro_f1(y_val, predictions)
    predictions = classes[np.argmax(logits, axis=1)]
    return float(np.mean(predictions == labels_val))


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


def to_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Real Subset Attribution Benchmark",
        "",
        "Frozen-feature real-data rescue benchmark using controlled ridge probes.",
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
        dataset_configs = payload.get("effective_dataset_configs", {}).get(dataset_name, [])
        config_line = ""
        if dataset_configs:
            first = dataset_configs[0]
            dataset_config = first["dataset_config"]
            config_line = (
                f"Train {dataset_config['num_train_samples']}, "
                f"Val {dataset_config['num_val_samples']}, "
                f"Anchors {dataset_config['num_anchors']}, "
                f"Candidates/anchor {dataset_config['candidates_per_anchor']}, "
                f"Global queries {first['global_queries']}, "
                f"Holdout queries {first['holdout_queries']}, "
                f"Projection dim <= {dataset_config['max_modality_dim']}"
            )
        base = result["base_summary"]
        ablation = result["ablation_summary"]
        sweep = result["sweep_summary"]
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                config_line if config_line else "",
                "",
                "### Base Rescue Benchmark",
                _method_line("Linear", base["linear"]),
                _method_line("RBF", base["rbf"]),
                _method_line("Structured no pair probes", base["structured_no_pair_probes"]),
                _method_line("Structured + pair probes", base["structured_pair_probes"]),
                "",
                "### Pair-Probe Ablation",
                _metric_line("Utility R2 gain", ablation["utility_r2_gain_pair_probes"]),
                _metric_line("LDS-Spearman gain", ablation["subset_lds_spearman_gain_pair_probes"]),
                _metric_line("Rescue AUPRC gain", ablation["rescue_auprc_gain_pair_probes"]),
                _metric_line("Rescue Recall@k gain", ablation["rescue_recall_gain_pair_probes"]),
                "",
                "### Sweep Calibration",
                _metric_line("Corr(corruption, mass) no-pair", sweep["corr_corruption_vs_mass_no_pair"]),
                _metric_line("Corr(corruption, mass) pair", sweep["corr_corruption_vs_mass_pair"]),
                _metric_line("Corr(purity, mass) no-pair", sweep["corr_purity_vs_mass_no_pair"]),
                _metric_line("Corr(purity, mass) pair", sweep["corr_purity_vs_mass_pair"]),
                "",
            ]
        )

    aggregate = payload["aggregate"]
    lines.extend(
        [
            "## Aggregate",
            "",
            _method_line("Linear", aggregate["base_methods"]["linear"]),
            _method_line("RBF", aggregate["base_methods"]["rbf"]),
            _method_line("Structured no pair probes", aggregate["base_methods"]["structured_no_pair_probes"]),
            _method_line("Structured + pair probes", aggregate["base_methods"]["structured_pair_probes"]),
            "",
            "### Pair-Probe Gains",
            _metric_line("Utility R2 gain", aggregate["pair_probe_ablation"]["utility_r2_gain_pair_probes"]),
            _metric_line("LDS-Spearman gain", aggregate["pair_probe_ablation"]["subset_lds_spearman_gain_pair_probes"]),
            _metric_line("Rescue AUPRC gain", aggregate["pair_probe_ablation"]["rescue_auprc_gain_pair_probes"]),
            _metric_line("Rescue Recall@k gain", aggregate["pair_probe_ablation"]["rescue_recall_gain_pair_probes"]),
            "",
            "### Sweep Calibration",
            _metric_line("Corr(corruption, mass) no-pair", aggregate["sweeps"]["corr_corruption_vs_mass_no_pair"]),
            _metric_line("Corr(corruption, mass) pair", aggregate["sweeps"]["corr_corruption_vs_mass_pair"]),
            _metric_line("Corr(purity, mass) no-pair", aggregate["sweeps"]["corr_purity_vs_mass_no_pair"]),
            _metric_line("Corr(purity, mass) pair", aggregate["sweeps"]["corr_purity_vs_mass_pair"]),
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
        f"utility R2 {summary['utility_r2']['mean']:.4f} ± {summary['utility_r2']['std']:.4f}, "
        f"AUPRC {summary['rescue_auprc']['mean']:.4f} ± {summary['rescue_auprc']['std']:.4f}, "
        f"Recall@k {summary['rescue_recall_at_k']['mean']:.4f} ± {summary['rescue_recall_at_k']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
