from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pid_estimator.benchmark_real_subset_attribution import SUPPORTED_DATASETS
from pid_estimator.benchmark_real_subset_pid import (
    DatasetRunConfig,
    load_processed_dataset,
    resolve_available_datasets,
    ridge_probe_score,
    select_focal_indices,
)
from pid_estimator.multimodal_tensor_kernel import (
    MultimodalTensorKernelConfig,
    compute_csmi_from_oracle,
    enumerate_multimodal_query_states,
    fit_multimodal_tensor_kernel_regressor,
)
from pid_estimator.multimodal_tensor_kernel_experiment import (
    FlatRBFKernelConfig,
    fit_flat_rbf_regressor,
    pearson_corr,
    r2_score,
    rmse,
)


@dataclass(frozen=True)
class RealTensorKernelExperimentConfig:
    dataset_name: str = "avmnist"
    seed: int = 0
    num_train_samples: int = 400
    num_val_samples: int = 200
    num_focal_samples: int = 3
    train_fraction: float = 0.25
    corruption_alpha: float = 1.0
    ridge_lambda: float = 1e-3
    kernel_ridge_lambda: float = 1e-3
    score_metric: str = "neg_loss"
    max_classes_food101: int = 10
    max_modality_dim: int = 32
    sample_gamma: float | None = 0.1
    modality_gamma: float | None = 0.1
    query_gamma: float | None = 0.1
    baseline_gamma: float | None = None


@dataclass
class RealMultimodalTensorBenchmark:
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
        self.num_samples = int(self.focal_indices.shape[0])
        self.num_modalities = 2
        self._cache: dict[bytes, float] = {}

    def evaluate(self, selection_matrix: np.ndarray, query_mask: np.ndarray) -> float:
        selection = np.asarray(selection_matrix, dtype=bool).reshape(self.num_samples, self.num_modalities)
        query = np.asarray(query_mask, dtype=bool).reshape(self.num_modalities)
        key = self._cache_key(selection, query)
        if key in self._cache:
            return self._cache[key]

        keep_scale = max(1.0 - float(self.corruption_alpha), 0.0)
        train_x_variant = self.train_x.copy()
        train_y_variant = self.train_y.copy()
        for local_index, train_index in enumerate(self.focal_indices):
            if not selection[local_index, 0]:
                train_x_variant[train_index] *= keep_scale
            if not selection[local_index, 1]:
                train_y_variant[train_index] *= keep_scale

        val_x_variant = self.val_x.copy()
        val_y_variant = self.val_y.copy()
        if not query[0]:
            val_x_variant *= keep_scale
        if not query[1]:
            val_y_variant *= keep_scale

        value = ridge_probe_score(
            train_features=np.concatenate([train_x_variant, train_y_variant], axis=1),
            train_targets=self.train_targets,
            val_features=np.concatenate([val_x_variant, val_y_variant], axis=1),
            val_targets=self.val_targets,
            ridge_lambda=self.ridge_lambda,
            task=self.task,
            metric=self.score_metric,
        )
        self._cache[key] = float(value)
        return float(value)

    def evaluate_batch(self, selection_matrices: np.ndarray, query_masks: np.ndarray) -> np.ndarray:
        selection = np.asarray(selection_matrices, dtype=bool)
        query = np.asarray(query_masks, dtype=bool)
        values = np.zeros(selection.shape[0], dtype=np.float64)
        for index in range(selection.shape[0]):
            values[index] = self.evaluate(selection[index], query[index])
        return values

    def _cache_key(self, selection_matrix: np.ndarray, query_mask: np.ndarray) -> bytes:
        bits = np.concatenate([selection_matrix.reshape(-1), query_mask.reshape(-1)]).astype(np.uint8)
        return np.packbits(bits, bitorder="little").tobytes()


def build_real_multimodal_tensor_benchmark(config: RealTensorKernelExperimentConfig) -> RealMultimodalTensorBenchmark:
    run_config = DatasetRunConfig(
        dataset_name=config.dataset_name,
        seed=config.seed,
        num_train_samples=config.num_train_samples,
        num_val_samples=config.num_val_samples,
        num_focal_samples=config.num_focal_samples,
        donor_num_anchors=4,
        donor_candidates_per_anchor=3,
        donor_partner_purity=1.0 / 3.0,
        train_queries=0,
        holdout_queries=0,
        lds_num_subsets=0,
        ridge_lambda=config.ridge_lambda,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        score_metric=config.score_metric,
        corruption_alpha=config.corruption_alpha,
        corruption_sweep=(config.corruption_alpha,),
        max_classes_food101=config.max_classes_food101,
        max_modality_dim=config.max_modality_dim,
    )
    processed = load_processed_dataset(run_config)
    focal_indices = select_focal_indices(
        processed["train_targets"],
        task=processed["task"],
        num_focal_samples=config.num_focal_samples,
        seed=config.seed,
    )
    return RealMultimodalTensorBenchmark(
        dataset_name=config.dataset_name,
        train_x=processed["train_x"],
        train_y=processed["train_y"],
        train_targets=processed["train_targets"],
        val_x=processed["val_x"],
        val_y=processed["val_y"],
        val_targets=processed["val_targets"],
        focal_indices=focal_indices,
        corruption_alpha=config.corruption_alpha,
        task=str(processed["task"]),
        ridge_lambda=config.ridge_lambda,
        score_metric=config.score_metric,
    )


def run_experiment(config: RealTensorKernelExperimentConfig) -> dict[str, object]:
    benchmark = build_real_multimodal_tensor_benchmark(config)
    all_selection_matrices, all_query_masks = enumerate_multimodal_query_states(
        benchmark.num_samples,
        benchmark.num_modalities,
    )
    all_targets = benchmark.evaluate_batch(all_selection_matrices, all_query_masks)

    rng = np.random.default_rng(config.seed)
    permutation = rng.permutation(all_targets.shape[0])
    train_count = max(16, int(round(config.train_fraction * all_targets.shape[0])))
    train_count = min(train_count, all_targets.shape[0] - 1)
    train_indices = np.sort(permutation[:train_count])
    test_indices = np.sort(permutation[train_count:])

    train_selection = all_selection_matrices[train_indices]
    train_query = all_query_masks[train_indices]
    train_targets = all_targets[train_indices]
    test_selection = all_selection_matrices[test_indices]
    test_query = all_query_masks[test_indices]
    test_targets = all_targets[test_indices]

    tensor_model = fit_multimodal_tensor_kernel_regressor(
        train_selection,
        train_query,
        train_targets,
        config=MultimodalTensorKernelConfig(
            ridge_lambda=config.kernel_ridge_lambda,
            sample_gamma=config.sample_gamma,
            modality_gamma=config.modality_gamma,
            query_gamma=config.query_gamma,
        ),
    )
    flat_model = fit_flat_rbf_regressor(
        train_selection,
        train_query,
        train_targets,
        config=FlatRBFKernelConfig(
            ridge_lambda=config.kernel_ridge_lambda,
            gamma=config.baseline_gamma,
        ),
    )

    tensor_test = tensor_model.predict_batch(test_selection, test_query)
    flat_test = flat_model.predict_batch(test_selection, test_query)
    tensor_all = tensor_model.predict_batch(all_selection_matrices, all_query_masks)
    flat_all = flat_model.predict_batch(all_selection_matrices, all_query_masks)

    exact_csmi, tensor_csmi, flat_csmi = evaluate_csmi_grid(benchmark, tensor_model, flat_model)
    contrast = evaluate_query_contrast(benchmark, tensor_model, flat_model)

    return {
        "experiment": "multimodal_tensor_kernel_real_learner",
        "config": asdict(config),
        "dataset_summary": {
            "task": benchmark.task,
            "num_train_rows": int(benchmark.train_x.shape[0]),
            "num_val_rows": int(benchmark.val_x.shape[0]),
            "num_focal_samples": int(benchmark.num_samples),
            "num_modalities": int(benchmark.num_modalities),
            "num_query_states": int(all_targets.shape[0]),
            "focal_indices": [int(index) for index in benchmark.focal_indices.tolist()],
        },
        "tensor": {
            "heldout_r2": r2_score(test_targets, tensor_test),
            "heldout_rmse": rmse(test_targets, tensor_test),
            "all_r2": r2_score(all_targets, tensor_all),
            "csmi_rmse": rmse(exact_csmi, tensor_csmi),
            "csmi_pearson": pearson_corr(exact_csmi, tensor_csmi),
            "sample_gamma": float(tensor_model.sample_gamma),
            "modality_gamma": float(tensor_model.modality_gamma),
            "query_gamma": float(tensor_model.query_gamma),
        },
        "flat_rbf": {
            "heldout_r2": r2_score(test_targets, flat_test),
            "heldout_rmse": rmse(test_targets, flat_test),
            "all_r2": r2_score(all_targets, flat_all),
            "csmi_rmse": rmse(exact_csmi, flat_csmi),
            "csmi_pearson": pearson_corr(exact_csmi, flat_csmi),
            "gamma": float(flat_model.gamma),
        },
        "contrast_case": contrast,
        "oracle_statistics": {
            "target_mean": float(np.mean(all_targets)),
            "target_std": float(np.std(all_targets)),
            "target_min": float(np.min(all_targets)),
            "target_max": float(np.max(all_targets)),
        },
    }


def evaluate_csmi_grid(
    benchmark: RealMultimodalTensorBenchmark,
    tensor_model,
    flat_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modality_subsets = enumerate_nonempty_masks(benchmark.num_modalities)
    exact_values = []
    tensor_values = []
    flat_values = []
    for sample_index in range(benchmark.num_samples):
        for train_subset in modality_subsets:
            for query_subset in modality_subsets:
                exact_values.append(
                    compute_csmi_from_oracle(
                        benchmark.evaluate,
                        num_samples=benchmark.num_samples,
                        num_modalities=benchmark.num_modalities,
                        sample_index=sample_index,
                        train_modalities=train_subset,
                        query_modalities=query_subset,
                    )
                )
                tensor_values.append(
                    tensor_model.csmi(
                        sample_index=sample_index,
                        train_modalities=train_subset,
                        query_modalities=query_subset,
                    )
                )
                flat_values.append(
                    flat_model.csmi(
                        sample_index=sample_index,
                        train_modalities=train_subset,
                        query_modalities=query_subset,
                    )
                )
    return (
        np.asarray(exact_values, dtype=np.float64),
        np.asarray(tensor_values, dtype=np.float64),
        np.asarray(flat_values, dtype=np.float64),
    )


def evaluate_query_contrast(
    benchmark: RealMultimodalTensorBenchmark,
    tensor_model,
    flat_model,
) -> dict[str, float]:
    full_selection = np.ones((benchmark.num_samples, benchmark.num_modalities), dtype=bool)
    modality0_only = np.asarray([True, False], dtype=bool)
    modality1_only = np.asarray([False, True], dtype=bool)
    full_query = np.asarray([True, True], dtype=bool)
    exact_gap = benchmark.evaluate(full_selection, full_query) - benchmark.evaluate(full_selection, modality0_only)
    tensor_gap = tensor_model.predict(full_selection, full_query) - tensor_model.predict(full_selection, modality0_only)
    flat_gap = flat_model.predict(full_selection, full_query) - flat_model.predict(full_selection, modality0_only)
    reverse_exact_gap = benchmark.evaluate(full_selection, full_query) - benchmark.evaluate(full_selection, modality1_only)
    reverse_tensor_gap = tensor_model.predict(full_selection, full_query) - tensor_model.predict(full_selection, modality1_only)
    reverse_flat_gap = flat_model.predict(full_selection, full_query) - flat_model.predict(full_selection, modality1_only)
    return {
        "full_minus_modality0_only_exact": float(exact_gap),
        "full_minus_modality0_only_tensor": float(tensor_gap),
        "full_minus_modality0_only_flat": float(flat_gap),
        "full_minus_modality1_only_exact": float(reverse_exact_gap),
        "full_minus_modality1_only_tensor": float(reverse_tensor_gap),
        "full_minus_modality1_only_flat": float(reverse_flat_gap),
    }


def enumerate_nonempty_masks(num_bits: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bitmask in range(1, 2**num_bits):
        mask = np.zeros(num_bits, dtype=bool)
        for index in range(num_bits):
            mask[index] = bool((bitmask >> index) & 1)
        masks.append(mask)
    return masks


def to_markdown(payload: dict[str, object]) -> str:
    config = payload["config"]
    dataset = payload["dataset_summary"]
    tensor = payload["tensor"]
    flat = payload["flat_rbf"]
    oracle = payload["oracle_statistics"]
    contrast = payload["contrast_case"]
    lines = [
        "# Multimodal Tensor Kernel Real-Learner Experiment",
        "",
        "## Config",
        f"- dataset: {config['dataset_name']}",
        f"- seed: {config['seed']}",
        f"- train samples: {dataset['num_train_rows']}",
        f"- val samples: {dataset['num_val_rows']}",
        f"- focal samples: {dataset['num_focal_samples']}",
        f"- query states: {dataset['num_query_states']}",
        f"- metric: {config['score_metric']}",
        f"- corruption alpha: {config['corruption_alpha']}",
        f"- focal indices: {dataset['focal_indices']}",
        "",
        "## Results",
        f"- tensor heldout R2: {tensor['heldout_r2']:.4f}",
        f"- flat heldout R2: {flat['heldout_r2']:.4f}",
        f"- tensor heldout RMSE: {tensor['heldout_rmse']:.4f}",
        f"- flat heldout RMSE: {flat['heldout_rmse']:.4f}",
        f"- tensor CSMI RMSE: {tensor['csmi_rmse']:.4f}",
        f"- flat CSMI RMSE: {flat['csmi_rmse']:.4f}",
        f"- tensor CSMI Pearson: {tensor['csmi_pearson']:.4f}",
        f"- flat CSMI Pearson: {flat['csmi_pearson']:.4f}",
        "",
        "## Oracle",
        f"- target mean: {oracle['target_mean']:.4f}",
        f"- target std: {oracle['target_std']:.4f}",
        f"- target min: {oracle['target_min']:.4f}",
        f"- target max: {oracle['target_max']:.4f}",
        "",
        "## Query Contrast",
        f"- exact full - modality0_only: {contrast['full_minus_modality0_only_exact']:.4f}",
        f"- tensor full - modality0_only: {contrast['full_minus_modality0_only_tensor']:.4f}",
        f"- flat full - modality0_only: {contrast['full_minus_modality0_only_flat']:.4f}",
        f"- exact full - modality1_only: {contrast['full_minus_modality1_only_exact']:.4f}",
        f"- tensor full - modality1_only: {contrast['full_minus_modality1_only_tensor']:.4f}",
        f"- flat full - modality1_only: {contrast['full_minus_modality1_only_flat']:.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multimodal tensor-kernel experiment against a real controlled learner.")
    parser.add_argument("--dataset-name", type=str, default="avmnist", choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train-samples", type=int, default=400)
    parser.add_argument("--num-val-samples", type=int, default=200)
    parser.add_argument("--num-focal-samples", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--corruption-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--kernel-ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--score-metric", type=str, default="neg_loss", choices=["accuracy", "neg_loss", "loss"])
    parser.add_argument("--max-classes-food101", type=int, default=10)
    parser.add_argument("--max-modality-dim", type=int, default=32)
    parser.add_argument("--sample-gamma", type=float, default=0.1)
    parser.add_argument("--modality-gamma", type=float, default=0.1)
    parser.add_argument("--query-gamma", type=float, default=0.1)
    parser.add_argument("--baseline-gamma", type=float, default=None)
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/pid_estimator/results/multimodal_tensor_kernel_real_experiment.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/multimodal_tensor_kernel_real_experiment.md",
    )
    args = parser.parse_args()

    resolved, skipped = resolve_available_datasets([args.dataset_name])
    if not resolved:
        reason = skipped[0]["reason"] if skipped else "unknown"
        raise RuntimeError(f"dataset {args.dataset_name!r} is not available: {reason}")

    config = RealTensorKernelExperimentConfig(
        dataset_name=args.dataset_name,
        seed=args.seed,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        num_focal_samples=args.num_focal_samples,
        train_fraction=args.train_fraction,
        corruption_alpha=args.corruption_alpha,
        ridge_lambda=args.ridge_lambda,
        kernel_ridge_lambda=args.kernel_ridge_lambda,
        score_metric=args.score_metric,
        max_classes_food101=args.max_classes_food101,
        max_modality_dim=args.max_modality_dim,
        sample_gamma=args.sample_gamma,
        modality_gamma=args.modality_gamma,
        query_gamma=args.query_gamma,
        baseline_gamma=args.baseline_gamma,
    )
    payload = run_experiment(config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload))


if __name__ == "__main__":
    main()
