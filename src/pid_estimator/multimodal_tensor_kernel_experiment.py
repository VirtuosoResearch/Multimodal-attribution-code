from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pid_estimator.multimodal_tensor_kernel import (
    MultimodalTensorKernelConfig,
    compute_csmi_from_oracle,
    enumerate_multimodal_query_states,
    fit_multimodal_tensor_kernel_regressor,
)


@dataclass(frozen=True)
class FlatRBFKernelConfig:
    ridge_lambda: float = 1e-3
    gamma: float | None = None


@dataclass
class FlatRBFKernelRegressor:
    train_inputs: np.ndarray
    dual_weights: np.ndarray
    gamma: float
    num_samples: int
    num_modalities: int

    def predict(self, selection_matrix: np.ndarray, query_mask: np.ndarray) -> float:
        return float(self.predict_batch(selection_matrix.reshape(1, self.num_samples, self.num_modalities), query_mask.reshape(1, self.num_modalities))[0])

    def predict_batch(self, selection_matrices: np.ndarray, query_masks: np.ndarray) -> np.ndarray:
        features = _flatten_inputs(selection_matrices, query_masks)
        kernel = _rbf_kernel(features, self.train_inputs, self.gamma)
        return kernel @ self.dual_weights

    def csmi(
        self,
        *,
        sample_index: int,
        train_modalities: Sequence[int] | Sequence[bool] | np.ndarray,
        query_modalities: Sequence[int] | Sequence[bool] | np.ndarray,
        base_selection_matrix: np.ndarray | None = None,
        base_query_mask: np.ndarray | None = None,
    ) -> float:
        base_selection = (
            np.ones((self.num_samples, self.num_modalities), dtype=bool)
            if base_selection_matrix is None
            else np.asarray(base_selection_matrix, dtype=bool).reshape(self.num_samples, self.num_modalities)
        )
        base_query = (
            np.ones(self.num_modalities, dtype=bool)
            if base_query_mask is None
            else np.asarray(base_query_mask, dtype=bool).reshape(self.num_modalities)
        )
        train_drop = _as_modality_mask(train_modalities, self.num_modalities)
        query_drop = _as_modality_mask(query_modalities, self.num_modalities)

        dropped_selection = base_selection.copy()
        dropped_selection[sample_index, train_drop] = False
        dropped_query = base_query.copy()
        dropped_query[query_drop] = False
        return float(
            self.predict(base_selection, base_query)
            - self.predict(dropped_selection, base_query)
            - self.predict(base_selection, dropped_query)
            + self.predict(dropped_selection, dropped_query)
        )


@dataclass(frozen=True)
class SyntheticOracleConfig:
    num_samples: int = 3
    num_modalities: int = 3
    singleton_scale: float = 0.20
    same_sample_scale: float = 1.10
    cross_sample_same_modality_scale: float = 0.85
    query_bias_scale: float = 0.12
    seed: int = 0


class SyntheticMultimodalOracle:
    def __init__(self, config: SyntheticOracleConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.singleton_weights = rng.uniform(0.7, 1.3, size=(config.num_samples, config.num_modalities))
        self.sample_pair_weights = rng.uniform(
            0.8,
            1.5,
            size=(config.num_samples, config.num_modalities, config.num_modalities),
        )
        self.modality_pair_weights = rng.uniform(
            0.8,
            1.4,
            size=(config.num_modalities, config.num_samples, config.num_samples),
        )
        self.query_weights = rng.uniform(0.5, 1.0, size=config.num_modalities)

    @property
    def num_samples(self) -> int:
        return self.config.num_samples

    @property
    def num_modalities(self) -> int:
        return self.config.num_modalities

    def evaluate(self, selection_matrix: np.ndarray, query_mask: np.ndarray) -> float:
        selection = np.asarray(selection_matrix, dtype=np.float64).reshape(self.num_samples, self.num_modalities)
        query = np.asarray(query_mask, dtype=np.float64).reshape(self.num_modalities)

        singleton = float(np.mean(self.singleton_weights * selection * query.reshape(1, -1)))

        same_sample_total = 0.0
        same_sample_count = 0
        for sample_index in range(self.num_samples):
            for left in range(self.num_modalities):
                for right in range(left + 1, self.num_modalities):
                    same_sample_total += (
                        self.sample_pair_weights[sample_index, left, right]
                        * selection[sample_index, left]
                        * selection[sample_index, right]
                        * query[left]
                        * query[right]
                    )
                    same_sample_count += 1
        same_sample = same_sample_total / max(same_sample_count, 1)

        cross_sample_same_modality_total = 0.0
        cross_sample_same_modality_count = 0
        for modality_index in range(self.num_modalities):
            for left in range(self.num_samples):
                for right in range(left + 1, self.num_samples):
                    cross_sample_same_modality_total += (
                        self.modality_pair_weights[modality_index, left, right]
                        * selection[left, modality_index]
                        * selection[right, modality_index]
                        * query[modality_index]
                    )
                    cross_sample_same_modality_count += 1
        cross_sample_same_modality = cross_sample_same_modality_total / max(cross_sample_same_modality_count, 1)

        query_bias = float(np.dot(self.query_weights, query) / max(self.num_modalities, 1))
        return float(
            self.config.singleton_scale * singleton
            + self.config.same_sample_scale * same_sample
            - self.config.cross_sample_same_modality_scale * cross_sample_same_modality
            + self.config.query_bias_scale * query_bias
        )

    def evaluate_batch(self, selection_matrices: np.ndarray, query_masks: np.ndarray) -> np.ndarray:
        selection = np.asarray(selection_matrices, dtype=bool)
        query = np.asarray(query_masks, dtype=bool)
        values = np.zeros(selection.shape[0], dtype=np.float64)
        for index in range(selection.shape[0]):
            values[index] = self.evaluate(selection[index], query[index])
        return values


@dataclass(frozen=True)
class MultimodalTensorKernelExperimentConfig:
    num_samples: int = 3
    num_modalities: int = 3
    train_fraction: float = 0.12
    num_repeats: int = 3
    split_seed: int = 0
    ridge_lambda: float = 1e-3
    sample_gamma: float | None = None
    modality_gamma: float | None = None
    query_gamma: float | None = None
    baseline_gamma: float | None = None


def fit_flat_rbf_regressor(
    selection_matrices: np.ndarray,
    query_masks: np.ndarray,
    targets: np.ndarray,
    *,
    config: FlatRBFKernelConfig,
) -> FlatRBFKernelRegressor:
    features = _flatten_inputs(selection_matrices, query_masks)
    gamma = _resolve_gamma(features, config.gamma)
    kernel = _rbf_kernel(features, features, gamma)
    regularized = kernel + float(config.ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64)
    try:
        dual_weights = np.linalg.solve(regularized, targets)
    except np.linalg.LinAlgError:
        dual_weights = np.linalg.pinv(regularized) @ targets
    return FlatRBFKernelRegressor(
        train_inputs=features,
        dual_weights=dual_weights,
        gamma=gamma,
        num_samples=selection_matrices.shape[1],
        num_modalities=selection_matrices.shape[2],
    )


def run_experiment(config: MultimodalTensorKernelExperimentConfig) -> dict[str, object]:
    oracle = SyntheticMultimodalOracle(
        SyntheticOracleConfig(
            num_samples=config.num_samples,
            num_modalities=config.num_modalities,
            seed=config.split_seed,
        )
    )
    all_selection_matrices, all_query_masks = enumerate_multimodal_query_states(
        config.num_samples,
        config.num_modalities,
    )
    all_targets = oracle.evaluate_batch(all_selection_matrices, all_query_masks)
    total_queries = all_targets.shape[0]
    base_rng = np.random.default_rng(config.split_seed)
    seeds = [int(seed) for seed in base_rng.integers(0, 1_000_000, size=config.num_repeats)]
    runs = [run_single_split(config, oracle, all_selection_matrices, all_query_masks, all_targets, split_seed=seed) for seed in seeds]

    tensor_better_fit = sum(
        float(run["tensor"]["heldout_r2"]) > float(run["flat_rbf"]["heldout_r2"])
        for run in runs
    )
    tensor_better_csmi = sum(
        float(run["tensor"]["csmi_rmse"]) < float(run["flat_rbf"]["csmi_rmse"])
        for run in runs
    )
    return {
        "experiment": "multimodal_tensor_kernel",
        "config": asdict(config),
        "oracle": {
            "num_queries_total": int(total_queries),
            "oracle_seed": int(config.split_seed),
        },
        "runs": runs,
        "summary": {
            "tensor_mean_heldout_r2": float(np.mean([float(run["tensor"]["heldout_r2"]) for run in runs])),
            "flat_mean_heldout_r2": float(np.mean([float(run["flat_rbf"]["heldout_r2"]) for run in runs])),
            "tensor_mean_csmi_rmse": float(np.mean([float(run["tensor"]["csmi_rmse"]) for run in runs])),
            "flat_mean_csmi_rmse": float(np.mean([float(run["flat_rbf"]["csmi_rmse"]) for run in runs])),
            "tensor_wins_heldout_r2": int(tensor_better_fit),
            "tensor_wins_csmi_rmse": int(tensor_better_csmi),
            "num_repeats": int(config.num_repeats),
        },
    }


def run_single_split(
    config: MultimodalTensorKernelExperimentConfig,
    oracle: SyntheticMultimodalOracle,
    all_selection_matrices: np.ndarray,
    all_query_masks: np.ndarray,
    all_targets: np.ndarray,
    *,
    split_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(split_seed)
    indices = np.arange(all_targets.shape[0], dtype=np.int64)
    rng.shuffle(indices)
    train_count = max(32, int(round(config.train_fraction * all_targets.shape[0])))
    train_count = min(train_count, all_targets.shape[0] - 1)
    train_indices = np.sort(indices[:train_count])
    test_indices = np.sort(indices[train_count:])

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
            ridge_lambda=config.ridge_lambda,
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
            ridge_lambda=config.ridge_lambda,
            gamma=config.baseline_gamma,
        ),
    )

    tensor_test = tensor_model.predict_batch(test_selection, test_query)
    flat_test = flat_model.predict_batch(test_selection, test_query)
    tensor_all = tensor_model.predict_batch(all_selection_matrices, all_query_masks)
    flat_all = flat_model.predict_batch(all_selection_matrices, all_query_masks)

    exact_csmi, tensor_csmi, flat_csmi = evaluate_csmi_grid(config, oracle, tensor_model, flat_model)
    contrast = evaluate_structured_contrast(oracle, tensor_model, flat_model)

    return {
        "split_seed": int(split_seed),
        "num_train": int(train_indices.shape[0]),
        "num_test": int(test_indices.shape[0]),
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
    }


def evaluate_csmi_grid(
    config: MultimodalTensorKernelExperimentConfig,
    oracle: SyntheticMultimodalOracle,
    tensor_model,
    flat_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact_values = []
    tensor_values = []
    flat_values = []
    modality_subsets = [
        np.asarray(mask, dtype=bool)
        for mask in enumerate_nonempty_masks(config.num_modalities)
    ]
    for sample_index in range(config.num_samples):
        for train_subset in modality_subsets:
            for query_subset in modality_subsets:
                exact_values.append(
                    compute_csmi_from_oracle(
                        oracle.evaluate,
                        num_samples=config.num_samples,
                        num_modalities=config.num_modalities,
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


def evaluate_structured_contrast(
    oracle: SyntheticMultimodalOracle,
    tensor_model,
    flat_model,
) -> dict[str, object]:
    full_query = np.ones(oracle.num_modalities, dtype=bool)
    same_sample_complete = np.zeros((oracle.num_samples, oracle.num_modalities), dtype=bool)
    same_sample_complete[0, :] = True
    cross_sample_same_modality = np.zeros((oracle.num_samples, oracle.num_modalities), dtype=bool)
    cross_sample_same_modality[0, 0] = True
    cross_sample_same_modality[1, 0] = True
    cross_sample_same_modality[2, 0] = True

    exact_same = oracle.evaluate(same_sample_complete, full_query)
    exact_cross = oracle.evaluate(cross_sample_same_modality, full_query)
    return {
        "description": "same Hamming weight but different sample/modality structure",
        "exact_gap": float(exact_same - exact_cross),
        "tensor_gap": float(
            tensor_model.predict(same_sample_complete, full_query)
            - tensor_model.predict(cross_sample_same_modality, full_query)
        ),
        "flat_gap": float(
            flat_model.predict(same_sample_complete, full_query)
            - flat_model.predict(cross_sample_same_modality, full_query)
        ),
    }


def enumerate_nonempty_masks(num_bits: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bitmask in range(1, 2**num_bits):
        mask = np.zeros(num_bits, dtype=bool)
        for index in range(num_bits):
            mask[index] = bool((bitmask >> index) & 1)
        masks.append(mask)
    return masks


def _flatten_inputs(selection_matrices: np.ndarray, query_masks: np.ndarray) -> np.ndarray:
    selection = np.asarray(selection_matrices, dtype=np.float64)
    query = np.asarray(query_masks, dtype=np.float64)
    if selection.ndim != 3:
        raise ValueError("selection_matrices must have shape [batch, num_samples, num_modalities]")
    if query.ndim != 2 or query.shape[0] != selection.shape[0]:
        raise ValueError("query masks must have shape [batch, num_modalities] and align with selection_matrices")
    return np.concatenate([selection.reshape(selection.shape[0], -1), query], axis=1)


def _rbf_kernel(lhs: np.ndarray, rhs: np.ndarray, gamma: float) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs_sq = np.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs * rhs, axis=1, keepdims=True).T
    squared_distance = np.clip(lhs_sq + rhs_sq - 2.0 * (lhs @ rhs.T), a_min=0.0, a_max=None)
    return np.exp(-float(gamma) * squared_distance)


def _resolve_gamma(features: np.ndarray, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    if features.shape[0] <= 1:
        return 1.0
    lhs_sq = np.sum(features * features, axis=1, keepdims=True)
    squared_distance = np.clip(lhs_sq + lhs_sq.T - 2.0 * (features @ features.T), a_min=0.0, a_max=None)
    upper = squared_distance[np.triu_indices_from(squared_distance, k=1)]
    positive = upper[upper > 1e-12]
    if positive.size == 0:
        return 1.0
    return float(1.0 / np.median(positive))


def _as_modality_mask(modalities: Sequence[int] | Sequence[bool] | np.ndarray, num_modalities: int) -> np.ndarray:
    values = np.asarray(modalities)
    if values.dtype == bool:
        mask = values.reshape(-1)
        if mask.shape[0] != num_modalities:
            raise ValueError(f"expected modality mask with {num_modalities} entries")
        return mask.astype(bool)
    indices = np.asarray(modalities, dtype=np.int64).reshape(-1)
    mask = np.zeros(num_modalities, dtype=bool)
    for index in indices:
        if not (0 <= int(index) < num_modalities):
            raise IndexError(f"modality index must lie in [0, {num_modalities}), got {int(index)}")
        mask[int(index)] = True
    return mask


def pearson_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if lhs.shape[0] != rhs.shape[0] or lhs.shape[0] == 0:
        return 0.0
    lhs_centered = lhs - lhs.mean()
    rhs_centered = rhs - rhs.mean()
    denominator = np.sqrt(np.sum(lhs_centered**2) * np.sum(rhs_centered**2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(lhs_centered * rhs_centered) / denominator)


def rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    return float(np.sqrt(np.mean((lhs - rhs) ** 2)))


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    if denominator <= 1e-12:
        return 0.0
    numerator = float(np.sum((targets - predictions) ** 2))
    return float(1.0 - numerator / denominator)


def to_markdown(payload: dict[str, object]) -> str:
    config = payload["config"]
    summary = payload["summary"]
    runs = payload["runs"]
    lines = [
        "# Multimodal Tensor Kernel Experiment",
        "",
        "## Config",
        f"- num_samples: {config['num_samples']}",
        f"- num_modalities: {config['num_modalities']}",
        f"- train_fraction: {config['train_fraction']}",
        f"- num_repeats: {config['num_repeats']}",
        f"- ridge_lambda: {config['ridge_lambda']}",
        "",
        "## Summary",
        f"- tensor mean heldout R2: {summary['tensor_mean_heldout_r2']:.4f}",
        f"- flat mean heldout R2: {summary['flat_mean_heldout_r2']:.4f}",
        f"- tensor mean CSMI RMSE: {summary['tensor_mean_csmi_rmse']:.4f}",
        f"- flat mean CSMI RMSE: {summary['flat_mean_csmi_rmse']:.4f}",
        f"- tensor wins heldout R2: {summary['tensor_wins_heldout_r2']}/{summary['num_repeats']}",
        f"- tensor wins CSMI RMSE: {summary['tensor_wins_csmi_rmse']}/{summary['num_repeats']}",
        "",
        "## Per-run",
    ]
    for run in runs:
        lines.extend(
            [
                f"- seed {run['split_seed']}: tensor R2={run['tensor']['heldout_r2']:.4f}, flat R2={run['flat_rbf']['heldout_r2']:.4f}, tensor CSMI RMSE={run['tensor']['csmi_rmse']:.4f}, flat CSMI RMSE={run['flat_rbf']['csmi_rmse']:.4f}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic validation for the multimodal tensor-kernel surrogate.")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-modalities", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.12)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--sample-gamma", type=float, default=None)
    parser.add_argument("--modality-gamma", type=float, default=None)
    parser.add_argument("--query-gamma", type=float, default=None)
    parser.add_argument("--baseline-gamma", type=float, default=None)
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/pid_estimator/results/multimodal_tensor_kernel_experiment.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/multimodal_tensor_kernel_experiment.md",
    )
    args = parser.parse_args()

    config = MultimodalTensorKernelExperimentConfig(
        num_samples=args.num_samples,
        num_modalities=args.num_modalities,
        train_fraction=args.train_fraction,
        num_repeats=args.num_repeats,
        split_seed=args.split_seed,
        ridge_lambda=args.ridge_lambda,
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
