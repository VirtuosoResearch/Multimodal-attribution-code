from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_surrogate.sks import sks_masks_and_metric


DEFAULT_OUTPUT_JSON = REPO_ROOT / "src" / "kernel_surrogate" / "results" / "synthetic_sks_loo.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "src" / "kernel_surrogate" / "results" / "synthetic_sks_loo.md"
NUM_MODALITIES = 2


@dataclass(frozen=True)
class Teacher:
    num_samples: int
    planted_pairs: tuple[tuple[int, int], ...]
    pair_gains: np.ndarray
    sample_drop_penalty: np.ndarray
    modality_drop_penalty: np.ndarray
    prototype_states: np.ndarray
    prototype_weights: np.ndarray
    prototype_gamma: float
    sks_rho: float
    noise_std: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LOO synthetic ablation for the SKS kernel surrogate. The teacher loss "
            "is a nonlinear structured SKS-kernel mixture over sample-modality "
            "selection states. Methods are compared by leave-one-out subset-loss "
            "prediction."
        )
    )
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--num-states", type=int, default=120)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--sample-keep-prob", type=float, default=0.72)
    parser.add_argument("--modality-keep-prob", type=float, default=0.68)
    parser.add_argument("--num-prototypes", type=int, default=20)
    parser.add_argument("--pair-gain-min", type=float, default=0.0)
    parser.add_argument("--pair-gain-max", type=float, default=0.0)
    parser.add_argument("--sample-penalty-scale", type=float, default=0.0)
    parser.add_argument("--modality-penalty-scale", type=float, default=0.0)
    parser.add_argument("--prototype-gamma", type=float, default=0.3)
    parser.add_argument("--prototype-weight-std", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--sks-rho", type=float, default=0.97)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=[1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    parser.add_argument("--gamma-scale-grid", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    methods = ("linear_structured", "flat_rbf", "structured_identity_rbf", "sks")
    seeds: list[dict[str, Any]] = []
    for offset in range(int(args.num_seeds)):
        seed = int(args.base_seed) + offset
        print(f"[seed {seed}] sampling states and teacher", flush=True)
        states = sample_states(
            num_states=int(args.num_states),
            num_samples=int(args.num_samples),
            sample_keep_prob=float(args.sample_keep_prob),
            modality_keep_prob=float(args.modality_keep_prob),
            seed=seed,
        )
        states[0, :, :] = True
        if states.shape[0] > 1:
            states[1, :, :] = False
        teacher = make_teacher(
            num_samples=int(args.num_samples),
            num_pairs=int(args.num_pairs),
            num_prototypes=int(args.num_prototypes),
            pair_gain_min=float(args.pair_gain_min),
            pair_gain_max=float(args.pair_gain_max),
            sample_penalty_scale=float(args.sample_penalty_scale),
            modality_penalty_scale=float(args.modality_penalty_scale),
            sample_keep_prob=float(args.sample_keep_prob),
            modality_keep_prob=float(args.modality_keep_prob),
            prototype_gamma=float(args.prototype_gamma),
            prototype_weight_std=float(args.prototype_weight_std),
            sks_rho=float(args.sks_rho),
            noise_std=float(args.noise_std),
            seed=10_000 + seed,
        )
        clean_targets = teacher_loss(teacher, states)
        noisy_targets = add_noise(clean_targets, noise_std=float(args.noise_std), seed=20_000 + seed)
        true_interactions = interaction_matrix_from_loss(teacher, int(args.num_samples))

        method_rows: list[dict[str, Any]] = []
        for method in methods:
            row = evaluate_method(
                method,
                states,
                noisy_targets,
                clean_targets=clean_targets,
                true_interactions=true_interactions,
                planted_pairs=teacher.planted_pairs,
                ridge_grid=[float(value) for value in args.ridge_grid],
                gamma_scale_grid=[float(value) for value in args.gamma_scale_grid],
                sks_rho=float(args.sks_rho),
            )
            method_rows.append(row)
            print(
                f"  {method}: LOO RMSE={row['loo_rmse']:.4f}, "
                f"LOO MAE={row['loo_mae']:.4f}, LOO R2={row['loo_r2']:.4f}",
                flush=True,
            )
        seeds.append(
            {
                "seed": seed,
                "planted_pairs": [[int(i), int(j)] for i, j in teacher.planted_pairs],
                "true_interactions": summarize_matrix(true_interactions),
                "methods": method_rows,
            }
        )

    output = {
        "config": {
            "num_samples": int(args.num_samples),
            "num_modalities": NUM_MODALITIES,
            "num_pairs": int(args.num_pairs),
            "num_states": int(args.num_states),
            "num_seeds": int(args.num_seeds),
            "base_seed": int(args.base_seed),
            "sample_keep_prob": float(args.sample_keep_prob),
            "modality_keep_prob": float(args.modality_keep_prob),
            "num_prototypes": int(args.num_prototypes),
            "pair_gain_min": float(args.pair_gain_min),
            "pair_gain_max": float(args.pair_gain_max),
            "sample_penalty_scale": float(args.sample_penalty_scale),
            "modality_penalty_scale": float(args.modality_penalty_scale),
            "prototype_gamma": float(args.prototype_gamma),
            "prototype_weight_std": float(args.prototype_weight_std),
            "noise_std": float(args.noise_std),
            "sks_rho": float(args.sks_rho),
            "ridge_grid": [float(value) for value in args.ridge_grid],
            "gamma_scale_grid": [float(value) for value in args.gamma_scale_grid],
            "methods": list(methods),
            "loo_protocol": (
                "For each method and hyperparameter candidate, fit n leave-one-out "
                "surrogates on n-1 state-loss pairs and predict the held-out state. "
                "The candidate with the lowest LOO RMSE against the held-out losses "
                "is selected as that method's fitting result."
            ),
        },
        "summary": summarize_all(seeds, methods),
        "claim_checks": claim_checks(seeds),
        "seeds": seeds,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"\nSaved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def make_teacher(
    *,
    num_samples: int,
    num_pairs: int,
    num_prototypes: int,
    pair_gain_min: float,
    pair_gain_max: float,
    sample_penalty_scale: float,
    modality_penalty_scale: float,
    sample_keep_prob: float,
    modality_keep_prob: float,
    prototype_gamma: float,
    prototype_weight_std: float,
    sks_rho: float,
    noise_std: float,
    seed: int,
) -> Teacher:
    if num_samples < 2 * num_pairs:
        raise ValueError("num_samples must be at least 2 * num_pairs")
    rng = np.random.default_rng(seed)
    planted_pairs = tuple((2 * pair_id, 2 * pair_id + 1) for pair_id in range(num_pairs))
    return Teacher(
        num_samples=num_samples,
        planted_pairs=planted_pairs,
        pair_gains=np.linspace(float(pair_gain_min), float(pair_gain_max), num_pairs, dtype=np.float64),
        sample_drop_penalty=(
            float(sample_penalty_scale) * rng.uniform(0.015, 0.045, size=num_samples).astype(np.float64)
        ),
        modality_drop_penalty=(
            float(modality_penalty_scale)
            * rng.uniform(0.004, 0.018, size=(num_samples, NUM_MODALITIES)).astype(np.float64)
        ),
        prototype_states=sample_states(
            num_states=num_prototypes,
            num_samples=num_samples,
            sample_keep_prob=sample_keep_prob,
            modality_keep_prob=modality_keep_prob,
            seed=seed + 1,
        ),
        prototype_weights=rng.normal(0.0, float(prototype_weight_std), size=num_prototypes).astype(np.float64),
        prototype_gamma=float(prototype_gamma),
        sks_rho=float(sks_rho),
        noise_std=float(noise_std),
    )


def teacher_loss(teacher: Teacher, states: np.ndarray) -> np.ndarray:
    states = as_state_batch(states, teacher.num_samples)
    present = states.any(axis=2)
    sample_deleted = (~present).astype(np.float64)
    modality_deleted = np.logical_and(present[:, :, None], ~states).astype(np.float64)

    loss = np.full(states.shape[0], 1.0, dtype=np.float64)
    loss += sample_deleted @ teacher.sample_drop_penalty
    loss += modality_deleted.reshape(states.shape[0], -1) @ teacher.modality_drop_penalty.reshape(-1)
    for pair_id, (left, right) in enumerate(teacher.planted_pairs):
        loss -= float(teacher.pair_gains[pair_id]) * states[:, left, 0] * states[:, right, 1]

    state_masks, metric = sks_features(states, rho=teacher.sks_rho)
    proto_masks, _ = sks_features(teacher.prototype_states, rho=teacher.sks_rho)
    loss += np.exp(-teacher.prototype_gamma * metric_sqdist(state_masks, proto_masks, metric)) @ teacher.prototype_weights
    return loss


def add_noise(values: np.ndarray, *, noise_std: float, seed: int) -> np.ndarray:
    if noise_std <= 0.0:
        return np.asarray(values, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    return np.asarray(values, dtype=np.float64) + rng.normal(0.0, noise_std, size=len(values))


def sample_states(
    *,
    num_states: int,
    num_samples: int,
    sample_keep_prob: float,
    modality_keep_prob: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    present = rng.random((num_states, num_samples)) < sample_keep_prob
    states = rng.random((num_states, num_samples, NUM_MODALITIES)) < modality_keep_prob
    states = np.logical_and(states, present[:, :, None])
    empty_present = np.logical_and(present, ~states.any(axis=2))
    rows, cols = np.where(empty_present)
    if rows.size:
        states[rows, cols, rng.integers(0, NUM_MODALITIES, size=rows.size)] = True
    return states.astype(bool, copy=False)


def as_state_batch(states: np.ndarray, num_samples: int) -> np.ndarray:
    states = np.asarray(states, dtype=bool)
    if states.ndim == 2:
        states = states.reshape(1, num_samples, NUM_MODALITIES)
    if states.shape[1:] != (num_samples, NUM_MODALITIES):
        raise ValueError(f"expected states shape (*, {num_samples}, {NUM_MODALITIES}), got {states.shape}")
    return states


def mask_blocks(states: np.ndarray) -> dict[str, np.ndarray]:
    present = states.any(axis=2)
    sample_deleted = (~present).astype(np.float64)
    modality_deleted = np.logical_and(present[:, :, None], ~states).astype(np.float64).reshape(states.shape[0], -1)
    flat_deleted = (~states).astype(np.float64).reshape(states.shape[0], -1)
    return {
        "sample_deleted": sample_deleted,
        "modality_deleted": modality_deleted,
        "flat_deleted": flat_deleted,
        "num_groups": np.asarray([states.shape[1]], dtype=np.int64),
    }


def sks_features(states: np.ndarray, *, rho: float) -> tuple[np.ndarray, np.ndarray]:
    blocks = mask_blocks(states)
    return sks_masks_and_metric(blocks, train_indices=np.arange(states.shape[0]), rho=rho)


def method_features(method: str, states: np.ndarray, *, rho: float) -> tuple[np.ndarray, np.ndarray]:
    blocks = mask_blocks(states)
    if method == "flat_rbf":
        features = blocks["flat_deleted"]
        return features, np.eye(features.shape[1], dtype=np.float64)
    if method == "structured_identity_rbf":
        features = np.concatenate([blocks["sample_deleted"], blocks["modality_deleted"]], axis=1)
        return features, np.eye(features.shape[1], dtype=np.float64)
    if method in {"sks", "linear_structured"}:
        return sks_masks_and_metric(blocks, train_indices=np.arange(states.shape[0]), rho=rho)
    raise ValueError(f"unknown method: {method}")


def evaluate_method(
    method: str,
    states: np.ndarray,
    targets: np.ndarray,
    *,
    clean_targets: np.ndarray,
    true_interactions: np.ndarray,
    planted_pairs: Sequence[tuple[int, int]],
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    sks_rho: float,
) -> dict[str, Any]:
    features, metric = method_features(method, states, rho=sks_rho)
    candidates = candidate_predictions(
        method,
        features,
        targets,
        metric=metric,
        ridge_grid=ridge_grid,
        gamma_scale_grid=gamma_scale_grid,
    )
    best = min(candidates, key=lambda item: (item["loo_rmse"], item["loo_mae"]))
    fit = fit_full(method, features, targets, metric=metric, hparams=best["hparams"], sks_rho=sks_rho)
    pred_interactions = interaction_matrix_from_fit(fit, states.shape[1])
    interaction_eval = evaluate_interactions(pred_interactions, true_interactions, planted_pairs)
    return {
        "method": method,
        "selected_hparams": best["hparams"],
        "loo_rmse": best["loo_rmse"],
        "loo_mae": best["loo_mae"],
        "loo_r2": best["loo_r2"],
        "loo_clean_rmse": rmse(best["loo_predictions"], clean_targets),
        "loo_clean_mae": mae(best["loo_predictions"], clean_targets),
        "loo_predictions": [float(value) for value in best["loo_predictions"].tolist()],
        "interaction_rmse": interaction_eval["rmse"],
        "interaction_spearman": interaction_eval["spearman"],
        "interaction_recall_at_p": interaction_eval["recall_at_p"],
        "interaction_hits_at_p": interaction_eval["hits_at_p"],
        "top_p_pairs": interaction_eval["top_p_pairs"],
        "top_10_pairs": interaction_eval["top_10_pairs"],
        "planted_pair_ranks": interaction_eval["planted_pair_ranks"],
    }


def candidate_predictions(
    method: str,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    metric: np.ndarray,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
) -> list[dict[str, Any]]:
    if method == "linear_structured":
        return [
            candidate_row({"ridge_lambda": float(ridge)}, loo_linear_predictions(features, targets, ridge_lambda=float(ridge)), targets)
            for ridge in ridge_grid
        ]

    distances = metric_sqdist(features, features, metric)
    base_gamma = median_gamma(distances)
    rows: list[dict[str, Any]] = []
    for ridge in ridge_grid:
        for gamma_scale in gamma_scale_grid:
            gamma = float(base_gamma * float(gamma_scale))
            kernel = np.exp(-gamma * distances)
            predictions = loo_krr_predictions(kernel, targets, ridge_lambda=float(ridge))
            rows.append(
                candidate_row(
                    {
                        "ridge_lambda": float(ridge),
                        "gamma_scale": float(gamma_scale),
                        "base_gamma": float(base_gamma),
                        "gamma": float(gamma),
                    },
                    predictions,
                    targets,
                )
            )
    return rows


def candidate_row(hparams: dict[str, float], predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    return {
        "hparams": hparams,
        "loo_predictions": predictions,
        "loo_rmse": rmse(predictions, targets),
        "loo_mae": mae(predictions, targets),
        "loo_r2": r2_score(predictions, targets),
    }


def loo_krr_predictions(kernel: np.ndarray, targets: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    system = kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64)
    inv_system = np.linalg.inv(system)
    alpha = inv_system @ targets
    denom = np.diag(inv_system)
    return targets - alpha / np.maximum(denom, 1e-12)


def loo_linear_predictions(features: np.ndarray, targets: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
    penalty = float(ridge_lambda) * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    inv_gram = np.linalg.inv(design.T @ design + penalty)
    hat = design @ inv_gram @ design.T
    fitted = hat @ targets
    leverage = np.clip(np.diag(hat), None, 1.0 - 1e-12)
    return (fitted - leverage * targets) / (1.0 - leverage)


def fit_full(
    method: str,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    metric: np.ndarray,
    hparams: dict[str, float],
    sks_rho: float,
) -> dict[str, Any]:
    if method == "linear_structured":
        design = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
        penalty = float(hparams["ridge_lambda"]) * np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return {
            "method": method,
            "kind": "linear",
            "coef": coef,
            "metric": metric,
            "sks_rho": float(sks_rho),
        }

    kernel = np.exp(-float(hparams["gamma"]) * metric_sqdist(features, features, metric))
    alpha = np.linalg.solve(
        kernel + float(hparams["ridge_lambda"]) * np.eye(kernel.shape[0], dtype=np.float64),
        targets,
    )
    return {
        "method": method,
        "kind": "krr",
        "alpha": alpha,
        "train_features": features,
        "metric": metric,
        "gamma": float(hparams["gamma"]),
        "sks_rho": float(sks_rho),
    }


def predict_fit(fit: dict[str, Any], states: np.ndarray) -> np.ndarray:
    features, _ = method_features(str(fit["method"]), states, rho=float(fit["sks_rho"]))
    if fit["kind"] == "linear":
        design = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
        return design @ fit["coef"]
    distances = metric_sqdist(features, fit["train_features"], fit["metric"])
    return np.exp(-float(fit["gamma"]) * distances) @ fit["alpha"]


def interaction_matrix_from_fit(fit: dict[str, Any], num_samples: int) -> np.ndarray:
    states, pairs = interaction_query_states(num_samples)
    values = predict_fit(fit, states)
    return finite_difference_matrix(values, pairs, num_samples)


def interaction_matrix_from_loss(teacher: Teacher, num_samples: int) -> np.ndarray:
    states, pairs = interaction_query_states(num_samples)
    values = teacher_loss(teacher, states)
    return finite_difference_matrix(values, pairs, num_samples)


def interaction_query_states(num_samples: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    reference = np.ones((num_samples, NUM_MODALITIES), dtype=bool)
    states: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    for left in range(num_samples):
        for right in range(num_samples):
            if left == right:
                continue
            pairs.append((left, right))
            for left_on, right_on in ((True, True), (False, True), (True, False), (False, False)):
                state = reference.copy()
                state[left, 0] = left_on
                state[right, 1] = right_on
                states.append(state)
    return np.stack(states, axis=0), pairs


def finite_difference_matrix(values: np.ndarray, pairs: Sequence[tuple[int, int]], num_samples: int) -> np.ndarray:
    scores = np.full((num_samples, num_samples), np.nan, dtype=np.float64)
    for pair_id, (left, right) in enumerate(pairs):
        offset = 4 * pair_id
        scores[left, right] = values[offset] - values[offset + 1] - values[offset + 2] + values[offset + 3]
    return scores


def evaluate_interactions(
    predicted: np.ndarray,
    truth: np.ndarray,
    planted_pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    planted = {(int(left), int(right)) for left, right in planted_pairs}
    finite = np.isfinite(predicted) & np.isfinite(truth)
    rows, cols = np.where(finite)
    predicted_benefit = -predicted[rows, cols]
    true_benefit = -truth[rows, cols]
    order = np.argsort(-predicted_benefit, kind="mergesort")
    ranked_pairs = [(int(rows[index]), int(cols[index])) for index in order]
    p = len(planted)
    top_p = ranked_pairs[:p]
    hits = sum(1 for pair in top_p if pair in planted)
    top_10 = []
    for index in order[:10]:
        left = int(rows[index])
        right = int(cols[index])
        top_10.append(
            {
                "pair": [left, right],
                "predicted_benefit": float(-predicted[left, right]),
                "true_benefit": float(-truth[left, right]),
                "is_planted": (left, right) in planted,
            }
        )
    return {
        "rmse": rmse(predicted[finite], truth[finite]),
        "spearman": spearmanr(predicted_benefit, true_benefit),
        "recall_at_p": float(hits / max(p, 1)),
        "hits_at_p": int(hits),
        "top_p_pairs": [[int(left), int(right)] for left, right in top_p],
        "top_10_pairs": top_10,
        "planted_pair_ranks": {
            f"{int(left)},{int(right)}": int(ranked_pairs.index((int(left), int(right))) + 1)
            for left, right in planted_pairs
        },
    }


def summarize_all(seeds: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, dict[str, float]]:
    metrics = ("loo_rmse", "loo_mae", "loo_r2", "loo_clean_rmse", "loo_clean_mae")
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        rows = [row for seed in seeds for row in seed["methods"] if row["method"] == method]
        summary[method] = {}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[method][f"{metric}_mean"] = float(np.nanmean(values))
            summary[method][f"{metric}_std"] = float(np.nanstd(values))
    return summary


def claim_checks(seeds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_all(seeds, ("linear_structured", "flat_rbf", "structured_identity_rbf", "sks"))
    sks = summary["sks"]
    linear = summary["linear_structured"]
    flat = summary["flat_rbf"]
    structured_identity = summary["structured_identity_rbf"]
    return {
        "nonlinearity_useful_sks_minus_linear": {
            "loo_rmse_reduction": float(linear["loo_rmse_mean"] - sks["loo_rmse_mean"]),
            "loo_mae_reduction": float(linear["loo_mae_mean"] - sks["loo_mae_mean"]),
            "loo_r2_gain": float(sks["loo_r2_mean"] - linear["loo_r2_mean"]),
        },
        "structure_useful_sks_minus_flat_rbf": {
            "loo_rmse_reduction": float(flat["loo_rmse_mean"] - sks["loo_rmse_mean"]),
            "loo_mae_reduction": float(flat["loo_mae_mean"] - sks["loo_mae_mean"]),
            "loo_r2_gain": float(sks["loo_r2_mean"] - flat["loo_r2_mean"]),
        },
        "structure_useful_sks_minus_structured_identity_rbf": {
            "loo_rmse_reduction": float(structured_identity["loo_rmse_mean"] - sks["loo_rmse_mean"]),
            "loo_mae_reduction": float(structured_identity["loo_mae_mean"] - sks["loo_mae_mean"]),
            "loo_r2_gain": float(sks["loo_r2_mean"] - structured_identity["loo_r2_mean"]),
        },
    }


def summarize_matrix(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def metric_sqdist(lhs: np.ndarray, rhs: np.ndarray, metric: np.ndarray) -> np.ndarray:
    lhs_metric = lhs @ metric
    lhs_sq = np.sum(lhs_metric * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum((rhs @ metric) * rhs, axis=1, keepdims=True).T
    return np.maximum(lhs_sq + rhs_sq - 2.0 * lhs_metric @ rhs.T, 0.0)


def median_gamma(distances: np.ndarray) -> float:
    values = distances[np.triu_indices(distances.shape[0], k=1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    return 1.0 / float(np.median(values))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2)))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    residual = float(np.sum((predictions - targets) ** 2))
    total = float(np.sum((targets - np.mean(targets)) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - residual / total)


def spearmanr(first: np.ndarray, second: np.ndarray) -> float:
    x = rankdata(np.asarray(first, dtype=np.float64))
    y = rankdata(np.asarray(second, dtype=np.float64))
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(x, y) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def to_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# Synthetic SKS LOO Fit Ablation",
        "",
        "## Protocol",
        "",
        f"- states per seed: {output['config']['num_states']}",
        f"- seeds: {output['config']['num_seeds']}",
        f"- LOO: {output['config']['loo_protocol']}",
        "",
        "## LOO Fit Summary",
        "",
        "| method | LOO RMSE | LOO MAE | LOO R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in output["config"]["methods"]:
        row = output["summary"][method]
        lines.append(
            f"| {method} | {row['loo_rmse_mean']:.4f} +/- {row['loo_rmse_std']:.4f} | "
            f"{row['loo_mae_mean']:.4f} +/- {row['loo_mae_std']:.4f} | "
            f"{row['loo_r2_mean']:.4f} +/- {row['loo_r2_std']:.4f} |"
        )
    lines.extend(["", "## Claim Checks", ""])
    for name, values in output["claim_checks"].items():
        fields = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
        lines.append(f"- {name}: {fields}")
    lines.extend(["", "## Per Seed", ""])
    for seed in output["seeds"]:
        lines.append(f"- seed {seed['seed']}, planted {seed['planted_pairs']}")
        for row in seed["methods"]:
            lines.append(
                f"  - {row['method']}: LOO RMSE={row['loo_rmse']:.4f}, "
                f"LOO MAE={row['loo_mae']:.4f}, LOO R2={row['loo_r2']:.4f}, "
                f"hparams={row['selected_hparams']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
