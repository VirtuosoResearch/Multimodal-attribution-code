from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "synthetic_exp.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "synthetic_exp.md"


@dataclass(frozen=True)
class SyntheticDataset:
    x1: np.ndarray
    x2: np.ndarray
    y: np.ndarray
    test_x1: np.ndarray
    test_x2: np.ndarray
    test_y: float
    planted_pairs: tuple[tuple[int, int], ...]
    target_mod1_indices: tuple[int, ...]
    target_mod2_indices: tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic sanity check in synthetic_exp.md: build a scalar "
            "two-modality dataset, fit a Mahalanobis-RBF kernel surrogate on random "
            "atom masks, and analytically extract cross-sample modal interactions."
        )
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-pairs", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--distractor-beta", type=float, default=0.2)
    parser.add_argument("--distractor-noise", type=float, default=0.005)
    parser.add_argument("--distractor-label-noise", type=float, default=0.002)
    parser.add_argument("--distractor-dot-threshold", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--num-masks", type=int, default=2000)
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--mask-seed", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=2)
    parser.add_argument("--init-std", type=float, default=1e-3)
    parser.add_argument("--init-mode", choices=["normal", "positive", "constant"], default="constant")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    parser.add_argument("--gamma-scale", type=float, default=1.0)
    parser.add_argument("--no-normalize-gradient-k", action="store_true")
    parser.add_argument("--batch-masks", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dataset = make_synthetic_dataset(
        num_samples=args.num_samples,
        num_pairs=args.num_pairs,
        epsilon=args.epsilon,
        beta=args.distractor_beta,
        distractor_noise=args.distractor_noise,
        label_noise=args.distractor_label_noise,
        dot_threshold=args.distractor_dot_threshold,
        seed=args.dataset_seed,
    )
    masks = sample_masks(
        num_masks=args.num_masks,
        num_atoms=args.num_samples * 2,
        mask_prob=args.mask_prob,
        seed=args.mask_seed,
    )

    print(
        f"Training {args.num_masks} masked predictors on {device} "
        f"({args.epochs} epochs, optimizer={args.optimizer})...",
        flush=True,
    )
    subset_losses = train_masked_predictors(
        dataset,
        masks,
        epsilon=args.epsilon,
        init_std=args.init_std,
        init_mode=args.init_mode,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        train_seed=args.train_seed,
        device=device,
        batch_masks=args.batch_masks,
    )
    gradient_k = build_gradient_metric(dataset, epsilon=args.epsilon, theta_value=args.init_std)
    if not args.no_normalize_gradient_k:
        gradient_k = normalize_psd_metric(gradient_k)

    fit = fit_mahalanobis_rbf_surrogate(
        masks.astype(np.float64),
        subset_losses,
        gradient_k,
        ridge_lambda=args.ridge_lambda,
        gamma_scale=args.gamma_scale,
    )
    score_derivative = analytical_cmis(
        masks.astype(np.float64),
        gradient_k,
        alpha=fit["alpha"],
        target_mean=float(fit["target_mean"]),
        gamma=float(fit["gamma"]),
        num_samples=args.num_samples,
    )
    score_loss_reduction = -score_derivative

    derivative_eval = evaluate_rankings(score_derivative, dataset.planted_pairs, descending=True)
    derivative_ascending_eval = evaluate_rankings(score_derivative, dataset.planted_pairs, descending=False)
    loss_reduction_eval = evaluate_rankings(score_loss_reduction, dataset.planted_pairs, descending=True)
    target_block_eval = summarize_target_block(score_derivative, dataset)
    degeneracy = detect_symmetry_degeneracy(score_derivative, dataset)

    output = {
        "config": {
            "num_samples": int(args.num_samples),
            "num_pairs": int(args.num_pairs),
            "epsilon": float(args.epsilon),
            "distractor_beta": float(args.distractor_beta),
            "distractor_noise": float(args.distractor_noise),
            "distractor_label_noise": float(args.distractor_label_noise),
            "distractor_dot_threshold": float(args.distractor_dot_threshold),
            "num_masks": int(args.num_masks),
            "mask_prob": float(args.mask_prob),
            "dataset_seed": int(args.dataset_seed),
            "mask_seed": int(args.mask_seed),
            "train_seed": int(args.train_seed),
            "init_std": float(args.init_std),
            "init_mode": str(args.init_mode),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "optimizer": str(args.optimizer),
            "ridge_lambda": float(args.ridge_lambda),
            "gamma_scale": float(args.gamma_scale),
            "normalize_gradient_k": not bool(args.no_normalize_gradient_k),
            "device": str(device),
        },
        "dataset": {
            "planted_pairs": [[int(i), int(j)] for i, j in dataset.planted_pairs],
            "target_mod1_indices": [int(i) for i in dataset.target_mod1_indices],
            "target_mod2_indices": [int(i) for i in dataset.target_mod2_indices],
            "feature_dim": int(dataset.x1.shape[1]),
            "test_x1": [float(v) for v in dataset.test_x1.tolist()],
            "test_x2": [float(v) for v in dataset.test_x2.tolist()],
            "test_y": float(dataset.test_y),
        },
        "surrogate": {
            "gamma": float(fit["gamma"]),
            "base_gamma": float(fit["base_gamma"]),
            "train_rmse": rmse(fit["predictions"], subset_losses),
            "train_spearman": spearmanr(fit["predictions"], subset_losses),
            "subset_loss_mean": float(np.mean(subset_losses)),
            "subset_loss_std": float(np.std(subset_losses)),
            "subset_loss_min": float(np.min(subset_losses)),
            "subset_loss_max": float(np.max(subset_losses)),
        },
        "ranking_by_raw_derivative_desc": derivative_eval,
        "ranking_by_raw_derivative_asc": derivative_ascending_eval,
        "ranking_by_loss_reduction_desc": loss_reduction_eval,
        "target_block_summary": target_block_eval,
        "symmetry_degeneracy": degeneracy,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"\nSaved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested} to cpu.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def make_synthetic_dataset(
    *,
    num_samples: int,
    num_pairs: int,
    epsilon: float,
    beta: float,
    distractor_noise: float,
    label_noise: float,
    dot_threshold: float,
    seed: int,
) -> SyntheticDataset:
    if num_samples < 2 * num_pairs:
        raise ValueError("num_samples must be at least 2 * num_pairs")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    rng = np.random.default_rng(seed)
    feature_dim = num_pairs
    x1 = np.empty((num_samples, feature_dim), dtype=np.float64)
    x2 = np.empty((num_samples, feature_dim), dtype=np.float64)
    y = np.empty(num_samples, dtype=np.float64)
    planted_pairs: list[tuple[int, int]] = []
    target_mod1: list[int] = []
    target_mod2: list[int] = []
    eps_vec = np.full(feature_dim, epsilon, dtype=np.float64)
    for pair_index in range(num_pairs):
        i = 2 * pair_index
        j = 2 * pair_index + 1
        basis = np.zeros(feature_dim, dtype=np.float64)
        basis[pair_index] = 1.0
        x1[i] = basis
        x2[i] = eps_vec
        x1[j] = eps_vec
        x2[j] = basis
        y[i] = 1.0
        y[j] = 1.0
        planted_pairs.append((i, j))
        target_mod1.append(i)
        target_mod2.append(j)
    if 2 * num_pairs < num_samples:
        for sample_index in range(2 * num_pairs, num_samples):
            x1[sample_index], x2[sample_index] = sample_orthogonal_distractor(
                rng,
                feature_dim=feature_dim,
                beta=beta,
                noise=distractor_noise,
                dot_threshold=dot_threshold,
            )
            y[sample_index] = float(np.dot(x1[sample_index], x2[sample_index])) + float(
                rng.normal(0.0, label_noise)
            )
    return SyntheticDataset(
        x1=x1,
        x2=x2,
        y=y,
        test_x1=np.ones(feature_dim, dtype=np.float64),
        test_x2=np.ones(feature_dim, dtype=np.float64),
        test_y=float(num_pairs),
        planted_pairs=tuple(planted_pairs),
        target_mod1_indices=tuple(target_mod1),
        target_mod2_indices=tuple(target_mod2),
    )


def sample_orthogonal_distractor(
    rng: np.random.Generator,
    *,
    feature_dim: int,
    beta: float,
    noise: float,
    dot_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(10_000):
        a = int(rng.integers(0, feature_dim))
        b = int(rng.integers(0, feature_dim - 1))
        if b >= a:
            b += 1
        x1 = rng.uniform(0.0, noise, size=feature_dim)
        x2 = rng.uniform(0.0, noise, size=feature_dim)
        x1[a] += beta
        x2[b] += beta
        if float(np.dot(x1, x2)) < dot_threshold:
            return x1.astype(np.float64, copy=False), x2.astype(np.float64, copy=False)
    raise RuntimeError("failed to sample distractor under dot_threshold")


def sample_masks(*, num_masks: int, num_atoms: int, mask_prob: float, seed: int) -> np.ndarray:
    if not 0.0 < mask_prob < 1.0:
        raise ValueError("mask_prob must be in (0, 1)")
    rng = np.random.default_rng(seed)
    masks = rng.random((num_masks, num_atoms)) < mask_prob
    # Add anchors that stabilize local derivatives around all-present masks.
    masks[0, :] = True
    if num_masks > 1:
        masks[1, :] = False
    return masks.astype(np.float32)


def train_masked_predictors(
    dataset: SyntheticDataset,
    masks: np.ndarray,
    *,
    epsilon: float,
    init_std: float,
    init_mode: str,
    epochs: int,
    lr: float,
    optimizer: str,
    train_seed: int,
    device: torch.device,
    batch_masks: int,
) -> np.ndarray:
    if batch_masks <= 0:
        raise ValueError("batch_masks must be positive")
    all_losses: list[np.ndarray] = []
    x1_base = torch.as_tensor(dataset.x1, dtype=torch.float32, device=device)
    x2_base = torch.as_tensor(dataset.x2, dtype=torch.float32, device=device)
    y = torch.as_tensor(dataset.y, dtype=torch.float32, device=device)
    test_x1 = torch.as_tensor(dataset.test_x1, dtype=torch.float32, device=device)
    test_x2 = torch.as_tensor(dataset.test_x2, dtype=torch.float32, device=device)
    test_y = torch.tensor(float(dataset.test_y), dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(train_seed))

    for start in range(0, masks.shape[0], batch_masks):
        end = min(start + batch_masks, masks.shape[0])
        mask = torch.as_tensor(masks[start:end].reshape(end - start, -1, 2), dtype=torch.float32, device=device)
        x1_eff = mask[:, :, 0:1] * x1_base.unsqueeze(0) + (1.0 - mask[:, :, 0:1]) * float(epsilon)
        x2_eff = mask[:, :, 1:2] * x2_base.unsqueeze(0) + (1.0 - mask[:, :, 1:2]) * float(epsilon)
        weight_shape = (end - start, 2, dataset.x1.shape[1])
        if init_mode == "constant":
            weights = torch.full(weight_shape, float(init_std), dtype=torch.float32, device=device)
        elif init_mode == "positive":
            weights = torch.abs(
                torch.randn(weight_shape, generator=generator, dtype=torch.float32, device=device)
            ) * float(init_std)
        else:
            weights = torch.randn(weight_shape, generator=generator, dtype=torch.float32, device=device) * float(init_std)
        weights.requires_grad_(True)
        if optimizer == "adam":
            opt = torch.optim.Adam([weights], lr=float(lr))
        else:
            opt = torch.optim.SGD([weights], lr=float(lr))
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            pred = torch.sum(
                (weights[:, 0:1, :] * x1_eff) * (weights[:, 1:2, :] * x2_eff),
                dim=2,
            )
            loss_per_mask = 0.5 * torch.mean((pred - y.unsqueeze(0)) ** 2, dim=1)
            loss = loss_per_mask.mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            test_pred = torch.sum((weights[:, 0, :] * test_x1) * (weights[:, 1, :] * test_x2), dim=1)
            test_loss = 0.5 * (test_pred - test_y) ** 2
            all_losses.append(test_loss.detach().cpu().numpy().astype(np.float64))
        print(f"  trained masks {start}-{end - 1}", flush=True)
    return np.concatenate(all_losses, axis=0)


def build_gradient_metric(dataset: SyntheticDataset, *, epsilon: float, theta_value: float) -> np.ndarray:
    feature_dim = dataset.x1.shape[1]
    w1 = np.full(feature_dim, theta_value, dtype=np.float64)
    w2 = np.full(feature_dim, theta_value, dtype=np.float64)
    gradients = np.empty((dataset.x1.shape[0] * 2, 2 * feature_dim), dtype=np.float64)
    eps_vec = np.full(feature_dim, epsilon, dtype=np.float64)
    for sample_index in range(dataset.x1.shape[0]):
        y = float(dataset.y[sample_index])
        masked_inputs = ((dataset.x1[sample_index], eps_vec), (eps_vec, dataset.x2[sample_index]))
        for modality_index, (mx1, mx2) in enumerate(masked_inputs):
            pred = float(np.sum((w1 * mx1) * (w2 * mx2)))
            error = pred - y
            gradients[sample_index * 2 + modality_index, :feature_dim] = error * w2 * mx1 * mx2
            gradients[sample_index * 2 + modality_index, feature_dim:] = error * w1 * mx1 * mx2
    return gradients @ gradients.T


def normalize_psd_metric(metric: np.ndarray) -> np.ndarray:
    diag_mean = float(np.mean(np.diag(metric)))
    if not math.isfinite(diag_mean) or diag_mean <= 0.0:
        return metric
    return metric / diag_mean


def fit_mahalanobis_rbf_surrogate(
    masks: np.ndarray,
    targets: np.ndarray,
    metric: np.ndarray,
    *,
    ridge_lambda: float,
    gamma_scale: float,
) -> dict[str, Any]:
    distances = mahalanobis_sqdist(masks, masks, metric)
    base_gamma = median_heuristic_gamma(distances)
    gamma = float(base_gamma * gamma_scale)
    kernel = np.exp(-gamma * distances)
    target_mean = float(np.mean(targets))
    alpha = np.linalg.solve(
        kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64),
        targets - target_mean,
    )
    predictions = target_mean + kernel @ alpha
    return {
        "alpha": alpha,
        "target_mean": target_mean,
        "gamma": gamma,
        "base_gamma": base_gamma,
        "predictions": predictions,
    }


def analytical_cmis(
    train_masks: np.ndarray,
    metric: np.ndarray,
    *,
    alpha: np.ndarray,
    target_mean: float,
    gamma: float,
    num_samples: int,
) -> np.ndarray:
    del target_mean
    anchor = np.ones(train_masks.shape[1], dtype=np.float64)
    diffs = anchor[None, :] - train_masks
    q = diffs @ metric
    kernel = np.exp(-float(gamma) * np.einsum("mi,ij,mj->m", diffs, metric, diffs))
    weighted = alpha * kernel
    scores = np.empty((num_samples, num_samples), dtype=np.float64)
    for u in range(num_samples):
        atom_u = 2 * u
        for v in range(num_samples):
            atom_v = 2 * v + 1
            second = np.sum(
                weighted
                * (
                    4.0 * float(gamma) ** 2 * q[:, atom_u] * q[:, atom_v]
                    - 2.0 * float(gamma) * metric[atom_u, atom_v]
                )
            )
            scores[u, v] = second
    return scores


def mahalanobis_sqdist(lhs: np.ndarray, rhs: np.ndarray, metric: np.ndarray) -> np.ndarray:
    lhs_metric = lhs @ metric
    lhs_sq = np.sum(lhs_metric * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum((rhs @ metric) * rhs, axis=1, keepdims=True).T
    distances = lhs_sq + rhs_sq - 2.0 * lhs_metric @ rhs.T
    return np.maximum(distances, 0.0)


def median_heuristic_gamma(distances: np.ndarray) -> float:
    values = distances[np.triu_indices(distances.shape[0], k=1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    return 1.0 / float(np.median(values))


def evaluate_rankings(
    scores: np.ndarray,
    planted_pairs: Sequence[tuple[int, int]],
    *,
    descending: bool,
) -> dict[str, Any]:
    num_samples = scores.shape[0]
    planted = {(int(i), int(j)) for i, j in planted_pairs}
    rows, cols = np.where(~np.eye(num_samples, dtype=bool))
    values = scores[rows, cols]
    order = np.argsort(-values if descending else values, kind="mergesort")
    ranked_pairs = [(int(rows[index]), int(cols[index])) for index in order]
    top_p = ranked_pairs[: len(planted)]
    top_10 = ranked_pairs[: min(10, len(ranked_pairs))]
    hits = sum(1 for pair in top_p if pair in planted)
    planted_scores = np.asarray([scores[i, j] for i, j in planted_pairs], dtype=np.float64)
    planted_tie_groups = {}
    for i, j in planted_pairs:
        score = float(scores[i, j])
        if descending:
            tie_mask = np.isclose(values, score, rtol=1e-10, atol=1e-12)
            better_mask = values > score
        else:
            tie_mask = np.isclose(values, score, rtol=1e-10, atol=1e-12)
            better_mask = values < score
        tie_pairs = {(int(rows[index]), int(cols[index])) for index in np.where(tie_mask)[0].tolist()}
        planted_tie_groups[f"{i},{j}"] = {
            "score": score,
            "num_strictly_better_pairs": int(np.count_nonzero(better_mask)),
            "tie_group_size": int(np.count_nonzero(tie_mask)),
            "planted_pairs_in_tie_group": int(sum(1 for pair in planted if pair in tie_pairs)),
        }
    planted_set = {tuple(pair) for pair in planted_pairs}
    return {
        "recall_at_p": float(hits / max(len(planted), 1)),
        "hits_at_p": int(hits),
        "p": int(len(planted)),
        "top_p_pairs": [[i, j] for i, j in top_p],
        "top_10_pairs": [
            {"pair": [i, j], "score": float(scores[i, j]), "is_planted": (i, j) in planted_set}
            for i, j in top_10
        ],
        "planted_pair_ranks": {
            f"{i},{j}": int(ranked_pairs.index((int(i), int(j))) + 1) for i, j in planted_pairs
        },
        "planted_pair_scores": {f"{i},{j}": float(scores[i, j]) for i, j in planted_pairs},
        "planted_score_min": float(np.min(planted_scores)),
        "planted_score_max": float(np.max(planted_scores)),
        "planted_tie_groups": planted_tie_groups,
    }


def summarize_target_block(scores: np.ndarray, dataset: SyntheticDataset) -> dict[str, Any]:
    target_rows = np.asarray(dataset.target_mod1_indices, dtype=np.int64)
    target_cols = np.asarray(dataset.target_mod2_indices, dtype=np.int64)
    block = scores[np.ix_(target_rows, target_cols)]
    planted_scores = np.asarray([scores[i, j] for i, j in dataset.planted_pairs], dtype=np.float64)
    off_pair_scores = []
    planted = set(dataset.planted_pairs)
    for i in target_rows.tolist():
        for j in target_cols.tolist():
            if (int(i), int(j)) not in planted:
                off_pair_scores.append(float(scores[int(i), int(j)]))
    off_pair = np.asarray(off_pair_scores, dtype=np.float64)
    diag = np.diag(block)
    offdiag = block[~np.eye(block.shape[0], dtype=bool)]
    target_family = {(int(i), int(j)) for i in target_rows.tolist() for j in target_cols.tolist() if int(i) != int(j)}
    rows, cols = np.where(~np.eye(scores.shape[0], dtype=bool))
    values = scores[rows, cols]
    order = np.argsort(-values, kind="mergesort")
    top_family = [(int(rows[index]), int(cols[index])) for index in order[: len(target_family)]]
    target_family_hits = sum(1 for pair in top_family if pair in target_family)
    return {
        "target_block_shape": [int(v) for v in block.shape],
        "target_block_min": float(np.min(block)),
        "target_block_max": float(np.max(block)),
        "target_block_std": float(np.std(block)),
        "planted_mean": float(np.mean(planted_scores)),
        "off_pair_mean": float(np.mean(off_pair)) if off_pair.size else None,
        "planted_minus_off_pair_mean": float(np.mean(planted_scores) - np.mean(off_pair)) if off_pair.size else None,
        "target_diag_mean": float(np.mean(diag)),
        "target_offdiag_mean": float(np.mean(offdiag)) if offdiag.size else None,
        "target_diag_minus_offdiag_mean": float(np.mean(diag) - np.mean(offdiag)) if offdiag.size else None,
        "target_diag_min": float(np.min(diag)),
        "target_offdiag_max": float(np.max(offdiag)) if offdiag.size else None,
        "target_diag_min_minus_offdiag_max": float(np.min(diag) - np.max(offdiag)) if offdiag.size else None,
        "target_family_size": int(len(target_family)),
        "target_family_hits_at_family_size": int(target_family_hits),
        "target_family_recall_at_family_size": float(target_family_hits / max(len(target_family), 1)),
    }


def detect_symmetry_degeneracy(scores: np.ndarray, dataset: SyntheticDataset) -> dict[str, Any]:
    target_rows = np.asarray(dataset.target_mod1_indices, dtype=np.int64)
    target_cols = np.asarray(dataset.target_mod2_indices, dtype=np.int64)
    block = scores[np.ix_(target_rows, target_cols)]
    max_abs_diff = float(np.max(np.abs(block - np.mean(block)))) if block.size else 0.0
    return {
        "target_block_nearly_tied": bool(max_abs_diff < 1e-8),
        "max_abs_deviation_from_target_block_mean": max_abs_diff,
        "note": (
            "The orthogonal design should break the scalar-design tie. If this flag is "
            "true, the target block is still numerically degenerate and the strict "
            "pair ranking is not identifiable."
        ),
    }


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2)))


def spearmanr(first: np.ndarray, second: np.ndarray) -> float:
    x = rankdata(np.asarray(first, dtype=np.float64))
    y = rankdata(np.asarray(second, dtype=np.float64))
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
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
    raw = output["ranking_by_raw_derivative_desc"]
    raw_asc = output["ranking_by_raw_derivative_asc"]
    reduction = output["ranking_by_loss_reduction_desc"]
    surrogate = output["surrogate"]
    block = output["target_block_summary"]
    degeneracy = output["symmetry_degeneracy"]
    lines = [
        "# Synthetic CMIS Experiment",
        "",
        "## Configuration",
        "",
        f"- samples: {output['config']['num_samples']}",
        f"- planted pairs: {output['dataset']['planted_pairs']}",
        f"- masks: {output['config']['num_masks']}",
        f"- device: {output['config']['device']}",
        f"- epochs: {output['config']['epochs']}",
        "",
        "## Surrogate Fit",
        "",
        f"- train RMSE: {surrogate['train_rmse']:.6g}",
        f"- train Spearman: {surrogate['train_spearman']:.6g}",
        f"- gamma: {surrogate['gamma']:.6g}",
        f"- subset loss mean/std: {surrogate['subset_loss_mean']:.6g} / {surrogate['subset_loss_std']:.6g}",
        "",
        "## Ranking Results",
        "",
        f"- raw derivative Recall@P: {raw['recall_at_p']:.6g} ({raw['hits_at_p']}/{raw['p']})",
        f"- raw derivative ascending Recall@P: {raw_asc['recall_at_p']:.6g} ({raw_asc['hits_at_p']}/{raw_asc['p']})",
        f"- raw derivative ascending top P pairs: {raw_asc['top_p_pairs']}",
        f"- loss-reduction Recall@P: {reduction['recall_at_p']:.6g} ({reduction['hits_at_p']}/{reduction['p']})",
        f"- loss-reduction top P pairs: {reduction['top_p_pairs']}",
        f"- loss-reduction planted ranks: {reduction['planted_pair_ranks']}",
        f"- target-family Recall@P^2: {block['target_family_recall_at_family_size']:.6g} "
        f"({block['target_family_hits_at_family_size']}/{block['target_family_size']})",
        "",
        "## Target Block",
        "",
        f"- target block min/max/std: {block['target_block_min']:.6g} / {block['target_block_max']:.6g} / {block['target_block_std']:.6g}",
        f"- planted mean: {block['planted_mean']:.6g}",
        f"- off-pair mean: {block['off_pair_mean']:.6g}" if block["off_pair_mean"] is not None else "- off-pair mean: n/a",
        f"- planted minus off-pair mean: {block['planted_minus_off_pair_mean']:.6g}"
        if block["planted_minus_off_pair_mean"] is not None
        else "- planted minus off-pair mean: n/a",
        f"- diagonal mean: {block['target_diag_mean']:.6g}",
        f"- off-diagonal mean: {block['target_offdiag_mean']:.6g}"
        if block["target_offdiag_mean"] is not None
        else "- off-diagonal mean: n/a",
        f"- diagonal minus off-diagonal mean: {block['target_diag_minus_offdiag_mean']:.6g}"
        if block["target_diag_minus_offdiag_mean"] is not None
        else "- diagonal minus off-diagonal mean: n/a",
        f"- min diagonal minus max off-diagonal: {block['target_diag_min_minus_offdiag_max']:.6g}"
        if block["target_diag_min_minus_offdiag_max"] is not None
        else "- min diagonal minus max off-diagonal: n/a",
        "",
        "## Degeneracy Check",
        "",
        f"- target block nearly tied: {degeneracy['target_block_nearly_tied']}",
        f"- max abs deviation from target block mean: {degeneracy['max_abs_deviation_from_target_block_mean']:.6g}",
        f"- note: {degeneracy['note']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
