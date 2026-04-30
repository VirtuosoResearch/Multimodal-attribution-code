"""Fisher Interaction Curvature (FIC) for cross-sample modal residuals.

FIC measures pairwise alignment of non-additive multimodal residual gradients
after conditioning on all additive unimodal Fisher information:

    FIC_raw[j,t] = c_j^T A_add^{-1} c_t

where

    A_add = damping * I + sum_j w_j (g_j0 g_j0^T + g_j1 g_j1^T)
    c_j   = h_j - g_j0 - g_j1.

The default parameter scope is `fusion`, i.e. the shared/fusion part of the
cross-attention model rather than only the classifier head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset
from fisher_geometry.fisher_metrics import compute_loss
from fisher_geometry.verify_cross_sample_modal_interaction import (
    DEFAULT_DATASETS,
    DEFAULT_OUTPUT_DIR as DEFAULT_CSMI_OUTPUT_DIR,
    DEFAULT_RESULT_DIR,
    build_additive_rows,
    build_residual_rows,
    norm_matched_random_rows,
)
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    compute_group_gradient_tables,
    flatten_grads,
    load_model,
    make_model_args,
    parameters_by_name,
    resolve_full_checkpoint,
    resolve_requested_device,
    select_named_parameters,
)


DEFAULT_OUTPUT_DIR = DEFAULT_CSMI_OUTPUT_DIR.parent / "fisher_interaction_curvature"


def inverse_gram_woodbury(
    query_rows: torch.Tensor,
    metric_rows: torch.Tensor,
    metric_weights: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    """Return query_rows @ A^{-1} @ query_rows.T for low-rank Fisher A.

    A = damping * I + U.T @ U, where U = sqrt(weights)[:, None] * metric_rows.
    This computes the exact Woodbury inverse quadratic form in the selected
    parameter subspace, without materializing A.
    """
    query_rows = query_rows.float()
    metric_rows = metric_rows.float()
    metric_weights = metric_weights.float()
    lam = float(damping)
    if lam <= 0.0:
        raise ValueError("--damping must be positive")
    if metric_rows.numel() == 0:
        return (query_rows @ query_rows.T) / lam

    weighted_rows = metric_rows * torch.sqrt(torch.clamp(metric_weights, min=0.0)).reshape(-1, 1)
    uu = (weighted_rows @ weighted_rows.T).double()
    qut = (query_rows @ weighted_rows.T).double()
    uu = 0.5 * (uu + uu.T)
    if not bool(torch.isfinite(uu).all()) or not bool(torch.isfinite(qut).all()):
        raise FloatingPointError("non-finite values in Woodbury system")
    uu_cpu = uu.cpu()
    qut_cpu = qut.cpu()
    qq_cpu = (query_rows.double() @ query_rows.double().T).cpu()
    try:
        evals, evecs = torch.linalg.eigh(uu_cpu)
    except torch.linalg.LinAlgError:
        evals, evecs = torch.linalg.eigh(uu_cpu + 1e-6 * torch.eye(uu_cpu.shape[0], dtype=torch.float64))
    evals = torch.clamp(evals, min=0.0)
    max_eval = float(evals.max().item()) if evals.numel() else 0.0
    keep = evals > max(lam * 1e-10, max_eval * 1e-12, 1e-20)
    if bool(keep.any()):
        kept_evals = evals[keep]
        kept_evecs = evecs[:, keep]
        coeff = (qut_cpu @ kept_evecs) / torch.sqrt(kept_evals).reshape(1, -1)
        rowspace_euclid = coeff @ coeff.T
        rowspace_inverse = (coeff / (lam + kept_evals).reshape(1, -1)) @ coeff.T
    else:
        rowspace_euclid = torch.zeros_like(qq_cpu)
        rowspace_inverse = torch.zeros_like(qq_cpu)
    orthogonal_euclid = 0.5 * ((qq_cpu - rowspace_euclid) + (qq_cpu - rowspace_euclid).T)
    diag = torch.clamp(torch.diag(orthogonal_euclid), min=0.0)
    orthogonal_euclid = orthogonal_euclid.clone()
    orthogonal_euclid[torch.arange(orthogonal_euclid.shape[0]), torch.arange(orthogonal_euclid.shape[0])] = diag
    gram_cpu = rowspace_inverse + orthogonal_euclid / lam
    gram_cpu = 0.5 * (gram_cpu + gram_cpu.T)
    return gram_cpu.to(device=query_rows.device, dtype=torch.float32)


def fic_from_inverse_gram(gram: torch.Tensor, *, eps: float) -> dict[str, Any]:
    diag = torch.clamp(torch.diag(gram), min=0.0)
    denom = torch.sqrt(torch.outer(diag, diag)) + float(eps)
    fic = gram / denom
    fic_plus = (gram * gram) / (torch.outer(diag, diag) + float(eps))
    fic = torch.clamp(fic, min=-1.0, max=1.0)
    fic_plus = torch.clamp(fic_plus, min=0.0, max=1.0)
    k = int(gram.shape[0])
    offdiag = ~torch.eye(k, dtype=torch.bool, device=gram.device)
    fic_off = fic[offdiag]
    fic_plus_off = fic_plus[offdiag]
    raw_off = gram[offdiag]
    fic_plus_matrix = fic_plus.detach().cpu().numpy().astype(np.float64)
    np.fill_diagonal(fic_plus_matrix, 0.0)
    if k > 1:
        eigvals = np.linalg.eigvalsh(fic_plus_matrix)
        positive_sum = float(np.sum(np.clip(eigvals, 0.0, None)))
        top_eig_frac = float(max(eigvals[-1], 0.0) / max(positive_sum, 1e-12))
    else:
        top_eig_frac = float("nan")
    return {
        "raw_matrix": gram.detach().cpu().tolist(),
        "fic_matrix": fic.detach().cpu().tolist(),
        "fic_plus_matrix": fic_plus.detach().cpu().tolist(),
        "summary": {
            "num_groups": k,
            "num_pairs": int(k * (k - 1)),
            "mean_abs_raw_fic": float(torch.mean(torch.abs(raw_off)).detach().cpu()) if raw_off.numel() else float("nan"),
            "mean_abs_fic": float(torch.mean(torch.abs(fic_off)).detach().cpu()) if fic_off.numel() else float("nan"),
            "mean_fic_plus": float(torch.mean(fic_plus_off).detach().cpu()) if fic_plus_off.numel() else float("nan"),
            "median_fic_plus": float(torch.median(fic_plus_off).detach().cpu()) if fic_plus_off.numel() else float("nan"),
            "mean_residual_self_inverse": float(torch.mean(diag).detach().cpu()) if diag.numel() else float("nan"),
            "fic_plus_top_eigen_fraction": top_eig_frac,
        },
    }


def fic_summary_from_rows(
    query_rows: torch.Tensor,
    additive_rows: torch.Tensor,
    additive_weights: torch.Tensor,
    *,
    damping: float,
    eps: float,
) -> dict[str, Any]:
    gram = inverse_gram_woodbury(
        query_rows,
        additive_rows,
        additive_weights,
        damping=float(damping),
    )
    return fic_from_inverse_gram(gram, eps=float(eps))


def structure_contrast(matrix: np.ndarray, buckets: Sequence[int]) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    bucket_array = np.asarray(buckets)
    k = values.shape[0]
    offdiag = ~np.eye(k, dtype=bool)
    same = (bucket_array[:, None] == bucket_array[None, :]) & offdiag
    diff = (bucket_array[:, None] != bucket_array[None, :]) & offdiag
    same_values = values[same]
    diff_values = values[diff]
    same_mean = float(np.mean(same_values)) if same_values.size else float("nan")
    diff_mean = float(np.mean(diff_values)) if diff_values.size else float("nan")
    return {
        "same_bucket_mean": same_mean,
        "different_bucket_mean": diff_mean,
        "same_minus_different": float(same_mean - diff_mean)
        if np.isfinite(same_mean) and np.isfinite(diff_mean)
        else float("nan"),
    }


def permutation_structure_control(
    matrix: np.ndarray,
    buckets: Sequence[int],
    *,
    num_controls: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    real = structure_contrast(matrix, buckets)
    contrasts = []
    for _ in range(int(num_controls)):
        perm = rng.permutation(matrix.shape[0])
        permuted = matrix[np.ix_(perm, perm)]
        contrasts.append(structure_contrast(permuted, buckets)["same_minus_different"])
    contrast_array = np.asarray(contrasts, dtype=np.float64)
    finite = contrast_array[np.isfinite(contrast_array)]
    real_contrast = float(real["same_minus_different"])
    return {
        **real,
        "permuted_same_minus_different_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "permuted_same_minus_different_std": float(np.std(finite)) if finite.size else float("nan"),
        "permutation_specific_contrast": float(real_contrast - np.mean(finite))
        if finite.size and np.isfinite(real_contrast)
        else float("nan"),
        "permutation_p_value_ge_real": float((1.0 + np.sum(finite >= real_contrast)) / (finite.size + 1.0))
        if finite.size and np.isfinite(real_contrast)
        else float("nan"),
    }


def remove_top_pcs_from_gram(gram: torch.Tensor, *, rank: int) -> torch.Tensor:
    gram_cpu = 0.5 * (gram.detach().double().cpu() + gram.detach().double().cpu().T)
    evals, evecs = torch.linalg.eigh(gram_cpu)
    evals = torch.clamp(evals, min=0.0)
    if int(rank) <= 0 or evals.numel() == 0:
        return gram.detach()
    idx = torch.argsort(evals, descending=True)
    keep_remove = idx[: min(int(rank), int(evals.numel()))]
    removed = gram_cpu - (evecs[:, keep_remove] * evals[keep_remove].reshape(1, -1)) @ evecs[:, keep_remove].T
    removed = 0.5 * (removed + removed.T)
    evals2, evecs2 = torch.linalg.eigh(removed)
    clipped = (evecs2 * torch.clamp(evals2, min=0.0).reshape(1, -1)) @ evecs2.T
    return clipped.to(device=gram.device, dtype=torch.float32)


def aggregate_fic_scores(
    residuals: torch.Tensor,
    residual_weights: torch.Tensor,
    additive_rows: torch.Tensor,
    additive_weights: torch.Tensor,
    base_gram: torch.Tensor,
    *,
    damping: float,
) -> list[float]:
    scores: list[float] = []
    k = int(residuals.shape[0])
    for target in range(k):
        mask = torch.ones(k, dtype=torch.bool, device=residuals.device)
        mask[target] = False
        rows = torch.cat([additive_rows, residuals[mask]], dim=0)
        weights = torch.cat([additive_weights, residual_weights[mask]], dim=0)
        res_gram = inverse_gram_woodbury(
            residuals[target : target + 1],
            rows,
            weights,
            damping=float(damping),
        )
        base = float(base_gram[target, target].detach().cpu())
        value = float(res_gram[0, 0].detach().cpu())
        scores.append((base - value) / max(base, 1e-12))
    return scores


def group_bucket_ids(
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    *,
    task: str,
    num_buckets: int,
) -> list[int]:
    labels = data["train_labels"].detach().cpu()
    group_values = []
    for indices in group_indices:
        group_label = labels[torch.as_tensor(indices, dtype=torch.long)]
        if task == "multilabel":
            value = float(group_label.float().mean().item())
        elif group_label.ndim == 0 or group_label.numel() == len(indices):
            values, counts = torch.unique(group_label.reshape(-1).long(), return_counts=True)
            value = float(values[torch.argmax(counts)].item())
        else:
            value = float(group_label.float().mean().item())
        group_values.append(value)
    unique_values = sorted(set(group_values))
    if 1 < len(unique_values) <= max(10, int(num_buckets)):
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        return [int(mapping[value]) for value in group_values]
    values_np = np.asarray(group_values, dtype=np.float64)
    if np.allclose(values_np, values_np[0]):
        return [0 for _ in group_values]
    quantiles = np.quantile(values_np, np.linspace(0.0, 1.0, int(num_buckets) + 1)[1:-1])
    return [int(np.searchsorted(quantiles, value, side="right")) for value in values_np]


def matched_group_permutation(
    buckets: Sequence[int],
    *,
    rng: np.random.Generator,
) -> list[int]:
    bucket_array = np.asarray(buckets)
    all_indices = np.arange(len(bucket_array))
    result = []
    for group_id, bucket in enumerate(bucket_array):
        candidates = all_indices[(bucket_array == bucket) & (all_indices != group_id)]
        if candidates.size == 0:
            candidates = all_indices[all_indices != group_id]
        if candidates.size == 0:
            candidates = np.asarray([group_id])
        result.append(int(rng.choice(candidates)))
    return result


def sampled_like_length(
    values: Sequence[int],
    *,
    size: int,
    rng: np.random.Generator,
) -> list[int]:
    source = np.asarray([int(value) for value in values], dtype=np.int64)
    if source.size == int(size):
        return source.tolist()
    return rng.choice(source, size=int(size), replace=source.size < int(size)).astype(np.int64).tolist()


def compute_shuffled_pair_gradient(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    x_indices: Sequence[int],
    y_indices: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    use_x: bool,
    use_y: bool,
) -> torch.Tensor:
    params = parameters_by_name(model, param_names)
    x_index = torch.as_tensor(x_indices, dtype=torch.long)
    y_index = torch.as_tensor(y_indices, dtype=torch.long)
    bx = data["train_x"][x_index].to(device)
    by = data["train_y"][y_index].to(device)
    labels = data["train_labels"][x_index].to(device)
    model.zero_grad(set_to_none=True)
    loss = compute_loss(task, model(bx, by, use_x=use_x, use_y=use_y), labels)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_grads(grads, params).numpy().astype(np.float32)


def compute_label_matched_shuffle_residuals(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    param_names: Sequence[str],
    single0: np.ndarray,
    buckets: Sequence[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    permutation = matched_group_permutation(buckets, rng=rng)
    residuals = []
    for group_id, source_group in enumerate(permutation):
        target_indices = [int(value) for value in group_indices[group_id]]
        source_indices = sampled_like_length(
            group_indices[source_group],
            size=len(target_indices),
            rng=rng,
        )
        full_shuffle = compute_shuffled_pair_gradient(
            model,
            data=data,
            x_indices=target_indices,
            y_indices=source_indices,
            task=task,
            device=device,
            param_names=param_names,
            use_x=True,
            use_y=True,
        )
        y_shuffle = compute_shuffled_pair_gradient(
            model,
            data=data,
            x_indices=target_indices,
            y_indices=source_indices,
            task=task,
            device=device,
            param_names=param_names,
            use_x=False,
            use_y=True,
        )
        residuals.append(full_shuffle - single0[group_id] - y_shuffle)
    return np.stack(residuals, axis=0).astype(np.float32), permutation


def run_dataset(
    dataset: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    payload_path = args.result_dir / f"{dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    data = load_dataset(
        dataset,
        int(args.seed),
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(args.feature_device or config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    model_args = make_model_args(config, device=str(device))
    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    full_checkpoint = resolve_full_checkpoint(
        None,
        roots=args.full_checkpoint_roots,
        dataset=dataset,
        seed=int(args.seed),
    )
    print(f"[{dataset}] loading checkpoint: {full_checkpoint}", flush=True)
    model = load_model(
        full_checkpoint,
        model_type=str(config.get("model", "cross_attention")),
        dim_x=int(data["train_x"].shape[1]),
        dim_y=int(data["train_y"].shape[1]),
        num_classes=int(data["num_classes"]),
        model_args=model_args,
        device=str(device),
    )
    named_params = select_named_parameters(model, mode=str(args.gradient_params))
    param_names = [name for name, _ in named_params]
    print(
        f"[{dataset}] parameter scope={args.gradient_params}, parameters={sum(p.numel() for _, p in named_params):,}",
        flush=True,
    )
    gradients = compute_group_gradient_tables(
        model,
        data=data,
        group_indices=group_indices,
        task=str(data["task"]),
        device=str(device),
        param_names=param_names,
    )
    additive_np, additive_weights_np = build_additive_rows(gradients, group_indices)
    residual_np, residual_weights_np = build_residual_rows(gradients, group_indices)
    single0_np = np.asarray(gradients["single"][0], dtype=np.float32)
    single1_np = np.asarray(gradients["single"][1], dtype=np.float32)
    additive_sum_np = single0_np + single1_np
    buckets = group_bucket_ids(
        data,
        group_indices,
        task=str(data["task"]),
        num_buckets=int(args.structure_buckets),
    )
    print(f"[{dataset}] computing label/difficulty-matched modality shuffle residuals", flush=True)
    label_shuffle_np, label_shuffle_sources = compute_label_matched_shuffle_residuals(
        model,
        data=data,
        group_indices=group_indices,
        task=str(data["task"]),
        device=str(device),
        param_names=param_names,
        single0=single0_np,
        buckets=buckets,
        rng=rng,
    )
    finite_mask = (
        np.isfinite(additive_np).all(axis=0)
        & np.isfinite(residual_np).all(axis=0)
        & np.isfinite(additive_sum_np).all(axis=0)
        & np.isfinite(label_shuffle_np).all(axis=0)
    )
    dropped_nonfinite = int(finite_mask.size - int(finite_mask.sum()))
    if dropped_nonfinite:
        print(
            f"[{dataset}] dropping {dropped_nonfinite:,} non-finite parameter coordinates "
            f"from FIC subspace",
            flush=True,
        )
        additive_np = additive_np[:, finite_mask]
        residual_np = residual_np[:, finite_mask]
        additive_sum_np = additive_sum_np[:, finite_mask]
        label_shuffle_np = label_shuffle_np[:, finite_mask]
    additive_rows = torch.as_tensor(additive_np, dtype=torch.float32, device=device)
    additive_weights = torch.as_tensor(additive_weights_np, dtype=torch.float32, device=device)
    residuals = torch.as_tensor(residual_np, dtype=torch.float32, device=device)
    residual_weights = torch.as_tensor(residual_weights_np, dtype=torch.float32, device=device)
    additive_sum_rows = torch.as_tensor(additive_sum_np, dtype=torch.float32, device=device)
    label_shuffle_rows = torch.as_tensor(label_shuffle_np, dtype=torch.float32, device=device)

    base_gram = inverse_gram_woodbury(
        residuals,
        additive_rows,
        additive_weights,
        damping=float(args.damping),
    )
    real = fic_from_inverse_gram(base_gram, eps=float(args.eps))
    real_fic_plus_matrix = np.asarray(real["fic_plus_matrix"], dtype=np.float64)
    permutation_control = permutation_structure_control(
        real_fic_plus_matrix,
        buckets,
        num_controls=int(args.permutation_controls),
        rng=rng,
    )
    additive_control = fic_summary_from_rows(
        additive_sum_rows,
        additive_rows,
        additive_weights,
        damping=float(args.damping),
        eps=float(args.eps),
    )
    label_shuffle_control = fic_summary_from_rows(
        label_shuffle_rows,
        additive_rows,
        additive_weights,
        damping=float(args.damping),
        eps=float(args.eps),
    )
    top_pc_controls = {}
    for rank in args.remove_top_pcs:
        removed_gram = remove_top_pcs_from_gram(base_gram, rank=int(rank))
        top_pc_controls[f"remove_top_{int(rank)}"] = fic_from_inverse_gram(
            removed_gram,
            eps=float(args.eps),
        )["summary"]
    real_agg = aggregate_fic_scores(
        residuals,
        residual_weights,
        additive_rows,
        additive_weights,
        base_gram,
        damping=float(args.damping),
    )
    real_agg_array = np.asarray(real_agg, dtype=np.float64)
    agg_valid = bool(
        np.all(np.isfinite(real_agg_array))
        and np.min(real_agg_array) >= -1e-6
        and np.max(real_agg_array) <= 1.0 + 1e-6
    )
    if not agg_valid:
        print(
            f"[{dataset}] residual-query AggFIC is numerically invalid in this subspace; "
            "reporting AggFIC as nan",
            flush=True,
        )

    random_mean_fic_plus: list[float] = []
    random_mean_abs_fic: list[float] = []
    random_agg_mean: list[float] = []
    for _ in range(int(args.random_controls)):
        random_residuals = norm_matched_random_rows(residuals, rng=rng)
        random_gram = inverse_gram_woodbury(
            random_residuals,
            additive_rows,
            additive_weights,
            damping=float(args.damping),
        )
        random = fic_from_inverse_gram(random_gram, eps=float(args.eps))
        random_mean_fic_plus.append(float(random["summary"]["mean_fic_plus"]))
        random_mean_abs_fic.append(float(random["summary"]["mean_abs_fic"]))
        if agg_valid:
            random_agg = aggregate_fic_scores(
                random_residuals,
                residual_weights,
                additive_rows,
                additive_weights,
                random_gram,
                damping=float(args.damping),
            )
            random_agg_mean.append(float(np.mean(random_agg)))

    real_summary = dict(real["summary"])
    real_agg_mean = float(np.mean(real_agg)) if agg_valid else float("nan")
    real_agg_median = float(np.median(real_agg)) if agg_valid else float("nan")
    real_agg_positive = float(np.mean(np.asarray(real_agg) > 0.0)) if agg_valid else float("nan")
    random_agg_mean_value = float(np.mean(random_agg_mean)) if random_agg_mean else float("nan")
    real_summary.update(
        {
            "agg_fic_valid": agg_valid,
            "mean_agg_fic": real_agg_mean,
            "median_agg_fic": real_agg_median,
            "agg_fic_positive_fraction": real_agg_positive,
            "random_mean_fic_plus": float(np.mean(random_mean_fic_plus)),
            "random_std_fic_plus": float(np.std(random_mean_fic_plus)),
            "fic_plus_specific_gain": float(real["summary"]["mean_fic_plus"] - np.mean(random_mean_fic_plus)),
            "random_p_value_fic_plus_ge_real": float(
                (1.0 + np.sum(np.asarray(random_mean_fic_plus) >= float(real["summary"]["mean_fic_plus"])))
                / (len(random_mean_fic_plus) + 1.0)
            ),
            "random_mean_abs_fic": float(np.mean(random_mean_abs_fic)),
            "abs_fic_specific_gain": float(real["summary"]["mean_abs_fic"] - np.mean(random_mean_abs_fic)),
            "random_mean_agg_fic": random_agg_mean_value,
            "random_std_agg_fic": float(np.std(random_agg_mean)) if random_agg_mean else float("nan"),
            "agg_fic_specific_gain": float(real_agg_mean - random_agg_mean_value)
            if np.isfinite(real_agg_mean) and np.isfinite(random_agg_mean_value)
            else float("nan"),
            "random_p_value_agg_ge_real": float(
                (1.0 + np.sum(np.asarray(random_agg_mean) >= float(real_agg_mean)))
                / (len(random_agg_mean) + 1.0)
            )
            if random_agg_mean and np.isfinite(real_agg_mean)
            else float("nan"),
            "bucket_same_fic_plus": float(permutation_control["same_bucket_mean"]),
            "bucket_diff_fic_plus": float(permutation_control["different_bucket_mean"]),
            "bucket_contrast": float(permutation_control["same_minus_different"]),
            "permuted_bucket_contrast": float(permutation_control["permuted_same_minus_different_mean"]),
            "bucket_contrast_specific_gain": float(permutation_control["permutation_specific_contrast"]),
            "bucket_contrast_p_value_ge_real": float(permutation_control["permutation_p_value_ge_real"]),
            "label_shuffle_mean_fic_plus": float(label_shuffle_control["summary"]["mean_fic_plus"]),
            "label_shuffle_specific_gain": float(
                real["summary"]["mean_fic_plus"] - label_shuffle_control["summary"]["mean_fic_plus"]
            ),
            "additive_mean_fic_plus": float(additive_control["summary"]["mean_fic_plus"]),
            "residual_minus_additive_fic_plus": float(
                real["summary"]["mean_fic_plus"] - additive_control["summary"]["mean_fic_plus"]
            ),
        }
    )
    for name, summary in top_pc_controls.items():
        real_summary[f"{name}_mean_fic_plus"] = float(summary["mean_fic_plus"])
        real_summary[f"{name}_retained_fraction"] = float(
            summary["mean_fic_plus"] / max(real["summary"]["mean_fic_plus"], 1e-12)
        )
    return {
        "dataset": dataset,
        "status": "ok",
        "num_groups": int(len(group_indices)),
        "num_parameters": int(sum(param.numel() for _, param in named_params)),
        "num_fic_parameters": int(additive_rows.shape[1]),
        "dropped_nonfinite_parameters": dropped_nonfinite,
        "parameter_scope": str(args.gradient_params),
        "summary": real_summary,
        "agg_fic_by_target": [float(value) for value in real_agg],
        "bucket_ids": [int(value) for value in buckets],
        "controls": {
            "residual_permutation_structure": permutation_control,
            "label_matched_shuffle_sources": [int(value) for value in label_shuffle_sources],
            "label_matched_shuffle": label_shuffle_control["summary"],
            "additive_gradient": additive_control["summary"],
            "top_pc_removal": top_pc_controls,
        },
        "pairwise": real,
    }


def summarize_all(datasets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mean_abs_fic",
        "mean_fic_plus",
        "random_mean_fic_plus",
        "fic_plus_specific_gain",
        "mean_agg_fic",
        "random_mean_agg_fic",
        "agg_fic_specific_gain",
        "fic_plus_top_eigen_fraction",
        "bucket_contrast",
        "permuted_bucket_contrast",
        "bucket_contrast_specific_gain",
        "label_shuffle_mean_fic_plus",
        "label_shuffle_specific_gain",
        "additive_mean_fic_plus",
        "residual_minus_additive_fic_plus",
        "remove_top_1_mean_fic_plus",
        "remove_top_1_retained_fraction",
        "remove_top_3_mean_fic_plus",
        "remove_top_3_retained_fraction",
    ]
    summary: dict[str, Any] = {"num_datasets": len(datasets)}
    for key in keys:
        values = [float(ds["summary"][key]) for ds in datasets if np.isfinite(float(ds["summary"][key]))]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    summary["datasets_fic_plus_beats_random"] = [
        ds["dataset"] for ds in datasets if float(ds["summary"]["fic_plus_specific_gain"]) > 0.0
    ]
    summary["datasets_agg_beats_random"] = [
        ds["dataset"] for ds in datasets if float(ds["summary"]["agg_fic_specific_gain"]) > 0.0
    ]
    summary["datasets_residual_beats_label_shuffle"] = [
        ds["dataset"] for ds in datasets if float(ds["summary"]["label_shuffle_specific_gain"]) > 0.0
    ]
    summary["datasets_residual_beats_additive"] = [
        ds["dataset"] for ds in datasets if float(ds["summary"]["residual_minus_additive_fic_plus"]) > 0.0
    ]
    return summary


def fmt(value: Any, precision: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.{precision}g}"


def to_markdown(output: dict[str, Any]) -> str:
    summary = output["overall"]
    lines = [
        "# Fisher Interaction Curvature",
        "",
        "FIC measures pairwise alignment of non-additive multimodal residual gradients after conditioning on all additive unimodal Fisher information.",
        "",
        "## Formulation",
        "",
        "\\[",
        "A_{\\mathrm{add}}=\\lambda I+\\sum_{j=1}^K w_j\\sum_{m=1}^M g_{j,m}g_{j,m}^\\top,",
        "\\qquad",
        "c_j=h_j-\\sum_{m=1}^M g_{j,m}.",
        "\\]",
        "",
        "\\[",
        "\\mathrm{FIC}^{\\mathrm{raw}}_{jt}=c_j^\\top A_{\\mathrm{add}}^{-1}c_t,",
        "\\]",
        "",
        "\\[",
        "\\mathrm{FIC}_{jt}=\\frac{c_j^\\top A_{\\mathrm{add}}^{-1}c_t}{\\sqrt{c_j^\\top A_{\\mathrm{add}}^{-1}c_j}\\sqrt{c_t^\\top A_{\\mathrm{add}}^{-1}c_t}+\\epsilon},",
        "\\qquad",
        "\\mathrm{FIC}^+_{jt}=\\mathrm{FIC}_{jt}^2.",
        "\\]",
        "",
        "The implementation computes \\(C A_{\\mathrm{add}}^{-1}C^\\top\\) exactly by Woodbury in the selected parameter subspace; no dense Hessian/Fisher inverse is materialized.",
        "",
        "## Configuration",
        "",
        f"- datasets: {', '.join(output['config']['datasets'])}",
        f"- parameter scope: `{output['config']['gradient_params']}`",
        f"- damping: {fmt(output['config']['damping'])}",
        f"- random controls: {output['config']['random_controls']}",
        f"- permutation controls: {output['config']['permutation_controls']}",
        f"- structure buckets: {output['config']['structure_buckets']}",
        f"- remove top PCs: {', '.join(str(v) for v in output['config']['remove_top_pcs'])}",
        "",
        "## Aggregate",
        "",
        f"- mean MeanFIC+: {fmt(summary['mean_mean_fic_plus'])}",
        f"- mean random MeanFIC+: {fmt(summary['mean_random_mean_fic_plus'])}",
        f"- mean FIC+ specific gain: {fmt(summary['mean_fic_plus_specific_gain'])}",
        f"- mean AggFIC: {fmt(summary['mean_mean_agg_fic'])}",
        f"- mean random AggFIC: {fmt(summary['mean_random_mean_agg_fic'])}",
        f"- mean label-matched shuffle MeanFIC+: {fmt(summary['mean_label_shuffle_mean_fic_plus'])}",
        f"- mean additive-gradient MeanFIC+: {fmt(summary['mean_additive_mean_fic_plus'])}",
        f"- mean remove-top-1 MeanFIC+: {fmt(summary['mean_remove_top_1_mean_fic_plus'])}",
        f"- mean remove-top-3 MeanFIC+: {fmt(summary['mean_remove_top_3_mean_fic_plus'])}",
        f"- datasets FIC+ > random: {', '.join(summary['datasets_fic_plus_beats_random']) or 'none'}",
        f"- datasets AggFIC > random: {', '.join(summary['datasets_agg_beats_random']) or 'none'}",
        f"- datasets residual FIC+ > label-shuffle FIC+: {', '.join(summary['datasets_residual_beats_label_shuffle']) or 'none'}",
        f"- datasets residual FIC+ > additive-gradient FIC+: {', '.join(summary['datasets_residual_beats_additive']) or 'none'}",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Params | FIC Params | Dropped | MeanAbsFIC | MeanFIC+ | Random FIC+ | FIC+ Gain | p(rand>=real) | LabelShuffle FIC+ | AddFIC+ | Top1Removed FIC+ | Top3Removed FIC+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in output["datasets"]:
        s = ds["summary"]
        lines.append(
            f"| {ds['dataset']} | {ds['num_parameters']} | {ds.get('num_fic_parameters', ds['num_parameters'])} | "
            f"{ds.get('dropped_nonfinite_parameters', 0)} | {fmt(s['mean_abs_fic'])} | "
            f"{fmt(s['mean_fic_plus'])} | {fmt(s['random_mean_fic_plus'])} | "
            f"{fmt(s['fic_plus_specific_gain'])} | {fmt(s['random_p_value_fic_plus_ge_real'], 3)} | "
            f"{fmt(s['label_shuffle_mean_fic_plus'])} | {fmt(s['additive_mean_fic_plus'])} | "
            f"{fmt(s.get('remove_top_1_mean_fic_plus', float('nan')))} | "
            f"{fmt(s.get('remove_top_3_mean_fic_plus', float('nan')))} |"
        )
    lines.extend(
        [
            "",
            "## Structure Controls",
            "",
            "| Dataset | SameBucket FIC+ | DiffBucket FIC+ | Contrast | Perm Contrast | Contrast Gain | p(perm>=real) | Residual-Additive | Residual-LabelShuffle | Top1 Retained | Top3 Retained |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ds in output["datasets"]:
        s = ds["summary"]
        lines.append(
            f"| {ds['dataset']} | {fmt(s['bucket_same_fic_plus'])} | {fmt(s['bucket_diff_fic_plus'])} | "
            f"{fmt(s['bucket_contrast'])} | {fmt(s['permuted_bucket_contrast'])} | "
            f"{fmt(s['bucket_contrast_specific_gain'])} | {fmt(s['bucket_contrast_p_value_ge_real'], 3)} | "
            f"{fmt(s['residual_minus_additive_fic_plus'])} | {fmt(s['label_shuffle_specific_gain'])} | "
            f"{fmt(s.get('remove_top_1_retained_fraction', float('nan')))} | "
            f"{fmt(s.get('remove_top_3_retained_fraction', float('nan')))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `MeanFIC+` summarizes pairwise cross-sample residual alignment in the inverse Fisher geometry left after the additive unimodal baseline.",
            "- `Random FIC+` uses norm-matched random residual directions, so the comparison controls for residual norm and generic PSD mass.",
            "- `LabelShuffle FIC+` recomputes residuals after same-label or same-difficulty modality-pair shuffling; it tests whether FIC is driven by true multimodal pairing rather than task direction alone.",
            "- `AddFIC+` computes the same inverse-Fisher pairwise alignment for additive gradients `g_j0 + g_j1`; it tests whether the residual signal is stronger than ordinary sample geometry.",
            "- `TopKRemoved FIC+` removes the leading residual PCs in the inverse-Fisher metric; it tests whether FIC is only a global common residual direction.",
            "- `AggFIC` sets the target query to the residual itself, measuring whether other groups' residual span reduces the target residual inverse-curvature.",
            "- A positive FIC+ gain indicates organized pairwise residual geometry beyond random directions; a positive AggFIC gain indicates that other samples' residuals span target residual directions beyond random controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--full-checkpoint-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT, DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--gradient-params", choices=["last_layer", "fusion", "all"], default="fusion")
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--random-controls", type=int, default=64)
    parser.add_argument("--permutation-controls", type=int, default=64)
    parser.add_argument("--structure-buckets", type=int, default=5)
    parser.add_argument("--remove-top-pcs", nargs="*", type=int, default=[1, 3])
    parser.add_argument("--random-seed", type=int, default=2028)
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    rng = np.random.default_rng(int(args.random_seed))
    datasets = [run_dataset(str(dataset), args=args, device=device, rng=rng) for dataset in args.datasets]
    output = {
        "config": {
            "datasets": [str(value) for value in args.datasets],
            "seed": int(args.seed),
            "result_dir": str(args.result_dir),
            "device": str(device),
            "gradient_params": str(args.gradient_params),
            "damping": float(args.damping),
            "eps": float(args.eps),
            "random_controls": int(args.random_controls),
            "permutation_controls": int(args.permutation_controls),
            "structure_buckets": int(args.structure_buckets),
            "remove_top_pcs": [int(value) for value in args.remove_top_pcs],
            "random_seed": int(args.random_seed),
            "score": "FIC_raw[j,t] = c_j^T A_add^{-1} c_t; FIC+ is squared normalized FIC.",
            "inverse": "exact Woodbury inverse for the low-rank Fisher baseline",
        },
        "overall": summarize_all(datasets),
        "datasets": datasets,
    }
    output_json = args.output_json or (args.output_dir / "fisher_interaction_curvature_all_datasets.json")
    output_md = args.output_md or output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
