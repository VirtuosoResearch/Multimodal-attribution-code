"""Fisher information geometry metrics.

Provides:
  - Per-example gradient collection
  - Eigendecomposition via dual Gram trick
  - Subspace overlap, residual energy, logdet gain, top-k spectrum mass
  - Interaction Fisher (F_XY - F_X - F_Y) analysis
  - Spectral shape metrics (effective rank, participation ratio, spectral decay)
  - Per-parameter-group gradient extraction
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_loss(task: str, logits: torch.Tensor,
                 labels: torch.Tensor) -> torch.Tensor:
    if task == "binary":
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float().view(-1))
    elif task == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    else:
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Per-example gradients
# ---------------------------------------------------------------------------

def per_example_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    max_samples: int,
    task: str,
    use_x: bool,
    use_y: bool,
    grad_batch_size: int = 512,
) -> torch.Tensor:
    """Collect per-sample gradient vectors for the empirical Fisher.

    Uses torch.func.vmap for batched per-example gradients on GPU.
    Returns tensor of shape [min(max_samples, N), num_params].
    """
    from torch.func import functional_call, vmap, grad

    count = min(max_samples, x.shape[0])
    model.eval()
    params_dict = {k: v.detach() for k, v in model.named_parameters()}
    param_names = list(params_dict.keys())
    params_tuple = tuple(params_dict[k] for k in param_names)

    def compute_loss_single(params_values, bx, by, bl):
        params_in = dict(zip(param_names, params_values))
        logits = functional_call(model, params_in,
                                 (bx.unsqueeze(0), by.unsqueeze(0)),
                                 kwargs={"use_x": use_x, "use_y": use_y})
        return compute_loss(task, logits, bl.unsqueeze(0))

    grad_fn = grad(compute_loss_single)
    vmapped_grad = vmap(grad_fn, in_dims=(None, 0, 0, 0))

    all_grads = []
    for start in range(0, count, grad_batch_size):
        end = min(start + grad_batch_size, count)
        bx = x[start:end].to(device)
        by = y[start:end].to(device)
        bl = labels[start:end].to(device)
        per_sample_grads = vmapped_grad(params_tuple, bx, by, bl)
        # Flatten and concatenate all parameter gradients per sample
        flat = torch.cat(
            [g.reshape(end - start, -1) for g in per_sample_grads], dim=1)
        all_grads.append(flat.detach().cpu())
    return torch.cat(all_grads, dim=0)


# ---------------------------------------------------------------------------
# Fisher eigendecomposition (dual Gram trick: n×n instead of d×d)
# ---------------------------------------------------------------------------

def fisher_eigen(grad_matrix: torch.Tensor,
                 top_k: int = 10) -> dict[str, Any]:
    """Full eigendecomposition of empirical Fisher via dual Gram matrix.

    Returns eigenvalues, top-k parameter-space eigenvectors, and scalar stats.
    """
    g = grad_matrix.float()
    n, d = g.shape
    scaled = g / math.sqrt(max(n, 1))
    gram = scaled @ scaled.T
    evals_all, evecs_small = torch.linalg.eigh(gram)
    evals_all = torch.clamp(evals_all, min=0.0)

    idx = torch.argsort(evals_all, descending=True)
    evals = evals_all[idx]
    evecs_small = evecs_small[:, idx]

    trace_val = float(evals.sum().item())
    positive_mask = evals > 1e-10
    pos_evals = evals[positive_mask]
    rank = int(pos_evals.numel())

    eps = 1e-6
    logdet = float(
        torch.log(pos_evals + eps).sum().item()
        + (d - rank) * math.log(eps))

    k = min(top_k, rank)
    if k > 0:
        top_small_vecs = evecs_small[:, :k]
        top_evals = evals[:k]
        param_vecs = ((scaled.T @ top_small_vecs)
                      / torch.sqrt(top_evals).unsqueeze(0).clamp(min=1e-12))
        param_vecs = torch.nn.functional.normalize(param_vecs, dim=0)
        # Guard against NaN from degenerate Fisher (e.g. tiny datasets)
        if not torch.isfinite(param_vecs).all():
            param_vecs = torch.zeros((d, k), dtype=torch.float32)
    else:
        param_vecs = torch.zeros((d, 0), dtype=torch.float32)
        top_evals = torch.tensor([])

    topk_mass = (float(top_evals.sum().item() / max(trace_val, 1e-12))
                 if k > 0 else 0.0)
    lambda_max = float(evals[0].item()) if evals.numel() > 0 else 0.0

    return {
        "eigenvalues": evals.cpu(),
        "top_k_vecs": param_vecs.cpu(),
        "trace": trace_val,
        "logdet": logdet,
        "lambda_max": lambda_max,
        "rank": rank,
        "topk_mass": topk_mass,
    }


# ---------------------------------------------------------------------------
# Subspace overlap
# ---------------------------------------------------------------------------

def subspace_overlap(vecs_a: torch.Tensor, vecs_b: torch.Tensor) -> float:
    """Mean cosine of principal angles between two subspaces.

    Args:
        vecs_a: d×k_a orthonormal columns.
        vecs_b: d×k_b orthonormal columns.
    """
    if vecs_a.shape[1] == 0 or vecs_b.shape[1] == 0:
        return 0.0
    M = vecs_a.float().T @ vecs_b.float()
    if not torch.isfinite(M).all():
        return 0.0
    try:
        svals = torch.linalg.svdvals(M)
    except torch._C._LinAlgError:
        return 0.0
    svals = torch.clamp(svals, 0.0, 1.0)
    return float(svals.mean().item())


# ---------------------------------------------------------------------------
# Residual energy
# ---------------------------------------------------------------------------

def residual_energy(grad_y: torch.Tensor,
                    top_vecs_x: torch.Tensor) -> float:
    """Fraction of Y gradient energy outside top-k eigenspace of F_X.

    residual = 1 - ||G_Y V_X||^2 / ||G_Y||^2
    """
    gy = grad_y.float()
    total_energy = float(gy.pow(2).sum().item())
    if total_energy < 1e-20 or top_vecs_x.shape[1] == 0:
        return 1.0
    proj = gy @ top_vecs_x.float()
    proj_energy = float(proj.pow(2).sum().item())
    return float(1.0 - proj_energy / total_energy)


# ---------------------------------------------------------------------------
# Sample-wise gradient cosine
# ---------------------------------------------------------------------------

def mean_sample_cosine(gx: torch.Tensor, gy: torch.Tensor) -> float:
    """Mean per-sample cosine similarity between two gradient matrices."""
    n = min(gx.shape[0], gy.shape[0])
    dots = (gx[:n].float() * gy[:n].float()).sum(dim=1)
    norms = (gx[:n].float().norm(dim=1)
             * gy[:n].float().norm(dim=1)).clamp(min=1e-12)
    return float((dots / norms).mean().item())


# ---------------------------------------------------------------------------
# Spectral shape metrics
# ---------------------------------------------------------------------------

def spectral_metrics(eigenvalues: torch.Tensor) -> dict[str, float]:
    """Compute spectral shape metrics from eigenvalues.

    Returns:
        effective_rank: exp(entropy of normalized eigenvalues)
        participation_ratio: (sum lambda)^2 / sum(lambda^2)
        spectral_decay: power-law exponent alpha from fit lambda_k ~ k^{-alpha}
        spectral_gap: lambda_1 / lambda_2
    """
    evals = eigenvalues.float()
    pos = evals[evals > 1e-10]
    if pos.numel() < 2:
        return {"effective_rank": 0.0, "participation_ratio": 0.0,
                "spectral_decay": 0.0, "spectral_gap": 0.0}

    # Effective rank = exp(Shannon entropy of normalized spectrum)
    p = pos / pos.sum()
    entropy = -float((p * torch.log(p + 1e-30)).sum().item())
    effective_rank = float(math.exp(entropy))

    # Participation ratio = (sum lambda)^2 / sum(lambda^2)
    s1 = float(pos.sum().item())
    s2 = float(pos.pow(2).sum().item())
    participation_ratio = s1 * s1 / max(s2, 1e-30)

    # Spectral decay: fit log(lambda_k) = -alpha * log(k) + c
    k = torch.arange(1, pos.numel() + 1, dtype=torch.float32)
    log_k = torch.log(k)
    log_lam = torch.log(pos.cpu())
    # Least squares: alpha = -cov(log_k, log_lam) / var(log_k)
    mean_lk = log_k.mean()
    mean_ll = log_lam.mean()
    cov = ((log_k - mean_lk) * (log_lam - mean_ll)).mean()
    var_k = ((log_k - mean_lk) ** 2).mean()
    spectral_decay = float((-cov / var_k.clamp(min=1e-30)).item())

    # Spectral gap
    spectral_gap = float(pos[0].item() / max(pos[1].item(), 1e-30))

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "spectral_decay": spectral_decay,
        "spectral_gap": spectral_gap,
    }


# ---------------------------------------------------------------------------
# Interaction Fisher: Delta = F_XY - F_X - F_Y
# ---------------------------------------------------------------------------

def interaction_fisher(
    gx: torch.Tensor, gy: torch.Tensor, gxy: torch.Tensor,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute the interaction Fisher Delta = F_XY - F_X - F_Y.

    Works in the dual (sample) space to avoid forming d×d matrices:
        F = G^T G / n  =>  Delta = G_XY^T G_XY / n - G_X^T G_X / n - G_Y^T G_Y / n

    We eigendecompose Delta by forming its n×n representation in a joint
    sample basis. Since n << d, this is tractable.

    Returns trace, positive/negative eigenvalue stats, and top-k eigenvectors.
    """
    n = min(gx.shape[0], gy.shape[0], gxy.shape[0])
    d = gx.shape[1]
    gx_f = gx[:n].float()
    gy_f = gy[:n].float()
    gxy_f = gxy[:n].float()

    # Form the Gram matrices (each n×n) for the three Fisher matrices
    # F = G^T G / n, so the n×n Gram in sample space is G G^T / n
    inv_n = 1.0 / max(n, 1)
    gram_xy = (gxy_f @ gxy_f.T) * inv_n
    gram_x = (gx_f @ gx_f.T) * inv_n
    gram_y = (gy_f @ gy_f.T) * inv_n
    delta_gram = gram_xy - gram_x - gram_y

    # Eigendecompose the interaction Gram
    evals, evecs = torch.linalg.eigh(delta_gram)
    idx = torch.argsort(evals.abs(), descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    trace_val = float(evals.sum().item())
    pos_evals = evals[evals > 1e-10]
    neg_evals = evals[evals < -1e-10]

    pos_trace = float(pos_evals.sum().item()) if pos_evals.numel() > 0 else 0.0
    neg_trace = float(neg_evals.sum().item()) if neg_evals.numel() > 0 else 0.0
    pos_rank = int(pos_evals.numel())
    neg_rank = int(neg_evals.numel())

    # Top-k eigenvectors (by absolute eigenvalue) in parameter space
    k = min(top_k, n)
    if k > 0:
        # Map back: v_param = G^T u / ||G^T u|| using the appropriate G
        # For interaction eigenvectors, use gxy as the basis
        top_evecs_small = evecs[:, :k]
        param_vecs = gxy_f.T @ top_evecs_small
        norms = param_vecs.norm(dim=0, keepdim=True).clamp(min=1e-12)
        param_vecs = param_vecs / norms
        if not torch.isfinite(param_vecs).all():
            param_vecs = torch.zeros((d, k), dtype=torch.float32)
    else:
        param_vecs = torch.zeros((d, 0), dtype=torch.float32)

    return {
        "trace": trace_val,
        "pos_trace": pos_trace,
        "neg_trace": neg_trace,
        "pos_rank": pos_rank,
        "neg_rank": neg_rank,
        "top_eigenvalues": evals[:min(20, evals.numel())].cpu().tolist(),
        "top_k_vecs": param_vecs.cpu(),
    }


# ---------------------------------------------------------------------------
# Per-parameter-group gradient extraction
# ---------------------------------------------------------------------------

def per_example_gradients_grouped(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    max_samples: int,
    task: str,
    use_x: bool,
    use_y: bool,
    grad_batch_size: int = 512,
    param_groups: dict[str, list[str]] | None = None,
) -> dict[str, torch.Tensor]:
    """Collect per-sample gradients partitioned by parameter group.

    If param_groups is None, returns {"all": full_grad_matrix}.
    Otherwise, param_groups maps group_name -> list of parameter name prefixes.
    Parameters not matching any group are placed in "other".

    Returns dict mapping group_name -> tensor of shape [n_samples, n_params_in_group].
    """
    from torch.func import functional_call, vmap, grad

    count = min(max_samples, x.shape[0])
    model.eval()
    params_dict = {k: v.detach() for k, v in model.named_parameters()}
    param_names = list(params_dict.keys())
    params_tuple = tuple(params_dict[k] for k in param_names)

    # Compute param sizes and group assignments
    param_sizes = [params_dict[k].numel() for k in param_names]

    if param_groups is None:
        param_groups = {"all": [""]}  # match everything

    # Assign each param to a group
    group_assignment: list[str] = []
    for pname in param_names:
        assigned = "other"
        for gname, prefixes in param_groups.items():
            if any(pname.startswith(p) or p in pname for p in prefixes):
                assigned = gname
                break
        group_assignment.append(assigned)

    def compute_loss_single(params_values, bx, by, bl):
        params_in = dict(zip(param_names, params_values))
        logits = functional_call(model, params_in,
                                 (bx.unsqueeze(0), by.unsqueeze(0)),
                                 kwargs={"use_x": use_x, "use_y": use_y})
        return compute_loss(task, logits, bl.unsqueeze(0))

    grad_fn = grad(compute_loss_single)
    vmapped_grad = vmap(grad_fn, in_dims=(None, 0, 0, 0))

    # Collect all groups
    all_group_names = sorted(set(group_assignment))
    group_grads = {g: [] for g in all_group_names}

    for start in range(0, count, grad_batch_size):
        end = min(start + grad_batch_size, count)
        bx = x[start:end].to(device)
        by = y[start:end].to(device)
        bl = labels[start:end].to(device)
        per_sample_grads = vmapped_grad(params_tuple, bx, by, bl)

        # Split into groups
        batch_groups = {g: [] for g in all_group_names}
        for i, (g_tensor, gname) in enumerate(
                zip(per_sample_grads, group_assignment)):
            batch_groups[gname].append(g_tensor.reshape(end - start, -1))

        for gname in all_group_names:
            if batch_groups[gname]:
                group_grads[gname].append(
                    torch.cat(batch_groups[gname], dim=1).detach().cpu())

    result = {}
    for gname in all_group_names:
        if group_grads[gname]:
            result[gname] = torch.cat(group_grads[gname], dim=0)
    return result
