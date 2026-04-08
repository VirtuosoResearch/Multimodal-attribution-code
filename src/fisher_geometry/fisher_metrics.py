"""Fisher information geometry metrics.

Provides:
  - Per-example gradient collection
  - Eigendecomposition via dual Gram trick
  - Subspace overlap, residual energy, logdet gain, top-k spectrum mass
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
) -> torch.Tensor:
    """Collect per-sample gradient vectors for the empirical Fisher.

    Returns tensor of shape [min(max_samples, N), num_params].
    """
    params = list(model.parameters())
    count = min(max_samples, x.shape[0])
    grads = []
    model.eval()
    for i in range(count):
        bx = x[i:i + 1].to(device)
        by = y[i:i + 1].to(device)
        bl = labels[i:i + 1].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(bx, by, use_x=use_x, use_y=use_y)
        loss = compute_loss(task, logits, bl)
        grad_tuple = torch.autograd.grad(
            loss, params, allow_unused=True, retain_graph=False)
        parts = []
        for g, p in zip(grad_tuple, params):
            parts.append(
                (g if g is not None else torch.zeros_like(p)).reshape(-1))
        grads.append(torch.cat(parts).detach().cpu())
    return torch.stack(grads, dim=0)


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
    svals = torch.linalg.svdvals(M)
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
