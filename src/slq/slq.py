"""Stochastic Lanczos Quadrature (SLQ) for spectral approximation.

Supports:
  - Explicit matrices (torch.Tensor)
  - Implicit matrix-vector product operators (callable)
  - Neural network Gauss-Newton / Fisher Hessian-vector products

References:
  Ubaru, Saad, Yousef (2017). "Fast estimation of tr(f(A)) via stochastic
  Lanczos quadrature."
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lanczos tridiagonalization
# ---------------------------------------------------------------------------

def lanczos(matvec: Callable[[torch.Tensor], torch.Tensor],
            dim: int,
            num_steps: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
            reorthogonalize: bool = False,
            return_basis: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Lanczos iteration with a random starting vector.

    Args:
        matvec: y = A @ x operator.
        dim: dimensionality of the vector space.
        num_steps: number of Lanczos steps (tridiagonal matrix size).
        device: torch device.
        dtype: torch dtype.
        reorthogonalize: if True, apply full reorthogonalization (O(m²d) cost).
        return_basis: if True, also return the Lanczos basis Q [m, dim].

    Returns:
        alpha: diagonal of tridiagonal matrix T, shape [num_steps].
        beta:  sub-diagonal of T, shape [num_steps - 1].
        Q (optional): Lanczos basis vectors, shape [m, dim].
    """
    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / q.norm()

    alpha = torch.zeros(num_steps, device=device, dtype=dtype)
    beta = torch.zeros(num_steps - 1, device=device, dtype=dtype)

    q_prev = torch.zeros_like(q)
    beta_prev = 0.0

    store_basis = reorthogonalize or return_basis
    if store_basis:
        Q = torch.zeros(num_steps, dim, device=device, dtype=dtype)

    for j in range(num_steps):
        if store_basis:
            Q[j] = q
        v = matvec(q)
        if j > 0:
            v = v - beta_prev * q_prev
        a = torch.dot(q, v)
        alpha[j] = a
        v = v - a * q
        # Full reorthogonalization: project out all previous Lanczos vectors
        if reorthogonalize and j > 0:
            coeffs = Q[:j + 1] @ v
            v = v - Q[:j + 1].T @ coeffs
        b = v.norm()
        if b < 1e-12:
            alpha = alpha[:j + 1]
            beta = beta[:j]
            if store_basis:
                Q = Q[:j + 1]
            break
        if j < num_steps - 1:
            beta[j] = b
        q_prev = q
        beta_prev = b
        q = v / b

    if return_basis:
        return alpha, beta, Q
    return alpha, beta


def tridiag_eigen(alpha: torch.Tensor,
                  beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose a symmetric tridiagonal matrix.

    Returns:
        eigenvalues: sorted ascending, shape [m].
        weights: squared first-component of each eigenvector, shape [m].
            These are the quadrature weights (|v_1^(j)|^2).
    """
    T = torch.diag(alpha)
    if beta.numel() > 0:
        T = T + torch.diag(beta, 1) + torch.diag(beta, -1)
    evals, evecs = torch.linalg.eigh(T)
    weights = evecs[0, :] ** 2  # squared first component
    return evals, weights


# ---------------------------------------------------------------------------
# Matvec helpers
# ---------------------------------------------------------------------------

def matrix_matvec(A: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap an explicit matrix into a matvec callable."""
    return lambda x: A @ x


def nn_fisher_matvec(
    model: nn.Module,
    data_loader,
    loss_fn: Callable,
    device: torch.device,
    loss_reduction: str = "mean",
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Build a Gauss-Newton (GGN) matrix-vector product operator.

    The GGN is  G = (1/N) J^T H_L J  where J = d(logits)/d(params) and
    H_L = d²L/d(logits)² is the Hessian of the loss w.r.t. logits only.

    This is always PSD and equals the Fisher for cross-entropy.

    Uses torch.func.jvp + torch.func.functional_call for clean Jv computation.

    Args:
        model: neural network (must be in eval mode).
        data_loader: iterable of (inputs, targets) batches.
        loss_fn: scalar loss function(logits, targets).
        device: torch device.
        loss_reduction: reduction used by loss_fn across the batch. Must be
            "mean" or "sum" so the dataset-average GGN/Fisher is scaled
            correctly.

    Returns:
        (matvec_fn, num_params)
    """
    from torch.func import jvp, functional_call

    if loss_reduction not in {"mean", "sum"}:
        raise ValueError("loss_reduction must be 'mean' or 'sum'.")

    named_params = list(model.named_parameters())
    params = [p for _, p in named_params if p.requires_grad]
    param_names = [n for n, p in named_params if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    batches = [(x.to(device), y.to(device)) for x, y in data_loader]
    total_samples = sum(x.shape[0] for x, _ in batches)

    # Build a dict of params for functional_call
    param_dict = dict(named_params)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        """Compute GGN-vector product: J^T H_L J v."""
        result = torch.zeros_like(v)

        # Unpack v into param-shaped tangent dict
        tangent_dict = {}
        offset = 0
        for name, p in named_params:
            if p.requires_grad:
                n = p.numel()
                tangent_dict[name] = v[offset:offset + n].reshape(p.shape)
                offset += n
            else:
                tangent_dict[name] = torch.zeros_like(p)

        for x_batch, y_batch in batches:
            bs = x_batch.shape[0]
            batch_weight = bs / total_samples if loss_reduction == "mean" else 1.0 / total_samples

            # --- Step 1: Jv via functional jvp ---
            def fwd_fn(p_dict):
                return functional_call(model, p_dict, (x_batch,))

            logits, Jv = jvp(fwd_fn, (param_dict,), (tangent_dict,))
            # logits: [B, C], Jv: [B, C]

            # --- Step 2: H_L @ Jv ---
            logits_d = logits.detach().requires_grad_(True)
            loss = loss_fn(logits_d, y_batch)
            dL_dz = torch.autograd.grad(loss, logits_d, create_graph=True)[0]
            # H_L Jv = d/d(logits) [ dL_dz · Jv ]
            HLJv = torch.autograd.grad(
                torch.sum(dL_dz * Jv.detach()), logits_d)[0]  # [B, C]

            # --- Step 3: J^T @ (H_L Jv) via backward ---
            model.zero_grad(set_to_none=True)
            logits2 = model(x_batch)
            JtHLJv = torch.autograd.grad(
                logits2, params, grad_outputs=HLJv.detach())
            JtHLJv_flat = torch.cat([g.reshape(-1) for g in JtHLJv])

            result += JtHLJv_flat.detach() * batch_weight
        return result

    return matvec, num_params


def nn_hessian_matvec(
    model: nn.Module,
    data_loader,
    loss_fn: Callable,
    device: torch.device,
    loss_reduction: str = "mean",
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Build a true Hessian-vector product operator for a neural network.

    Computes Hv = d²L/dθ² · v exactly via two backward passes.
    Unlike the Fisher/GGN, this captures the full curvature including
    negative eigenvalues from the Hessian of the loss w.r.t. logits.

    Args:
        model: neural network.
        data_loader: iterable of (inputs, targets) batches.
        loss_fn: scalar loss function(logits, targets).
        device: torch device.
        loss_reduction: reduction used by loss_fn across the batch. Must be
            "mean" or "sum" so the dataset-average Hessian is scaled correctly.

    Returns:
        (matvec_fn, num_params)
    """
    if loss_reduction not in {"mean", "sum"}:
        raise ValueError("loss_reduction must be 'mean' or 'sum'.")

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    batches = [(x.to(device), y.to(device)) for x, y in data_loader]
    total_samples = sum(x.shape[0] for x, _ in batches)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian-vector product Hv."""
        result = torch.zeros_like(v)
        for x_batch, y_batch in batches:
            bs = x_batch.shape[0]
            batch_weight = bs / total_samples if loss_reduction == "mean" else 1.0 / total_samples
            model.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            # Flatten and dot with v
            g_flat = torch.cat([g.reshape(-1) for g in grads])
            gv = torch.dot(g_flat, v)
            # Second backward through the graph to get Hv
            hvp = torch.autograd.grad(gv, params, retain_graph=False)
            hvp_flat = torch.cat([h.reshape(-1) for h in hvp])
            result += hvp_flat.detach() * batch_weight
        return result

    return matvec, num_params


# ---------------------------------------------------------------------------
# Main SLQ interface
# ---------------------------------------------------------------------------

class SLQ:
    """Stochastic Lanczos Quadrature estimator.

    Usage:
        # Explicit matrix
        slq = SLQ(matrix=A)

        # Neural network (Fisher/GGN — PSD, default)
        slq = SLQ(model=model, data_loader=dl, loss_fn=F.cross_entropy, device='cuda')

        # Neural network (true Hessian — may have negative eigenvalues)
        slq = SLQ(model=model, data_loader=dl, loss_fn=F.cross_entropy,
                  device='cuda', mode='hessian')

        # Custom operator
        slq = SLQ(matvec=my_fn, dim=1000, device='cuda')

        # Estimate spectral density
        nodes, weights = slq.spectral_density(num_lanczos=50, num_probes=10)

        # Estimate trace
        tr = slq.trace(num_lanczos=50, num_probes=10)

        # Estimate log-determinant
        ld = slq.logdet(num_lanczos=50, num_probes=10)
    """

    def __init__(
        self,
        matrix: Optional[torch.Tensor] = None,
        matvec: Optional[Callable] = None,
        dim: Optional[int] = None,
        model: Optional[nn.Module] = None,
        data_loader=None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        mode: str = "fisher",
        loss_reduction: str = "mean",
    ):
        """
        Args:
            matrix: explicit symmetric matrix.
            matvec: custom matrix-vector product callable.
            dim: dimension (required with matvec).
            model: neural network.
            data_loader: iterable of (inputs, targets).
            loss_fn: scalar loss function(logits, targets).
            device: torch device.
            mode: "fisher" (GGN/Fisher, PSD) or "hessian" (true Hessian,
                  may have negative eigenvalues). Only used with model.
            loss_reduction: reduction used by loss_fn across the batch. Must
                  be "mean" or "sum". Only used with model.
        """
        if matrix is not None:
            self.dim = matrix.shape[0]
            self.device = matrix.device
            self.dtype = matrix.dtype
            self.matvec = matrix_matvec(matrix)
        elif model is not None:
            assert data_loader is not None and loss_fn is not None
            dev = device or next(model.parameters()).device
            if mode == "hessian":
                self.matvec, self.dim = nn_hessian_matvec(
                    model, data_loader, loss_fn, dev,
                    loss_reduction=loss_reduction)
            else:
                self.matvec, self.dim = nn_fisher_matvec(
                    model, data_loader, loss_fn, dev,
                    loss_reduction=loss_reduction)
            self.device = dev
            self.dtype = torch.float32
        elif matvec is not None:
            assert dim is not None
            self.matvec = matvec
            self.dim = dim
            self.device = device or torch.device('cpu')
            self.dtype = torch.float32
        else:
            raise ValueError("Provide matrix, (matvec, dim), or (model, data_loader, loss_fn).")

    def _run_probes(self, num_lanczos: int, num_probes: int,
                    reorthogonalize: bool = False):
        """Run multiple Lanczos probes, return list of (evals, weights)."""
        results = []
        m = min(num_lanczos, self.dim)
        for _ in range(num_probes):
            alpha, beta = lanczos(self.matvec, self.dim, m, self.device,
                                  self.dtype, reorthogonalize=reorthogonalize)
            evals, weights = tridiag_eigen(alpha, beta)
            results.append((evals, weights))
        return results

    def spectral_density(self, num_lanczos: int = 50, num_probes: int = 10,
                         reorthogonalize: bool = False):
        """Estimate spectral density as a weighted set of quadrature nodes.

        Returns:
            nodes: eigenvalue quadrature nodes, shape [num_probes * m].
            weights: corresponding weights (sum ≈ 1), shape [num_probes * m].
        """
        probes = self._run_probes(num_lanczos, num_probes, reorthogonalize)
        all_nodes = []
        all_weights = []
        for evals, w in probes:
            all_nodes.append(evals)
            all_weights.append(w / num_probes)
        return torch.cat(all_nodes), torch.cat(all_weights)

    def trace(self, num_lanczos: int = 50, num_probes: int = 10,
              reorthogonalize: bool = False) -> float:
        """Estimate tr(A) = dim * E[v^T A v]."""
        probes = self._run_probes(num_lanczos, num_probes, reorthogonalize)
        estimates = []
        for evals, weights in probes:
            estimates.append(float(torch.dot(weights, evals).item()))
        return self.dim * sum(estimates) / len(estimates)

    def logdet(self, num_lanczos: int = 50, num_probes: int = 10,
               eps: float = 1e-7, reorthogonalize: bool = False) -> float:
        """Estimate log det(A) = tr(log(A)) via SLQ.

        Uses log(max(λ, eps)) to handle near-zero eigenvalues.
        """
        probes = self._run_probes(num_lanczos, num_probes, reorthogonalize)
        estimates = []
        for evals, weights in probes:
            log_evals = torch.log(torch.clamp(evals, min=eps))
            estimates.append(float(torch.dot(weights, log_evals).item()))
        return self.dim * sum(estimates) / len(estimates)

    def spectral_norm(self, num_lanczos: int = 100, num_probes: int = 5,
                      reorthogonalize: bool = False) -> float:
        """Estimate the spectral norm max(|lambda_i|) of a symmetric matrix."""
        best = -float('inf')
        m = min(num_lanczos, self.dim)
        for _ in range(num_probes):
            alpha, beta = lanczos(self.matvec, self.dim, m, self.device,
                                  self.dtype, reorthogonalize=reorthogonalize)
            evals, _ = tridiag_eigen(alpha, beta)
            best = max(best, float(evals.abs().max().item()))
        return best

    def top_k_eigen(self, k: int, num_lanczos: int = 100,
                    reorthogonalize: bool = True,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate top-k eigenvalues and eigenvectors via Lanczos Ritz vectors.

        The Ritz vectors are y_i = Q^T s_i where Q is the Lanczos basis and
        s_i are the eigenvectors of the tridiagonal matrix T.

        Reorthogonalization is strongly recommended here — without it, ghost
        eigenvalues produce spurious/duplicate Ritz vectors.

        Args:
            k: number of top eigenpairs to return.
            num_lanczos: Lanczos steps (should be > k, ideally 2-5x).
            reorthogonalize: full reorthogonalization (default True).

        Returns:
            eigenvalues: shape [k], descending.
            eigenvectors: shape [dim, k], columns are approximate eigenvectors.
        """
        m = min(num_lanczos, self.dim)
        alpha, beta, Q = lanczos(self.matvec, self.dim, m, self.device,
                                 self.dtype, reorthogonalize=reorthogonalize,
                                 return_basis=True)
        # Build and diagonalize tridiagonal matrix
        T = torch.diag(alpha)
        if beta.numel() > 0:
            T = T + torch.diag(beta, 1) + torch.diag(beta, -1)
        evals, evecs = torch.linalg.eigh(T)

        # Sort descending, take top-k
        idx = torch.argsort(evals, descending=True)[:k]
        top_evals = evals[idx]
        top_evecs_small = evecs[:, idx]  # [m, k]

        # Ritz vectors: project back to full space
        ritz_vectors = Q.T @ top_evecs_small  # [dim, k]
        # Normalize (should already be ~unit norm if Q is orthonormal)
        ritz_vectors = torch.nn.functional.normalize(ritz_vectors, dim=0)

        return top_evals, ritz_vectors


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hutchinson trace estimator
# ---------------------------------------------------------------------------

def hutchinson_trace(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_probes: int = 50,
    distribution: str = "rademacher",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> float:
    """Hutchinson's stochastic trace estimator.

    tr(A) = E[z^T A z] where z is a random probe vector.

    Args:
        matvec: y = A @ x operator.
        dim: vector space dimension.
        num_probes: number of random probe vectors.
        distribution: "rademacher" (±1) or "gaussian".
        device: torch device.
        dtype: torch dtype.

    Returns:
        Estimated trace (scalar).
    """
    total = 0.0
    for _ in range(num_probes):
        if distribution == "rademacher":
            z = torch.randint(0, 2, (dim,), device=device, dtype=dtype) * 2 - 1
        else:
            z = torch.randn(dim, device=device, dtype=dtype)
        Az = matvec(z)
        total += float(torch.dot(z, Az).item())
    return total / num_probes


def hutchinson_trace_matrix(A: torch.Tensor, num_probes: int = 50,
                            distribution: str = "rademacher") -> float:
    """Convenience: Hutchinson trace of an explicit matrix."""
    return hutchinson_trace(matrix_matvec(A), A.shape[0], num_probes,
                            distribution, A.device, A.dtype)


def slq_spectral_density(A: torch.Tensor, num_lanczos: int = 50,
                         num_probes: int = 10):
    """Convenience: spectral density of an explicit matrix."""
    return SLQ(matrix=A).spectral_density(num_lanczos, num_probes)


def slq_trace(A: torch.Tensor, num_lanczos: int = 50,
              num_probes: int = 10) -> float:
    """Convenience: trace estimate of an explicit matrix."""
    return SLQ(matrix=A).trace(num_lanczos, num_probes)


def slq_logdet(A: torch.Tensor, num_lanczos: int = 50,
               num_probes: int = 10) -> float:
    """Convenience: log-determinant estimate of an explicit matrix."""
    return SLQ(matrix=A).logdet(num_lanczos, num_probes)
