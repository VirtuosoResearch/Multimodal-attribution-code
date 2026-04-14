"""Tests for Stochastic Lanczos Quadrature."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from slq import (SLQ, slq_trace, slq_logdet, slq_spectral_density,
                 nn_fisher_matvec, nn_hessian_matvec)


def _exact_hessian(model, params, loss):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = torch.cat([g.reshape(-1) for g in grads])
    H = torch.zeros(g_flat.numel(), g_flat.numel())
    for i in range(g_flat.numel()):
        basis = torch.zeros_like(g_flat)
        basis[i] = 1.0
        hv = torch.autograd.grad(torch.dot(g_flat, basis), params,
                                 retain_graph=True)
        H[:, i] = torch.cat([h.reshape(-1) for h in hv]).detach()
    return H


def test_trace():
    """Test trace estimation against exact trace."""
    torch.manual_seed(42)
    d = 200
    M = torch.randn(d, d)
    A = M @ M.T  # symmetric PSD
    exact_trace = float(torch.trace(A).item())
    est_trace = slq_trace(A, num_lanczos=80, num_probes=30)
    rel_err = abs(est_trace - exact_trace) / abs(exact_trace)
    print(f"[Trace]  exact={exact_trace:.4f}  est={est_trace:.4f}  rel_err={rel_err:.4e}")
    assert rel_err < 0.05, f"Trace relative error too large: {rel_err}"


def test_logdet():
    """Test log-determinant estimation against exact."""
    torch.manual_seed(7)
    d = 100
    M = torch.randn(d, d)
    A = M @ M.T + 0.1 * torch.eye(d)  # ensure positive definite
    exact_logdet = float(torch.logdet(A).item())
    est_logdet = slq_logdet(A, num_lanczos=80, num_probes=30)
    abs_err = abs(est_logdet - exact_logdet)
    rel_err = abs_err / abs(exact_logdet)
    print(f"[LogDet] exact={exact_logdet:.4f}  est={est_logdet:.4f}  rel_err={rel_err:.4e}")
    assert rel_err < 0.10, f"LogDet relative error too large: {rel_err}"


def test_spectral_density():
    """Test that spectral density nodes cover the true eigenvalue range."""
    torch.manual_seed(0)
    d = 150
    eigs_true = torch.linspace(0.1, 10.0, d)
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    A = Q @ torch.diag(eigs_true) @ Q.T

    nodes, weights = slq_spectral_density(A, num_lanczos=100, num_probes=20)
    print(f"[Spectral] node range=[{nodes.min():.4f}, {nodes.max():.4f}]  "
          f"true range=[0.1, 10.0]  weight_sum={weights.sum():.4f}")
    assert nodes.min() < 0.5, "Spectral density misses low end"
    assert nodes.max() > 9.0, "Spectral density misses high end"
    assert abs(weights.sum() - 1.0) < 0.05, "Weights don't sum to ~1"


def test_spectral_norm():
    """Test spectral norm (largest eigenvalue) estimation."""
    torch.manual_seed(3)
    d = 200
    eigs_true = torch.linspace(1.0, 100.0, d)
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    A = Q @ torch.diag(eigs_true) @ Q.T

    slq = SLQ(matrix=A)
    est_norm = slq.spectral_norm(num_lanczos=100, num_probes=5)
    rel_err = abs(est_norm - 100.0) / 100.0
    print(f"[SpecNorm] exact=100.0  est={est_norm:.4f}  rel_err={rel_err:.4e}")
    assert rel_err < 0.02, f"Spectral norm error too large: {rel_err}"


def test_spectral_norm_indefinite():
    """Spectral norm should use max absolute eigenvalue for symmetric A."""
    A = torch.diag(torch.tensor([-10.0, 1.0, 0.5]))
    slq = SLQ(matrix=A)
    est_norm = slq.spectral_norm(num_lanczos=3, num_probes=1,
                                 reorthogonalize=True)
    rel_err = abs(est_norm - 10.0) / 10.0
    print(f"[SpecNorm Indef] exact=10.0  est={est_norm:.4f}  rel_err={rel_err:.4e}")
    assert rel_err < 1e-5, f"Indefinite spectral norm error too large: {rel_err}"


def test_custom_matvec():
    """Test with a custom matvec operator (diagonal matrix)."""
    torch.manual_seed(1)
    d = 300
    diag = torch.linspace(1, 50, d)
    matvec = lambda x: diag * x
    exact_trace = float(diag.sum().item())

    slq = SLQ(matvec=matvec, dim=d)
    est_trace = slq.trace(num_lanczos=80, num_probes=30)
    rel_err = abs(est_trace - exact_trace) / abs(exact_trace)
    print(f"[CustomMV] exact_trace={exact_trace:.4f}  est={est_trace:.4f}  rel_err={rel_err:.4e}")
    assert rel_err < 0.05


def test_nn_fisher():
    """Test SLQ on a small neural network's Fisher/GGN matrix."""
    torch.manual_seed(99)
    d_in, d_hidden, d_out = 10, 20, 3
    n_samples = 50

    model = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )
    model.eval()

    X = torch.randn(n_samples, d_in)
    Y = torch.randint(0, d_out, (n_samples,))
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=25)

    slq = SLQ(model=model, data_loader=loader,
              loss_fn=F.cross_entropy, device=torch.device('cpu'))

    print(f"[NN Fisher] dim={slq.dim}")
    tr = slq.trace(num_lanczos=30, num_probes=5)
    print(f"[NN Fisher] trace_est={tr:.6f}")
    norm = slq.spectral_norm(num_lanczos=30, num_probes=3)
    print(f"[NN Fisher] spectral_norm_est={norm:.6f}")

    # Sanity: trace and norm should be non-negative for PSD Fisher
    assert tr >= -1e-3, f"Fisher trace should be non-negative, got {tr}"
    assert norm >= -1e-3, f"Fisher spectral norm should be non-negative, got {norm}"
    print("[NN Fisher] PASSED")


def test_nn_hessian_sum_reduction_matches_exact():
    """`loss_reduction='sum'` should still produce the dataset-average Hessian."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.2]]))

    X = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    Y = torch.tensor([[0.5], [-1.0]])
    loader = DataLoader(TensorDataset(X, Y), batch_size=2, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]

    def mse_sum(logits, targets):
        return ((logits - targets) ** 2).sum()

    exact = _exact_hessian(model, params, mse_sum(model(X), Y)) / len(X)
    matvec, dim = nn_hessian_matvec(model, loader, mse_sum, torch.device('cpu'),
                                    loss_reduction='sum')
    v = torch.tensor([0.7, -1.3])
    est = matvec(v)
    ref = exact @ v

    print(f"[NN Hess Sum] dim={dim}  max_abs_err={(est - ref).abs().max():.4e}")
    assert torch.allclose(est, ref, atol=1e-6, rtol=1e-6)


def test_nn_fisher_sum_reduction_matches_exact():
    """`loss_reduction='sum'` should still produce the dataset-average GGN."""
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.4, -0.1], [0.2, 0.3]]))

    X = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    Y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(X, Y), batch_size=2, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]

    def ce_sum(logits, targets):
        return F.cross_entropy(logits, targets, reduction='sum')

    # Linear model => exact Hessian in parameter space equals GGN/Fisher.
    exact = _exact_hessian(model, params, ce_sum(model(X), Y)) / len(X)
    matvec, dim = nn_fisher_matvec(model, loader, ce_sum, torch.device('cpu'),
                                   loss_reduction='sum')
    v = torch.tensor([0.5, -0.7, 1.1, 0.2])
    est = matvec(v)
    ref = exact @ v

    print(f"[NN Fish Sum] dim={dim}  max_abs_err={(est - ref).abs().max():.4e}")
    assert torch.allclose(est, ref, atol=1e-6, rtol=1e-6)


def test_nn_fisher_handles_frozen_params():
    """Frozen parameters should not break JVP structure."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    for p in model[0].parameters():
        p.requires_grad_(False)

    loader = DataLoader(
        TensorDataset(torch.randn(6, 3), torch.randint(0, 2, (6,))),
        batch_size=3,
        shuffle=False,
    )
    slq = SLQ(model=model, data_loader=loader, loss_fn=F.cross_entropy,
              device=torch.device('cpu'))
    v = torch.randn(slq.dim)
    out = slq.matvec(v)

    print(f"[Frozen Params] dim={slq.dim}  out_norm={out.norm():.4f}")
    assert out.shape == v.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    tests = [test_trace, test_logdet, test_spectral_density,
             test_spectral_norm, test_spectral_norm_indefinite,
             test_custom_matvec, test_nn_fisher,
             test_nn_hessian_sum_reduction_matches_exact,
             test_nn_fisher_sum_reduction_matches_exact,
             test_nn_fisher_handles_frozen_params]
    for t in tests:
        print(f"\n{'='*60}")
        print(f"Running {t.__name__}...")
        t()
        print(f"{t.__name__} PASSED")
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
