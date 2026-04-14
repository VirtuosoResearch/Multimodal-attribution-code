"""Test Fisher vs Hessian mode on a small neural network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from slq import SLQ

torch.manual_seed(42)

d_in, d_hidden, d_out = 10, 20, 3
n_samples = 80

model = nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.ReLU(),
    nn.Linear(d_hidden, d_out),
)
model.eval()

X = torch.randn(n_samples, d_in)
Y = torch.randint(0, d_out, (n_samples,))
loader = DataLoader(TensorDataset(X, Y), batch_size=40)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params: {num_params}")

# --- Fisher/GGN mode ---
slq_fisher = SLQ(model=model, data_loader=loader,
                 loss_fn=F.cross_entropy, device=torch.device('cpu'),
                 mode='fisher')

# --- Hessian mode ---
slq_hessian = SLQ(model=model, data_loader=loader,
                  loss_fn=F.cross_entropy, device=torch.device('cpu'),
                  mode='hessian')

print(f"\n{'Metric':<25s} {'Fisher/GGN':>12s} {'Hessian':>12s}")
print("-" * 52)

for m, nv in [(30, 10), (50, 10)]:
    torch.manual_seed(0)
    tr_f = slq_fisher.trace(m, nv)
    torch.manual_seed(0)
    tr_h = slq_hessian.trace(m, nv)
    print(f"trace (m={m}, nv={nv})     {tr_f:>12.6f} {tr_h:>12.6f}")

    torch.manual_seed(0)
    sn_f = slq_fisher.spectral_norm(m, 5)
    torch.manual_seed(0)
    sn_h = slq_hessian.spectral_norm(m, 5)
    print(f"λ_max (m={m}, nv={nv})     {sn_f:>12.6f} {sn_h:>12.6f}")

# --- Spectral comparison: check if Hessian has negative eigenvalues ---
print("\n--- Spectral density comparison (m=50, 20 probes) ---")
torch.manual_seed(0)
nodes_f, weights_f = slq_fisher.spectral_density(50, 20)
torch.manual_seed(0)
nodes_h, weights_h = slq_hessian.spectral_density(50, 20)

print(f"Fisher  range: [{float(nodes_f.min()):.6f}, {float(nodes_f.max()):.6f}]")
print(f"Hessian range: [{float(nodes_h.min()):.6f}, {float(nodes_h.max()):.6f}]")

neg_weight_f = float(weights_f[nodes_f < -1e-6].sum())
neg_weight_h = float(weights_h[nodes_h < -1e-6].sum())
print(f"Fisher  negative spectral mass: {neg_weight_f:.6f}")
print(f"Hessian negative spectral mass: {neg_weight_h:.6f}")

# --- Top-k eigenvectors: compare Fisher vs Hessian subspaces ---
print("\n--- Top-5 eigenvalues ---")
torch.manual_seed(42)
evals_f, evecs_f = slq_fisher.top_k_eigen(5, num_lanczos=80)
torch.manual_seed(42)
evals_h, evecs_h = slq_hessian.top_k_eigen(5, num_lanczos=80)

print(f"{'i':>3s} {'Fisher λ':>12s} {'Hessian λ':>12s}")
print("-" * 30)
for i in range(5):
    print(f"{i+1:>3d} {float(evals_f[i]):>12.6f} {float(evals_h[i]):>12.6f}")

# Subspace overlap between Fisher and Hessian top-5
M = evecs_f.T @ evecs_h
svals = torch.linalg.svdvals(M)
print(f"\nFisher-Hessian top-5 subspace overlap: {float(svals.mean()):.4f}")
print(f"Principal angles (cos): {[f'{float(s):.4f}' for s in svals]}")
