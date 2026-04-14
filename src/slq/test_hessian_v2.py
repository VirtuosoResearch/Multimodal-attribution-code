"""Fisher vs Hessian after training (where they diverge)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from slq import SLQ

torch.manual_seed(42)

d_in, d_hidden, d_out = 10, 32, 3
n_samples = 100

# Use tanh (smooth) so Hessian of f w.r.t. params is nonzero everywhere
model = nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.Tanh(),
    nn.Linear(d_hidden, d_hidden),
    nn.Tanh(),
    nn.Linear(d_hidden, d_out),
)

X = torch.randn(n_samples, d_in)
Y = torch.randint(0, d_out, (n_samples,))
loader = DataLoader(TensorDataset(X, Y), batch_size=50)

# Train for a while so residuals are nonzero -> Fisher != Hessian
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(200):
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(f"Final train loss: {float(loss):.4f}")
model.eval()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {num_params}\n")

slq_f = SLQ(model=model, data_loader=loader,
            loss_fn=F.cross_entropy, device=torch.device('cpu'), mode='fisher')
slq_h = SLQ(model=model, data_loader=loader,
            loss_fn=F.cross_entropy, device=torch.device('cpu'), mode='hessian')

print(f"{'Metric':<25s} {'Fisher/GGN':>12s} {'Hessian':>12s}")
print("-" * 52)

torch.manual_seed(0)
tr_f = slq_f.trace(50, 10)
torch.manual_seed(0)
tr_h = slq_h.trace(50, 10)
print(f"trace                    {tr_f:>12.6f} {tr_h:>12.6f}")

torch.manual_seed(0)
sn_f = slq_f.spectral_norm(50, 5)
torch.manual_seed(0)
sn_h = slq_h.spectral_norm(50, 5)
print(f"λ_max                    {sn_f:>12.6f} {sn_h:>12.6f}")

# Spectral density
print("\n--- Spectral density (m=80, 20 probes) ---")
torch.manual_seed(0)
nodes_f, weights_f = slq_f.spectral_density(80, 20)
torch.manual_seed(0)
nodes_h, weights_h = slq_h.spectral_density(80, 20)

print(f"Fisher  range: [{float(nodes_f.min()):.6f}, {float(nodes_f.max()):.6f}]")
print(f"Hessian range: [{float(nodes_h.min()):.6f}, {float(nodes_h.max()):.6f}]")

neg_f = float(weights_f[nodes_f < -1e-6].sum())
neg_h = float(weights_h[nodes_h < -1e-6].sum())
print(f"Fisher  negative spectral mass: {neg_f:.6f}")
print(f"Hessian negative spectral mass: {neg_h:.6f}")

# Top-5 eigenvalues
print("\n--- Top-5 eigenvalues ---")
torch.manual_seed(42)
evals_f, evecs_f = slq_f.top_k_eigen(5, num_lanczos=100)
torch.manual_seed(42)
evals_h, evecs_h = slq_h.top_k_eigen(5, num_lanczos=100)

print(f"{'i':>3s} {'Fisher λ':>12s} {'Hessian λ':>12s}")
print("-" * 30)
for i in range(5):
    print(f"{i+1:>3d} {float(evals_f[i]):>12.6f} {float(evals_h[i]):>12.6f}")

# Bottom-5 (most negative) for Hessian
print("\n--- Bottom-5 Hessian eigenvalues (Lanczos on -H) ---")
slq_neg = SLQ(matvec=lambda v: -slq_h.matvec(v), dim=slq_h.dim)
torch.manual_seed(42)
evals_neg, _ = slq_neg.top_k_eigen(5, num_lanczos=100)
print(f"Most negative Hessian eigenvalues: {[-float(e) for e in evals_neg[:5]]}")

# Subspace overlap
M = evecs_f.T @ evecs_h
svals = torch.linalg.svdvals(M)
print(f"\nFisher-Hessian top-5 subspace overlap: {float(svals.mean()):.4f}")
print(f"Principal angles (cos): {[f'{float(s):.4f}' for s in svals]}")
