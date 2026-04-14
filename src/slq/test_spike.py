"""Test SLQ on spiked Wishart model: 3 signal eigenvalues + bulk noise."""
import numpy as np
import torch
from slq import SLQ

np.random.seed(0)
n = 2000
X = np.zeros((n, n))
X[0, 0] = 5
X[1, 1] = 4
X[2, 2] = 3
Z = np.random.randn(n, n)
A = X + (1 / n) * (Z @ Z.T)

A_torch = torch.from_numpy(A).float()

# Exact spectrum
print("Computing exact eigenvalues...")
eigs_exact = torch.linalg.eigvalsh(A_torch)
eigs_exact_sorted = eigs_exact.sort(descending=True).values

exact_trace = float(eigs_exact.sum())
exact_logdet = float(torch.log(eigs_exact.clamp(min=1e-7)).sum())
exact_top3 = eigs_exact_sorted[:5].tolist()
exact_norm = float(eigs_exact_sorted[0])

print(f"Exact trace    = {exact_trace:.4f}")
print(f"Exact logdet   = {exact_logdet:.4f}")
print(f"Exact top-5 λ  = {[f'{v:.4f}' for v in exact_top3]}")
print(f"Exact λ_max    = {exact_norm:.4f}")
print(f"Bulk range     ≈ [{float(eigs_exact_sorted[-1]):.4f}, {float(eigs_exact_sorted[3]):.4f}]")

print("\n" + "=" * 90)
slq = SLQ(matrix=A_torch)

for reorth in [False, True]:
    tag = "reorth" if reorth else "no_reorth"
    print(f"\n--- {tag} ---")
    for nlz in [50, 100, 200]:
        tr = slq.trace(nlz, 20, reorthogonalize=reorth)
        ld = slq.logdet(nlz, 20, reorthogonalize=reorth)
        sn = slq.spectral_norm(nlz, 5, reorthogonalize=reorth)
        tr_err = abs(tr - exact_trace) / abs(exact_trace)
        ld_err = abs(ld - exact_logdet) / abs(exact_logdet)
        sn_err = abs(sn - exact_norm) / abs(exact_norm)
        print(f"  m={nlz:3d}  trace={tr:10.2f} (err={tr_err:.4e})  "
              f"logdet={ld:10.2f} (err={ld_err:.4e})  "
              f"λ_max={sn:8.4f} (err={sn_err:.4e})")

# Spectral density visualization
print("\n--- Spectral density (m=200, 30 probes, reorth=True) ---")
nodes, weights = slq.spectral_density(200, 30, reorthogonalize=True)
# Bin into histogram
bins = torch.linspace(float(nodes.min()) - 0.1, float(nodes.max()) + 0.1, 50)
hist = torch.zeros(len(bins) - 1)
for i in range(len(bins) - 1):
    mask = (nodes >= bins[i]) & (nodes < bins[i + 1])
    hist[i] = weights[mask].sum()

print(f"  Spectral range: [{float(nodes.min()):.4f}, {float(nodes.max()):.4f}]")
print(f"  Weight sum: {float(weights.sum()):.4f}")
# Show where the mass is
top_bins = hist.topk(5)
print("  Top 5 density bins:")
for idx, val in zip(top_bins.indices, top_bins.values):
    lo, hi = float(bins[idx]), float(bins[idx + 1])
    print(f"    [{lo:.3f}, {hi:.3f}): weight={float(val):.4f}")
