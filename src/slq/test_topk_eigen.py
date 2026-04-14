"""Compare SLQ top-k eigenvector estimation to ground truth on spiked Wishart."""
import numpy as np
import torch
from slq import SLQ

np.random.seed(0)
torch.manual_seed(0)

n = 2000
X = np.zeros((n, n))
X[0, 0] = 5; X[1, 1] = 4; X[2, 2] = 3
Z = np.random.randn(n, n)
A = X + (1 / n) * (Z @ Z.T)
A_torch = torch.from_numpy(A).float()

# Ground truth
print("Computing exact eigen-decomposition...")
evals_exact, evecs_exact = torch.linalg.eigh(A_torch)
# Sort descending
idx = torch.argsort(evals_exact, descending=True)
evals_exact = evals_exact[idx]
evecs_exact = evecs_exact[:, idx]

k = 5
print(f"\nExact top-{k} eigenvalues: {evals_exact[:k].tolist()}")


def cosine_similarity(v1, v2):
    """Absolute cosine similarity (sign ambiguity in eigenvectors)."""
    return float(torch.abs(torch.dot(v1, v2)).item())


def subspace_overlap(V_est, V_true):
    """Mean of singular values of V_est^T V_true (1.0 = perfect overlap)."""
    M = V_est.T @ V_true
    svals = torch.linalg.svdvals(M)
    return float(svals.mean().item())


slq = SLQ(matrix=A_torch)

print("\n" + "=" * 95)
header = (f"{'Config':<35s} | {'λ_err':>8s} | "
          + " ".join([f"cos(v{i+1})" for i in range(k)])
          + f" | {'subspace':>8s}")
print(header)
print("-" * len(header))

for reorth in [False, True]:
    for m in [20, 50, 100, 200, 500]:
        torch.manual_seed(42)
        tag = "reorth" if reorth else "no_reo"
        label = f"SLQ m={m:3d} {tag}"

        top_evals, top_evecs = slq.top_k_eigen(
            k, num_lanczos=m, reorthogonalize=reorth)

        # Eigenvalue relative error (avg over top-k)
        eval_err = float((torch.abs(top_evals - evals_exact[:k]) /
                          evals_exact[:k].abs().clamp(min=1e-12)).mean())

        # Per-vector cosine similarity
        cosines = []
        for i in range(k):
            c = cosine_similarity(top_evecs[:, i], evecs_exact[:, i])
            cosines.append(c)

        # Subspace overlap
        so = subspace_overlap(top_evecs, evecs_exact[:, :k])

        cos_str = " ".join([f"  {c:.4f}" for c in cosines])
        print(f"{label:<35s} | {eval_err:>8.2e} | {cos_str} | {so:>8.4f}")

    print()

# --- Detailed look at best config ---
print("=" * 95)
print("Detailed: m=200, reorth=True")
torch.manual_seed(42)
top_evals, top_evecs = slq.top_k_eigen(k, num_lanczos=200, reorthogonalize=True)
print(f"\n{'i':>3s} {'λ_exact':>10s} {'λ_est':>10s} {'|Δλ|/λ':>10s} {'|cos|':>8s}")
print("-" * 45)
for i in range(k):
    le = float(evals_exact[i])
    la = float(top_evals[i])
    err = abs(la - le) / abs(le)
    c = cosine_similarity(top_evecs[:, i], evecs_exact[:, i])
    print(f"{i+1:>3d} {le:>10.4f} {la:>10.4f} {err:>10.2e} {c:>8.4f}")

# --- Check residual ||Av - λv|| / ||λv|| ---
print(f"\nResidual norms ||Av - λv|| / |λ|:")
for i in range(k):
    v = top_evecs[:, i]
    Av = A_torch @ v
    lv = top_evals[i] * v
    res = float((Av - lv).norm() / abs(float(top_evals[i])))
    print(f"  v{i+1}: {res:.6e}")
