"""Compare Hutchinson vs SLQ trace estimation on spiked Wishart model."""
import time
import numpy as np
import torch
from slq import SLQ, hutchinson_trace, matrix_matvec

np.random.seed(0)
torch.manual_seed(0)

n = 2000
X = np.zeros((n, n))
X[0, 0] = 5; X[1, 1] = 4; X[2, 2] = 3
Z = np.random.randn(n, n)
A = X + (1 / n) * (Z @ Z.T)
A_torch = torch.from_numpy(A).float()

exact_trace = float(torch.trace(A_torch))
print(f"n={n}, exact trace = {exact_trace:.4f}\n")

matvec = matrix_matvec(A_torch)
slq = SLQ(matrix=A_torch)

# ---------- Vary number of matvec calls (= budget) ----------
# Hutchinson: 1 matvec per probe
# SLQ: m matvecs per probe (Lanczos steps)
# Fair comparison: fix total matvec budget = num_probes * m for SLQ = num_probes for Hutchinson

header = f"{'Method':<25s} {'Budget':>6s} {'Estimate':>12s} {'RelErr':>10s} {'Time(s)':>8s}"
print(header)
print("-" * len(header))

for budget in [20, 50, 100, 200, 500, 1000]:
    # --- Hutchinson with `budget` probes ---
    for dist in ["rademacher", "gaussian"]:
        torch.manual_seed(0)
        t0 = time.time()
        est = hutchinson_trace(matvec, n, num_probes=budget, distribution=dist,
                               device=A_torch.device, dtype=A_torch.dtype)
        dt = time.time() - t0
        err = abs(est - exact_trace) / abs(exact_trace)
        label = f"Hutch-{dist[:3]}"
        print(f"{label:<25s} {budget:>6d} {est:>12.2f} {err:>10.4e} {dt:>8.3f}")

    # --- SLQ: split budget into (m Lanczos steps) x (num_probes) ---
    for m_ratio in [0.5, 0.8]:  # fraction of budget spent on Lanczos depth
        m = max(10, int(budget * m_ratio))
        nv = max(1, budget // m)
        actual_budget = m * nv
        torch.manual_seed(0)
        t0 = time.time()
        est = slq.trace(num_lanczos=m, num_probes=nv, reorthogonalize=False)
        dt = time.time() - t0
        err = abs(est - exact_trace) / abs(exact_trace)
        label = f"SLQ(m={m},nv={nv})"
        print(f"{label:<25s} {actual_budget:>6d} {est:>12.2f} {err:>10.4e} {dt:>8.3f}")

    print()

# ---------- Variance comparison: repeat 50 times at budget=100 ----------
print("\n=== Variance comparison (budget=100, 50 repetitions) ===\n")
results = {"Hutch-rad": [], "Hutch-gau": [], "SLQ(m=50,nv=2)": [], "SLQ(m=20,nv=5)": []}

for trial in range(50):
    seed = trial + 1000
    torch.manual_seed(seed)
    results["Hutch-rad"].append(
        hutchinson_trace(matvec, n, 100, "rademacher", A_torch.device, A_torch.dtype))
    torch.manual_seed(seed)
    results["Hutch-gau"].append(
        hutchinson_trace(matvec, n, 100, "gaussian", A_torch.device, A_torch.dtype))
    torch.manual_seed(seed)
    results["SLQ(m=50,nv=2)"].append(
        slq.trace(50, 2, reorthogonalize=False))
    torch.manual_seed(seed)
    results["SLQ(m=20,nv=5)"].append(
        slq.trace(20, 5, reorthogonalize=False))

print(f"{'Method':<20s} {'Mean':>12s} {'Std':>12s} {'MeanRelErr':>12s} {'Std/Exact':>12s}")
print("-" * 68)
for name, vals in results.items():
    vals_t = torch.tensor(vals)
    mean = float(vals_t.mean())
    std = float(vals_t.std())
    mean_err = abs(mean - exact_trace) / abs(exact_trace)
    rel_std = std / abs(exact_trace)
    print(f"{name:<20s} {mean:>12.2f} {std:>12.2f} {mean_err:>12.4e} {rel_std:>12.4e}")

print(f"\nExact trace = {exact_trace:.4f}")
