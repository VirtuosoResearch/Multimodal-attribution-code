"""Compare SLQ with and without reorthogonalization."""
import torch
from slq import SLQ


def compare(label, exact, est_no, est_re):
    err_no = abs(est_no - exact) / max(abs(exact), 1e-12)
    err_re = abs(est_re - exact) / max(abs(exact), 1e-12)
    better = "reorth wins" if err_re < err_no else "no reorth wins" if err_no < err_re else "tie"
    print(f"  {label:12s}  exact={exact:12.4f}  no_reorth={est_no:12.4f} (err={err_no:.4e})  "
          f"reorth={est_re:12.4f} (err={err_re:.4e})  [{better}]")
    return err_no, err_re


torch.manual_seed(42)
print("=" * 100)

# --- Test 1: Well-conditioned matrix (d=200) ---
print("\n[1] Well-conditioned PSD matrix, d=200")
d = 200
M = torch.randn(d, d); A = M @ M.T
exact_tr = float(torch.trace(A))
exact_ld = float(torch.logdet(A + 0.1 * torch.eye(d)))
A_pd = A + 0.1 * torch.eye(d)

slq = SLQ(matrix=A)
slq_pd = SLQ(matrix=A_pd)
for nlz in [30, 60, 100]:
    print(f"  num_lanczos={nlz}, num_probes=20")
    compare("trace", exact_tr,
            slq.trace(nlz, 20, reorthogonalize=False),
            slq.trace(nlz, 20, reorthogonalize=True))
    compare("logdet", exact_ld,
            slq_pd.logdet(nlz, 20, reorthogonalize=False),
            slq_pd.logdet(nlz, 20, reorthogonalize=True))

# --- Test 2: Ill-conditioned matrix (κ=1e6) ---
print("\n[2] Ill-conditioned matrix, d=200, κ=1e6")
eigs = torch.logspace(0, 6, d)
Q, _ = torch.linalg.qr(torch.randn(d, d))
A_ill = Q @ torch.diag(eigs) @ Q.T
exact_tr_ill = float(eigs.sum())
exact_ld_ill = float(torch.log(eigs).sum())

slq_ill = SLQ(matrix=A_ill)
for nlz in [30, 60, 100, 150]:
    print(f"  num_lanczos={nlz}, num_probes=20")
    compare("trace", exact_tr_ill,
            slq_ill.trace(nlz, 20, reorthogonalize=False),
            slq_ill.trace(nlz, 20, reorthogonalize=True))
    compare("logdet", exact_ld_ill,
            slq_ill.logdet(nlz, 20, reorthogonalize=False),
            slq_ill.logdet(nlz, 20, reorthogonalize=True))

# --- Test 3: Large-ish matrix with few Lanczos steps (stress test) ---
print("\n[3] d=500, few Lanczos steps (stress ghost eigenvalues)")
d = 500
M = torch.randn(d, d); A = M @ M.T
exact_tr = float(torch.trace(A))
slq = SLQ(matrix=A)
for nlz in [20, 50, 100]:
    print(f"  num_lanczos={nlz}, num_probes=20")
    compare("trace", exact_tr,
            slq.trace(nlz, 20, reorthogonalize=False),
            slq.trace(nlz, 20, reorthogonalize=True))

print("\n" + "=" * 100)
print("Done.")
