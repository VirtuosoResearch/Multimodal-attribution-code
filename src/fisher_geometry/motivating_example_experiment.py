"""Motivating example for Fisher Interaction Curvature (FIC).

This script implements the toy experiment designed in motivating_example.md:
the same two-mode XOR data distribution has pure PID synergy, while additive,
shared-bilinear, and gated-bilinear learners induce different cross-sample
Fisher interaction geometry.

Example:
    conda run -n vlm --no-capture-output bash -lc \
        'python src/fisher_geometry/motivating_example_experiment.py \
        --n 128 --epochs 300 --assert-expected'
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "src" / "fisher_geometry" / "results" / "motivating_example"
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ToyXORData:
    x1: torch.Tensor
    x2: torch.Tensor
    labels: torch.Tensor
    y_sign: torch.Tensor
    z: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


def make_two_mode_xor(n: int, *, seed: int, noise: float = 0.0) -> ToyXORData:
    """Create a balanced two-mode XOR dataset.

    The base support contains every (z, u, v) combination equally often, which
    keeps each unimodal view independent of the label in finite samples when
    noise=0.
    """
    if n < 8:
        raise ValueError("--n must be at least 8")
    rng = np.random.default_rng(seed)
    base = np.array(
        [(z, u, v) for z in (0, 1) for u in (-1, 1) for v in (-1, 1)],
        dtype=np.int64,
    )
    repeats = int(math.ceil(n / len(base)))
    rows = np.tile(base, (repeats, 1))[:n].copy()
    rng.shuffle(rows)

    z_np = rows[:, 0]
    u_np = rows[:, 1]
    v_np = rows[:, 2]
    y_np = u_np * v_np
    if noise > 0.0:
        flips = rng.random(n) < float(noise)
        y_np = np.where(flips, -y_np, y_np)

    x1_np = np.stack([z_np, u_np], axis=1).astype(np.float32)
    x2_np = np.stack([z_np, v_np], axis=1).astype(np.float32)
    labels_np = (y_np > 0).astype(np.float32)
    return ToyXORData(
        x1=torch.from_numpy(x1_np),
        x2=torch.from_numpy(x2_np),
        labels=torch.from_numpy(labels_np),
        y_sign=torch.from_numpy(y_np.astype(np.float32)),
        z=torch.from_numpy(z_np.astype(np.int64)),
        u=torch.from_numpy(u_np.astype(np.float32)),
        v=torch.from_numpy(v_np.astype(np.float32)),
    )


class AdditiveXORLearner(nn.Module):
    """Additive learner f(x1, x2)=a(x1)+b(x2), unable to represent XOR."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1, bias=False)
        self.b = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.a.weight)
        nn.init.zeros_(self.b.weight)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        use_x1: bool = True,
        use_x2: bool = True,
    ) -> torch.Tensor:
        out = torch.zeros(x1.shape[0], 1, dtype=x1.dtype, device=x1.device)
        if use_x1:
            out = out + self.a(x1)
        if use_x2:
            out = out + self.b(x2)
        return out.view(-1)


class SharedBilinearXORLearner(nn.Module):
    """Learner f(x1, x2)=theta*alpha_z*u*v with one shared parameter.

    The small alpha_z variation keeps all residuals on one learned direction
    while avoiding a constant K matrix in the training-effect correlation test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        use_x1: bool = True,
        use_x2: bool = True,
    ) -> torch.Tensor:
        if not (use_x1 and use_x2):
            return torch.zeros(x1.shape[0], dtype=x1.dtype, device=x1.device)
        alpha = 1.0 + 0.25 * x1[:, 0]
        return self.theta * alpha * x1[:, 1] * x2[:, 1]


class GatedBilinearXORLearner(nn.Module):
    """Learner f(x1, x2)=theta_z*u*v with separate A/B interaction parameters."""

    def __init__(self, *, scale_a: float = 1.0, scale_b: float = 1.0) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(2))
        self.register_buffer(
            "scales", torch.tensor([scale_a, scale_b], dtype=torch.float32)
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        use_x1: bool = True,
        use_x2: bool = True,
    ) -> torch.Tensor:
        if not (use_x1 and use_x2):
            return torch.zeros(x1.shape[0], dtype=x1.dtype, device=x1.device)
        z = x1[:, 0].long().clamp(0, 1)
        return self.scales[z] * self.theta[z] * x1[:, 1] * x2[:, 1]


def build_model(name: str, *, gated_scale_a: float, gated_scale_b: float) -> nn.Module:
    if name == "additive":
        return AdditiveXORLearner()
    if name == "shared":
        return SharedBilinearXORLearner()
    if name == "gated":
        return GatedBilinearXORLearner(scale_a=gated_scale_a, scale_b=gated_scale_b)
    raise ValueError(f"Unknown model: {name}")


def train_model(
    model: nn.Module,
    data: ToyXORData,
    *,
    lr: float,
    epochs: int,
    device: torch.device,
) -> dict[str, float]:
    model.to(device)
    x1 = data.x1.to(device)
    x2 = data.x2.to(device)
    labels = data.labels.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x1, x2, use_x1=True, use_x2=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        logits = model(x1, x2, use_x1=True, use_x2=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        preds = (logits >= 0.0).float()
        acc = (preds == labels).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(acc.item())}


def _flat_grad(model: nn.Module) -> torch.Tensor:
    grads = []
    for param in model.parameters():
        if param.requires_grad:
            grad = param.grad
            if grad is None:
                grads.append(torch.zeros_like(param).reshape(-1))
            else:
                grads.append(grad.detach().reshape(-1))
    return torch.cat(grads)


def per_sample_gradients(
    model: nn.Module,
    data: ToyXORData,
    *,
    device: torch.device,
    use_x1: bool,
    use_x2: bool,
) -> torch.Tensor:
    model.eval()
    rows = []
    x1 = data.x1.to(device)
    x2 = data.x2.to(device)
    labels = data.labels.to(device)
    for idx in range(x1.shape[0]):
        model.zero_grad(set_to_none=True)
        logits = model(
            x1[idx : idx + 1],
            x2[idx : idx + 1],
            use_x1=use_x1,
            use_x2=use_x2,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels[idx : idx + 1])
        if loss.requires_grad:
            loss.backward()
            rows.append(_flat_grad(model).detach().cpu())
        else:
            rows.append(torch.zeros_like(parameters_to_vector(model.parameters())).cpu())
    model.zero_grad(set_to_none=True)
    return torch.stack(rows, dim=0)


def fisher_interaction(
    h: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    *,
    damping: float,
    energy_threshold: float,
) -> dict[str, torch.Tensor]:
    c = (h - g1 - g2).float()
    n, d = c.shape
    eye = torch.eye(d, dtype=torch.float32)
    a_add = (g1.float().T @ g1.float() + g2.float().T @ g2.float()) / max(n, 1)
    a_add = a_add + float(damping) * eye
    a_inv = torch.linalg.inv(a_add)
    k = c @ a_inv @ c.T
    energy = torch.diag(k).clamp(min=0.0)
    denom = energy[:, None] * energy[None, :]
    fic = torch.zeros_like(k)
    valid = denom >= float(energy_threshold)
    fic[valid] = (k[valid].pow(2) / denom[valid].clamp(min=1e-30)).clamp(0.0, 1.0)
    return {
        "c": c,
        "A_add": a_add,
        "A_inv": a_inv,
        "K": k,
        "FIC": fic,
        "energy": energy,
    }


def cosine_matrix(rows: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    rows = rows.float()
    denom = rows.norm(dim=1)[:, None] * rows.norm(dim=1)[None, :]
    return (rows @ rows.T) / denom.clamp(min=eps)


def entropy(keys: list[tuple[Any, ...]]) -> float:
    total = len(keys)
    if total == 0:
        return 0.0
    counts: dict[tuple[Any, ...], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def mutual_information(x: list[tuple[Any, ...]], y: list[tuple[Any, ...]]) -> float:
    return entropy(x) + entropy(y) - entropy([a + b for a, b in zip(x, y)])


def pid_summary(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    x1_keys = [tuple(map(int, row.tolist())) for row in x1]
    x2_keys = [tuple(map(int, row.tolist())) for row in x2]
    y_keys = [(int(value),) for value in y.tolist()]
    x12_keys = [a + b for a, b in zip(x1_keys, x2_keys)]
    i1 = mutual_information(x1_keys, y_keys)
    i2 = mutual_information(x2_keys, y_keys)
    i12 = mutual_information(x12_keys, y_keys)
    # For this XOR construction the single-view terms are zero, so the
    # whole-minus-sum interaction equals the PID synergy in the clean case.
    synergy = max(0.0, i12 - i1 - i2)
    redundancy = min(i1, i2)
    unique_1 = max(0.0, i1 - redundancy)
    unique_2 = max(0.0, i2 - redundancy)
    return {
        "I_x1_y": float(i1),
        "I_x2_y": float(i2),
        "I_x12_y": float(i12),
        "R": float(redundancy),
        "U1": float(unique_1),
        "U2": float(unique_2),
        "S": float(synergy),
    }


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if x.size < 2 or np.std(x) < 1e-30 or np.std(y) < 1e-30:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if x.size < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.astype(np.float64)
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_average(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.astype(np.float64)
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(labels.size) + 1)
    return float((precision * sorted_labels).sum() / positives)


def pair_vectors(
    matrix: torch.Tensor,
    z: torch.Tensor,
    *,
    upper_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if upper_only:
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    else:
        mask = ~torch.eye(n, dtype=torch.bool)
    labels = (z[:, None] == z[None, :]).long()
    return matrix.detach().cpu()[mask].numpy(), labels[mask].numpy()


def tau_values(model: nn.Module, data: ToyXORData, *, device: torch.device) -> torch.Tensor:
    model.eval()
    x1 = data.x1.to(device)
    x2 = data.x2.to(device)
    labels = data.labels.to(device)
    with torch.no_grad():
        fused = model(x1, x2, use_x1=True, use_x2=True)
        one = model(x1, x2, use_x1=True, use_x2=False)
        two = model(x1, x2, use_x1=False, use_x2=True)
        logp_fused = -F.binary_cross_entropy_with_logits(fused, labels, reduction="none")
        logp_one = -F.binary_cross_entropy_with_logits(one, labels, reduction="none")
        logp_two = -F.binary_cross_entropy_with_logits(two, labels, reduction="none")
        pi = labels.mean().clamp(1e-6, 1.0 - 1e-6)
        log_pi = torch.where(labels > 0.5, torch.log(pi), torch.log1p(-pi))
        return (logp_fused - logp_one - logp_two + log_pi).detach().cpu()


def training_effect_proxy(
    model: nn.Module,
    data: ToyXORData,
    c: torch.Tensor,
    a_inv: torch.Tensor,
    *,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    params = [param for param in model.parameters() if param.requires_grad]
    base_vector = parameters_to_vector(params).detach().clone()
    base_tau = tau_values(model, data, device=device)
    updates = c.float() @ a_inv.float().T
    columns = []
    try:
        for idx in range(c.shape[0]):
            new_vector = base_vector.cpu() - float(eta) * updates[idx].cpu()
            vector_to_parameters(new_vector.to(device), params)
            columns.append(tau_values(model, data, device=device) - base_tau)
    finally:
        vector_to_parameters(base_vector.to(device), params)
    return torch.stack(columns, dim=1)


def summarize_matrix_by_group(matrix: torch.Tensor, z: torch.Tensor) -> dict[str, float]:
    n = matrix.shape[0]
    offdiag = ~torch.eye(n, dtype=torch.bool)
    same = (z[:, None] == z[None, :]) & offdiag
    diff = (z[:, None] != z[None, :]) & offdiag
    return {
        "offdiag_mean": float(matrix[offdiag].mean().item()) if offdiag.any() else 0.0,
        "same_z_mean": float(matrix[same].mean().item()) if same.any() else 0.0,
        "diff_z_mean": float(matrix[diff].mean().item()) if diff.any() else 0.0,
    }


def run_one_model(
    name: str,
    data: ToyXORData,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model = build_model(
        name,
        gated_scale_a=float(args.gated_scale_a),
        gated_scale_b=float(args.gated_scale_b),
    )
    train_stats = train_model(
        model,
        data,
        lr=float(args.lr),
        epochs=int(args.epochs),
        device=device,
    )
    h = per_sample_gradients(model, data, device=device, use_x1=True, use_x2=True)
    g1 = per_sample_gradients(model, data, device=device, use_x1=True, use_x2=False)
    g2 = per_sample_gradients(model, data, device=device, use_x1=False, use_x2=True)
    fisher = fisher_interaction(
        h,
        g1,
        g2,
        damping=float(args.damping),
        energy_threshold=float(args.energy_threshold),
    )
    fic = fisher["FIC"]
    k = fisher["K"]
    c = fisher["c"]
    add_cos = cosine_matrix(g1 + g2).nan_to_num(0.0)
    residual_cos = cosine_matrix(c).nan_to_num(0.0)
    model.eval()
    with torch.inference_mode():
        logits = model(data.x1.to(device), data.x2.to(device), use_x1=True, use_x2=True)
        pred = (logits.detach().cpu() >= 0.0).long()
    model_pid = pid_summary(data.x1.long(), data.x2.long(), pred)

    k_scores, pair_labels = pair_vectors(k, data.z)
    fic_scores, _ = pair_vectors(fic, data.z)
    pid_pair = np.ones_like(fic_scores)
    residual_cos_scores, _ = pair_vectors(residual_cos, data.z)
    add_cos_scores, _ = pair_vectors(add_cos, data.z)

    delta_tau = training_effect_proxy(
        model,
        data,
        c,
        fisher["A_inv"],
        eta=float(args.eta),
        device=device,
    )
    offdiag = ~torch.eye(k.shape[0], dtype=torch.bool)
    k_for_delta = k.T[offdiag].detach().cpu().numpy()
    delta_flat = delta_tau[offdiag].detach().cpu().numpy()

    result = {
        "train": train_stats,
        "model_pid": model_pid,
        "parameters": {
            key: value.detach().cpu().reshape(-1).tolist()
            for key, value in model.state_dict().items()
            if value.is_floating_point()
        },
        "residual_energy": {
            "mean": float(fisher["energy"].mean().item()),
            "median": float(fisher["energy"].median().item()),
            "max": float(fisher["energy"].max().item()),
        },
        "K_summary": summarize_matrix_by_group(k, data.z),
        "FIC_summary": summarize_matrix_by_group(fic, data.z),
        "pair_recovery": {
            "fic_auc": binary_auc(fic_scores, pair_labels),
            "fic_ap": average_precision(fic_scores, pair_labels),
            "pidpair_auc": binary_auc(pid_pair, pair_labels),
            "pidpair_ap": average_precision(pid_pair, pair_labels),
            "residual_cos_auc": binary_auc(residual_cos_scores, pair_labels),
            "residual_cos_ap": average_precision(residual_cos_scores, pair_labels),
            "additive_cos_auc": binary_auc(add_cos_scores, pair_labels),
            "additive_cos_ap": average_precision(add_cos_scores, pair_labels),
        },
        "training_effect": {
            "pearson_K_delta_tau": pearson(k_for_delta, delta_flat),
            "spearman_K_delta_tau": spearman(k_for_delta, delta_flat),
            "delta_tau_abs_mean": float(np.mean(np.abs(delta_flat))),
        },
    }
    arrays = {
        "K": k,
        "FIC": fic,
        "C": c,
        "energy": fisher["energy"],
        "residual_cos": residual_cos,
        "additive_cos": add_cos,
        "delta_tau": delta_tau,
        "A_inv": fisher["A_inv"],
    }
    return result, arrays


def make_plots(arrays: dict[str, dict[str, torch.Tensor]], data: ToyXORData, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping plots because matplotlib is unavailable: {exc}", flush=True)
        return

    order = torch.argsort(data.z)
    names = ["additive", "shared", "gated"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, name in zip(axes, names):
        matrix = arrays[name]["FIC"][order][:, order].detach().cpu().numpy()
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.8, label="FIC")
    fig.suptitle("Same PID, Different Fisher Interaction Geometry")
    path = output_dir / "fic_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    colors = np.where(data.z.numpy() == 0, "#377eb8", "#e41a1c")
    for ax, name in zip(axes, names):
        c = arrays[name]["C"].float()
        a_inv = arrays[name].get("A_inv")
        if a_inv is not None:
            whitened = c @ torch.linalg.cholesky(a_inv)
        else:
            whitened = c
        if whitened.shape[1] == 1:
            points = torch.cat([whitened, torch.zeros_like(whitened)], dim=1)
        elif whitened.shape[1] >= 2:
            points = whitened[:, :2]
        else:
            points = torch.zeros((c.shape[0], 2))
        ax.scatter(points[:, 0].numpy(), points[:, 1].numpy(), c=colors, s=18)
        ax.set_title(name)
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.axvline(0.0, color="0.8", linewidth=0.8)
    fig.suptitle("Fisher-Whitened Residual Schematic")
    path = output_dir / "whitened_residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def assert_expected(results: dict[str, Any]) -> None:
    additive = results["models"]["additive"]
    shared = results["models"]["shared"]
    gated = results["models"]["gated"]
    checks = [
        (
            additive["residual_energy"]["mean"] < 1e-8,
            "additive residual energy should be near zero",
        ),
        (shared["train"]["accuracy"] >= 0.99, "shared learner should solve XOR"),
        (gated["train"]["accuracy"] >= 0.99, "gated learner should solve XOR"),
        (
            shared["FIC_summary"]["offdiag_mean"] >= 0.95,
            "shared learner FIC should be dense",
        ),
        (
            gated["FIC_summary"]["same_z_mean"] >= 0.95
            and gated["FIC_summary"]["diff_z_mean"] <= 0.05,
            "gated learner FIC should be block diagonal",
        ),
        (
            gated["pair_recovery"]["fic_auc"] >= 0.99,
            "gated learner FIC should recover same-z pairs",
        ),
        (
            abs(gated["pair_recovery"]["pidpair_auc"] - 0.5) <= 0.01,
            "PIDPair baseline should be tied at AUC 0.5",
        ),
        (
            shared["training_effect"]["pearson_K_delta_tau"] >= 0.98,
            "shared K should match the finite-difference training effect",
        ),
        (
            gated["training_effect"]["pearson_K_delta_tau"] >= 0.98,
            "gated K should match the finite-difference training effect",
        ),
    ]
    failures = [message for ok, message in checks if not ok]
    if failures:
        raise AssertionError("; ".join(failures))


def compact_print(results: dict[str, Any]) -> str:
    lines = [
        "# Motivating Example Results",
        "",
        f"Data PID S={results['data_pid']['S']:.4f}, "
        f"I(X1;Y)={results['data_pid']['I_x1_y']:.4f}, "
        f"I(X2;Y)={results['data_pid']['I_x2_y']:.4f}",
        "",
        "| model | acc | model PID S | E mean | FIC offdiag | FIC same-z | FIC diff-z | gated AUC | K/delta Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results["models"].items():
        pair = result["pair_recovery"]
        lines.append(
            f"| {name} | {result['train']['accuracy']:.3f} | "
            f"{result['model_pid']['S']:.3f} | "
            f"{result['residual_energy']['mean']:.3e} | "
            f"{result['FIC_summary']['offdiag_mean']:.3f} | "
            f"{result['FIC_summary']['same_z_mean']:.3f} | "
            f"{result['FIC_summary']['diff_z_mean']:.3f} | "
            f"{pair['fic_auc']:.3f} | "
            f"{result['training_effect']['pearson_K_delta_tau']:.3f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-mode XOR motivating example for FIC."
    )
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--energy-threshold", type=float, default=1e-16)
    parser.add_argument("--eta", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--assert-expected", action="store_true")
    parser.add_argument(
        "--gated-scale-a",
        type=float,
        default=1.0,
        help="Parameterization scale for z=A gated interaction.",
    )
    parser.add_argument(
        "--gated-scale-b",
        type=float,
        default=3.0,
        help="Parameterization scale for z=B gated interaction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_two_mode_xor(int(args.n), seed=int(args.seed), noise=float(args.noise))
    data_pid = pid_summary(data.x1.long(), data.x2.long(), data.labels.long())

    model_results: dict[str, Any] = {}
    model_arrays: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("additive", "shared", "gated"):
        result, arrays = run_one_model(name, data, args=args, device=device)
        model_results[name] = result
        model_arrays[name] = arrays

    results = {
        "config": {
            "n": int(args.n),
            "seed": int(args.seed),
            "noise": float(args.noise),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "damping": float(args.damping),
            "energy_threshold": float(args.energy_threshold),
            "eta": float(args.eta),
            "device": str(device),
            "gated_scale_a": float(args.gated_scale_a),
            "gated_scale_b": float(args.gated_scale_b),
        },
        "data_pid": data_pid,
        "models": model_results,
    }

    json_path = output_dir / "motivating_example_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    npz_path = output_dir / "motivating_example_matrices.npz"
    np.savez_compressed(
        npz_path,
        z=data.z.numpy(),
        labels=data.labels.numpy(),
        **{
            f"{model}_{key}": value.detach().cpu().numpy()
            for model, arrays in model_arrays.items()
            for key, value in arrays.items()
        },
    )
    if not args.no_plots:
        make_plots(model_arrays, data, output_dir)

    if args.assert_expected:
        assert_expected(results)

    print(compact_print(results), flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
