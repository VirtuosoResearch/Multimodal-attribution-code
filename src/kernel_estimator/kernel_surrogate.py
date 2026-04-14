from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pid_estimator.data import MultimodalDataset
from pid_estimator.lsmi import ClassificationNetwork


@dataclass(frozen=True)
class KernelSurrogateConfig:
    num_subcomponents_per_modality: int = 4
    num_masks: int = 128
    mask_prob: float = 0.5
    poly_order: int = 2
    ridge_lambda: float = 1e-2
    kernel_gamma: float | None = None
    classifier_hidden_dim: int = 64
    classifier_epochs: int = 20
    classifier_lr: float = 1e-3
    classifier_batch_size: int = 64
    oracle_train_fraction: float = 0.7
    seed: int | None = None
    verbose: bool = False


@dataclass
class KernelSurrogateResult:
    mean_components: torch.Tensor
    poly_coeffs: np.ndarray
    masks: np.ndarray
    g_values: np.ndarray
    mobius_main: np.ndarray
    mobius_cross: np.ndarray
    g_mod1: float
    g_mod2: float
    g_all: float
    subset_masks: np.ndarray
    subset_values: np.ndarray
    kernel_alpha: np.ndarray
    kernel_gamma: float

    @property
    def redundancy(self) -> torch.Tensor:
        return self.mean_components[0]

    @property
    def unique_1(self) -> torch.Tensor:
        return self.mean_components[1]

    @property
    def unique_2(self) -> torch.Tensor:
        return self.mean_components[2]

    @property
    def synergy(self) -> torch.Tensor:
        return self.mean_components[3]


def estimate_kernel_surrogate_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: KernelSurrogateConfig | None = None,
    device: str | torch.device | None = None,
) -> KernelSurrogateResult:
    config = config or KernelSurrogateConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = _ensure_2d(x1).float()
    x2 = _ensure_2d(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)

    train_tensors, test_tensors = _resolve_train_test_tensors(
        x1=x1,
        x2=x2,
        labels=labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_fraction=config.oracle_train_fraction,
        seed=config.seed,
    )
    x1_train, x2_train, y_train = train_tensors
    x1_test, x2_test, y_test = test_tensors

    idx1 = _partition_dims(x1_train.shape[1], config.num_subcomponents_per_modality)
    idx2 = _partition_dims(x2_train.shape[1], config.num_subcomponents_per_modality)
    K1 = len(idx1)
    K2 = len(idx2)
    K = K1 + K2

    rng = np.random.RandomState(config.seed or 0)
    masks = _generate_masks(K, K1, config.num_masks, config.mask_prob, rng)
    g_values = np.zeros(len(masks), dtype=np.float64)
    for index, mask in enumerate(masks):
        g_values[index] = _estimate_mask_information(
            x1_train=x1_train,
            x2_train=x2_train,
            y_train=y_train,
            x1_test=x1_test,
            x2_test=x2_test,
            y_test=y_test,
            mask=mask,
            idx1=idx1,
            idx2=idx2,
            K1=K1,
            num_labels=num_labels,
            config=config,
            device=device,
        )
        if config.verbose and (index + 1) % 10 == 0:
            print(f"[kernel/oracle] mask {index + 1}/{len(masks)} G={g_values[index]:.4f}")

    kernel_gamma = _resolve_kernel_gamma(masks, config.kernel_gamma)
    kernel_alpha = _fit_kernel_ridge(
        train_masks=masks,
        targets=g_values,
        ridge_lambda=config.ridge_lambda,
        kernel_gamma=kernel_gamma,
    )

    subset_masks = _enumerate_all_masks(K)
    subset_values = _predict_kernel_ridge(
        train_masks=masks,
        train_alpha=kernel_alpha,
        query_masks=subset_masks,
        kernel_gamma=kernel_gamma,
    )
    subset_values[0] = 0.0

    mobius_coeffs = _mobius_transform(subset_values)
    mobius_main = np.array([mobius_coeffs[1 << bit] for bit in range(K)], dtype=np.float64)
    mobius_cross = np.zeros((K1, K2), dtype=np.float64)
    for i in range(K1):
        for j in range(K2):
            mobius_cross[i, j] = mobius_coeffs[(1 << i) | (1 << (K1 + j))]

    all_code = (1 << K) - 1
    mod1_code = (1 << K1) - 1
    mod2_code = ((1 << K2) - 1) << K1
    all_mask = _code_to_mask(all_code, K)
    mod1_mask = _code_to_mask(mod1_code, K)
    mod2_mask = _code_to_mask(mod2_code, K)
    all_stats = _estimate_mask_statistics(
        x1_train=x1_train,
        x2_train=x2_train,
        y_train=y_train,
        x1_test=x1_test,
        x2_test=x2_test,
        y_test=y_test,
        mask=all_mask,
        idx1=idx1,
        idx2=idx2,
        K1=K1,
        num_labels=num_labels,
        config=config,
        device=device,
    )
    mod1_stats = _estimate_mask_statistics(
        x1_train=x1_train,
        x2_train=x2_train,
        y_train=y_train,
        x1_test=x1_test,
        x2_test=x2_test,
        y_test=y_test,
        mask=mod1_mask,
        idx1=idx1,
        idx2=idx2,
        K1=K1,
        num_labels=num_labels,
        config=config,
        device=device,
    )
    mod2_stats = _estimate_mask_statistics(
        x1_train=x1_train,
        x2_train=x2_train,
        y_train=y_train,
        x1_test=x1_test,
        x2_test=x2_test,
        y_test=y_test,
        mask=mod2_mask,
        idx1=idx1,
        idx2=idx2,
        K1=K1,
        num_labels=num_labels,
        config=config,
        device=device,
    )
    g_all = max(all_stats.mi, mod1_stats.mi, mod2_stats.mi)
    g_mod1 = mod1_stats.mi
    g_mod2 = mod2_stats.mi

    raw_redundancy = _compute_imin_redundancy(
        specific_info_1=mod1_stats.specific_info,
        specific_info_2=mod2_stats.specific_info,
        label_prior=mod1_stats.label_prior,
    )
    redundancy = _project_redundancy(
        redundancy=raw_redundancy,
        g_mod1=g_mod1,
        g_mod2=g_mod2,
        g_all=g_all,
    )
    unique_1 = g_mod1 - redundancy
    unique_2 = g_mod2 - redundancy
    synergy = g_all - g_mod1 - g_mod2 + redundancy

    if config.verbose:
        print(f"[kernel] g_all={g_all:.4f} g_mod1={g_mod1:.4f} g_mod2={g_mod2:.4f}")
        print(f"[kernel] R={redundancy:.4f} U1={unique_1:.4f} U2={unique_2:.4f} S={synergy:.4f}")

    mean_components = torch.tensor(
        [redundancy, unique_1, unique_2, synergy],
        dtype=torch.float32,
    )

    return KernelSurrogateResult(
        mean_components=mean_components,
        poly_coeffs=mobius_coeffs,
        masks=masks,
        g_values=g_values,
        mobius_main=mobius_main,
        mobius_cross=mobius_cross,
        g_mod1=g_mod1,
        g_mod2=g_mod2,
        g_all=g_all,
        subset_masks=subset_masks,
        subset_values=subset_values,
        kernel_alpha=kernel_alpha,
        kernel_gamma=kernel_gamma,
    )


def _estimate_mask_information(
    *,
    x1_train: torch.Tensor,
    x2_train: torch.Tensor,
    y_train: torch.Tensor,
    x1_test: torch.Tensor,
    x2_test: torch.Tensor,
    y_test: torch.Tensor,
    mask: np.ndarray,
    idx1: list[np.ndarray],
    idx2: list[np.ndarray],
    K1: int,
    num_labels: int,
    config: KernelSurrogateConfig,
    device: torch.device,
) -> float:
    return _estimate_mask_statistics(
        x1_train=x1_train,
        x2_train=x2_train,
        y_train=y_train,
        x1_test=x1_test,
        x2_test=x2_test,
        y_test=y_test,
        mask=mask,
        idx1=idx1,
        idx2=idx2,
        K1=K1,
        num_labels=num_labels,
        config=config,
        device=device,
    ).mi


@dataclass(frozen=True)
class MaskOracleStats:
    mi: float
    specific_info: np.ndarray
    label_prior: np.ndarray


def _estimate_mask_statistics(
    *,
    x1_train: torch.Tensor,
    x2_train: torch.Tensor,
    y_train: torch.Tensor,
    x1_test: torch.Tensor,
    x2_test: torch.Tensor,
    y_test: torch.Tensor,
    mask: np.ndarray,
    idx1: list[np.ndarray],
    idx2: list[np.ndarray],
    K1: int,
    num_labels: int,
    config: KernelSurrogateConfig,
    device: torch.device,
) -> MaskOracleStats:
    if int(mask.sum()) == 0:
        return MaskOracleStats(
            mi=0.0,
            specific_info=np.zeros(num_labels, dtype=np.float64),
            label_prior=np.zeros(num_labels, dtype=np.float64),
        )

    seed_offset = _mask_to_code(mask)
    _maybe_set_seed(None if config.seed is None else config.seed + seed_offset)

    x_train = _mask_and_cat(x1_train, x2_train, mask, idx1, idx2, K1)
    x_test = _mask_and_cat(x1_test, x2_test, mask, idx1, idx2, K1)

    model = ClassificationNetwork(
        input_dim=x_train.shape[1],
        hidden_dim=config.classifier_hidden_dim,
        output_dim=num_labels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.classifier_lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.classifier_batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _epoch in range(config.classifier_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device))
        log_probs = F.log_softmax(logits, dim=1)
        prior = _estimate_label_prior(y_train, num_labels).to(device)
        y_test_device = y_test.to(device)
        rows = torch.arange(y_test.shape[0], device=device)
        pointwise = log_probs[rows, y_test_device] - torch.log(prior[y_test_device] + 1e-8)

        specific_info = []
        for class_index in range(num_labels):
            class_mask = y_test_device == class_index
            if class_mask.any():
                value = (log_probs[class_mask, class_index] - torch.log(prior[class_index] + 1e-8)).mean()
                specific_info.append(float(torch.clamp_min(value, 0.0).item()))
            else:
                specific_info.append(0.0)

    return MaskOracleStats(
        mi=float(pointwise.mean().clamp_min(0.0).item()),
        specific_info=np.asarray(specific_info, dtype=np.float64),
        label_prior=prior.detach().cpu().numpy().astype(np.float64),
    )


def _fit_kernel_ridge(
    *,
    train_masks: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
    kernel_gamma: float,
) -> np.ndarray:
    kernel_matrix = _rbf_kernel(train_masks, train_masks, kernel_gamma)
    system = kernel_matrix + ridge_lambda * np.eye(kernel_matrix.shape[0], dtype=np.float64)
    return np.linalg.solve(system, targets)


def _predict_kernel_ridge(
    *,
    train_masks: np.ndarray,
    train_alpha: np.ndarray,
    query_masks: np.ndarray,
    kernel_gamma: float,
) -> np.ndarray:
    return _rbf_kernel(query_masks, train_masks, kernel_gamma) @ train_alpha


def _rbf_kernel(lhs: np.ndarray, rhs: np.ndarray, gamma: float) -> np.ndarray:
    lhs_sq = np.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs * rhs, axis=1, keepdims=True).T
    sq_dist = lhs_sq + rhs_sq - 2.0 * lhs @ rhs.T
    return np.exp(-gamma * np.clip(sq_dist, a_min=0.0, a_max=None))


def _mobius_transform(values: np.ndarray) -> np.ndarray:
    coeffs = values.astype(np.float64).copy()
    total_masks = coeffs.shape[0]
    num_bits = (total_masks - 1).bit_length()
    for bit in range(num_bits):
        step = 1 << bit
        for mask in range(total_masks):
            if mask & step:
                coeffs[mask] -= coeffs[mask ^ step]
    return coeffs


def _sum_cross_modal_mobius(coeffs: np.ndarray, K1: int, K2: int) -> float:
    total = 0.0
    for mask_code, coeff in enumerate(coeffs):
        if mask_code == 0:
            continue
        has_mod1 = (mask_code & ((1 << K1) - 1)) != 0
        has_mod2 = (mask_code >> K1) != 0
        if has_mod1 and has_mod2:
            total += float(coeff)
    return total


def _compute_imin_redundancy(
    *,
    specific_info_1: np.ndarray,
    specific_info_2: np.ndarray,
    label_prior: np.ndarray,
) -> float:
    return float(np.sum(label_prior * np.minimum(specific_info_1, specific_info_2)))


def _project_redundancy(
    *,
    redundancy: float,
    g_mod1: float,
    g_mod2: float,
    g_all: float,
) -> float:
    lower = max(0.0, g_mod1 + g_mod2 - g_all)
    upper = min(g_mod1, g_mod2)
    if lower > upper:
        return float(upper)
    return float(min(max(redundancy, lower), upper))


def _generate_masks(
    K: int,
    K1: int,
    num_masks: int,
    prob: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    max_masks = 1 << K
    target_masks = min(max(num_masks, 1), max_masks)
    seen: set[tuple[int, ...]] = set()
    masks: list[tuple[int, ...]] = []

    def _add(mask: np.ndarray) -> None:
        key = tuple(mask.astype(int).tolist())
        if key not in seen:
            seen.add(key)
            masks.append(key)

    _add(np.zeros(K))
    _add(np.ones(K))
    mod1 = np.zeros(K); mod1[:K1] = 1.0; _add(mod1)
    mod2 = np.zeros(K); mod2[K1:] = 1.0; _add(mod2)
    for bit in range(K):
        singleton = np.zeros(K); singleton[bit] = 1.0; _add(singleton)
    for bit in range(K):
        leave_one_out = np.ones(K); leave_one_out[bit] = 0.0; _add(leave_one_out)
    for i in range(K1):
        for j in range(K1, K):
            pair = np.zeros(K); pair[i] = 1.0; pair[j] = 1.0; _add(pair)

    if target_masks == max_masks:
        for code in range(max_masks):
            _add(_code_to_mask(code, K))
    else:
        while len(masks) < target_masks:
            _add(rng.binomial(1, prob, size=K).astype(np.float64))

    return np.array(masks[:target_masks], dtype=np.float64)


def _enumerate_all_masks(K: int) -> np.ndarray:
    return np.stack([_code_to_mask(code, K) for code in range(1 << K)], axis=0)


def _code_to_mask(code: int, K: int) -> np.ndarray:
    bits = [(code >> bit) & 1 for bit in range(K)]
    return np.array(bits, dtype=np.float64)


def _mask_to_code(mask: np.ndarray) -> int:
    code = 0
    for bit, value in enumerate(mask.astype(int).tolist()):
        if value:
            code |= 1 << bit
    return code


def _resolve_kernel_gamma(masks: np.ndarray, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    if len(masks) <= 1:
        return 1.0
    diffs = masks[:, None, :] - masks[None, :, :]
    sq_dists = np.sum(diffs * diffs, axis=-1)
    positives = sq_dists[sq_dists > 0]
    if positives.size == 0:
        return 1.0
    return float(1.0 / np.median(positives))


def _resolve_train_test_tensors(
    *,
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    train_dataset: Dataset | None,
    test_dataset: Dataset | None,
    train_fraction: float,
    seed: int | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if train_dataset is not None and test_dataset is not None:
        return _dataset_to_tensors(train_dataset), _dataset_to_tensors(test_dataset)

    total = x1.shape[0]
    train_size = max(1, min(total - 1, int(round(total * train_fraction))))
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed + 999)
    perm = torch.randperm(total, generator=generator)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    return (
        x1[train_idx],
        x2[train_idx],
        labels[train_idx],
    ), (
        x1[test_idx],
        x2[test_idx],
        labels[test_idx],
    )


def _dataset_to_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(dataset, MultimodalDataset):
        x1 = _ensure_2d(torch.as_tensor(dataset.modalities[0])).float()
        x2 = _ensure_2d(torch.as_tensor(dataset.modalities[1])).float()
        labels = _prepare_labels(torch.as_tensor(dataset.labels))
        return x1, x2, labels

    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    xs1: list[torch.Tensor] = []
    xs2: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for batch in loader:
        xs1.append(_ensure_2d(torch.as_tensor(batch[0])).float())
        xs2.append(_ensure_2d(torch.as_tensor(batch[1])).float())
        ys.append(_prepare_labels(torch.as_tensor(batch[2])))
    return torch.cat(xs1, dim=0), torch.cat(xs2, dim=0), torch.cat(ys, dim=0)


def _mask_and_cat(
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: np.ndarray,
    idx1: list[np.ndarray],
    idx2: list[np.ndarray],
    K1: int,
) -> torch.Tensor:
    x1_masked = x1.clone()
    x2_masked = x2.clone()
    for block_index, indices in enumerate(idx1):
        if mask[block_index] == 0:
            x1_masked[:, indices] = 0.0
    for block_index, indices in enumerate(idx2):
        if mask[K1 + block_index] == 0:
            x2_masked[:, indices] = 0.0
    return torch.cat([x1_masked, x2_masked], dim=1)


def _partition_dims(total_dim: int, num_parts: int) -> list[np.ndarray]:
    return [chunk for chunk in np.array_split(np.arange(total_dim), num_parts) if len(chunk) > 0]


def _estimate_label_prior(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_labels).float()
    counts /= counts.sum().clamp_min(1.0)
    return counts


def _ensure_2d(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1) if value.ndim == 1 else value


def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    return labels.squeeze(-1) if labels.ndim > 1 else labels


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "KernelSurrogateConfig",
    "KernelSurrogateResult",
    "estimate_kernel_surrogate_pid",
]
