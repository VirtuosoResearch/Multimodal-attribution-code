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
class SubsetKernelSurrogateConfig:
    num_blocks: int = 8
    num_calibration_subsets: int = 24
    kernel_gamma: float | None = None
    ridge_lambda: float = 1e-2
    predictor_hidden_dim: int = 64
    predictor_epochs: int = 10
    predictor_lr: float = 1e-3
    predictor_batch_size: int = 64
    oracle_train_fraction: float = 1.0
    center_targets: bool = True
    seed: int | None = None
    verbose: bool = False


@dataclass
class SubsetCalibrationRecord:
    subset_weights: torch.Tensor
    selected_indices: torch.Tensor
    target_components: torch.Tensor
    oracle_components: torch.Tensor


@dataclass
class KernelSubsetQueryEngine:
    train_subset_weights: torch.Tensor
    alpha: torch.Tensor
    kernel_gamma: float
    full_components: torch.Tensor
    center_targets: bool

    def query(self, subset_weights: torch.Tensor) -> torch.Tensor:
        query = subset_weights.float().view(1, -1)
        train = self.train_subset_weights.float()
        kernel = _rbf_kernel_torch(query, train, self.kernel_gamma)
        prediction = (kernel @ self.alpha.float()).squeeze(0)
        if self.center_targets:
            prediction = prediction + self.full_components.float()
        return _project_components(prediction)


@dataclass
class SubsetKernelSurrogateResult:
    mean_components: torch.Tensor
    query_engine: KernelSubsetQueryEngine
    block_indices: list[torch.Tensor]
    calibration_records: list[SubsetCalibrationRecord]
    kernel_gamma: float
    alpha: torch.Tensor
    train_subset_weights: torch.Tensor
    train_targets: torch.Tensor
    full_oracle_components: torch.Tensor

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


def estimate_subset_kernel_surrogate_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: SubsetKernelSurrogateConfig | None = None,
    device: str | torch.device | None = None,
) -> SubsetKernelSurrogateResult:
    config = config or SubsetKernelSurrogateConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = _ensure_feature_batch(x1).float()
    x2 = _ensure_feature_batch(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)
    dataset = test_dataset or train_dataset or MultimodalDataset([x1, x2], labels)

    block_indices = _partition_indices(len(dataset), config.num_blocks, config.seed)
    full_dataset = _subset_dataset(dataset, torch.arange(len(dataset)))
    full_oracle_components = _estimate_subset_oracle_components(
        subset_dataset=full_dataset,
        num_labels=num_labels,
        config=config,
        device=device,
    )

    subset_weights_list = _generate_subset_weights(
        num_blocks=len(block_indices),
        num_subsets=config.num_calibration_subsets,
        seed=config.seed,
    )

    calibration_records: list[SubsetCalibrationRecord] = []
    for subset_weights in subset_weights_list:
        selected_indices = _indices_from_subset_weights(block_indices, subset_weights)
        subset_dataset = _subset_dataset(dataset, selected_indices)
        oracle_components = _estimate_subset_oracle_components(
            subset_dataset=subset_dataset,
            num_labels=num_labels,
            config=config,
            device=device,
        )
        target_components = (
            oracle_components - full_oracle_components
            if config.center_targets else oracle_components
        )
        calibration_records.append(
            SubsetCalibrationRecord(
                subset_weights=subset_weights.detach().cpu(),
                selected_indices=selected_indices.detach().cpu(),
                target_components=target_components.detach().cpu(),
                oracle_components=oracle_components.detach().cpu(),
            )
        )
        if config.verbose:
            print(
                "[subset-kernel/calib] "
                f"subset={subset_weights.tolist()} "
                f"oracle={[round(float(v), 4) for v in oracle_components.tolist()]}"
            )

    train_subset_weights = torch.stack([record.subset_weights for record in calibration_records], dim=0).float()
    train_targets = torch.stack([record.target_components for record in calibration_records], dim=0).float()
    kernel_gamma = _resolve_kernel_gamma(train_subset_weights, config.kernel_gamma)
    alpha = _fit_kernel_ridge(
        train_subset_weights=train_subset_weights,
        train_targets=train_targets,
        ridge_lambda=config.ridge_lambda,
        kernel_gamma=kernel_gamma,
    )

    query_engine = KernelSubsetQueryEngine(
        train_subset_weights=train_subset_weights.detach().cpu(),
        alpha=alpha.detach().cpu(),
        kernel_gamma=kernel_gamma,
        full_components=full_oracle_components.detach().cpu(),
        center_targets=config.center_targets,
    )
    full_weights = torch.ones(len(block_indices), dtype=torch.float32)
    mean_components = query_engine.query(full_weights)

    return SubsetKernelSurrogateResult(
        mean_components=mean_components,
        query_engine=query_engine,
        block_indices=block_indices,
        calibration_records=calibration_records,
        kernel_gamma=kernel_gamma,
        alpha=alpha.detach().cpu(),
        train_subset_weights=train_subset_weights.detach().cpu(),
        train_targets=train_targets.detach().cpu(),
        full_oracle_components=full_oracle_components.detach().cpu(),
    )


def evaluate_subset_queries(
    result: SubsetKernelSurrogateResult,
    dataset: Dataset,
    *,
    num_labels: int,
    num_eval_subsets: int,
    config: SubsetKernelSurrogateConfig,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    device_obj = _resolve_device(device)
    eval_subsets = _generate_subset_weights(
        num_blocks=len(result.block_indices),
        num_subsets=num_eval_subsets,
        seed=None if config.seed is None else config.seed + 1000,
    )
    train_subset_set = {tuple(weight.tolist()) for weight in result.train_subset_weights}

    sq_errors = []
    abs_errors = []
    oracle_norms = []
    num_evaluated = 0
    for subset_weights in eval_subsets:
        if tuple(subset_weights.tolist()) in train_subset_set:
            continue
        selected_indices = _indices_from_subset_weights(result.block_indices, subset_weights)
        subset_dataset = _subset_dataset(dataset, selected_indices)
        oracle = _estimate_subset_oracle_components(
            subset_dataset=subset_dataset,
            num_labels=num_labels,
            config=config,
            device=device_obj,
        ).cpu()
        prediction = result.query_engine.query(subset_weights).cpu()
        sq_errors.append((prediction - oracle).pow(2))
        abs_errors.append((prediction - oracle).abs())
        oracle_norms.append(oracle.abs().mean())
        num_evaluated += 1

    if num_evaluated == 0:
        return {"num_eval_subsets": 0.0, "rmse": 0.0, "mae": 0.0, "oracle_mean_abs": 0.0}

    sq_tensor = torch.stack(sq_errors, dim=0)
    abs_tensor = torch.stack(abs_errors, dim=0)
    return {
        "num_eval_subsets": float(num_evaluated),
        "rmse": float(torch.sqrt(sq_tensor.mean()).item()),
        "mae": float(abs_tensor.mean().item()),
        "oracle_mean_abs": float(torch.stack(oracle_norms).mean().item()),
    }


def estimate_subset_oracle_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    dataset: Dataset | None = None,
    config: SubsetKernelSurrogateConfig | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    config = config or SubsetKernelSurrogateConfig()
    device_obj = _resolve_device(device)
    x1 = _ensure_feature_batch(x1).float()
    x2 = _ensure_feature_batch(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)
    subset_dataset = dataset or MultimodalDataset([x1, x2], labels)
    return _estimate_subset_oracle_components(
        subset_dataset=subset_dataset,
        num_labels=num_labels,
        config=config,
        device=device_obj,
    )


def _estimate_subset_oracle_components(
    *,
    subset_dataset: Dataset,
    num_labels: int,
    config: SubsetKernelSurrogateConfig,
    device: torch.device,
) -> torch.Tensor:
    x1, x2, y = _dataset_to_tensors(subset_dataset)
    train_tensors, test_tensors = _split_subset_tensors(
        x1=x1,
        x2=x2,
        y=y,
        train_fraction=config.oracle_train_fraction,
        seed=config.seed,
    )
    x1_train, x2_train, y_train = train_tensors
    x1_test, x2_test, y_test = test_tensors

    stats_all = _train_and_score_predictor(
        x_train=torch.cat([x1_train, x2_train], dim=1),
        y_train=y_train,
        x_test=torch.cat([x1_test, x2_test], dim=1),
        y_test=y_test,
        num_labels=num_labels,
        config=config,
        device=device,
        seed_offset=0,
    )
    stats_1 = _train_and_score_predictor(
        x_train=x1_train,
        y_train=y_train,
        x_test=x1_test,
        y_test=y_test,
        num_labels=num_labels,
        config=config,
        device=device,
        seed_offset=1,
    )
    stats_2 = _train_and_score_predictor(
        x_train=x2_train,
        y_train=y_train,
        x_test=x2_test,
        y_test=y_test,
        num_labels=num_labels,
        config=config,
        device=device,
        seed_offset=2,
    )

    g_all = max(stats_all.mi, stats_1.mi, stats_2.mi)
    g_mod1 = stats_1.mi
    g_mod2 = stats_2.mi
    redundancy = _compute_imin_redundancy(
        specific_info_1=stats_1.specific_info,
        specific_info_2=stats_2.specific_info,
        label_prior=stats_1.label_prior,
    )
    redundancy = _project_redundancy(
        redundancy=redundancy,
        g_mod1=g_mod1,
        g_mod2=g_mod2,
        g_all=g_all,
    )
    unique_1 = max(g_mod1 - redundancy, 0.0)
    unique_2 = max(g_mod2 - redundancy, 0.0)
    synergy = max(g_all - g_mod1 - g_mod2 + redundancy, 0.0)
    return torch.tensor([redundancy, unique_1, unique_2, synergy], dtype=torch.float32)


@dataclass(frozen=True)
class PredictorStats:
    mi: float
    specific_info: np.ndarray
    label_prior: np.ndarray


def _train_and_score_predictor(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    num_labels: int,
    config: SubsetKernelSurrogateConfig,
    device: torch.device,
    seed_offset: int,
) -> PredictorStats:
    _maybe_set_seed(None if config.seed is None else config.seed + seed_offset)
    model = ClassificationNetwork(
        input_dim=x_train.shape[1],
        hidden_dim=config.predictor_hidden_dim,
        output_dim=num_labels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.predictor_lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(config.predictor_batch_size, max(len(x_train), 1)),
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(config.predictor_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    prior = _estimate_label_prior(y_train, num_labels).to(device)
    y_test_dev = y_test.to(device)
    with torch.no_grad():
        logits = model(x_test.to(device))
        log_probs = F.log_softmax(logits, dim=1)
        rows = torch.arange(y_test_dev.shape[0], device=device)
        pointwise = log_probs[rows, y_test_dev] - torch.log(prior[y_test_dev] + 1e-8)
        specific_info = []
        for class_index in range(num_labels):
            class_mask = y_test_dev == class_index
            if class_mask.any():
                value = (log_probs[class_mask, class_index] - torch.log(prior[class_index] + 1e-8)).mean()
                specific_info.append(float(torch.clamp_min(value, 0.0).item()))
            else:
                specific_info.append(0.0)
    return PredictorStats(
        mi=float(pointwise.mean().clamp_min(0.0).item()),
        specific_info=np.asarray(specific_info, dtype=np.float64),
        label_prior=prior.detach().cpu().numpy().astype(np.float64),
    )


def _fit_kernel_ridge(
    *,
    train_subset_weights: torch.Tensor,
    train_targets: torch.Tensor,
    ridge_lambda: float,
    kernel_gamma: float,
) -> torch.Tensor:
    kernel = _rbf_kernel_torch(train_subset_weights, train_subset_weights, kernel_gamma)
    regularized = kernel + ridge_lambda * torch.eye(kernel.shape[0], dtype=kernel.dtype)
    return torch.linalg.solve(regularized, train_targets)


def _rbf_kernel_torch(lhs: torch.Tensor, rhs: torch.Tensor, gamma: float) -> torch.Tensor:
    lhs = lhs.float()
    rhs = rhs.float()
    lhs_sq = (lhs * lhs).sum(dim=1, keepdim=True)
    rhs_sq = (rhs * rhs).sum(dim=1, keepdim=True).T
    sq_dist = lhs_sq + rhs_sq - 2.0 * lhs @ rhs.T
    return torch.exp(-gamma * torch.clamp(sq_dist, min=0.0))


def _resolve_kernel_gamma(subset_weights: torch.Tensor, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    if subset_weights.shape[0] <= 1:
        return 1.0
    diffs = subset_weights[:, None, :] - subset_weights[None, :, :]
    sq_dists = (diffs * diffs).sum(dim=-1)
    positives = sq_dists[sq_dists > 0]
    if positives.numel() == 0:
        return 1.0
    return float(1.0 / positives.median().item())


def _project_components(components: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(components.float(), min=0.0)
    info_1 = values[0] + values[1]
    info_2 = values[0] + values[2]
    info_12 = torch.clamp(values.sum(), min=torch.maximum(info_1, info_2))
    redundancy = _project_redundancy(
        redundancy=float(values[0].item()),
        g_mod1=float(info_1.item()),
        g_mod2=float(info_2.item()),
        g_all=float(info_12.item()),
    )
    unique_1 = max(float(info_1.item()) - redundancy, 0.0)
    unique_2 = max(float(info_2.item()) - redundancy, 0.0)
    synergy = max(float(info_12.item()) - redundancy - unique_1 - unique_2, 0.0)
    return torch.tensor([redundancy, unique_1, unique_2, synergy], dtype=torch.float32)


def _project_redundancy(*, redundancy: float, g_mod1: float, g_mod2: float, g_all: float) -> float:
    lower = max(0.0, g_mod1 + g_mod2 - g_all)
    upper = min(g_mod1, g_mod2)
    if lower > upper:
        return float(upper)
    return float(min(max(redundancy, lower), upper))


def _compute_imin_redundancy(
    *,
    specific_info_1: np.ndarray,
    specific_info_2: np.ndarray,
    label_prior: np.ndarray,
) -> float:
    return float(np.sum(label_prior * np.minimum(specific_info_1, specific_info_2)))


def _generate_subset_weights(num_blocks: int, num_subsets: int, seed: int | None) -> list[torch.Tensor]:
    generator = random.Random(seed)
    candidates: list[tuple[int, ...]] = [tuple([1] * num_blocks)]
    for block in range(num_blocks):
        single = tuple(1 if idx == block else 0 for idx in range(num_blocks))
        candidates.append(single)
        leave_one_out = tuple(0 if idx == block else 1 for idx in range(num_blocks))
        candidates.append(leave_one_out)
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            pair = tuple(1 if idx in {i, j} else 0 for idx in range(num_blocks))
            candidates.append(pair)
            if len(candidates) >= num_subsets * 2:
                break
        if len(candidates) >= num_subsets * 2:
            break
    while len(candidates) < num_subsets * 2:
        size = generator.randint(1, num_blocks)
        selected = set(generator.sample(range(num_blocks), k=size))
        candidates.append(tuple(1 if idx in selected else 0 for idx in range(num_blocks)))

    unique: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if sum(candidate) == 0 or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(torch.tensor(candidate, dtype=torch.float32))
        if len(unique) >= num_subsets:
            break
    return unique


def _partition_indices(num_samples: int, num_blocks: int, seed: int | None) -> list[torch.Tensor]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    chunks = torch.chunk(permutation, min(num_blocks, num_samples))
    return [chunk for chunk in chunks if len(chunk) > 0]


def _indices_from_subset_weights(block_indices: list[torch.Tensor], subset_weights: torch.Tensor) -> torch.Tensor:
    selected = [block for block, weight in zip(block_indices, subset_weights.tolist()) if weight > 0]
    if not selected:
        return block_indices[0]
    return torch.cat(selected, dim=0)


def _subset_dataset(dataset: Dataset, indices: torch.Tensor) -> MultimodalDataset:
    x1, x2, y = _dataset_to_tensors(dataset)
    return MultimodalDataset([x1[indices], x2[indices]], y[indices])


def _dataset_to_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(dataset, MultimodalDataset):
        x1 = _ensure_feature_batch(dataset.modalities[0]).float()
        x2 = _ensure_feature_batch(dataset.modalities[1]).float()
        y = _prepare_labels(dataset.labels)
        return x1, x2, y

    x1_list = []
    x2_list = []
    y_list = []
    for item in dataset:
        x1_list.append(torch.as_tensor(item[0]))
        x2_list.append(torch.as_tensor(item[1]))
        y_list.append(torch.as_tensor(item[2]))
    x1 = _ensure_feature_batch(torch.stack(x1_list)).float()
    x2 = _ensure_feature_batch(torch.stack(x2_list)).float()
    y = _prepare_labels(torch.stack(y_list))
    return x1, x2, y


def _split_subset_tensors(
    *,
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float,
    seed: int | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    total = x1.shape[0]
    if train_fraction >= 1.0 - 1e-8:
        return (x1, x2, y), (x1, x2, y)
    if total <= 1:
        return (x1, x2, y), (x1, x2, y)
    train_size = max(1, min(total - 1, int(round(total * train_fraction))))
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed + total)
    perm = torch.randperm(total, generator=generator)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    return (
        x1[train_idx],
        x2[train_idx],
        y[train_idx],
    ), (
        x1[test_idx],
        x2[test_idx],
        y[test_idx],
    )


def _estimate_label_prior(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_labels).float()
    counts /= counts.sum().clamp_min(1.0)
    return counts


def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    return labels


def _ensure_feature_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(-1)
    return value


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
    "KernelSubsetQueryEngine",
    "SubsetCalibrationRecord",
    "SubsetKernelSurrogateConfig",
    "SubsetKernelSurrogateResult",
    "estimate_subset_oracle_pid",
    "estimate_subset_kernel_surrogate_pid",
    "evaluate_subset_queries",
]
