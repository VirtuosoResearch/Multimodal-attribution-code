from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_NUM_GROUPS = 50
DEFAULT_GROUP_BATCH_SIZE = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "subset_group_indices"
DEFAULT_GROUP_METHOD = "random"


@dataclass(frozen=True)
class GroupPartitionMetadata:
    dataset_name: str
    dataset_size: int
    num_groups: int
    seed: int | None
    group_method: str
    cache_path: Path


@dataclass(frozen=True)
class ModalitySubsetMetadata:
    modality_names: tuple[str, ...]
    num_modalities: int


def _sanitize_dataset_name(dataset_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", dataset_name.strip()).strip("._-")
    return normalized or "dataset"


def _infer_dataset_name(dataset: Dataset) -> str:
    if isinstance(dataset, Subset):
        base_name = _infer_dataset_name(dataset.dataset)
        return f"{base_name}_subset"
    return dataset.__class__.__name__.lower()


def _to_numpy_bool_array(value: Sequence[int | float | bool] | torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(value, dtype=bool)


def _clone_sample_value(sample_value: Any) -> Any:
    if isinstance(sample_value, torch.Tensor):
        return sample_value.clone()
    if isinstance(sample_value, np.ndarray):
        return sample_value.copy()
    return sample_value


def _zero_like(sample_value: Any) -> Any:
    if isinstance(sample_value, torch.Tensor):
        return torch.zeros_like(sample_value)
    if isinstance(sample_value, np.ndarray):
        return np.zeros_like(sample_value)
    raise TypeError(f"unsupported modality value type: {type(sample_value)!r}")


def _infer_modality_metadata(dataset: Dataset) -> ModalitySubsetMetadata:
    if len(dataset) <= 0:
        raise ValueError("cannot infer modalities from an empty dataset")

    sample = dataset[0]
    if (
        isinstance(sample, tuple)
        and len(sample) == 4
        and isinstance(sample[0], dict)
    ):
        modality_names = tuple(str(name) for name in sample[0].keys())
        return ModalitySubsetMetadata(
            modality_names=modality_names,
            num_modalities=len(modality_names),
        )
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        num_modalities = len(sample) - 1
        modality_names = tuple(f"modality_{index}" for index in range(num_modalities))
        return ModalitySubsetMetadata(
            modality_names=modality_names,
            num_modalities=num_modalities,
        )
    raise TypeError(
        "unsupported dataset sample format for modality-aware subset loading: "
        f"{type(sample)!r}"
    )


def _resolve_sample_modality_mask(
    modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray,
    *,
    dataset_size: int,
    num_groups: int,
    num_modalities: int,
    selected_indices: torch.Tensor,
    group_ids: Sequence[int],
    group_indices: Sequence[torch.Tensor],
) -> np.ndarray:
    raw_mask = _to_numpy_bool_array(modality_mask)

    if raw_mask.ndim == 1:
        if raw_mask.shape[0] == num_modalities:
            return np.broadcast_to(raw_mask.reshape(1, num_modalities), (len(selected_indices), num_modalities)).copy()
        if raw_mask.shape[0] == dataset_size * num_modalities:
            raw_mask = raw_mask.reshape(dataset_size, num_modalities)
        elif raw_mask.shape[0] == len(selected_indices) * num_modalities:
            raw_mask = raw_mask.reshape(len(selected_indices), num_modalities)
        elif raw_mask.shape[0] == len(group_ids) * num_modalities:
            raw_mask = raw_mask.reshape(len(group_ids), num_modalities)
        elif raw_mask.shape[0] == num_groups * num_modalities:
            raw_mask = raw_mask.reshape(num_groups, num_modalities)
        else:
            raise ValueError(
                "unsupported modality_mask length: "
                f"got {raw_mask.shape[0]}, expected one of "
                f"{num_modalities}, {dataset_size * num_modalities}, "
                f"{len(selected_indices) * num_modalities}, {len(group_ids) * num_modalities}, "
                f"{num_groups * num_modalities}"
            )

    if raw_mask.ndim != 2:
        raise ValueError(
            f"modality_mask must be 1D or 2D, got shape {tuple(raw_mask.shape)}"
        )
    if raw_mask.shape[1] != num_modalities:
        raise ValueError(
            f"modality_mask width mismatch: expected {num_modalities}, got {raw_mask.shape[1]}"
        )

    if raw_mask.shape[0] == len(selected_indices):
        return raw_mask.copy()
    if raw_mask.shape[0] == dataset_size:
        return raw_mask[selected_indices.cpu().numpy()].copy()
    if raw_mask.shape[0] == len(group_ids):
        expanded_masks: list[np.ndarray] = []
        for local_group_index, group_id in enumerate(group_ids):
            group_size = int(group_indices[group_id].numel())
            expanded_masks.append(
                np.broadcast_to(raw_mask[local_group_index], (group_size, num_modalities))
            )
        return np.concatenate(expanded_masks, axis=0).copy()
    if raw_mask.shape[0] == num_groups:
        expanded_masks = []
        for group_id in group_ids:
            group_size = int(group_indices[group_id].numel())
            expanded_masks.append(
                np.broadcast_to(raw_mask[group_id], (group_size, num_modalities))
            )
        return np.concatenate(expanded_masks, axis=0).copy()
    raise ValueError(
        "unsupported modality_mask shape: "
        f"{tuple(raw_mask.shape)} for dataset_size={dataset_size}, "
        f"selected_size={len(selected_indices)}, selected_groups={len(group_ids)}, "
        f"num_groups={num_groups}, num_modalities={num_modalities}"
    )


class ModalityMaskedSubset(Dataset):
    """Subset wrapper that can independently keep/drop modalities per selected sample."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        sample_modality_mask: Sequence[Sequence[bool]] | np.ndarray | torch.Tensor,
        modality_names: Sequence[str],
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.modality_names = tuple(str(name) for name in modality_names)
        self.sample_modality_mask = _to_numpy_bool_array(sample_modality_mask)
        if self.sample_modality_mask.ndim != 2:
            raise ValueError("sample_modality_mask must have shape [num_samples, num_modalities]")
        if self.sample_modality_mask.shape != (len(self.indices), len(self.modality_names)):
            raise ValueError(
                "sample_modality_mask shape mismatch: "
                f"expected {(len(self.indices), len(self.modality_names))}, "
                f"got {tuple(self.sample_modality_mask.shape)}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_index = self.indices[index]
        keep_mask = self.sample_modality_mask[index]
        sample = self.dataset[dataset_index]

        if (
            isinstance(sample, tuple)
            and len(sample) == 4
            and isinstance(sample[0], dict)
        ):
            modalities, original_mask, label, sample_id = sample
            masked_modalities: dict[str, Any] = {}
            for modality_index, modality_name in enumerate(self.modality_names):
                value = _clone_sample_value(modalities[modality_name])
                masked_modalities[modality_name] = value if keep_mask[modality_index] else _zero_like(value)
            resolved_mask = torch.as_tensor(keep_mask, dtype=torch.bool)
            if original_mask is not None:
                base_mask = torch.as_tensor(original_mask, dtype=torch.bool).view(-1)
                if base_mask.shape[0] != len(self.modality_names):
                    raise ValueError(
                        "original modality mask width mismatch: "
                        f"expected {len(self.modality_names)}, got {base_mask.shape[0]}"
                    )
                resolved_mask = base_mask & resolved_mask
            return masked_modalities, resolved_mask, label, sample_id

        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            items = list(sample)
            num_modalities = len(items) - 1
            if num_modalities != len(self.modality_names):
                raise ValueError(
                    f"sample modality count mismatch: expected {len(self.modality_names)}, got {num_modalities}"
                )
            for modality_index in range(num_modalities):
                value = _clone_sample_value(items[modality_index])
                items[modality_index] = value if keep_mask[modality_index] else _zero_like(value)
            return tuple(items)

        raise TypeError(
            "unsupported dataset sample format for modality masking: "
            f"{type(sample)!r}"
        )


def partition_indices(
    num_samples: int,
    num_groups: int = DEFAULT_NUM_GROUPS,
    seed: int | None = 42,
) -> list[torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    chunks = torch.chunk(permutation, min(num_groups, num_samples))
    return [chunk.clone().cpu() for chunk in chunks if len(chunk) > 0]


def _flatten_modality_tensor(values: torch.Tensor) -> torch.Tensor:
    tensor = values.detach().cpu()
    if tensor.ndim == 0:
        tensor = tensor.view(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.to(dtype=torch.float32)


def _tensor_dataset_group_features(dataset: Dataset) -> torch.Tensor | None:
    tensors = getattr(dataset, "tensors", None)
    if not isinstance(tensors, tuple) or len(tensors) < 2:
        return None

    modality_tensors = tensors[:-1]
    if not modality_tensors:
        return None

    flattened: list[torch.Tensor] = []
    for modality_tensor in modality_tensors:
        if not isinstance(modality_tensor, torch.Tensor):
            return None
        flattened.append(F.normalize(_flatten_modality_tensor(modality_tensor), dim=1))
    return torch.cat(flattened, dim=1)


def _sample_modalities_to_feature_row(sample: Any) -> torch.Tensor:
    if (
        isinstance(sample, tuple)
        and len(sample) == 4
        and isinstance(sample[0], dict)
    ):
        modalities = sample[0]
        feature_parts: list[torch.Tensor] = []
        for modality_name in sorted(modalities):
            value = modalities[modality_name]
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                raise TypeError(
                    "unsupported modality value type for clustering: "
                    f"{type(value)!r}"
                )
            value_tensor = torch.as_tensor(value, dtype=torch.float32).reshape(1, -1)
            feature_parts.append(F.normalize(value_tensor, dim=1))
        return torch.cat(feature_parts, dim=1).squeeze(0).cpu()

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        feature_parts = []
        for value in sample[:-1]:
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                raise TypeError(
                    "unsupported modality value type for clustering: "
                    f"{type(value)!r}"
                )
            value_tensor = torch.as_tensor(value, dtype=torch.float32).reshape(1, -1)
            feature_parts.append(F.normalize(value_tensor, dim=1))
        return torch.cat(feature_parts, dim=1).squeeze(0).cpu()

    raise TypeError(
        "unsupported dataset sample format for clustering-based grouping: "
        f"{type(sample)!r}"
    )


def dataset_group_features(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        base_features = dataset_group_features(dataset.dataset)
        subset_indices = torch.as_tensor(dataset.indices, dtype=torch.long).view(-1).cpu()
        return base_features.index_select(0, subset_indices)

    tensor_features = _tensor_dataset_group_features(dataset)
    if tensor_features is not None:
        return tensor_features

    if len(dataset) <= 0:
        raise ValueError("cannot build group features from an empty dataset")
    rows = [_sample_modalities_to_feature_row(dataset[index]) for index in range(len(dataset))]
    return torch.stack(rows, dim=0)


def cluster_indices_by_features(
    features: torch.Tensor,
    num_groups: int = DEFAULT_NUM_GROUPS,
    seed: int | None = 42,
) -> list[torch.Tensor]:
    num_samples = int(features.shape[0])
    if num_samples <= 0:
        raise ValueError("features must contain at least one sample")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    resolved_num_groups = min(int(num_groups), num_samples)
    if resolved_num_groups >= num_samples:
        return [torch.tensor([index], dtype=torch.long) for index in range(num_samples)]

    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "group_method='kmeans' requires scikit-learn to be installed"
        ) from exc

    feature_matrix = torch.as_tensor(features, dtype=torch.float32).cpu().numpy()
    kmeans = KMeans(
        n_clusters=resolved_num_groups,
        random_state=seed,
        n_init=10,
    )
    assignments = np.asarray(kmeans.fit_predict(feature_matrix), dtype=np.int64)
    group_indices: list[torch.Tensor] = []
    for cluster_id in range(resolved_num_groups):
        member_indices = np.flatnonzero(assignments == cluster_id)
        if member_indices.size == 0:
            raise RuntimeError(f"kmeans produced an empty cluster: {cluster_id}")
        group_indices.append(torch.as_tensor(member_indices, dtype=torch.long).cpu())
    return group_indices


def save_group_indices(
    *,
    group_indices: Sequence[torch.Tensor],
    dataset_name: str,
    dataset_size: int,
    num_groups: int,
    seed: int | None,
    group_method: str,
    cache_path: str | Path,
) -> Path:
    resolved_cache_path = Path(cache_path)
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name,
        "dataset_size": int(dataset_size),
        "num_groups": int(num_groups),
        "seed": seed,
        "group_method": str(group_method),
        "group_indices": [indices.clone().cpu().long() for indices in group_indices],
    }
    torch.save(payload, resolved_cache_path)
    return resolved_cache_path


def load_group_indices(cache_path: str | Path) -> dict[str, object]:
    payload = torch.load(Path(cache_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid group cache payload: {cache_path}")
    if "group_indices" not in payload:
        raise ValueError(f"missing group_indices in cache: {cache_path}")
    payload["group_indices"] = [torch.as_tensor(indices, dtype=torch.long).cpu() for indices in payload["group_indices"]]
    return payload


def resolve_group_indices(
    *,
    dataset: Dataset,
    dataset_name: str | None = None,
    num_groups: int = DEFAULT_NUM_GROUPS,
    seed: int | None = 42,
    group_method: str = DEFAULT_GROUP_METHOD,
    cache_dir: str | Path | None = None,
) -> tuple[list[torch.Tensor], GroupPartitionMetadata]:
    resolved_dataset_name = _sanitize_dataset_name(dataset_name or _infer_dataset_name(dataset))
    dataset_size = len(dataset)
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    resolved_num_groups = min(int(num_groups), max(dataset_size, 1))
    seed_token = "none" if seed is None else str(seed)
    method_token = _sanitize_dataset_name(group_method)
    cache_path = (
        resolved_cache_dir
        / f"{resolved_dataset_name}_n{dataset_size}_g{resolved_num_groups}_{method_token}_seed{seed_token}.pt"
    )

    if cache_path.exists():
        payload = load_group_indices(cache_path)
        cached_size = int(payload["dataset_size"])
        cached_groups = int(payload["num_groups"])
        cached_seed = payload.get("seed")
        cached_method = str(payload.get("group_method", DEFAULT_GROUP_METHOD))
        if (
            cached_size != dataset_size
            or cached_groups != resolved_num_groups
            or cached_seed != seed
            or cached_method != group_method
        ):
            raise ValueError(
                "cached group indices do not match the requested configuration: "
                f"path={cache_path}, cached_size={cached_size}, requested_size={dataset_size}, "
                f"cached_groups={cached_groups}, requested_groups={resolved_num_groups}, "
                f"cached_seed={cached_seed}, requested_seed={seed}, "
                f"cached_method={cached_method}, requested_method={group_method}"
            )
        group_indices = payload["group_indices"]
    else:
        if group_method == "random":
            group_indices = partition_indices(dataset_size, resolved_num_groups, seed)
        elif group_method == "kmeans":
            group_indices = cluster_indices_by_features(
                dataset_group_features(dataset),
                resolved_num_groups,
                seed,
            )
        else:
            raise ValueError(f"unsupported group_method: {group_method}")
        save_group_indices(
            group_indices=group_indices,
            dataset_name=resolved_dataset_name,
            dataset_size=dataset_size,
            num_groups=resolved_num_groups,
            seed=seed,
            group_method=group_method,
            cache_path=cache_path,
        )

    metadata = GroupPartitionMetadata(
        dataset_name=resolved_dataset_name,
        dataset_size=dataset_size,
        num_groups=len(group_indices),
        seed=seed,
        group_method=str(group_method),
        cache_path=cache_path,
    )
    return group_indices, metadata


def subset_indices_from_group_ids(
    group_indices: Sequence[torch.Tensor],
    group_ids: Sequence[int],
) -> torch.Tensor:
    if not group_indices:
        raise ValueError("group_indices must not be empty")

    resolved_group_ids = [int(group_id) for group_id in group_ids]
    if not resolved_group_ids:
        return group_indices[0].clone().cpu().long()

    selected: list[torch.Tensor] = []
    for group_id in resolved_group_ids:
        if group_id < 0 or group_id >= len(group_indices):
            raise IndexError(f"group id out of range: {group_id}")
        selected.append(group_indices[group_id].clone().cpu().long())
    return torch.cat(selected, dim=0)


def subset_indices_from_weights(
    group_indices: Sequence[torch.Tensor],
    subset_weights: Sequence[int | float | bool] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(subset_weights, torch.Tensor):
        weight_list = subset_weights.detach().cpu().view(-1).tolist()
    else:
        weight_list = list(subset_weights)
    if len(weight_list) != len(group_indices):
        raise ValueError(
            f"subset_weights length mismatch: expected {len(group_indices)}, got {len(weight_list)}"
        )
    group_ids = [index for index, weight in enumerate(weight_list) if float(weight) > 0.0]
    return subset_indices_from_group_ids(group_indices, group_ids)


def _clone_loader_kwargs(
    base_loader: DataLoader,
    *,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool | None,
) -> dict[str, object]:
    resolved_batch_size = batch_size if batch_size is not None else base_loader.batch_size
    if resolved_batch_size is None:
        raise ValueError("subset loader requires an explicit batch_size when base_loader.batch_size is None")

    loader_kwargs: dict[str, object] = {
        "batch_size": resolved_batch_size,
        "shuffle": shuffle,
        "num_workers": base_loader.num_workers,
        "collate_fn": base_loader.collate_fn,
        "pin_memory": base_loader.pin_memory,
        "drop_last": base_loader.drop_last if drop_last is None else drop_last,
        "timeout": base_loader.timeout,
        "worker_init_fn": base_loader.worker_init_fn,
        "multiprocessing_context": base_loader.multiprocessing_context,
        "generator": base_loader.generator,
    }

    if base_loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = base_loader.persistent_workers
        if getattr(base_loader, "prefetch_factor", None) is not None:
            loader_kwargs["prefetch_factor"] = base_loader.prefetch_factor
    if getattr(base_loader, "pin_memory_device", ""):
        loader_kwargs["pin_memory_device"] = base_loader.pin_memory_device
    return loader_kwargs


class GroupedSubsetLoader:
    """Create cached group partitions and build subset train dataloaders from them."""

    def __init__(
        self,
        train_loader: DataLoader,
        *,
        dataset_name: str | None = None,
        num_groups: int = DEFAULT_NUM_GROUPS,
        seed: int | None = 42,
        group_method: str = DEFAULT_GROUP_METHOD,
        cache_dir: str | Path | None = None,
        default_group_batch_size: int = DEFAULT_GROUP_BATCH_SIZE,
    ) -> None:
        if train_loader.dataset is None:
            raise ValueError("train_loader.dataset must not be None")
        if default_group_batch_size <= 0:
            raise ValueError("default_group_batch_size must be positive")

        self.train_loader = train_loader
        self.dataset = train_loader.dataset
        self.group_indices, self.metadata = resolve_group_indices(
            dataset=self.dataset,
            dataset_name=dataset_name,
            num_groups=num_groups,
            seed=seed,
            group_method=group_method,
            cache_dir=cache_dir,
        )
        self.modality_metadata = _infer_modality_metadata(self.dataset)
        self.default_group_batch_size = int(default_group_batch_size)

    @property
    def cache_path(self) -> Path:
        return self.metadata.cache_path

    @property
    def num_groups(self) -> int:
        return len(self.group_indices)

    @property
    def group_sizes(self) -> list[int]:
        return [int(indices.numel()) for indices in self.group_indices]

    @property
    def modality_names(self) -> tuple[str, ...]:
        return self.modality_metadata.modality_names

    @property
    def num_modalities(self) -> int:
        return self.modality_metadata.num_modalities

    def get_subset_indices(self, group_ids: Sequence[int]) -> torch.Tensor:
        return subset_indices_from_group_ids(self.group_indices, group_ids)

    def get_subset_indices_from_weights(
        self,
        subset_weights: Sequence[int | float | bool] | torch.Tensor,
    ) -> torch.Tensor:
        return subset_indices_from_weights(self.group_indices, subset_weights)

    def build_subset_dataset(
        self,
        group_ids: Sequence[int],
        *,
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray | None = None,
    ) -> Dataset:
        indices = self.get_subset_indices(group_ids)
        if modality_mask is None:
            return Subset(self.dataset, indices.tolist())
        sample_modality_mask = _resolve_sample_modality_mask(
            modality_mask,
            dataset_size=len(self.dataset),
            num_groups=self.num_groups,
            num_modalities=self.num_modalities,
            selected_indices=indices,
            group_ids=group_ids,
            group_indices=self.group_indices,
        )
        return ModalityMaskedSubset(
            self.dataset,
            indices.tolist(),
            sample_modality_mask=sample_modality_mask,
            modality_names=self.modality_names,
        )

    def build_subset_dataset_from_weights(
        self,
        subset_weights: Sequence[int | float | bool] | torch.Tensor,
        *,
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray | None = None,
    ) -> Dataset:
        indices = self.get_subset_indices_from_weights(subset_weights)
        if modality_mask is None:
            return Subset(self.dataset, indices.tolist())
        selected_group_ids = [
            index
            for index, weight in enumerate(subset_weights.detach().cpu().view(-1).tolist() if isinstance(subset_weights, torch.Tensor) else list(subset_weights))
            if float(weight) > 0.0
        ]
        sample_modality_mask = _resolve_sample_modality_mask(
            modality_mask,
            dataset_size=len(self.dataset),
            num_groups=self.num_groups,
            num_modalities=self.num_modalities,
            selected_indices=indices,
            group_ids=selected_group_ids,
            group_indices=self.group_indices,
        )
        return ModalityMaskedSubset(
            self.dataset,
            indices.tolist(),
            sample_modality_mask=sample_modality_mask,
            modality_names=self.modality_names,
        )

    def build_subset_dataset_from_indices(
        self,
        indices: Sequence[int] | torch.Tensor | np.ndarray,
        *,
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray | None = None,
    ) -> Dataset:
        resolved_indices = torch.as_tensor(indices, dtype=torch.long).view(-1).cpu()
        if modality_mask is None:
            return Subset(self.dataset, resolved_indices.tolist())

        raw_mask = _to_numpy_bool_array(modality_mask)
        if raw_mask.ndim == 1:
            if raw_mask.shape[0] == self.num_modalities:
                sample_modality_mask = np.broadcast_to(
                    raw_mask.reshape(1, self.num_modalities),
                    (int(resolved_indices.numel()), self.num_modalities),
                ).copy()
            elif raw_mask.shape[0] == int(resolved_indices.numel()) * self.num_modalities:
                sample_modality_mask = raw_mask.reshape(int(resolved_indices.numel()), self.num_modalities).copy()
            elif raw_mask.shape[0] == len(self.dataset) * self.num_modalities:
                sample_modality_mask = raw_mask.reshape(len(self.dataset), self.num_modalities)[
                    resolved_indices.numpy()
                ].copy()
            else:
                raise ValueError(
                    "unsupported modality_mask length for explicit indices: "
                    f"got {raw_mask.shape[0]}, expected one of "
                    f"{self.num_modalities}, {int(resolved_indices.numel()) * self.num_modalities}, "
                    f"{len(self.dataset) * self.num_modalities}"
                )
        elif raw_mask.ndim == 2:
            if raw_mask.shape[1] != self.num_modalities:
                raise ValueError(
                    f"modality_mask width mismatch: expected {self.num_modalities}, got {raw_mask.shape[1]}"
                )
            if raw_mask.shape[0] == int(resolved_indices.numel()):
                sample_modality_mask = raw_mask.copy()
            elif raw_mask.shape[0] == len(self.dataset):
                sample_modality_mask = raw_mask[resolved_indices.numpy()].copy()
            else:
                raise ValueError(
                    "unsupported modality_mask shape for explicit indices: "
                    f"{tuple(raw_mask.shape)} for selected_size={int(resolved_indices.numel())}, "
                    f"dataset_size={len(self.dataset)}, num_modalities={self.num_modalities}"
                )
        else:
            raise ValueError(
                f"modality_mask must be 1D or 2D, got shape {tuple(raw_mask.shape)}"
            )

        return ModalityMaskedSubset(
            self.dataset,
            resolved_indices.tolist(),
            sample_modality_mask=sample_modality_mask,
            modality_names=self.modality_names,
        )

    def build_subset_loader(
        self,
        group_ids: Sequence[int],
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray | None = None,
    ) -> DataLoader:
        subset_dataset = self.build_subset_dataset(group_ids, modality_mask=modality_mask)
        loader_kwargs = _clone_loader_kwargs(
            self.train_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(subset_dataset, **loader_kwargs)

    def build_subset_loader_from_weights(
        self,
        subset_weights: Sequence[int | float | bool] | torch.Tensor,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray | None = None,
    ) -> DataLoader:
        subset_dataset = self.build_subset_dataset_from_weights(
            subset_weights,
            modality_mask=modality_mask,
        )
        loader_kwargs = _clone_loader_kwargs(
            self.train_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(subset_dataset, **loader_kwargs)

    def build_subset_loader_from_indices(
        self,
        indices: Sequence[int] | torch.Tensor | np.ndarray,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray | None = None,
    ) -> DataLoader:
        subset_dataset = self.build_subset_dataset_from_indices(
            indices,
            modality_mask=modality_mask,
        )
        loader_kwargs = _clone_loader_kwargs(
            self.train_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(subset_dataset, **loader_kwargs)

    def build_modality_subset_dataset(
        self,
        group_ids: Sequence[int],
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray,
    ) -> Dataset:
        return self.build_subset_dataset(group_ids, modality_mask=modality_mask)

    def build_modality_subset_loader(
        self,
        group_ids: Sequence[int],
        modality_mask: Sequence[int | float | bool] | torch.Tensor | np.ndarray,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
    ) -> DataLoader:
        return self.build_subset_loader(
            group_ids,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            modality_mask=modality_mask,
        )

    def iter_group_batches(
        self,
        *,
        group_batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> Iterator[list[int]]:
        resolved_group_batch_size = (
            self.default_group_batch_size if group_batch_size is None else int(group_batch_size)
        )
        if resolved_group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")

        order = torch.arange(self.num_groups)
        if shuffle:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
            order = order[torch.randperm(self.num_groups, generator=generator)]

        total_batches = math.ceil(self.num_groups / resolved_group_batch_size)
        for batch_index in range(total_batches):
            start = batch_index * resolved_group_batch_size
            end = min(start + resolved_group_batch_size, self.num_groups)
            if drop_last and end - start < resolved_group_batch_size:
                continue
            yield order[start:end].tolist()

    def iter_subset_loaders(
        self,
        *,
        group_batch_size: int | None = None,
        shuffle_groups: bool = True,
        drop_last_groups: bool = False,
        group_seed: int | None = None,
        subset_batch_size: int | None = None,
        subset_shuffle: bool = True,
        subset_drop_last: bool | None = None,
    ) -> Iterator[tuple[list[int], DataLoader]]:
        for group_ids in self.iter_group_batches(
            group_batch_size=group_batch_size,
            shuffle=shuffle_groups,
            drop_last=drop_last_groups,
            seed=group_seed,
        ):
            yield group_ids, self.build_subset_loader(
                group_ids,
                batch_size=subset_batch_size,
                shuffle=subset_shuffle,
                drop_last=subset_drop_last,
            )


def build_grouped_subset_loader(
    train_loader: DataLoader,
    *,
    dataset_name: str | None = None,
    num_groups: int = DEFAULT_NUM_GROUPS,
    seed: int | None = 42,
    group_method: str = DEFAULT_GROUP_METHOD,
    cache_dir: str | Path | None = None,
    default_group_batch_size: int = DEFAULT_GROUP_BATCH_SIZE,
) -> GroupedSubsetLoader:
    return GroupedSubsetLoader(
        train_loader,
        dataset_name=dataset_name,
        num_groups=num_groups,
        seed=seed,
        group_method=group_method,
        cache_dir=cache_dir,
        default_group_batch_size=default_group_batch_size,
    )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_GROUP_BATCH_SIZE",
    "DEFAULT_GROUP_METHOD",
    "DEFAULT_NUM_GROUPS",
    "GroupPartitionMetadata",
    "GroupedSubsetLoader",
    "ModalityMaskedSubset",
    "ModalitySubsetMetadata",
    "build_grouped_subset_loader",
    "cluster_indices_by_features",
    "dataset_group_features",
    "load_group_indices",
    "partition_indices",
    "resolve_group_indices",
    "save_group_indices",
    "subset_indices_from_group_ids",
    "subset_indices_from_weights",
]
