from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ArrayLike = np.ndarray | torch.Tensor | Sequence[float]


def _to_tensor(value: ArrayLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


class MultimodalDataset(Dataset):
    """Simple dataset that returns `(modality_1, ..., modality_k, label)` tuples."""

    def __init__(self, modalities: Sequence[ArrayLike], labels: ArrayLike):
        if not modalities:
            raise ValueError("modalities cannot be empty")

        self.modalities = [_to_tensor(modality) for modality in modalities]
        self.labels = _to_tensor(labels)

        length = len(self.labels)
        for idx, modality in enumerate(self.modalities):
            if len(modality) != length:
                raise ValueError(
                    f"modality {idx} has length {len(modality)} but labels have length {length}"
                )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(modality[index] for modality in self.modalities) + (self.labels[index],)


class SyntheticDataset(Dataset):
    """Dataset wrapper for the pickled synthetic-data format used in PID."""

    def __init__(self, data: dict[str, np.ndarray], keys: Sequence[str], modalities: Sequence[int]):
        if not keys:
            raise ValueError("keys cannot be empty")
        self.data = data
        self.keys = list(keys)
        self.modalities = list(modalities)

    def __len__(self) -> int:
        return int(len(self.data[self.keys[-1]]))

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        sample: list[torch.Tensor] = []
        for offset, _modality in enumerate(self.modalities):
            key = self.keys[offset]
            if key not in self.data:
                raise KeyError(f"key '{key}' not found in synthetic dataset split")
            sample.append(torch.as_tensor(self.data[key][index]))
        sample.append(torch.as_tensor(self.data[self.keys[-1]][index]))
        return sample


def process_synthetic_batch(inputs: Sequence[Sequence[torch.Tensor]]) -> list[torch.Tensor]:
    processed_input: list[torch.Tensor] = []
    feature_count = len(inputs[0]) - 1

    for feature_index in range(feature_count):
        stacked = [sample[feature_index] for sample in inputs]
        processed_input.append(torch.stack(stacked))

    labels = [sample[-1] for sample in inputs]
    processed_input.append(torch.as_tensor(labels).view(len(inputs)))
    return processed_input


def save_pickle_dataset(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle_dataset(path: str | Path | Sequence[str | Path]) -> dict:
    if isinstance(path, (list, tuple)):
        datasets = [_load_single_pickle(candidate) for candidate in path]
        if not datasets:
            raise ValueError("path list cannot be empty")
        merged = {"train": {}}
        for key in datasets[0]["train"]:
            segments = []
            for index, dataset in enumerate(datasets):
                values = dataset["train"][key]
                if index == 0:
                    segments.append(values)
                else:
                    segments.append(values[: max(int(0.1 * len(values)), 1)])
            merged["train"][key] = np.concatenate(segments, axis=0)
        return merged
    return _load_single_pickle(path)


def _load_single_pickle(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def get_synthetic_dataloaders(
    path: str | Path | Sequence[str | Path],
    *,
    keys: Sequence[str] = ("0", "1", "label"),
    modalities: Sequence[int] = (0, 1),
    batch_size: int = 32,
    num_workers: int = 0,
    test_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None, DataLoader]:
    data = load_pickle_dataset(path)

    if test_path is not None:
        test_data = load_pickle_dataset(test_path)
        data["valid1"] = test_data["valid1"]
        data["valid2"] = test_data["valid2"]
        data["test"] = test_data["test"]

    label_key = keys[-1]
    train_loader = DataLoader(
        SyntheticDataset(data["train"], keys, modalities=modalities),
        shuffle=True,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=process_synthetic_batch,
    )

    if "valid" in data:
        valid_loader_1 = DataLoader(
            SyntheticDataset(data["valid"], keys, modalities=modalities),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=process_synthetic_batch,
        )
        valid_loader_2 = None
    else:
        valid_loader_1 = DataLoader(
            SyntheticDataset(data["valid1"], keys, modalities=modalities),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=process_synthetic_batch,
        )
        valid_loader_2 = DataLoader(
            SyntheticDataset(data["valid2"], keys, modalities=modalities),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=process_synthetic_batch,
        )

    test_loader = DataLoader(
        SyntheticDataset(data["test"], keys, modalities=modalities),
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=process_synthetic_batch,
    )

    if label_key not in data["train"] or label_key not in data["test"]:
        raise KeyError(f"label key '{label_key}' not found in synthetic dataset")

    return train_loader, valid_loader_1, valid_loader_2, test_loader


__all__ = [
    "ArrayLike",
    "MultimodalDataset",
    "SyntheticDataset",
    "get_synthetic_dataloaders",
    "load_pickle_dataset",
    "process_synthetic_batch",
    "save_pickle_dataset",
]
