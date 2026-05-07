"""Standardized dataloaders for common datasets used under MultiBench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets as tv_datasets
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = Path(
    os.environ.get("MULTIBENCH_DATASETS_ROOT", REPO_ROOT / "data" / "datasets")
).expanduser()

IMDB_PATH = DATASETS_ROOT / "multimodal_imdb.hdf5"
MOSI_PATH = DATASETS_ROOT / "mosi" / "mosi_data.pkl"
HUMOR_PATH = DATASETS_ROOT / "humor" / "humor.pkl"
SARCASM_PATH = DATASETS_ROOT / "sarcasm" / "sarcasm.pkl"
SARCAM_PATH = SARCASM_PATH
MOSEI_PATH = DATASETS_ROOT / "mosei" / "mosei_senti_data.pkl"
AVMNIST_ROOT = DATASETS_ROOT / "avmnist"
FOOD101_ROOT = DATASETS_ROOT / "food101"
OXFORD_PET_ROOT = DATASETS_ROOT / "oxford_pet"

SUPPORTED_DATASETS = (
    "imdb",
    "mosi",
    "humor",
    "sarcasm",
    "sarcam",
    "mosei",
    "avmnist",
    "food101",
    "oxford_pet",
    "oxford-pet",
    "oxford pet",
)

_DATASET_ALIASES = {
    "imdb": "imdb",
    "mosi": "mosi",
    "humor": "humor",
    "sarcasm": "sarcasm",
    "sarcam": "sarcasm",
    "scarcam": "sarcasm",
    "mosei": "mosei",
    "avmnist": "avmnist",
    "food101": "food101",
    "oxford_pet": "oxford_pet",
    "oxford-pet": "oxford_pet",
    "oxford pet": "oxford_pet",
    "oxfordpet": "oxford_pet",
}

_IMDB_SPLITS = {
    "train": (0, 15552),
    "valid": (15552, 18160),
    "test": (18160, 25959),
}


@dataclass(frozen=True)
class StandardDatasetBundle:
    name: str
    task: str
    modality_names: tuple[str, ...]
    train: Dataset
    valid: Dataset
    test: Dataset
    num_classes: int | None = None
    collate_fn: Callable[..., Any] | None = None


class ArrayBackedDataset(Dataset):
    """Dataset wrapper for array-backed multimodal splits."""

    def __init__(
        self,
        *,
        modalities: dict[str, np.ndarray],
        labels: np.ndarray,
        ids: list[str] | None = None,
        mask: np.ndarray | None = None,
        task: str = "binary",
    ) -> None:
        if not modalities:
            raise ValueError("modalities must not be empty")

        self.modalities = {name: np.asarray(values) for name, values in modalities.items()}
        self.labels = np.asarray(labels)
        self.mask = None if mask is None else np.asarray(mask)
        self.task = task

        sizes = {name: values.shape[0] for name, values in self.modalities.items()}
        sizes["labels"] = self.labels.shape[0]
        if len(set(sizes.values())) != 1:
            raise ValueError(f"inconsistent split sizes: {sizes}")

        self.size = self.labels.shape[0]
        self.ids = ids if ids is not None else [f"sample_{index}" for index in range(self.size)]

    def __len__(self) -> int:
        return int(self.size)

    def __getitem__(self, index: int):
        modalities = {
            name: torch.as_tensor(values[index], dtype=torch.float32)
            for name, values in self.modalities.items()
        }
        mask = None if self.mask is None else torch.as_tensor(self.mask[index], dtype=torch.bool)
        if self.task == "multiclass":
            label = torch.as_tensor(self.labels[index], dtype=torch.long)
        else:
            label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return modalities, mask, label, self.ids[index]


class TorchvisionDatasetAdapter(Dataset):
    """Adapter that exposes torchvision classification datasets in the same format."""

    def __init__(self, dataset: Dataset, *, split_name: str) -> None:
        self.dataset = dataset
        self.split_name = split_name

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        modalities = {"image": image}
        return modalities, None, torch.as_tensor(label, dtype=torch.long), f"{self.split_name}_{index}"


class StandardizedDatasetAdapter(Dataset):
    """Wrap an existing dataset and adapt samples to the standardized format."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_adapter: Callable[[Any, int], tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor, str]],
    ) -> None:
        self.dataset = dataset
        self.sample_adapter = sample_adapter

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int):
        return self.sample_adapter(self.dataset[index], index)


def standardized_collate(
    items: list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor, str]]
):
    names = list(items[0][0].keys())
    modalities = {
        name: torch.stack([row[0][name] for row in items], dim=0)
        for name in names
    }
    masks = None if items[0][1] is None else torch.stack([row[1] for row in items], dim=0)
    labels = torch.stack([row[2] for row in items], dim=0)
    ids = [row[3] for row in items]
    return modalities, masks, labels, ids


def build_dataloaders(
    dataset_name: str,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    train_shuffle: bool = True,
    pin_memory: bool = True,
    max_seq_len: int = 50,
    image_size: int = 224,
    use_vgg_features: bool = True,
    valid_ratio: float = 0.1,
    seed: int = 42,
    download: bool = False,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardDatasetBundle]:
    """Build train/valid/test dataloaders with a standardized batch format."""

    bundle = load_dataset(
        dataset_name,
        max_seq_len=max_seq_len,
        image_size=image_size,
        use_vgg_features=use_vgg_features,
        valid_ratio=valid_ratio,
        seed=seed,
        download=download,
        max_samples=max_samples,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": bundle.collate_fn or standardized_collate,
    }
    train_loader = DataLoader(bundle.train, shuffle=train_shuffle, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(bundle.valid, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(bundle.test, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader, bundle


def load_dataset(
    dataset_name: str,
    *,
    max_seq_len: int = 50,
    image_size: int = 224,
    use_vgg_features: bool = True,
    valid_ratio: float = 0.1,
    seed: int = 42,
    download: bool = False,
    max_samples: int | None = None,
) -> StandardDatasetBundle:
    """Load a dataset bundle with unified split metadata."""

    canonical_name = _canonicalize_dataset_name(dataset_name)
    if canonical_name == "imdb":
        return _load_imdb_bundle(use_vgg_features=use_vgg_features, max_samples=max_samples)
    if canonical_name == "mosi":
        return _load_mosi_bundle(max_seq_len=max_seq_len, max_samples=max_samples)
    if canonical_name == "humor":
        return _load_affect_bundle(HUMOR_PATH, dataset_name="humor", max_seq_len=max_seq_len, max_samples=max_samples)
    if canonical_name == "sarcasm":
        return _load_affect_bundle(SARCASM_PATH, dataset_name="sarcasm", max_seq_len=max_seq_len, max_samples=max_samples)
    if canonical_name == "mosei":
        return _load_mosei_bundle(max_seq_len=max_seq_len, max_samples=max_samples)
    if canonical_name == "avmnist":
        return _load_avmnist_bundle(max_samples=max_samples)
    if canonical_name == "food101":
        return _load_food101_bundle(
            image_size=image_size,
            valid_ratio=valid_ratio,
            seed=seed,
            download=download,
            max_samples=max_samples,
        )
    if canonical_name == "oxford_pet":
        return _load_oxford_pet_bundle(
            image_size=image_size,
            valid_ratio=valid_ratio,
            seed=seed,
            download=download,
            max_samples=max_samples,
        )
    raise ValueError(f"unsupported dataset: {dataset_name}")


def _canonicalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower().replace("-", "_")
    key = " ".join(key.split())
    key = key.replace(" ", "_")
    if key not in _DATASET_ALIASES:
        raise ValueError(
            f"unsupported dataset '{dataset_name}'. supported values: {', '.join(SUPPORTED_DATASETS)}"
        )
    return _DATASET_ALIASES[key]


def _normalize_ids(raw_ids: Any, *, split_name: str, expected_size: int) -> list[str]:
    if raw_ids is None:
        return [f"{split_name}_{index}" for index in range(expected_size)]

    normalized: list[str] = []
    for index, item in enumerate(raw_ids):
        if isinstance(item, np.ndarray):
            if item.size == 0:
                normalized.append(f"{split_name}_{index}")
            else:
                normalized.append(str(item.reshape(-1)[0]))
        elif isinstance(item, (list, tuple)):
            normalized.append(str(item[0] if item else f"{split_name}_{index}"))
        else:
            normalized.append(str(item))
    return normalized


def _slice_limit(values: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None:
        return values
    return values[:max_samples]


def _limit_ids(ids: list[str], max_samples: int | None) -> list[str]:
    if max_samples is None:
        return ids
    return ids[:max_samples]


def _load_affect_bundle(
    path: Path,
    *,
    dataset_name: str,
    max_seq_len: int,
    max_samples: int | None,
) -> StandardDatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {path}")

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    def _build(split_name: str) -> ArrayBackedDataset:
        split = payload[split_name]
        text = np.asarray(split["text"], dtype=np.float32)
        audio = np.asarray(split["audio"], dtype=np.float32)
        video = np.asarray(split["vision"], dtype=np.float32)
        labels = np.asarray(split["labels"], dtype=np.float32).reshape(text.shape[0], -1)[:, 0]

        if max_seq_len is not None:
            text = text[:, :max_seq_len]
            audio = audio[:, :max_seq_len]
            video = video[:, :max_seq_len]

        text[~np.isfinite(text)] = 0.0
        audio[~np.isfinite(audio)] = 0.0
        video[~np.isfinite(video)] = 0.0

        mask = np.abs(text).sum(axis=-1) > 0
        keep = mask.any(axis=1)
        ids = _normalize_ids(split.get("id"), split_name=split_name, expected_size=text.shape[0])

        text = text[keep]
        audio = audio[keep]
        video = video[keep]
        labels = (labels[keep] > 0).astype(np.float32)
        mask = mask[keep].astype(bool)
        ids = [item for item, flag in zip(ids, keep.tolist()) if flag]

        text = _slice_limit(text, max_samples)
        audio = _slice_limit(audio, max_samples)
        video = _slice_limit(video, max_samples)
        labels = _slice_limit(labels, max_samples)
        mask = _slice_limit(mask, max_samples)
        ids = _limit_ids(ids, max_samples)

        return ArrayBackedDataset(
            modalities={"text": text, "audio": audio, "video": video},
            labels=labels,
            ids=ids,
            mask=mask,
            task="binary",
        )

    return StandardDatasetBundle(
        name=dataset_name,
        task="binary",
        modality_names=("text", "audio", "video"),
        train=_build("train"),
        valid=_build("valid"),
        test=_build("test"),
        num_classes=2,
        collate_fn=standardized_collate,
    )


def _load_imdb_bundle(
    *,
    use_vgg_features: bool,
    max_samples: int | None,
) -> StandardDatasetBundle:
    if not IMDB_PATH.exists():
        raise FileNotFoundError(f"imdb file not found: {IMDB_PATH}")

    image_key = "vgg_features" if use_vgg_features else "images"
    with h5py.File(IMDB_PATH, "r") as handle:
        split_datasets: dict[str, ArrayBackedDataset] = {}
        for split_name, (start, stop) in _IMDB_SPLITS.items():
            text = np.asarray(handle["features"][start:stop], dtype=np.float32)
            image = np.asarray(handle[image_key][start:stop], dtype=np.float32)
            if image.ndim > 2:
                image = image.reshape(image.shape[0], -1)
            labels = np.asarray(handle["genres"][start:stop], dtype=np.float32)
            raw_ids = np.asarray(handle["imdb_ids"][start:stop])
            ids = [
                item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                for item in raw_ids
            ]

            text = _slice_limit(text, max_samples)
            image = _slice_limit(image, max_samples)
            labels = _slice_limit(labels, max_samples)
            ids = _limit_ids(ids, max_samples)

            split_datasets[split_name] = ArrayBackedDataset(
                modalities={"text": text, "image": image},
                labels=labels,
                ids=ids,
                mask=None,
                task="multilabel",
            )

    return StandardDatasetBundle(
        name="imdb",
        task="multilabel",
        modality_names=("text", "image"),
        train=split_datasets["train"],
        valid=split_datasets["valid"],
        test=split_datasets["test"],
        num_classes=23,
        collate_fn=standardized_collate,
    )


def _load_mosei_bundle(
    *,
    max_seq_len: int,
    max_samples: int | None,
) -> StandardDatasetBundle:
    if not MOSEI_PATH.exists():
        raise FileNotFoundError(f"mosei file not found: {MOSEI_PATH}")

    from .affect.get_data import get_dataloader as get_affect_dataloader

    train_loader, valid_loader, test_loader = get_affect_dataloader(
        str(MOSEI_PATH),
        batch_size=1,
        max_seq_len=max_seq_len,
        max_pad=True,
        train_shuffle=False,
        num_workers=0,
        flatten_time_series=False,
        task="classification",
        robust_test=False,
        data_type="mosei",
    )

    train = StandardizedDatasetAdapter(
        _maybe_limit_dataset(train_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_affect_sample(sample, split_name="train", index=index),
    )
    valid = StandardizedDatasetAdapter(
        _maybe_limit_dataset(valid_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_affect_sample(sample, split_name="valid", index=index),
    )
    test = StandardizedDatasetAdapter(
        _maybe_limit_dataset(test_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_affect_sample(sample, split_name="test", index=index),
    )

    return StandardDatasetBundle(
        name="mosei",
        task="binary",
        modality_names=("text", "audio", "video"),
        train=train,
        valid=valid,
        test=test,
        num_classes=2,
        collate_fn=standardized_collate,
    )


def _load_mosi_bundle(
    *,
    max_seq_len: int,
    max_samples: int | None,
) -> StandardDatasetBundle:
    if not MOSI_PATH.exists():
        raise FileNotFoundError(f"mosi file not found: {MOSI_PATH}")

    from .affect.get_data import get_dataloader as get_affect_dataloader

    train_loader, valid_loader, test_loader = get_affect_dataloader(
        str(MOSI_PATH),
        batch_size=1,
        max_seq_len=max_seq_len,
        max_pad=True,
        train_shuffle=False,
        num_workers=0,
        flatten_time_series=False,
        task="classification",
        robust_test=False,
        data_type="mosi",
    )

    train = StandardizedDatasetAdapter(
        _maybe_limit_dataset(train_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_affect_sample(sample, split_name="train", index=index),
    )
    valid = StandardizedDatasetAdapter(
        _maybe_limit_dataset(valid_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_affect_sample(sample, split_name="valid", index=index),
    )
    test = StandardizedDatasetAdapter(
        _maybe_limit_dataset(test_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_affect_sample(sample, split_name="test", index=index),
    )

    return StandardDatasetBundle(
        name="mosi",
        task="binary",
        modality_names=("text", "audio", "video"),
        train=train,
        valid=valid,
        test=test,
        num_classes=2,
        collate_fn=standardized_collate,
    )


def _load_avmnist_bundle(
    *,
    max_samples: int | None,
) -> StandardDatasetBundle:
    if not AVMNIST_ROOT.exists():
        raise FileNotFoundError(f"avmnist root not found: {AVMNIST_ROOT}")

    from .avmnist.get_data import get_dataloader as get_avmnist_dataloader

    train_loader, valid_loader, test_loader = get_avmnist_dataloader(
        str(AVMNIST_ROOT),
        batch_size=1,
        num_workers=0,
        train_shuffle=False,
        flatten_audio=False,
        flatten_image=False,
        unsqueeze_channel=True,
        generate_sample=False,
        normalize_image=True,
        normalize_audio=True,
    )

    train = StandardizedDatasetAdapter(
        _maybe_limit_dataset(train_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_avmnist_sample(sample, split_name="train", index=index),
    )
    valid = StandardizedDatasetAdapter(
        _maybe_limit_dataset(valid_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_avmnist_sample(sample, split_name="valid", index=index),
    )
    test = StandardizedDatasetAdapter(
        _maybe_limit_dataset(test_loader.dataset, max_samples),
        sample_adapter=lambda sample, index: _adapt_avmnist_sample(sample, split_name="test", index=index),
    )

    return StandardDatasetBundle(
        name="avmnist",
        task="multiclass",
        modality_names=("image", "audio"),
        train=train,
        valid=valid,
        test=test,
        num_classes=10,
        collate_fn=standardized_collate,
    )


def _load_food101_bundle(
    *,
    image_size: int,
    valid_ratio: float,
    seed: int,
    download: bool,
    max_samples: int | None,
) -> StandardDatasetBundle:
    train_transform = _build_image_transform(image_size=image_size, train=True)
    eval_transform = _build_image_transform(image_size=image_size, train=False)

    try:
        train_full = tv_datasets.Food101(root=str(FOOD101_ROOT), split="train", transform=train_transform, download=download)
        train_full_eval = tv_datasets.Food101(root=str(FOOD101_ROOT), split="train", transform=eval_transform, download=download)
        test = tv_datasets.Food101(root=str(FOOD101_ROOT), split="test", transform=eval_transform, download=download)
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"food101 not found under {FOOD101_ROOT}. Set download=True to fetch it automatically."
        ) from exc

    train_indices, valid_indices = _split_indices(len(train_full), valid_ratio=valid_ratio, seed=seed)
    train = TorchvisionDatasetAdapter(Subset(train_full, train_indices), split_name="train")
    valid = TorchvisionDatasetAdapter(Subset(train_full_eval, valid_indices), split_name="valid")
    test = TorchvisionDatasetAdapter(_maybe_limit_dataset(test, max_samples), split_name="test")
    train = TorchvisionDatasetAdapter(_maybe_limit_dataset(train.dataset, max_samples), split_name="train")
    valid = TorchvisionDatasetAdapter(_maybe_limit_dataset(valid.dataset, max_samples), split_name="valid")

    return StandardDatasetBundle(
        name="food101",
        task="multiclass",
        modality_names=("image",),
        train=train,
        valid=valid,
        test=test,
        num_classes=101,
        collate_fn=standardized_collate,
    )


def _load_oxford_pet_bundle(
    *,
    image_size: int,
    valid_ratio: float,
    seed: int,
    download: bool,
    max_samples: int | None,
) -> StandardDatasetBundle:
    train_transform = _build_image_transform(image_size=image_size, train=True)
    eval_transform = _build_image_transform(image_size=image_size, train=False)

    try:
        trainval = tv_datasets.OxfordIIITPet(
            root=str(OXFORD_PET_ROOT),
            split="trainval",
            target_types="category",
            transform=train_transform,
            download=download,
        )
        trainval_eval = tv_datasets.OxfordIIITPet(
            root=str(OXFORD_PET_ROOT),
            split="trainval",
            target_types="category",
            transform=eval_transform,
            download=download,
        )
        test = tv_datasets.OxfordIIITPet(
            root=str(OXFORD_PET_ROOT),
            split="test",
            target_types="category",
            transform=eval_transform,
            download=download,
        )
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"oxford pet not found under {OXFORD_PET_ROOT}. Set download=True to fetch it automatically."
        ) from exc

    train_indices, valid_indices = _split_indices(len(trainval), valid_ratio=valid_ratio, seed=seed)
    train = TorchvisionDatasetAdapter(Subset(trainval, train_indices), split_name="train")
    valid = TorchvisionDatasetAdapter(Subset(trainval_eval, valid_indices), split_name="valid")
    test = TorchvisionDatasetAdapter(_maybe_limit_dataset(test, max_samples), split_name="test")
    train = TorchvisionDatasetAdapter(_maybe_limit_dataset(train.dataset, max_samples), split_name="train")
    valid = TorchvisionDatasetAdapter(_maybe_limit_dataset(valid.dataset, max_samples), split_name="valid")

    return StandardDatasetBundle(
        name="oxford_pet",
        task="multiclass",
        modality_names=("image",),
        train=train,
        valid=valid,
        test=test,
        num_classes=37,
        collate_fn=standardized_collate,
    )


def _build_image_transform(*, image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _split_indices(total_size: int, *, valid_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    valid_size = max(1, int(round(total_size * valid_ratio)))
    train_size = total_size - valid_size
    if train_size <= 0:
        raise ValueError(f"valid_ratio={valid_ratio} leaves no training samples")

    generator = torch.Generator().manual_seed(seed)
    train_subset, valid_subset = random_split(range(total_size), [train_size, valid_size], generator=generator)
    return list(train_subset.indices), list(valid_subset.indices)


def _maybe_limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _adapt_affect_sample(
    sample: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    split_name: str,
    index: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, str]:
    vision = torch.as_tensor(sample[0], dtype=torch.float32)
    audio = torch.as_tensor(sample[1], dtype=torch.float32)
    text = torch.as_tensor(sample[2], dtype=torch.float32)
    label = torch.as_tensor(sample[-1], dtype=torch.float32).reshape(-1)[0]
    mask = torch.abs(text).sum(dim=-1) > 0
    modalities = {"text": text, "audio": audio, "video": vision}
    return modalities, mask, label, f"{split_name}_{index}"


def _adapt_avmnist_sample(
    sample: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    split_name: str,
    index: int,
) -> tuple[dict[str, torch.Tensor], None, torch.Tensor, str]:
    image = torch.as_tensor(sample[0], dtype=torch.float32)
    audio = torch.as_tensor(sample[1], dtype=torch.float32)
    label = torch.as_tensor(sample[2], dtype=torch.long)
    modalities = {"image": image, "audio": audio}
    return modalities, None, label, f"{split_name}_{index}"
