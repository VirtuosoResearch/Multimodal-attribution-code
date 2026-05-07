"""Dataset loading for data selection experiments without pre-pooling."""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASETS_ROOT = Path(
    os.environ.get("MULTIBENCH_DATASETS_ROOT", REPO_ROOT / "data" / "datasets")
).expanduser()
_IMDB_SPLITS = {
    "train": (0, 15552),
    "valid": (15552, 18160),
    "test": (18160, 25959),
}


def drop_entry(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop affect samples with empty text, matching MultiBench behavior."""
    drop = []
    for index, text in enumerate(dataset["text"]):
        if text.sum() == 0:
            drop.append(index)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


@dataclass(frozen=True)
class SplitTensors:
    main: torch.Tensor
    aux: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class DatasetTensors:
    train: SplitTensors
    valid: SplitTensors
    eval: SplitTensors
    eval_split: str
    task: str
    num_classes: int
    main_shape: tuple[int, ...]
    aux_shape: tuple[int, ...]
    full_modality: str
    budgeted_modality: str


def _nan_to_num(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _standardize_by_feature(train: torch.Tensor, *others: torch.Tensor
                            ) -> tuple[torch.Tensor, ...]:
    flat = train.reshape(-1, train.shape[-1]) if train.ndim >= 3 else train
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        return (x - mean.view(*([1] * (x.ndim - 1)), -1)) / std.view(
            *([1] * (x.ndim - 1)), -1)

    return (_apply(train), *(_apply(x) for x in others))


def _standardize_image(train: torch.Tensor, *others: torch.Tensor
                       ) -> tuple[torch.Tensor, ...]:
    dims = tuple(range(0, train.ndim))
    if train.ndim >= 4:
        dims = (0, 2, 3)
    mean = train.mean(dim=dims, keepdim=True)
    std = train.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return ((train - mean) / std, *((x - mean) / std for x in others))


def _choose_standardizer(train: torch.Tensor, *others: torch.Tensor
                         ) -> tuple[torch.Tensor, ...]:
    if train.ndim == 4:
        return _standardize_image(train, *others)
    return _standardize_by_feature(train, *others)


def _prepare_pair(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_labels: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    valid_labels: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_labels: torch.Tensor,
    full_modality: str,
    standardize: bool,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    eval_split: str,
    task: str,
    num_classes: int,
) -> DatasetTensors:
    if full_modality == "x":
        train_main, train_aux = train_x, train_y
        valid_main, valid_aux = valid_x, valid_y
        eval_main, eval_aux = eval_x, eval_y
        budgeted_modality = "y"
    else:
        train_main, train_aux = train_y, train_x
        valid_main, valid_aux = valid_y, valid_x
        eval_main, eval_aux = eval_y, eval_x
        budgeted_modality = "x"

    if max_train_samples is not None:
        limit = min(int(max_train_samples), train_main.shape[0])
        train_main = train_main[:limit]
        train_aux = train_aux[:limit]
        train_labels = train_labels[:limit]
    if max_eval_samples is not None:
        limit = min(int(max_eval_samples), eval_main.shape[0])
        eval_main = eval_main[:limit]
        eval_aux = eval_aux[:limit]
        eval_labels = eval_labels[:limit]

    if standardize:
        train_main, valid_main, eval_main = _choose_standardizer(
            train_main, valid_main, eval_main)
        train_aux, valid_aux, eval_aux = _choose_standardizer(
            train_aux, valid_aux, eval_aux)

    train_main = _nan_to_num(train_main)
    train_aux = _nan_to_num(train_aux)
    valid_main = _nan_to_num(valid_main)
    valid_aux = _nan_to_num(valid_aux)
    eval_main = _nan_to_num(eval_main)
    eval_aux = _nan_to_num(eval_aux)

    return DatasetTensors(
        train=SplitTensors(train_main, train_aux, train_labels),
        valid=SplitTensors(valid_main, valid_aux, valid_labels),
        eval=SplitTensors(eval_main, eval_aux, eval_labels),
        eval_split=eval_split,
        task=task,
        num_classes=num_classes,
        main_shape=tuple(train_main.shape[1:]),
        aux_shape=tuple(train_aux.shape[1:]),
        full_modality=full_modality,
        budgeted_modality=budgeted_modality,
    )


def _load_avmnist() -> dict[str, Any]:
    base = DATASETS_ROOT / "avmnist"
    train_img = torch.from_numpy(
        np.load(base / "image" / "train_data.npy").astype(np.float32))
    train_aud = torch.from_numpy(
        np.load(base / "audio" / "train_data.npy").astype(np.float32))
    train_labels = torch.from_numpy(
        np.load(base / "train_labels.npy").astype(np.int64))
    test_img = torch.from_numpy(
        np.load(base / "image" / "test_data.npy").astype(np.float32))
    test_aud = torch.from_numpy(
        np.load(base / "audio" / "test_data.npy").astype(np.float32))
    test_labels = torch.from_numpy(
        np.load(base / "test_labels.npy").astype(np.int64))

    train_img = train_img.reshape(-1, 1, 28, 28) / 255.0
    test_img = test_img.reshape(-1, 1, 28, 28) / 255.0
    train_aud = train_aud.reshape(-1, 1, 112, 112) / 255.0
    test_aud = test_aud.reshape(-1, 1, 112, 112) / 255.0
    return {
        "train_x": train_img[:55000],
        "train_y": train_aud[:55000],
        "train_labels": train_labels[:55000],
        "valid_x": train_img[55000:],
        "valid_y": train_aud[55000:],
        "valid_labels": train_labels[55000:],
        "test_x": test_img,
        "test_y": test_aud,
        "test_labels": test_labels,
        "task": "multiclass",
        "num_classes": 10,
    }


def _affect_path(dataset_name: str) -> Path:
    direct = DATASETS_ROOT / dataset_name / f"{dataset_name}.pkl"
    if direct.exists():
        return direct
    if dataset_name == "mosei":
        candidates = [
            DATASETS_ROOT / "mosei" / "mosei_senti_data.pkl",
            DATASETS_ROOT / "mosei" / "mosei_data.pkl",
        ]
    else:
        candidates = [DATASETS_ROOT / dataset_name / f"{dataset_name}_data.pkl"]
    for path in candidates:
        if path.exists():
            return path
    checked = [direct, *candidates]
    raise FileNotFoundError(
        "Could not find affect dataset file. Checked: "
        + ", ".join(str(path) for path in checked))


def _load_affect(dataset_name: str, max_seq_len: int) -> dict[str, Any]:
    with open(_affect_path(dataset_name), "rb") as handle:
        payload = pickle.load(handle)

    def _split(split_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        split = drop_entry(payload[split_name])
        text = torch.from_numpy(np.asarray(split["text"], dtype=np.float32))
        audio = torch.from_numpy(np.asarray(split["audio"], dtype=np.float32))
        vision = torch.from_numpy(np.asarray(split["vision"], dtype=np.float32))
        text = text[:, :max_seq_len]
        audio = audio[:, :max_seq_len]
        vision = vision[:, :max_seq_len]
        aux = torch.cat([audio, vision], dim=-1)
        labels_raw = np.asarray(split["labels"], dtype=np.float32).reshape(-1)
        labels = torch.from_numpy((labels_raw > 0).astype(np.int64))
        return text, aux, labels

    train_x, train_y, train_labels = _split("train")
    valid_x, valid_y, valid_labels = _split("valid")
    test_x, test_y, test_labels = _split("test")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_labels": train_labels,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "valid_labels": valid_labels,
        "test_x": test_x,
        "test_y": test_y,
        "test_labels": test_labels,
        "task": "binary",
        "num_classes": 1,
    }


def _load_imdb() -> dict[str, Any]:
    env_path = os.environ.get("MMIMDB_HDF5_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        DATASETS_ROOT / "multimodal_imdb.hdf5",
        REPO_ROOT / "data" / "datasets" / "multimodal_imdb.hdf5",
    ]
    imdb_path = next((p for p in candidates if p is not None and p.exists()), None)
    if imdb_path is None:
        checked = ", ".join(str(p) for p in candidates if p is not None)
        raise FileNotFoundError(
            f"MM-IMDb file not found. Checked: {checked}. "
            "Set MMIMDB_HDF5_PATH if needed.")

    def _split(handle: h5py.File, split_name: str
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, stop = _IMDB_SPLITS[split_name]
        text = torch.from_numpy(
            np.asarray(handle["features"][start:stop], dtype=np.float32))
        image = torch.from_numpy(
            np.asarray(handle["vgg_features"][start:stop], dtype=np.float32))
        labels = torch.from_numpy(
            np.asarray(handle["genres"][start:stop], dtype=np.float32))
        return text, image, labels

    with h5py.File(imdb_path, "r") as handle:
        train_x, train_y, train_labels = _split(handle, "train")
        valid_x, valid_y, valid_labels = _split(handle, "valid")
        test_x, test_y, test_labels = _split(handle, "test")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_labels": train_labels,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "valid_labels": valid_labels,
        "test_x": test_x,
        "test_y": test_y,
        "test_labels": test_labels,
        "task": "multilabel",
        "num_classes": int(train_labels.shape[1]),
    }


def load_dataset(
    dataset_name: str,
    *,
    seed: int,
    full_modality: str,
    eval_split: str,
    standardize: bool,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    max_seq_len: int,
) -> DatasetTensors:
    del seed
    if dataset_name == "avmnist":
        raw = _load_avmnist()
    elif dataset_name == "imdb":
        raw = _load_imdb()
    elif dataset_name in {"mosi", "mosei", "humor", "sarcasm"}:
        raw = _load_affect(dataset_name, max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Unsupported data_selection dataset: {dataset_name}")

    if eval_split == "auto":
        eval_split = "test"
    prefix = "test" if eval_split == "test" else "valid"
    return _prepare_pair(
        train_x=raw["train_x"],
        train_y=raw["train_y"],
        train_labels=raw["train_labels"],
        valid_x=raw["valid_x"],
        valid_y=raw["valid_y"],
        valid_labels=raw["valid_labels"],
        eval_x=raw[f"{prefix}_x"],
        eval_y=raw[f"{prefix}_y"],
        eval_labels=raw[f"{prefix}_labels"],
        full_modality=full_modality,
        standardize=standardize,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        eval_split=eval_split,
        task=raw["task"],
        num_classes=raw["num_classes"],
    )
