"""Data loading for Fisher geometry experiments.

Loads pre-extracted features for five multimodal datasets:
  - Food-101: CLIP ViT-B/16 image + text prototypes
  - Oxford Pets: CLIP ViT-B/32 image + text descriptions
  - MM-IMDb: GloVe text (mean-pooled) + VGG image
  - Humor: GloVe text (mean-pooled) + audio+video (mean-pooled, concatenated)
  - Sarcasm: same modality structure as Humor

All outputs are flat float32 tensors of shape [N, D].
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UML_ROOT = Path("/data/zhenshuo/multimodal/Unpaired-Multimodal-Learning"
                "/vision_language")


def _load_food101(seed: int) -> dict[str, Any]:
    feature_dir = UML_ROOT / "features"
    clip_encoder = "ViT-B-16"
    image_path = (feature_dir / "image" / clip_encoder / "food101"
                  / "crop" / "shot_-1-seed_1.pth")
    text_path = (feature_dir / "text" / clip_encoder / "food101"
                 / "template_mining.pth")
    image_trainval = torch.load(image_path, map_location="cpu",
                                weights_only=False)
    text_payload = torch.load(text_path, map_location="cpu",
                              weights_only=False)

    text_features = text_payload["features"].float()
    text_labels = text_payload["labels"].long()
    num_classes = int(text_labels.max().item()) + 1
    text_by_class = torch.zeros(num_classes, text_features.shape[1])
    for c in range(num_classes):
        text_by_class[c] = text_features[text_labels == c].mean(dim=0)

    train = image_trainval["train"]
    val = image_trainval["val"]
    return {
        "train_x": train["features"].float(),
        "train_y": text_by_class[train["labels"].long()],
        "train_labels": train["labels"].long(),
        "val_x": val["features"].float(),
        "val_y": text_by_class[val["labels"].long()],
        "val_labels": val["labels"].long(),
        "num_classes": num_classes, "task": "multiclass", "name": "food101",
    }


def _load_oxford_pets(seed: int) -> dict[str, Any]:
    cache_path = (UML_ROOT / "features" / "joint_oxford_pets_cache"
                  / f"oxford_pets_clip_cache_seed{seed}_val8.pt")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Oxford Pets cache not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_labels = payload["train_labels"].long()
    val_labels = payload["val_labels"].long()
    text_by_class = payload["text_by_class"].float()
    num_classes = int(train_labels.max().item()) + 1
    return {
        "train_x": payload["train_image"].float(),
        "train_y": text_by_class[train_labels],
        "train_labels": train_labels,
        "val_x": payload["val_image"].float(),
        "val_y": text_by_class[val_labels],
        "val_labels": val_labels,
        "num_classes": num_classes, "task": "multiclass",
        "name": "oxford_pets",
    }


def _load_affect_features(dataset_name: str) -> dict[str, Any]:
    from opt_motivation.core import load_affect

    pkl = (f"/data/zhenshuo/datasets/{dataset_name}/{dataset_name}.pkl")
    ds = load_affect(pkl, dataset_name=dataset_name, max_seq_len=50)

    def _pool(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float()
        return t.mean(dim=1) if t.ndim == 3 else t

    def _split(split):
        text = _pool(split.modalities["text"])
        audio = _pool(split.modalities["audio"])
        video = _pool(split.modalities["video"])
        aux = torch.cat([audio, video], dim=1)
        raw = split.labels
        labels = torch.from_numpy(
            (raw > 0).astype(np.int64) if raw.dtype in (np.float32, np.float64)
            else raw).long()
        return text, aux, labels

    tr_x, tr_y, tr_l = _split(ds.train)
    va_x, va_y, va_l = _split(ds.valid)
    return {
        "train_x": tr_x, "train_y": tr_y, "train_labels": tr_l,
        "val_x": va_x, "val_y": va_y, "val_labels": va_l,
        "num_classes": 1, "task": "binary", "name": dataset_name,
    }


def _load_imdb_features() -> dict[str, Any]:
    from opt_motivation.core import load_imdb

    ds = load_imdb("/data/zhenshuo/datasets/multimodal_imdb.hdf5",
                   train_max_samples=None, valid_max_samples=None,
                   test_max_samples=None)

    def _split(split):
        text_raw = torch.from_numpy(split.modalities["text"]).float()
        text = text_raw.mean(dim=1) if text_raw.ndim == 3 else text_raw
        image = torch.from_numpy(split.modalities["image"]).float()
        labels = torch.from_numpy(split.labels).float()
        return text, image, labels

    tr_x, tr_y, tr_l = _split(ds.train)
    va_x, va_y, va_l = _split(ds.valid)
    return {
        "train_x": tr_x, "train_y": tr_y, "train_labels": tr_l,
        "val_x": va_x, "val_y": va_y, "val_labels": va_l,
        "num_classes": int(tr_l.shape[1]), "task": "multilabel",
        "name": "imdb",
    }


def load_dataset(name: str, seed: int) -> dict[str, Any]:
    """Load a dataset by name. Returns dict with train/val tensors."""
    loaders = {
        "food101": _load_food101,
        "oxford_pets": _load_oxford_pets,
        "humor": lambda s: _load_affect_features("humor"),
        "sarcasm": lambda s: _load_affect_features("sarcasm"),
        "imdb": lambda s: _load_imdb_features(),
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name](seed)


def permute_aux(data: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy with shuffled auxiliary features (negative control)."""
    rng = np.random.default_rng(seed)
    out = dict(data)
    out["train_y"] = data["train_y"][
        torch.from_numpy(rng.permutation(data["train_y"].shape[0]))]
    out["val_y"] = data["val_y"][
        torch.from_numpy(rng.permutation(data["val_y"].shape[0]))]
    return out


def make_loader(x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor,
                batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y, labels),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False)
