"""Data loading for Fisher geometry experiments.

Supports cached CLIP features and optional frozen backbone features for
image-text datasets, while keeping the existing affect / imdb loaders.
All outputs are flat float32 tensors of shape [N, D].
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision import datasets as tv_datasets
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# opt_motivation/core.py lives in the old repo copy
_OLD_REPO = Path("/data/zhenshuo/multimodal/old_multimodal-representation-and-attribution")
if str(_OLD_REPO) not in sys.path:
    sys.path.insert(0, str(_OLD_REPO))

from src.backbone import load_text_backbone, load_vision_backbone

UML_ROOT = Path("/data/zhenshuo/multimodal/Unpaired-Multimodal-Learning"
                "/vision_language")
DATASETS_ROOT = Path("/data/zhenshuo/datasets")
BACKBONE_CACHE_ROOT = DATASETS_ROOT / "backbone_cache"


def _slug(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace("-", "_")


def _backbone_cache_path(
    dataset_name: str,
    *,
    vision_backbone: str,
    text_backbone: str,
    cache_dir: str | None,
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else BACKBONE_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{dataset_name}_{_slug(vision_backbone)}_{_slug(text_backbone)}.pt"


def _encode_dataset_images(
    dataset: Any,
    *,
    vision_backbone: Any,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for start in range(0, len(dataset), batch_size):
        batch_images = []
        batch_labels = []
        for index in range(start, min(start + batch_size, len(dataset))):
            image, label = dataset[index]
            batch_images.append(image)
            batch_labels.append(int(label))
        features.append(vision_backbone.encode(batch_images, batch_size=len(batch_images)))
        labels.append(torch.tensor(batch_labels, dtype=torch.long))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _food101_prompt(class_name: str) -> str:
    clean = class_name.replace("_", " ")
    return f"A photo of {clean}, a type of food."


def _oxford_pets_prompt(class_name: str) -> str:
    clean = class_name.replace("_", " ")
    return f"A photo of a {clean} pet."


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


def _load_food101_from_backbones(
    *,
    feature_device: str,
    backbone_batch_size: int,
    download_datasets: bool,
    text_backbone_name: str,
    vision_backbone_name: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    cache_path = _backbone_cache_path(
        "food101",
        vision_backbone=vision_backbone_name,
        text_backbone=text_backbone_name,
        cache_dir=cache_dir,
    )
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    train_ds = tv_datasets.Food101(
        root=str(DATASETS_ROOT / "food101"),
        split="train",
        download=download_datasets,
    )
    test_ds = tv_datasets.Food101(
        root=str(DATASETS_ROOT / "food101"),
        split="test",
        download=download_datasets,
    )

    text_backbone = load_text_backbone(text_backbone_name, device=feature_device)
    vision_backbone = load_vision_backbone(vision_backbone_name, device=feature_device)

    class_prompts = [_food101_prompt(name) for name in train_ds.classes]
    text_by_class = text_backbone.encode(class_prompts, batch_size=backbone_batch_size)
    train_x, train_labels = _encode_dataset_images(
        train_ds, vision_backbone=vision_backbone, batch_size=backbone_batch_size)
    val_x, val_labels = _encode_dataset_images(
        test_ds, vision_backbone=vision_backbone, batch_size=backbone_batch_size)

    payload = {
        "train_x": train_x.float(),
        "train_y": text_by_class[train_labels].float(),
        "train_labels": train_labels.long(),
        "val_x": val_x.float(),
        "val_y": text_by_class[val_labels].float(),
        "val_labels": val_labels.long(),
        "num_classes": len(train_ds.classes),
        "task": "multiclass",
        "name": "food101",
    }
    torch.save(payload, cache_path)
    return payload


def _load_oxford_pets_from_backbones(
    *,
    feature_device: str,
    backbone_batch_size: int,
    download_datasets: bool,
    text_backbone_name: str,
    vision_backbone_name: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    cache_path = _backbone_cache_path(
        "oxford_pets",
        vision_backbone=vision_backbone_name,
        text_backbone=text_backbone_name,
        cache_dir=cache_dir,
    )
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    train_ds = tv_datasets.OxfordIIITPet(
        root=str(DATASETS_ROOT / "oxford_pet"),
        split="trainval",
        target_types="category",
        download=download_datasets,
    )
    test_ds = tv_datasets.OxfordIIITPet(
        root=str(DATASETS_ROOT / "oxford_pet"),
        split="test",
        target_types="category",
        download=download_datasets,
    )

    text_backbone = load_text_backbone(text_backbone_name, device=feature_device)
    vision_backbone = load_vision_backbone(vision_backbone_name, device=feature_device)

    class_prompts = [_oxford_pets_prompt(name) for name in train_ds.classes]
    text_by_class = text_backbone.encode(class_prompts, batch_size=backbone_batch_size)
    train_x, train_labels = _encode_dataset_images(
        train_ds, vision_backbone=vision_backbone, batch_size=backbone_batch_size)
    val_x, val_labels = _encode_dataset_images(
        test_ds, vision_backbone=vision_backbone, batch_size=backbone_batch_size)

    payload = {
        "train_x": train_x.float(),
        "train_y": text_by_class[train_labels].float(),
        "train_labels": train_labels.long(),
        "val_x": val_x.float(),
        "val_y": text_by_class[val_labels].float(),
        "val_labels": val_labels.long(),
        "num_classes": len(train_ds.classes),
        "task": "multiclass",
        "name": "oxford_pets",
    }
    torch.save(payload, cache_path)
    return payload


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


def _load_avmnist_features() -> dict[str, Any]:
    """Load AV-MNIST: image (784-d) + audio (12544-d flattened spectrogram)."""
    base = DATASETS_ROOT / "avmnist"
    train_img = torch.from_numpy(
        np.load(base / "image" / "train_data.npy").astype(np.float32))
    train_aud = torch.from_numpy(
        np.load(base / "audio" / "train_data.npy").reshape(-1, 112 * 112).astype(np.float32))
    train_labels = torch.from_numpy(
        np.load(base / "train_labels.npy").astype(np.int64))
    test_img = torch.from_numpy(
        np.load(base / "image" / "test_data.npy").astype(np.float32))
    test_aud = torch.from_numpy(
        np.load(base / "audio" / "test_data.npy").reshape(-1, 112 * 112).astype(np.float32))
    test_labels = torch.from_numpy(
        np.load(base / "test_labels.npy").astype(np.int64))
    # Normalize to [0, 1]
    train_img = train_img / 255.0
    test_img = test_img / 255.0
    train_aud = train_aud / 255.0
    test_aud = test_aud / 255.0
    return {
        "train_x": train_img, "train_y": train_aud,
        "train_labels": train_labels,
        "val_x": test_img, "val_y": test_aud,
        "val_labels": test_labels,
        "num_classes": 10, "task": "multiclass", "name": "avmnist",
    }


def _load_affect_mosi_mosei(dataset_name: str) -> dict[str, Any]:
    """Load MOSI or MOSEI from their pickle files."""
    pkl_path = DATASETS_ROOT / dataset_name / f"{dataset_name}_data.pkl"
    if dataset_name == "mosei":
        pkl_path = DATASETS_ROOT / "mosei" / "mosei_senti_data.pkl"
    import pickle
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    def _split(split_name):
        split = payload[split_name]
        text = torch.from_numpy(
            np.asarray(split["text"], dtype=np.float32))
        audio = torch.from_numpy(
            np.asarray(split["audio"], dtype=np.float32))
        video = torch.from_numpy(
            np.asarray(split["vision"], dtype=np.float32))
        # Mean-pool over time dimension
        text = text.mean(dim=1) if text.ndim == 3 else text
        audio = audio.mean(dim=1) if audio.ndim == 3 else audio
        video = video.mean(dim=1) if video.ndim == 3 else video
        text[~torch.isfinite(text)] = 0.0
        audio[~torch.isfinite(audio)] = 0.0
        video[~torch.isfinite(video)] = 0.0
        aux = torch.cat([audio, video], dim=1)
        labels_raw = np.asarray(split["labels"], dtype=np.float32).reshape(-1)
        labels = torch.from_numpy((labels_raw > 0).astype(np.int64))
        return text, aux, labels

    tr_x, tr_y, tr_l = _split("train")
    va_x, va_y, va_l = _split("valid")
    return {
        "train_x": tr_x, "train_y": tr_y, "train_labels": tr_l,
        "val_x": va_x, "val_y": va_y, "val_labels": va_l,
        "num_classes": 1, "task": "binary", "name": dataset_name,
    }


def load_dataset(
    name: str,
    seed: int,
    *,
    text_backbone: str = "clip_cached",
    vision_backbone: str = "clip_cached",
    feature_device: str = "cuda:0",
    backbone_batch_size: int = 32,
    download_datasets: bool = False,
    backbone_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Load a dataset by name. Returns dict with train/val tensors."""
    if name == "food101":
        if text_backbone == "clip_cached" and vision_backbone == "clip_cached":
            return _load_food101(seed)
        return _load_food101_from_backbones(
            feature_device=feature_device,
            backbone_batch_size=backbone_batch_size,
            download_datasets=download_datasets,
            text_backbone_name=text_backbone,
            vision_backbone_name=vision_backbone,
            cache_dir=backbone_cache_dir,
        )
    if name == "oxford_pets":
        if text_backbone == "clip_cached" and vision_backbone == "clip_cached":
            return _load_oxford_pets(seed)
        return _load_oxford_pets_from_backbones(
            feature_device=feature_device,
            backbone_batch_size=backbone_batch_size,
            download_datasets=download_datasets,
            text_backbone_name=text_backbone,
            vision_backbone_name=vision_backbone,
            cache_dir=backbone_cache_dir,
        )

    if text_backbone != "clip_cached" or vision_backbone != "clip_cached":
        raise ValueError(
            f"Custom backbone extraction is only wired for food101 / oxford_pets, got dataset={name}"
        )

    if name == "avmnist":
        return _load_avmnist_features()

    loaders = {
        "humor": lambda s: _load_affect_features("humor"),
        "sarcasm": lambda s: _load_affect_features("sarcasm"),
        "mosi": lambda s: _load_affect_mosi_mosei("mosi"),
        "mosei": lambda s: _load_affect_mosi_mosei("mosei"),
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


def perturb_aux(data: dict[str, Any], noise_level: float,
                seed: int, mode: str = "gaussian") -> dict[str, Any]:
    """Return a copy with noisy auxiliary features.

    mode="gaussian" (default):
        Y_perturbed = (1 - alpha) * Y + alpha * N(0, I)
        At alpha=0: original Y. At alpha=1: pure standard Gaussian noise.
    mode="additive":
        Y_perturbed = Y + alpha * std(Y) * N(0, I)
        Additive noise scaled by alpha (legacy).
    mode="permute":
        Y_perturbed = (1 - alpha) * Y + alpha * Y_permuted
        Interpolation with shuffled samples (legacy).

    alpha = noise_level. At alpha=0 all modes return the original.
    """
    if noise_level <= 0.0:
        return dict(data)
    rng = np.random.default_rng(seed)
    out = dict(data)
    alpha = float(noise_level)

    for split in ["train_y", "val_y"]:
        y = data[split]
        if mode == "gaussian":
            noise = torch.from_numpy(
                rng.standard_normal(y.shape).astype(np.float32))
            out[split] = (1.0 - alpha) * y + alpha * noise
        elif mode == "permute":
            perm = torch.from_numpy(rng.permutation(y.shape[0]))
            y_perm = y[perm]
            out[split] = (1.0 - alpha) * y + alpha * y_perm
        elif mode == "additive":
            std = y.std(dim=0, keepdim=True).clamp(min=1e-8)
            noise = torch.from_numpy(
                rng.standard_normal(y.shape).astype(np.float32))
            out[split] = y + alpha * std * noise
        else:
            raise ValueError(f"Unknown perturbation mode: {mode}")

    return out


def make_loader(x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor,
                batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y, labels),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False)
