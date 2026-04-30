"""Data loading for Fisher geometry experiments.

Supports cached CLIP features and optional frozen backbone features for
image-text datasets, while keeping the existing affect / imdb loaders.
All outputs are flat float32 tensors of shape [N, D].
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torchvision import datasets as tv_datasets
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
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
TRAFFIC_PROJECT_ROOT = ROOT.parent / "Satellite-images-for-accident-analysis"
_IMDB_SPLITS = {
    "train": (0, 15552),
    "valid": (15552, 18160),
    "test": (18160, 25959),
}


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
    import h5py

    env_path = os.environ.get("MMIMDB_HDF5_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        DATASETS_ROOT / "multimodal_imdb.hdf5",
        ROOT / "src" / "datasets" / "multimodal_imdb.hdf5",
        ROOT / "data" / "datasets" / "multimodal_imdb.hdf5",
    ]
    imdb_path = next((path for path in candidates if path is not None and path.exists()), None)
    if imdb_path is None:
        checked = ", ".join(str(path) for path in candidates if path is not None)
        raise FileNotFoundError(
            "MM-IMDb file not found. Checked: "
            f"{checked}. Set MMIMDB_HDF5_PATH to your multimodal_imdb.hdf5 path."
        )

    def _split(handle: h5py.File, split_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, stop = _IMDB_SPLITS[split_name]
        text = torch.from_numpy(np.asarray(handle["features"][start:stop], dtype=np.float32))
        if "vgg_features" in handle:
            image = torch.from_numpy(np.asarray(handle["vgg_features"][start:stop], dtype=np.float32))
        elif "images" in handle:
            image_raw = np.asarray(handle["images"][start:stop], dtype=np.float32)
            image = torch.from_numpy(image_raw.reshape(image_raw.shape[0], -1))
        else:
            raise KeyError("MM-IMDb hdf5 missing both 'vgg_features' and 'images' datasets")
        labels = torch.from_numpy(np.asarray(handle["genres"][start:stop], dtype=np.float32))
        return text, image, labels

    try:
        with h5py.File(imdb_path, "r") as handle:
            tr_x, tr_y, tr_l = _split(handle, "train")
            va_x, va_y, va_l = _split(handle, "valid")
            te_x, te_y, te_l = _split(handle, "test")
    except OSError as exc:
        raise OSError(
            f"Failed to open MM-IMDb file at {imdb_path}. "
            "The file may be incomplete/corrupted (for example, interrupted download)."
        ) from exc
    return {
        "train_x": tr_x, "train_y": tr_y, "train_labels": tr_l,
        "val_x": va_x, "val_y": va_y, "val_labels": va_l,
        "test_x": te_x, "test_y": te_y, "test_labels": te_l,
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


_TRAFFIC_STATIC_EDGE_KEYS = [
    "oneway",
    "access_ramp",
    "bus_stop",
    "crossing",
    "disused",
    "elevator",
    "escape",
    "living_street",
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "residential",
    "rest_area",
    "road",
    "secondary",
    "secondary_link",
    "stairs",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "unclassified",
    "unsurfaced",
]


def _adjust_feature_rows(features: torch.Tensor, target_rows: int) -> torch.Tensor:
    if int(features.shape[0]) == int(target_rows):
        return features
    if int(features.shape[0]) > int(target_rows):
        return features[:target_rows]
    pad_rows = int(target_rows) - int(features.shape[0])
    return torch.cat([features, torch.zeros(pad_rows, int(features.shape[1]), dtype=features.dtype)], dim=0)


def _traffic_edge_index(data_dir: Path, *, state_name: str) -> torch.Tensor:
    adj_path = data_dir / state_name / "adj_matrix.pt"
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing traffic adjacency matrix file: {adj_path}")
    edge_index = torch.load(adj_path, map_location="cpu", weights_only=False).coalesce().indices().long().T
    return edge_index


def _traffic_static_edge_features(
    data_dir: Path,
    *,
    state_name: str,
    num_edges: int,
) -> torch.Tensor:
    feature_path = data_dir / state_name / "Edges" / "edge_features.pt"
    if not feature_path.exists():
        return torch.zeros(num_edges, 0, dtype=torch.float32)

    edge_feature_dict = torch.load(feature_path, map_location="cpu", weights_only=False)
    edge_lengths = edge_feature_dict["length"].coalesce().values().float()
    length_mean = edge_lengths[~torch.isnan(edge_lengths)].mean()
    edge_lengths = edge_lengths.clone()
    edge_lengths[torch.isnan(edge_lengths)] = 0.0 if torch.isnan(length_mean) else length_mean
    length_std = edge_lengths.std()
    if float(length_std) == 0.0:
        length_std = torch.tensor(1.0, dtype=edge_lengths.dtype)
    normalized_length = ((edge_lengths - edge_lengths.mean()) / length_std).view(-1, 1)

    columns: list[torch.Tensor] = []
    for key in _TRAFFIC_STATIC_EDGE_KEYS:
        if key not in edge_feature_dict:
            values = torch.zeros_like(edge_lengths)
        else:
            values = edge_feature_dict[key].coalesce().values().float()
            values = values.clone()
            values[torch.isnan(values)] = 0.0
        columns.append(values.view(-1, 1))
    columns.append(normalized_length)
    stacked = torch.cat(columns, dim=1).float()
    return _adjust_feature_rows(stacked, num_edges)


def _traffic_dynamic_edge_features(
    data_dir: Path,
    *,
    state_name: str,
    year: int,
    num_edges: int,
) -> torch.Tensor:
    yearly_path = data_dir / state_name / "Edges" / f"edge_features_{year}.pt"
    if not yearly_path.exists():
        return torch.zeros(num_edges, 1, dtype=torch.float32)

    yearly_dict = torch.load(yearly_path, map_location="cpu", weights_only=False)
    aadt_values = yearly_dict["AADT"].coalesce().values().float()
    finite_values = aadt_values[~torch.isnan(aadt_values)]
    fill_value = 0.0 if finite_values.numel() == 0 else float(finite_values.mean())
    aadt_values = aadt_values.clone()
    aadt_values[torch.isnan(aadt_values)] = fill_value
    return _adjust_feature_rows(aadt_values.view(-1, 1), num_edges)


def _monthly_node_features(
    data_dir: Path,
    *,
    state_name: str,
    year: int,
    month: int,
) -> torch.Tensor:
    node_feature_file = data_dir / state_name / "Nodes" / f"node_features_{year}_{month}.csv"
    if not node_feature_file.exists():
        raise FileNotFoundError(f"Missing traffic node feature file: {node_feature_file}")
    node_df = pd.read_csv(node_feature_file)
    weather_columns = ["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]
    missing_columns = [column for column in weather_columns if column not in node_df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing weather columns {missing_columns} in {node_feature_file}"
        )
    node_df = node_df[weather_columns].fillna(node_df[weather_columns].mean()).fillna(0.0)
    return torch.from_numpy(node_df.to_numpy(dtype=np.float32))


def _monthly_accident_edges(
    accidents_df: pd.DataFrame,
    *,
    year: int,
    month: int,
    label_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    month_df = accidents_df[
        (accidents_df["year"] == int(year))
        & (accidents_df["month"] == int(month))
    ][["node_1_idx", "node_2_idx", "acc_count"]]
    if month_df.empty:
        return (
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )

    grouped = month_df.groupby(["node_1_idx", "node_2_idx"], as_index=False).agg({"acc_count": "sum"})
    edges = torch.from_numpy(grouped[["node_1_idx", "node_2_idx"]].to_numpy(dtype=np.int64))
    labels = torch.from_numpy((grouped["acc_count"].to_numpy(dtype=np.float32) > float(label_threshold)).astype(np.int64))
    keep = labels == 1
    return edges[keep], labels[keep]


def _build_edge_samples(
    *,
    edge_pairs: torch.Tensor,
    edge_ids: torch.Tensor,
    node_tabular: torch.Tensor,
    edge_tabular: torch.Tensor,
    node_vision: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = edge_pairs[:, 0].long()
    v = edge_pairs[:, 1].long()
    x = torch.cat([edge_tabular[edge_ids], node_tabular[u], node_tabular[v]], dim=1).float()
    y = torch.cat([node_vision[u], node_vision[v]], dim=1).float()
    return x, y, labels.long()


def _build_traffic_edge_split(
    *,
    years: list[int],
    months: list[int],
    state_name: str,
    data_dir: Path,
    tabular_static: torch.Tensor,
    vision_features: torch.Tensor,
    edge_index_pairs: torch.Tensor,
    static_edge_features: torch.Tensor,
    accidents_df: pd.DataFrame,
    include_dynamic_node_features: bool,
    include_static_edge_features: bool,
    include_dynamic_edge_features: bool,
    negative_ratio: float,
    label_threshold: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_tabular: list[torch.Tensor] = []
    split_vision: list[torch.Tensor] = []
    split_labels: list[torch.Tensor] = []
    num_nodes = int(tabular_static.shape[0])
    num_edges = int(edge_index_pairs.shape[0])
    edge_to_id = {tuple(edge.tolist()): index for index, edge in enumerate(edge_index_pairs)}

    for year in years:
        for month in months:
            try:
                dynamic_features = _monthly_node_features(
                    data_dir,
                    state_name=state_name,
                    year=int(year),
                    month=int(month),
                )
            except FileNotFoundError:
                # Some states (e.g., IA in late 2023/2024) do not have full
                # month coverage. Skip missing months instead of failing the
                # entire split construction.
                continue
            if dynamic_features.shape[0] != num_nodes:
                raise ValueError(
                    "Traffic node count mismatch between static and dynamic features: "
                    f"{num_nodes} vs {dynamic_features.shape[0]} in {year}-{month:02d}"
                )
            node_tabular = torch.cat([tabular_static, dynamic_features], dim=1) if include_dynamic_node_features else tabular_static

            edge_modalities: list[torch.Tensor] = []
            if include_static_edge_features and int(static_edge_features.shape[1]) > 0:
                edge_modalities.append(static_edge_features)
            if include_dynamic_edge_features:
                edge_modalities.append(
                    _traffic_dynamic_edge_features(
                        data_dir,
                        state_name=state_name,
                        year=int(year),
                        num_edges=num_edges,
                    )
                )
            if edge_modalities:
                edge_tabular = torch.cat(edge_modalities, dim=1).float()
            else:
                edge_tabular = torch.zeros(num_edges, 0, dtype=torch.float32)

            positive_edges, positive_labels = _monthly_accident_edges(
                accidents_df,
                year=int(year),
                month=int(month),
                label_threshold=label_threshold,
            )
            if int(positive_edges.shape[0]) == 0:
                continue

            valid_mask = torch.as_tensor(
                [tuple(edge.tolist()) in edge_to_id for edge in positive_edges],
                dtype=torch.bool,
            )
            if int(valid_mask.sum().item()) == 0:
                continue
            positive_edges = positive_edges[valid_mask]
            positive_labels = positive_labels[valid_mask]
            positive_edge_ids = torch.as_tensor(
                [edge_to_id[tuple(edge.tolist())] for edge in positive_edges],
                dtype=torch.long,
            )

            positive_x, positive_y, positive_l = _build_edge_samples(
                edge_pairs=positive_edges,
                edge_ids=positive_edge_ids,
                node_tabular=node_tabular,
                edge_tabular=edge_tabular,
                node_vision=vision_features,
                labels=positive_labels,
            )
            split_tabular.append(positive_x)
            split_vision.append(positive_y)
            split_labels.append(positive_l)

            num_pos = int(positive_edges.shape[0])
            num_neg = int(round(float(negative_ratio) * num_pos))
            if num_neg <= 0:
                continue
            all_edge_ids = np.arange(num_edges, dtype=np.int64)
            pos_unique = np.unique(positive_edge_ids.numpy())
            candidate_neg_ids = np.setdiff1d(all_edge_ids, pos_unique, assume_unique=False)
            if candidate_neg_ids.size == 0:
                continue
            num_neg = min(num_neg, int(candidate_neg_ids.size))
            rng = np.random.default_rng(int(seed) * 1_000_000 + int(year) * 100 + int(month))
            selected_neg_ids = rng.choice(candidate_neg_ids, size=num_neg, replace=False)
            negative_edge_ids = torch.from_numpy(selected_neg_ids.astype(np.int64))
            negative_edges = edge_index_pairs[negative_edge_ids]
            negative_labels = torch.zeros(num_neg, dtype=torch.long)

            negative_x, negative_y, negative_l = _build_edge_samples(
                edge_pairs=negative_edges,
                edge_ids=negative_edge_ids,
                node_tabular=node_tabular,
                edge_tabular=edge_tabular,
                node_vision=vision_features,
                labels=negative_labels,
            )
            split_tabular.append(negative_x)
            split_vision.append(negative_y)
            split_labels.append(negative_l)

    if not split_tabular:
        raise ValueError(
            f"No traffic split samples were constructed for years={years} and months={months}"
        )
    return (
        torch.cat(split_tabular, dim=0).float(),
        torch.cat(split_vision, dim=0).float(),
        torch.cat(split_labels, dim=0).long(),
    )


def _resolve_state_embedding_file(
    embedding_root: Path,
    *,
    feature_type: str,
    state_name: str,
) -> Path:
    feature_dir = embedding_root / feature_type
    exact = feature_dir / f"{state_name}.npy"
    if exact.exists():
        return exact

    candidates = sorted(feature_dir.glob(f"{state_name}_*.npy"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        preferred_by_name = feature_dir / f"{state_name}_128.npy"
        if preferred_by_name in candidates:
            return preferred_by_name
        non_lessnode = [path for path in candidates if "lessnode" not in path.stem.lower()]
        if len(non_lessnode) == 1:
            return non_lessnode[0]
        raise FileNotFoundError(
            f"Multiple candidate files found for state={state_name}, feature_type={feature_type}: "
            f"{[str(path) for path in candidates]}. Please pass a more specific feature type or rename files."
        )
    raise FileNotFoundError(
        f"Missing traffic embedding file for state={state_name}, feature_type={feature_type}. "
        f"Checked exact path {exact} and wildcard pattern {feature_dir / (state_name + '_*.npy')}."
    )


def _load_traffic_accident_features(
    *,
    state_name: str,
    data_dir: str | None,
    embeddings_dir: str | None,
    node_feature_type: str,
    vision_feature_type: str,
    train_years: list[int],
    valid_years: list[int],
    test_years: list[int],
    months: list[int],
    include_dynamic_node_features: bool,
    include_static_edge_features: bool,
    include_dynamic_edge_features: bool,
    negative_ratio: float,
    label_threshold: float,
    normalize_modalities: bool,
    seed: int,
) -> dict[str, Any]:
    data_root = Path(data_dir) if data_dir is not None else TRAFFIC_PROJECT_ROOT / "data"
    embedding_root = Path(embeddings_dir) if embeddings_dir is not None else TRAFFIC_PROJECT_ROOT / "embeddings"

    accidents_file = data_root / state_name / "accidents_monthly.csv"
    if not accidents_file.exists():
        raise FileNotFoundError(f"Missing traffic accidents file: {accidents_file}")
    accidents_df = pd.read_csv(accidents_file)

    tabular_static_file = _resolve_state_embedding_file(
        embedding_root,
        feature_type=node_feature_type,
        state_name=state_name,
    )
    vision_feature_file = _resolve_state_embedding_file(
        embedding_root,
        feature_type=vision_feature_type,
        state_name=state_name,
    )

    tabular_static = torch.from_numpy(np.load(tabular_static_file).astype(np.float32))
    vision_features = torch.from_numpy(np.load(vision_feature_file).astype(np.float32))
    if tabular_static.shape[0] != vision_features.shape[0]:
        raise ValueError(
            "Traffic node count mismatch between tabular and vision features: "
            f"{tabular_static.shape[0]} vs {vision_features.shape[0]}"
        )
    edge_index_pairs = _traffic_edge_index(data_root, state_name=state_name)
    static_edge_features = _traffic_static_edge_features(
        data_root,
        state_name=state_name,
        num_edges=int(edge_index_pairs.shape[0]),
    )

    train_x, train_y, train_labels = _build_traffic_edge_split(
        years=[int(year) for year in train_years],
        months=[int(month) for month in months],
        state_name=state_name,
        data_dir=data_root,
        tabular_static=tabular_static,
        vision_features=vision_features,
        edge_index_pairs=edge_index_pairs,
        static_edge_features=static_edge_features,
        accidents_df=accidents_df,
        include_dynamic_node_features=include_dynamic_node_features,
        include_static_edge_features=include_static_edge_features,
        include_dynamic_edge_features=include_dynamic_edge_features,
        negative_ratio=negative_ratio,
        label_threshold=label_threshold,
        seed=seed,
    )
    val_x, val_y, val_labels = _build_traffic_edge_split(
        years=[int(year) for year in valid_years],
        months=[int(month) for month in months],
        state_name=state_name,
        data_dir=data_root,
        tabular_static=tabular_static,
        vision_features=vision_features,
        edge_index_pairs=edge_index_pairs,
        static_edge_features=static_edge_features,
        accidents_df=accidents_df,
        include_dynamic_node_features=include_dynamic_node_features,
        include_static_edge_features=include_static_edge_features,
        include_dynamic_edge_features=include_dynamic_edge_features,
        negative_ratio=negative_ratio,
        label_threshold=label_threshold,
        seed=seed + 1,
    )
    test_x, test_y, test_labels = _build_traffic_edge_split(
        years=[int(year) for year in test_years],
        months=[int(month) for month in months],
        state_name=state_name,
        data_dir=data_root,
        tabular_static=tabular_static,
        vision_features=vision_features,
        edge_index_pairs=edge_index_pairs,
        static_edge_features=static_edge_features,
        accidents_df=accidents_df,
        include_dynamic_node_features=include_dynamic_node_features,
        include_static_edge_features=include_static_edge_features,
        include_dynamic_edge_features=include_dynamic_edge_features,
        negative_ratio=negative_ratio,
        label_threshold=label_threshold,
        seed=seed + 2,
    )

    if normalize_modalities:
        def _normalize_by_train_stats(
            train_tensor: torch.Tensor,
            val_tensor: torch.Tensor,
            test_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mean = train_tensor.mean(dim=0, keepdim=True)
            std = train_tensor.std(dim=0, keepdim=True)
            std = std.clamp(min=1e-6)
            return (
                (train_tensor - mean) / std,
                (val_tensor - mean) / std,
                (test_tensor - mean) / std,
            )

        train_x, val_x, test_x = _normalize_by_train_stats(train_x, val_x, test_x)
        train_y, val_y, test_y = _normalize_by_train_stats(train_y, val_y, test_y)

    node_tabular_dim = int(tabular_static.shape[1] + (6 if include_dynamic_node_features else 0))
    edge_tabular_dim = int(
        (int(static_edge_features.shape[1]) if include_static_edge_features else 0)
        + (1 if include_dynamic_edge_features else 0)
    )

    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_labels": train_labels,
        "val_x": val_x,
        "val_y": val_y,
        "val_labels": val_labels,
        "test_x": test_x,
        "test_y": test_y,
        "test_labels": test_labels,
        "num_classes": 1,
        "task": "binary",
        "name": f"traffic_accident_{state_name.lower()}",
        "traffic_node_tabular_dim": node_tabular_dim,
        "traffic_edge_tabular_dim": edge_tabular_dim,
        "traffic_vision_node_dim": int(vision_features.shape[1]),
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
    traffic_state_name: str = "DE",
    traffic_data_dir: str | None = None,
    traffic_embeddings_dir: str | None = None,
    traffic_node_feature_type: str = "node2vec",
    traffic_vision_feature_type: str = "clip_image",
    traffic_train_years: list[int] | None = None,
    traffic_valid_years: list[int] | None = None,
    traffic_test_years: list[int] | None = None,
    traffic_months: list[int] | None = None,
    traffic_include_dynamic_node_features: bool = True,
    traffic_include_static_edge_features: bool = True,
    traffic_include_dynamic_edge_features: bool = True,
    traffic_negative_ratio: float = 1.0,
    traffic_label_threshold: float = 0.0,
    traffic_normalize_modalities: bool = True,
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

    is_traffic_dataset = (
        name in {"traffic_accident", "traffic_accident_prediction"}
        or name.startswith("traffic_accident_")
    )
    if (
        (text_backbone != "clip_cached" or vision_backbone != "clip_cached")
        and name not in {"avmnist"}
        and not is_traffic_dataset
    ):
        raise ValueError(
            f"Custom backbone extraction is only wired for food101 / oxford_pets, got dataset={name}"
        )

    if name == "avmnist":
        return _load_avmnist_features()
    if is_traffic_dataset:
        traffic_state_defaults: dict[str, dict[str, Any]] = {
            "de": {
                "train_years": [2009, 2010, 2011, 2012, 2013],
                "valid_years": [2014, 2015, 2016, 2017, 2018],
                "test_years": [2019, 2020, 2021, 2022],
                "months": list(range(1, 13)),
            },
            "ma": {
                "train_years": [2009, 2010, 2011, 2012, 2013, 2014],
                "valid_years": [2015, 2016, 2017, 2018, 2019, 2020],
                "test_years": [2021, 2022],
                "months": list(range(1, 13)),
            },
            "ia": {
                "train_years": [2014, 2015, 2016, 2017],
                "valid_years": [2019, 2020, 2023, 2024],
                "test_years": [2018, 2021, 2022],
                "months": list(range(1, 13)),
            },
            "nv": {
                "train_years": [2016, 2017, 2018, 2023],
                "valid_years": [2019, 2020, 2024],
                "test_years": [2021, 2022],
                "months": list(range(1, 13)),
            },
            "mt": {
                "train_years": [2016, 2017, 2018, 2023],
                "valid_years": [2019, 2020, 2024],
                "test_years": [2021, 2022],
                "months": list(range(1, 13)),
            },
            "md": {
                "train_years": [2016, 2017, 2020],
                "valid_years": [2018, 2019],
                "test_years": [2015, 2021, 2022],
                "months": list(range(1, 13)),
            },
        }
        state_name = traffic_state_name
        if name.startswith("traffic_accident_"):
            suffix = name.replace("traffic_accident_", "", 1)
            if suffix:
                state_name = suffix.upper()
        state_key = str(state_name).strip().lower()
        default_split = traffic_state_defaults.get(
            state_key,
            traffic_state_defaults["de"],
        )
        return _load_traffic_accident_features(
            state_name=state_name,
            data_dir=traffic_data_dir,
            embeddings_dir=traffic_embeddings_dir,
            node_feature_type=traffic_node_feature_type,
            vision_feature_type=traffic_vision_feature_type,
            train_years=traffic_train_years if traffic_train_years is not None else list(default_split["train_years"]),
            valid_years=traffic_valid_years if traffic_valid_years is not None else list(default_split["valid_years"]),
            test_years=traffic_test_years if traffic_test_years is not None else list(default_split["test_years"]),
            months=traffic_months if traffic_months is not None else list(default_split["months"]),
            include_dynamic_node_features=traffic_include_dynamic_node_features,
            include_static_edge_features=traffic_include_static_edge_features,
            include_dynamic_edge_features=traffic_include_dynamic_edge_features,
            negative_ratio=traffic_negative_ratio,
            label_threshold=traffic_label_threshold,
            normalize_modalities=traffic_normalize_modalities,
            seed=int(seed),
        )

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
