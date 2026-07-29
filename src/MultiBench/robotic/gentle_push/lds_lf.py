from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import math
import multiprocessing as mp
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import fannypack
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

if not hasattr(np, "string_"):
    np.string_ = np.bytes_


ROOT = Path(__file__).resolve().parents[3]
ROBOTIC_ROOT = ROOT / "src" / "robotic"
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT, ROBOTIC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets.gentle_push.data_loader import (  # noqa: E402
    PushTask,
    SubsequenceDataset,
    _load_trajectories,
    dataset_urls,
)
from fusions.common_fusions import ConcatWithLinear  # noqa: E402
from pid_estimator.subset_loader import build_grouped_subset_loader  # noqa: E402
from training_structures.Supervised_Learning import MMDL  # noqa: E402
from unimodals.common_models import Identity, Reshape, Sequential, Transpose  # noqa: E402
from unimodals.gentle_push.head import GentlePushLateLSTM  # noqa: E402
import unimodals.gentle_push.layers as layers  # noqa: E402


Task = PushTask
DATASET_NAME = "gentle_push_lf"
MODALITY_NAMES = ("gripper_pos", "gripper_sensors", "image", "control")
DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train"
)
DEFAULT_RESULT_ROOT = (
    ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
)
DEFAULT_GENTLE_PUSH_CACHE = Path("/data/zhenshuo/datasets/gentle_push")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _drive_id_from_url(url: str) -> str:
    marker = "/d/"
    if marker not in url:
        raise ValueError(f"unsupported Google Drive URL: {url}")
    return url.split(marker, 1)[1].split("/", 1)[0]


def _ensure_fannypack_cache_aliases(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, url in dataset_urls.items():
        source = cache_dir / name
        if not source.exists():
            continue
        alias = cache_dir / f"{_drive_id_from_url(url)}-{name}"
        if alias.exists():
            try:
                if alias.samefile(source):
                    continue
            except OSError:
                pass
            if alias.stat().st_size == source.stat().st_size:
                continue
            alias.unlink()
        try:
            alias.symlink_to(source)
        except OSError:
            shutil.copy2(source, alias)


def _result_path(result_dir: Path, *, dataset_name: str, seed: int) -> Path:
    return result_dir / f"{dataset_name}_seed{seed}.json"


def _subset_artifact_relpath(
    *, dataset_name: str, seed: int, sample_index: int
) -> Path:
    return Path(dataset_name) / f"seed{seed}" / f"subset{sample_index + 1:03d}"


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    dataset_name: str,
    seed: int,
    sample_index: int,
    repeat_index: int,
) -> Path:
    return (
        checkpoint_dir
        / _subset_artifact_relpath(
            dataset_name=dataset_name,
            seed=seed,
            sample_index=sample_index,
        )
        / "checkpoints"
        / f"checkpoint_repeat{repeat_index + 1:02d}.pt"
    )


def _subset_metric_path(
    result_dir: Path, *, dataset_name: str, seed: int, sample_index: int
) -> Path:
    return (
        result_dir
        / _subset_artifact_relpath(
            dataset_name=dataset_name,
            seed=seed,
            sample_index=sample_index,
        )
        / "result.json"
    )


def _subset_group_index_path(
    result_dir: Path, *, dataset_name: str, seed: int, sample_index: int
) -> Path:
    return (
        result_dir
        / _subset_artifact_relpath(
            dataset_name=dataset_name,
            seed=seed,
            sample_index=sample_index,
        )
        / "group_index.json"
    )


def _subset_manifest_path(
    result_dir: Path, *, dataset_name: str, seed: int, sample_index: int
) -> Path:
    return (
        result_dir
        / _subset_artifact_relpath(
            dataset_name=dataset_name,
            seed=seed,
            sample_index=sample_index,
        )
        / "manifest.json"
    )


def _rate_token(rate: float) -> str:
    percent = float(rate) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return str(int(round(percent)))
    return f"{percent:.4g}".replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    if args.intervention == "auxiliary_acquisition":
        return (
            f"{DATASET_NAME}_"
            f"group_unit_{args.group_method}_"
            f"auxacq{_rate_token(args.data_keep_rate)}_"
            f"mod{_rate_token(args.modality_keep_rate)}_"
            f"{args.aux_mask_granularity}_"
            f"main-{args.main_modality}_"
            f"g{args.num_groups}_"
            f"trainavg{len(args.train_repeat_seeds)}"
        )
    return (
        f"{DATASET_NAME}_"
        f"group_unit_{args.group_method}_"
        f"data{_rate_token(args.data_keep_rate)}_"
        f"mod{_rate_token(args.modality_keep_rate)}_"
        f"g{args.num_groups}_"
        f"trainavg{len(args.train_repeat_seeds)}"
    )


def _resolve_output_dirs(args: argparse.Namespace) -> None:
    args.run_name = args.run_name or _default_run_name(args)
    if args.result_dir is None:
        args.result_dir = args.result_root / args.run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.checkpoint_root / args.run_name


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for key in (
        "checkpoint_dir",
        "checkpoint_root",
        "result_dir",
        "result_root",
        "group_cache_dir",
        "gentle_push_cache_dir",
    ):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(value)
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _config_matches(existing: dict[str, Any], args: argparse.Namespace) -> bool:
    existing_config = existing.get("config")
    if not isinstance(existing_config, dict):
        return False
    current_config = _serialize_args(args)
    existing_config = dict(existing_config)
    for key in ("num_subset_samples",):
        current_config.pop(key, None)
        existing_config.pop(key, None)
    return current_config == existing_config


def _resolved_groups_per_subset(num_groups: int, data_keep_rate: float) -> int:
    keep_rate = float(data_keep_rate)
    if keep_rate <= 0.0:
        raise ValueError(f"data_keep_rate={data_keep_rate} keeps no data")
    return min(int(num_groups), max(1, int(math.ceil(int(num_groups) * keep_rate))))


def _sample_subset_specs(
    *,
    grouped_loader,
    num_subset_samples: int,
    groups_per_subset: int,
    modality_keep_prob: float,
    subset_sampling_seed_base: int,
    existing_signatures: set[tuple[tuple[int, ...], bytes]],
) -> list[dict[str, Any]]:
    if groups_per_subset > grouped_loader.num_groups:
        raise ValueError(
            f"groups_per_subset={groups_per_subset} exceeds "
            f"available groups={grouped_loader.num_groups}"
        )

    sampled: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(int(num_subset_samples) * 500, 500)
    while len(sampled) < int(num_subset_samples):
        if attempts >= max_attempts:
            raise RuntimeError(
                "failed to sample enough unique LDS subsets; "
                f"requested={num_subset_samples}, collected={len(sampled)}"
            )
        subset_seed = int(subset_sampling_seed_base + attempts)
        rng = np.random.default_rng(subset_seed)
        group_ids = sorted(
            rng.choice(
                grouped_loader.num_groups,
                size=int(groups_per_subset),
                replace=False,
            ).tolist()
        )
        subset_indices = grouped_loader.get_subset_indices(group_ids).clone().cpu()
        modality_mask = (
            rng.random((int(subset_indices.numel()), grouped_loader.num_modalities))
            < float(modality_keep_prob)
        )
        signature = (
            tuple(int(group_id) for group_id in group_ids),
            np.packbits(
                modality_mask.reshape(-1).astype(np.uint8),
                bitorder="little",
            ).tobytes(),
        )
        attempts += 1
        if signature in existing_signatures:
            continue
        existing_signatures.add(signature)
        sampled.append(
            {
                "subset_seed": subset_seed,
                "group_ids": [int(group_id) for group_id in group_ids],
                "subset_indices": subset_indices,
                "modality_mask": modality_mask.astype(bool, copy=True),
            }
        )
    return sampled


def _sample_subset_specs_for_range(
    *,
    grouped_loader,
    start_index: int,
    stop_index: int,
    groups_per_subset: int,
    modality_keep_prob: float,
    subset_sampling_seed_base: int,
    intervention: str = "subset_retrain",
    main_modality_index: int = 0,
    aux_mask_granularity: str = "sample",
) -> list[dict[str, Any]]:
    if groups_per_subset > grouped_loader.num_groups:
        raise ValueError(
            f"groups_per_subset={groups_per_subset} exceeds "
            f"available groups={grouped_loader.num_groups}"
        )
    if start_index < 0 or stop_index < start_index:
        raise ValueError(
            f"invalid subset range: start_index={start_index}, stop_index={stop_index}"
        )

    sampled: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[int, ...], bytes]] = set()
    accepted_index = 0
    attempts = 0
    max_attempts = max(int(stop_index) * 500, 500)
    while accepted_index < int(stop_index):
        if attempts >= max_attempts:
            raise RuntimeError(
                "failed to sample enough unique LDS subsets; "
                f"requested_stop_index={stop_index}, collected={accepted_index}"
            )
        subset_seed = int(subset_sampling_seed_base + attempts)
        rng = np.random.default_rng(subset_seed)
        group_ids = sorted(
            rng.choice(
                grouped_loader.num_groups,
                size=int(groups_per_subset),
                replace=False,
            ).tolist()
        )
        if intervention == "auxiliary_acquisition":
            data_group_ids = list(range(int(grouped_loader.num_groups)))
            subset_indices = grouped_loader.get_subset_indices(data_group_ids).clone().cpu()
            modality_mask = np.zeros(
                (int(subset_indices.numel()), grouped_loader.num_modalities),
                dtype=bool,
            )
            modality_mask[:, int(main_modality_index)] = True
            aux_indices = [
                index
                for index in range(int(grouped_loader.num_modalities))
                if index != int(main_modality_index)
            ]
            selected_groups = set(int(group_id) for group_id in group_ids)
            start = 0
            for data_group_id, group_size in enumerate(grouped_loader.group_sizes):
                stop = start + int(group_size)
                if data_group_id in selected_groups:
                    if aux_mask_granularity == "group":
                        aux_keep = rng.random(len(aux_indices)) < float(modality_keep_prob)
                        modality_mask[start:stop, aux_indices] = aux_keep
                    else:
                        modality_mask[start:stop, aux_indices] = (
                            rng.random((int(group_size), len(aux_indices)))
                            < float(modality_keep_prob)
                        )
                start = stop
        else:
            data_group_ids = [int(group_id) for group_id in group_ids]
            subset_indices = grouped_loader.get_subset_indices(group_ids).clone().cpu()
            modality_mask = (
                rng.random((int(subset_indices.numel()), grouped_loader.num_modalities))
                < float(modality_keep_prob)
            )
        signature = (
            tuple(int(group_id) for group_id in group_ids),
            np.packbits(
                modality_mask.reshape(-1).astype(np.uint8),
                bitorder="little",
            ).tobytes(),
        )
        attempts += 1
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        if accepted_index >= int(start_index):
            sampled.append(
                {
                    "sample_index": int(accepted_index),
                    "subset_seed": subset_seed,
                    "group_ids": [int(group_id) for group_id in group_ids],
                    "data_group_ids": data_group_ids,
                    "subset_indices": subset_indices,
                    "modality_mask": modality_mask.astype(bool, copy=True),
                }
            )
        accepted_index += 1
    return sampled


def _existing_subset_signatures(
    runs: list[dict[str, Any]],
) -> set[tuple[tuple[int, ...], bytes]]:
    signatures: set[tuple[tuple[int, ...], bytes]] = set()
    for run in runs:
        group_ids = tuple(int(group_id) for group_id in run.get("group_ids", []))
        modality_mask = np.asarray(run.get("sample_modality_mask", []), dtype=bool)
        signatures.add(
            (
                group_ids,
                np.packbits(
                    modality_mask.reshape(-1).astype(np.uint8),
                    bitorder="little",
                ).tobytes(),
            )
        )
    return signatures


def _recover_completed_subset_runs(
    *,
    result_dir: Path,
    dataset_name: str,
    seed: int,
    num_subset_samples: int,
    existing_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recovered_by_index: dict[int, dict[str, Any]] = {}
    for run in existing_runs:
        if not isinstance(run, dict) or "sample_index" not in run:
            continue
        recovered_by_index[int(run["sample_index"])] = run

    for sample_index in range(int(num_subset_samples)):
        if sample_index in recovered_by_index:
            continue
        metric_path = _subset_metric_path(
            result_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=sample_index,
        )
        if not metric_path.exists():
            continue
        try:
            run_payload = json.loads(metric_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(run_payload, dict):
            continue
        if int(run_payload.get("sample_index", -1)) != sample_index:
            continue
        checkpoint_paths = [
            str(path)
            for path in run_payload.get("checkpoint_paths", [])
            if str(path)
        ]
        checkpoint_path = str(run_payload.get("checkpoint_path", ""))
        if checkpoint_path and checkpoint_path not in checkpoint_paths:
            checkpoint_paths.append(checkpoint_path)
        if checkpoint_paths and not all(Path(path).exists() for path in checkpoint_paths):
            continue
        recovered_by_index[sample_index] = run_payload

    return [
        recovered_by_index[index]
        for index in sorted(recovered_by_index)
        if index < int(num_subset_samples)
    ]


def _next_subset_index_from_artifacts(
    *,
    result_dir: Path,
    dataset_name: str,
    seed: int,
    completed_runs: list[dict[str, Any]],
) -> int:
    max_index = -1
    for run in completed_runs:
        if isinstance(run, dict) and "sample_index" in run:
            max_index = max(max_index, int(run["sample_index"]))

    subset_root = result_dir / Path(dataset_name) / f"seed{seed}"
    if subset_root.exists():
        for path in subset_root.glob("subset*"):
            if not path.is_dir():
                continue
            token = path.name.removeprefix("subset")
            if not token.isdigit():
                continue
            max_index = max(max_index, int(token) - 1)
    return max_index + 1


def _group_partition_payload(grouped_loader) -> dict[str, Any]:
    return {
        "cache_path": str(grouped_loader.cache_path),
        "group_method": str(grouped_loader.metadata.group_method),
        "num_groups": int(grouped_loader.num_groups),
        "num_modalities": int(grouped_loader.num_modalities),
        "modality_names": list(MODALITY_NAMES),
        "group_sizes": [int(size) for size in grouped_loader.group_sizes],
        "group_indices": [
            [int(index) for index in indices.tolist()]
            for indices in grouped_loader.group_indices
        ],
    }


def _subset_group_index_payload(
    *,
    dataset_name: str,
    seed: int,
    partition_seed: int,
    sample_index: int,
    subset_seed: int,
    group_ids: list[int],
    data_group_ids: list[int],
    subset_indices: torch.Tensor,
    sample_modality_mask: np.ndarray,
    grouped_loader,
) -> dict[str, Any]:
    group_index = []
    for group_id in group_ids:
        group_indices = grouped_loader.group_indices[int(group_id)].detach().cpu()
        group_index.append(
            {
                "group_id": int(group_id),
                "data_indices": [int(index) for index in group_indices.tolist()],
            }
        )
    return {
        "dataset": dataset_name,
        "seed": int(seed),
        "partition_seed": int(partition_seed),
        "sample_index": int(sample_index),
        "subset_seed": int(subset_seed),
        "num_groups": int(grouped_loader.num_groups),
        "group_method": str(grouped_loader.metadata.group_method),
        "num_modalities": int(grouped_loader.num_modalities),
        "modality_names": list(MODALITY_NAMES),
        "group_cache_path": str(grouped_loader.cache_path),
        "group_ids": [int(group_id) for group_id in group_ids],
        "data_group_ids": [int(group_id) for group_id in data_group_ids],
        "group_index": group_index,
        "data_indices": [int(index) for index in subset_indices.tolist()],
        "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    if not runs:
        return {
            "num_completed_subsets": 0,
            "mean_test_loss": float("nan"),
            "std_test_loss": float("nan"),
            "mean_test_score": float("nan"),
            "std_test_score": float("nan"),
            "mean_effective_sample_keep_ratio": float("nan"),
            "mean_modality_keep_ratio": float("nan"),
        }
    losses = np.asarray([float(run["test_loss"]) for run in runs], dtype=np.float64)
    scores = np.asarray([float(run["test_score"]) for run in runs], dtype=np.float64)
    sample_keep = np.asarray(
        [float(run["effective_sample_keep_ratio"]) for run in runs],
        dtype=np.float64,
    )
    modality_keep = np.asarray(
        [float(run["modality_keep_ratio_over_selected"]) for run in runs],
        dtype=np.float64,
    )
    return {
        "num_completed_subsets": int(len(runs)),
        "mean_test_loss": float(losses.mean()),
        "std_test_loss": float(losses.std()),
        "mean_test_score": float(scores.mean()),
        "std_test_score": float(scores.std()),
        "mean_effective_sample_keep_ratio": float(sample_keep.mean()),
        "mean_modality_keep_ratio": float(modality_keep.mean()),
    }


def _initial_result_payload(
    *,
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
    grouped_loader,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "seed": int(seed),
        "task": "regression",
        "metric_name": "mse",
        "config": _serialize_args(args),
        "group_partition": _group_partition_payload(grouped_loader),
        "runs": [],
        "summary": _summarize_runs([]),
    }


def build_lf_model(subsequence_length: int) -> MMDL:
    encoders = [
        Sequential(
            Transpose(0, 1),
            layers.observation_pos_layers(64),
            GentlePushLateLSTM(64, 256),
            Transpose(0, 1),
        ),
        Sequential(
            Transpose(0, 1),
            layers.observation_sensors_layers(64),
            GentlePushLateLSTM(64, 256),
            Transpose(0, 1),
        ),
        Sequential(
            Transpose(0, 1),
            Reshape([-1, 1, 32, 32]),
            layers.observation_image_layers(64),
            Reshape([int(subsequence_length), -1, 64]),
            GentlePushLateLSTM(64, 256),
            Transpose(0, 1),
        ),
        Sequential(
            Transpose(0, 1),
            layers.control_layers(64),
            GentlePushLateLSTM(64, 256),
            Transpose(0, 1),
        ),
    ]
    fusion = ConcatWithLinear(256 * 4, 2, concat_dim=2)
    return MMDL(encoders, fusion, Identity())


def _batch_to_device(
    batch: list[torch.Tensor] | tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    inputs = [item.float().to(device) for item in batch[:-1]]
    target = batch[-1].float().to(device)
    return inputs, target


def _train_one_model(
    *,
    train_loader,
    valid_loader,
    args: argparse.Namespace,
    train_seed: int,
) -> tuple[MMDL, list[float], list[float], float]:
    set_seed(train_seed)
    device = torch.device(args.device)
    model = build_lf_model(args.subsequence_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.MSELoss()
    epoch_train_losses: list[float] = []
    epoch_valid_losses: list[float] = []
    best_valid_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            inputs, target = _batch_to_device(batch, device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_val))
            optimizer.step()
            batch_size = int(target.shape[0])
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        train_loss = running_loss / max(total_samples, 1)
        valid_metrics = _evaluate_regression(
            model,
            valid_loader,
            device=args.device,
            max_sample_losses=0,
        )
        valid_loss = float(valid_metrics["loss"])
        epoch_train_losses.append(float(train_loss))
        epoch_valid_losses.append(float(valid_loss))
        print(
            f"    epoch {epoch + 1}/{args.epochs}: "
            f"train_mse={train_loss:.6g} valid_mse={valid_loss:.6g}",
            flush=True,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, epoch_train_losses, epoch_valid_losses, float(best_valid_loss)


def _evaluate_regression(
    model: torch.nn.Module,
    loader,
    *,
    device: str,
    max_sample_losses: int,
) -> dict[str, float | list[float] | int | str]:
    resolved_device = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    sample_losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, target = _batch_to_device(batch, resolved_device)
            pred = model(inputs)
            per_sample_loss = F.mse_loss(pred, target, reduction="none").reshape(
                target.shape[0],
                -1,
            ).mean(dim=1)
            batch_size = int(target.shape[0])
            total_loss += float(per_sample_loss.sum().item())
            total_samples += batch_size
            if len(sample_losses) < int(max_sample_losses):
                remaining = int(max_sample_losses) - len(sample_losses)
                sample_losses.extend(
                    float(value)
                    for value in per_sample_loss[:remaining].detach().cpu().tolist()
                )
    if total_samples == 0:
        raise ValueError("evaluation loader produced no batches")
    loss = total_loss / total_samples
    return {
        "loss": float(loss),
        "mse": float(loss),
        "score": float(-loss),
        "metric_name": "mse",
        "sample_losses": sample_losses,
        "num_sample_losses": int(len(sample_losses)),
    }


def _build_clean_dataloaders(args: argparse.Namespace):
    train_dataset = SubsequenceDataset(
        Task.get_train_trajectories(**args.dataset_args),
        int(args.subsequence_length),
    )
    valid_dataset = SubsequenceDataset(
        Task.get_eval_trajectories(**args.dataset_args),
        int(args.subsequence_length),
    )
    test_dataset = SubsequenceDataset(
        _load_trajectories("gentle_push_300.hdf5", **args.dataset_args),
        int(args.subsequence_length),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=bool(args.drop_last),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.data_keep_rate <= 1.0:
        raise ValueError(f"data_keep_rate must be in (0, 1], got {args.data_keep_rate}")
    if not 0.0 < args.modality_keep_rate <= 1.0:
        raise ValueError(
            f"modality_keep_rate must be in (0, 1], got {args.modality_keep_rate}"
        )
    if args.num_subset_samples <= 0:
        raise ValueError("num_subset_samples must be positive")
    if args.num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if not args.train_repeat_seeds:
        raise ValueError("train_repeat_seeds must contain at least one seed")
    if args.max_test_loss_samples < 0:
        raise ValueError("max_test_loss_samples must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device} but CUDA is unavailable")
    if int(args.parallel_workers) <= 0:
        raise ValueError("parallel_workers must be positive")


def _run_one_subset(
    *,
    train_loader,
    valid_loader,
    test_loader,
    grouped_loader,
    args: argparse.Namespace,
    dataset_name: str,
    seed: int,
    partition_seed: int,
    offset: int,
    subset_spec: dict[str, Any],
    groups_per_subset: int,
    test_loader_name: str,
) -> dict[str, Any]:
    subset_indices = torch.as_tensor(subset_spec["subset_indices"], dtype=torch.long).cpu()
    group_ids = [int(group_id) for group_id in subset_spec["group_ids"]]
    data_group_ids = [
        int(group_id)
        for group_id in subset_spec.get("data_group_ids", subset_spec["group_ids"])
    ]
    sample_modality_mask = np.asarray(subset_spec["modality_mask"], dtype=bool)
    kept_atom_indices = []
    for row_index, dataset_index in enumerate(subset_indices.tolist()):
        for modality_index, keep in enumerate(sample_modality_mask[row_index].tolist()):
            if keep:
                kept_atom_indices.append(
                    int(dataset_index) * grouped_loader.num_modalities + int(modality_index)
                )
    effective_sample_count = int(sample_modality_mask.any(axis=1).sum())
    subset_artifact_relpath = _subset_artifact_relpath(
        dataset_name=dataset_name,
        seed=seed,
        sample_index=offset,
    )
    metric_path = _subset_metric_path(
        args.result_dir,
        dataset_name=dataset_name,
        seed=seed,
        sample_index=offset,
    )
    group_index_path = _subset_group_index_path(
        args.result_dir,
        dataset_name=dataset_name,
        seed=seed,
        sample_index=offset,
    )
    manifest_path = _subset_manifest_path(
        args.result_dir,
        dataset_name=dataset_name,
        seed=seed,
        sample_index=offset,
    )
    _save_json(
        group_index_path,
        _subset_group_index_payload(
            dataset_name=dataset_name,
            seed=seed,
            partition_seed=partition_seed,
            sample_index=offset,
            subset_seed=int(subset_spec["subset_seed"]),
            group_ids=group_ids,
            data_group_ids=data_group_ids,
            subset_indices=subset_indices,
            sample_modality_mask=sample_modality_mask,
            grouped_loader=grouped_loader,
        ),
    )

    subset_train_loader = grouped_loader.build_subset_loader(
        data_group_ids,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        modality_mask=sample_modality_mask,
    )
    print(
        f"[{dataset_name}] seed={seed}: subset "
        f"{offset + 1}/{args.num_subset_samples} training "
        f"{len(args.train_repeat_seeds)} repeat seeds with {len(group_ids)} "
        f"intervention groups / {len(data_group_ids)} data groups / "
        f"{int(subset_indices.numel())} samples...",
        flush=True,
    )

    repeat_payloads: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    repeat_test_losses: list[float] = []
    repeat_test_scores: list[float] = []
    repeat_final_train_losses: list[float] = []
    repeat_best_valid_losses: list[float] = []
    repeat_sample_losses: list[list[float]] = []

    for repeat_index, repeat_seed in enumerate(args.train_repeat_seeds):
        train_seed = int(seed) * 1_000_000 + offset * 10_000 + int(repeat_seed)
        print(
            f"[{dataset_name}] seed={seed}: subset "
            f"{offset + 1}/{args.num_subset_samples} repeat "
            f"{repeat_index + 1}/{len(args.train_repeat_seeds)} "
            f"train_seed={train_seed}",
            flush=True,
        )
        model, epoch_train_losses, epoch_valid_losses, best_valid_loss = _train_one_model(
            train_loader=subset_train_loader,
            valid_loader=valid_loader,
            args=args,
            train_seed=train_seed,
        )
        test_metrics = _evaluate_regression(
            model,
            test_loader,
            device=args.device,
            max_sample_losses=int(args.max_test_loss_samples),
        )
        checkpoint_path = _checkpoint_path(
            args.checkpoint_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
            repeat_index=repeat_index,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "model_config": {
                    "model": "gentle_push_lf",
                    "subsequence_length": int(args.subsequence_length),
                },
                "dataset": dataset_name,
                "task": "regression",
                "metric_name": "mse",
                "seed": int(seed),
                "partition_seed": int(partition_seed),
                "subset_seed": int(subset_spec["subset_seed"]),
                "train_seed": int(train_seed),
                "train_repeat_seed": int(repeat_seed),
                "train_repeat_index": int(repeat_index),
                "sample_index": int(offset),
                "subset_artifact_relpath": str(subset_artifact_relpath),
                "num_groups": int(grouped_loader.num_groups),
                "group_method": str(grouped_loader.metadata.group_method),
                "groups_per_subset": int(groups_per_subset),
                "selected_group_count": int(len(group_ids)),
                "group_ids": [int(group_id) for group_id in group_ids],
                "data_group_ids": [int(group_id) for group_id in data_group_ids],
                "data_indices": [int(index) for index in subset_indices.tolist()],
                "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
                "kept_atom_indices": kept_atom_indices,
                "effective_sample_count": int(effective_sample_count),
                "epoch_train_losses": [float(loss) for loss in epoch_train_losses],
                "epoch_valid_losses": [float(loss) for loss in epoch_valid_losses],
                "best_valid_loss": float(best_valid_loss),
                "test_metrics": {
                    "loss": float(test_metrics["loss"]),
                    "mse": float(test_metrics["mse"]),
                    "score": float(test_metrics["score"]),
                    "metric_name": "mse",
                    "num_sample_losses": int(test_metrics["num_sample_losses"]),
                },
                "test_loader_name": test_loader_name,
                "group_cache_path": str(grouped_loader.cache_path),
                "group_index_path": str(group_index_path),
                "result_json_path": str(metric_path),
            },
            checkpoint_path,
        )
        checkpoint_paths.append(str(checkpoint_path))
        repeat_test_losses.append(float(test_metrics["loss"]))
        repeat_test_scores.append(float(test_metrics["score"]))
        repeat_final_train_losses.append(float(epoch_train_losses[-1]))
        repeat_best_valid_losses.append(float(best_valid_loss))
        repeat_sample_losses.append([float(loss) for loss in test_metrics["sample_losses"]])
        repeat_payloads.append(
            {
                "train_repeat_index": int(repeat_index),
                "train_repeat_seed": int(repeat_seed),
                "train_seed": int(train_seed),
                "checkpoint_path": str(checkpoint_path),
                "final_train_loss": float(epoch_train_losses[-1]),
                "best_valid_loss": float(best_valid_loss),
                "test_loss": float(test_metrics["loss"]),
                "test_mse": float(test_metrics["mse"]),
                "test_score": float(test_metrics["score"]),
                "num_test_sample_losses": int(test_metrics["num_sample_losses"]),
            }
        )

    min_sample_loss_count = min((len(losses) for losses in repeat_sample_losses), default=0)
    if min_sample_loss_count:
        sample_loss_matrix = np.asarray(
            [losses[:min_sample_loss_count] for losses in repeat_sample_losses],
            dtype=np.float64,
        )
        test_sample_losses = sample_loss_matrix.mean(axis=0).tolist()
        test_sample_loss_std = sample_loss_matrix.std(axis=0).tolist()
    else:
        test_sample_losses = []
        test_sample_loss_std = []

    test_loss_array = np.asarray(repeat_test_losses, dtype=np.float64)
    test_score_array = np.asarray(repeat_test_scores, dtype=np.float64)
    final_train_loss_array = np.asarray(repeat_final_train_losses, dtype=np.float64)
    best_valid_loss_array = np.asarray(repeat_best_valid_losses, dtype=np.float64)

    run_payload = {
        "sample_index": int(offset),
        "subset_seed": int(subset_spec["subset_seed"]),
        "subset_artifact_relpath": str(subset_artifact_relpath),
        "train_repeat_seeds": [int(repeat_seed) for repeat_seed in args.train_repeat_seeds],
        "train_repeats": repeat_payloads,
        "group_ids": [int(group_id) for group_id in group_ids],
        "data_group_ids": [int(group_id) for group_id in data_group_ids],
        "data_indices": [int(index) for index in subset_indices.tolist()],
        "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
        "kept_atom_indices": kept_atom_indices,
        "groups_per_subset": int(groups_per_subset),
        "selected_group_count": int(len(group_ids)),
        "selected_sample_count": int(subset_indices.numel()),
        "effective_sample_count": int(effective_sample_count),
        "subset_size": int(subset_indices.numel()),
        "effective_sample_keep_ratio": float(
            effective_sample_count / max(len(train_loader.dataset), 1)
        ),
        "selected_sample_keep_ratio": float(
            subset_indices.numel() / max(len(train_loader.dataset), 1)
        ),
        "modality_keep_ratio_over_selected": float(sample_modality_mask.mean()),
        "modality_missing_ratio_over_selected": float(1.0 - sample_modality_mask.mean()),
        "checkpoint_path": checkpoint_paths[0] if checkpoint_paths else "",
        "checkpoint_paths": checkpoint_paths,
        "final_train_loss": float(final_train_loss_array.mean()),
        "final_train_loss_std": float(final_train_loss_array.std()),
        "best_valid_loss": float(best_valid_loss_array.mean()),
        "best_valid_loss_std": float(best_valid_loss_array.std()),
        "test_loss": float(test_loss_array.mean()),
        "test_loss_std": float(test_loss_array.std()),
        "test_mse": float(test_loss_array.mean()),
        "test_mse_std": float(test_loss_array.std()),
        "test_score": float(test_score_array.mean()),
        "test_score_std": float(test_score_array.std()),
        "test_metric_name": "mse",
        "test_loader_name": test_loader_name,
        "max_test_loss_samples": int(args.max_test_loss_samples),
        "num_test_sample_losses": int(len(test_sample_losses)),
        "test_sample_indices": [int(index) for index in range(len(test_sample_losses))],
        "test_sample_losses": [float(loss) for loss in test_sample_losses],
        "test_sample_loss_std": [float(loss) for loss in test_sample_loss_std],
        "metric_json_path": str(metric_path),
        "result_json_path": str(metric_path),
        "group_index_path": str(group_index_path),
        "manifest_json_path": str(manifest_path),
    }
    _save_json(metric_path, run_payload)
    _save_json(
        manifest_path,
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "sample_index": int(offset),
            "subset_seed": int(subset_spec["subset_seed"]),
            "subset_artifact_relpath": str(subset_artifact_relpath),
            "group_ids": [int(group_id) for group_id in group_ids],
            "data_group_ids": [int(group_id) for group_id in data_group_ids],
            "group_index_path": str(group_index_path),
            "checkpoint_path": checkpoint_paths[0] if checkpoint_paths else "",
            "checkpoint_paths": checkpoint_paths,
            "result_json_path": str(metric_path),
            "test_loss": float(run_payload["test_loss"]),
            "test_mse": float(run_payload["test_mse"]),
            "test_score": float(run_payload["test_score"]),
            "test_metric_name": "mse",
        },
    )
    return run_payload


def _parallel_subset_worker(job: dict[str, Any]) -> dict[str, Any]:
    args = job["args"]
    seed = int(job["seed"])
    partition_seed = int(job["partition_seed"])
    dataset_name = str(job["dataset_name"])
    _ensure_fannypack_cache_aliases(args.gentle_push_cache_dir)
    fannypack.data.set_cache_path(str(args.gentle_push_cache_dir))
    set_seed(seed)
    train_loader, valid_loader, test_loader = _build_clean_dataloaders(args)
    grouped_loader = build_grouped_subset_loader(
        train_loader,
        dataset_name=dataset_name,
        num_groups=int(args.num_groups),
        seed=partition_seed,
        group_method=str(args.group_method),
        cache_dir=args.group_cache_dir,
    )
    return _run_one_subset(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        grouped_loader=grouped_loader,
        args=args,
        dataset_name=dataset_name,
        seed=seed,
        partition_seed=partition_seed,
        offset=int(job["offset"]),
        subset_spec=job["subset_spec"],
        groups_per_subset=int(job["groups_per_subset"]),
        test_loader_name=str(job["test_loader_name"]),
    )


def run_single(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    dataset_name = DATASET_NAME
    result_path = _result_path(args.result_dir, dataset_name=dataset_name, seed=seed)
    partition_seed = int(args.partition_seed if args.partition_seed is not None else seed)

    print(f"\n[{dataset_name}] seed={seed}: loading gentle_push data...", flush=True)
    set_seed(seed)
    train_loader, valid_loader, test_loader = _build_clean_dataloaders(args)
    test_loader_name = "clean_gentle_push_300"
    grouped_loader = build_grouped_subset_loader(
        train_loader,
        dataset_name=dataset_name,
        num_groups=int(args.num_groups),
        seed=partition_seed,
        group_method=str(args.group_method),
        cache_dir=args.group_cache_dir,
    )
    groups_per_subset = _resolved_groups_per_subset(
        grouped_loader.num_groups,
        args.data_keep_rate,
    )

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not _config_matches(payload, args):
            print(
                f"[{dataset_name}] seed={seed}: existing result uses "
                "different config, overwriting",
                flush=True,
            )
            payload = _initial_result_payload(
                dataset_name=dataset_name,
                seed=seed,
                args=args,
                grouped_loader=grouped_loader,
            )
        else:
            payload["config"] = _serialize_args(args)
            print(
                f"[{dataset_name}] seed={seed}: resuming with "
                f"{len(payload.get('runs', []))} finished subsets",
                flush=True,
            )
    else:
        payload = _initial_result_payload(
            dataset_name=dataset_name,
            seed=seed,
            args=args,
            grouped_loader=grouped_loader,
        )

    completed_runs = _recover_completed_subset_runs(
        result_dir=args.result_dir,
        dataset_name=dataset_name,
        seed=seed,
        num_subset_samples=int(args.num_subset_samples),
        existing_runs=payload.get("runs", []),
    )
    if len(completed_runs) != len(payload.get("runs", [])):
        print(
            f"[{dataset_name}] seed={seed}: recovered {len(completed_runs)} "
            "completed subset result artifacts",
            flush=True,
        )
        payload["runs"] = completed_runs
        payload["summary"] = _summarize_runs(completed_runs)
        _save_json(result_path, payload)
    next_subset_index = _next_subset_index_from_artifacts(
        result_dir=args.result_dir,
        dataset_name=dataset_name,
        seed=seed,
        completed_runs=completed_runs,
    )
    target_subset_count = int(args.num_subset_samples)
    if next_subset_index >= target_subset_count:
        payload["summary"] = _summarize_runs(completed_runs)
        _save_json(result_path, payload)
        return payload
    if next_subset_index > len(completed_runs):
        print(
            f"[{dataset_name}] seed={seed}: next subset index is "
            f"{next_subset_index + 1} from existing subset artifacts; "
            f"{len(completed_runs)} completed result files are recorded",
            flush=True,
        )

    subset_specs = _sample_subset_specs_for_range(
        grouped_loader=grouped_loader,
        start_index=next_subset_index,
        stop_index=target_subset_count,
        groups_per_subset=groups_per_subset,
        modality_keep_prob=float(args.modality_keep_rate),
        subset_sampling_seed_base=int(args.subset_sampling_seed_base) + int(seed) * 100_000,
        intervention=str(args.intervention),
        main_modality_index=MODALITY_NAMES.index(str(args.main_modality)),
        aux_mask_granularity=str(args.aux_mask_granularity),
    )

    if int(args.parallel_workers) > 1:
        jobs = []
        for subset_spec in subset_specs:
            offset = int(subset_spec["sample_index"])
            jobs.append(
                {
                "args": args,
                "seed": int(seed),
                "partition_seed": int(partition_seed),
                "dataset_name": dataset_name,
                "offset": int(offset),
                "subset_spec": subset_spec,
                "groups_per_subset": int(groups_per_subset),
                "test_loader_name": test_loader_name,
                }
            )
        print(
            f"[{dataset_name}] seed={seed}: training {len(jobs)} subsets with "
            f"{int(args.parallel_workers)} parallel workers",
            flush=True,
        )
        del train_loader, valid_loader, test_loader, grouped_loader
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(args.parallel_workers),
            mp_context=mp_context,
        ) as executor:
            future_to_offset = {
                executor.submit(_parallel_subset_worker, job): int(job["offset"])
                for job in jobs
            }
            for future in as_completed(future_to_offset):
                offset = future_to_offset[future]
                run_payload = future.result()
                completed_runs.append(run_payload)
                completed_runs.sort(key=lambda run: int(run["sample_index"]))
                payload["runs"] = completed_runs
                payload["summary"] = _summarize_runs(completed_runs)
                _save_json(result_path, payload)
                print(
                    f"[{dataset_name}] seed={seed}: subset {offset + 1}/"
                    f"{args.num_subset_samples} finished",
                    flush=True,
                )
        return payload

    for subset_spec in subset_specs:
        offset = int(subset_spec["sample_index"])
        subset_indices = subset_spec["subset_indices"]
        group_ids = subset_spec["group_ids"]
        data_group_ids = subset_spec.get("data_group_ids", group_ids)
        sample_modality_mask = np.asarray(subset_spec["modality_mask"], dtype=bool)
        kept_atom_indices = []
        for row_index, dataset_index in enumerate(subset_indices.tolist()):
            for modality_index, keep in enumerate(sample_modality_mask[row_index].tolist()):
                if keep:
                    kept_atom_indices.append(
                        int(dataset_index) * grouped_loader.num_modalities
                        + int(modality_index)
                    )
        effective_sample_count = int(sample_modality_mask.any(axis=1).sum())
        subset_artifact_relpath = _subset_artifact_relpath(
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
        )
        metric_path = _subset_metric_path(
            args.result_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
        )
        group_index_path = _subset_group_index_path(
            args.result_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
        )
        manifest_path = _subset_manifest_path(
            args.result_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
        )
        _save_json(
            group_index_path,
            _subset_group_index_payload(
                dataset_name=dataset_name,
                seed=seed,
                partition_seed=partition_seed,
                sample_index=offset,
                subset_seed=int(subset_spec["subset_seed"]),
                group_ids=group_ids,
                data_group_ids=data_group_ids,
                subset_indices=subset_indices,
                sample_modality_mask=sample_modality_mask,
                grouped_loader=grouped_loader,
            ),
        )

        subset_train_loader = grouped_loader.build_subset_loader(
            data_group_ids,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            modality_mask=sample_modality_mask,
        )
        print(
            f"[{dataset_name}] seed={seed}: subset "
            f"{offset + 1}/{args.num_subset_samples} training "
            f"{len(args.train_repeat_seeds)} repeat seeds on {len(group_ids)} "
            f"groups / {int(subset_indices.numel())} selected samples...",
            flush=True,
        )

        repeat_payloads: list[dict[str, Any]] = []
        checkpoint_paths: list[str] = []
        repeat_test_losses: list[float] = []
        repeat_test_scores: list[float] = []
        repeat_final_train_losses: list[float] = []
        repeat_best_valid_losses: list[float] = []
        repeat_sample_losses: list[list[float]] = []

        for repeat_index, repeat_seed in enumerate(args.train_repeat_seeds):
            train_seed = int(seed) * 1_000_000 + offset * 10_000 + int(repeat_seed)
            print(
                f"[{dataset_name}] seed={seed}: subset "
                f"{offset + 1}/{args.num_subset_samples} repeat "
                f"{repeat_index + 1}/{len(args.train_repeat_seeds)} "
                f"train_seed={train_seed}",
                flush=True,
            )
            model, epoch_train_losses, epoch_valid_losses, best_valid_loss = _train_one_model(
                train_loader=subset_train_loader,
                valid_loader=valid_loader,
                args=args,
                train_seed=train_seed,
            )
            test_metrics = _evaluate_regression(
                model,
                test_loader,
                device=args.device,
                max_sample_losses=int(args.max_test_loss_samples),
            )
            checkpoint_path = _checkpoint_path(
                args.checkpoint_dir,
                dataset_name=dataset_name,
                seed=seed,
                sample_index=offset,
                repeat_index=repeat_index,
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_config": {
                        "model": "gentle_push_lf",
                        "subsequence_length": int(args.subsequence_length),
                    },
                    "dataset": dataset_name,
                    "task": "regression",
                    "metric_name": "mse",
                    "seed": int(seed),
                    "partition_seed": int(partition_seed),
                    "subset_seed": int(subset_spec["subset_seed"]),
                    "train_seed": int(train_seed),
                    "train_repeat_seed": int(repeat_seed),
                    "train_repeat_index": int(repeat_index),
                    "sample_index": int(offset),
                    "subset_artifact_relpath": str(subset_artifact_relpath),
                    "num_groups": int(grouped_loader.num_groups),
                    "group_method": str(grouped_loader.metadata.group_method),
                    "groups_per_subset": int(groups_per_subset),
                    "selected_group_count": int(len(group_ids)),
                    "group_ids": [int(group_id) for group_id in group_ids],
                    "data_group_ids": [int(group_id) for group_id in data_group_ids],
                    "data_indices": [int(index) for index in subset_indices.tolist()],
                    "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
                    "kept_atom_indices": kept_atom_indices,
                    "effective_sample_count": int(effective_sample_count),
                    "epoch_train_losses": [float(loss) for loss in epoch_train_losses],
                    "epoch_valid_losses": [float(loss) for loss in epoch_valid_losses],
                    "best_valid_loss": float(best_valid_loss),
                    "test_metrics": {
                        "loss": float(test_metrics["loss"]),
                        "mse": float(test_metrics["mse"]),
                        "score": float(test_metrics["score"]),
                        "metric_name": "mse",
                        "num_sample_losses": int(test_metrics["num_sample_losses"]),
                    },
                    "test_loader_name": test_loader_name,
                    "group_cache_path": str(grouped_loader.cache_path),
                    "group_index_path": str(group_index_path),
                    "result_json_path": str(metric_path),
                },
                checkpoint_path,
            )
            checkpoint_paths.append(str(checkpoint_path))
            repeat_test_losses.append(float(test_metrics["loss"]))
            repeat_test_scores.append(float(test_metrics["score"]))
            repeat_final_train_losses.append(float(epoch_train_losses[-1]))
            repeat_best_valid_losses.append(float(best_valid_loss))
            repeat_sample_losses.append(
                [float(loss) for loss in test_metrics["sample_losses"]]
            )
            repeat_payloads.append(
                {
                    "train_repeat_index": int(repeat_index),
                    "train_repeat_seed": int(repeat_seed),
                    "train_seed": int(train_seed),
                    "checkpoint_path": str(checkpoint_path),
                    "final_train_loss": float(epoch_train_losses[-1]),
                    "best_valid_loss": float(best_valid_loss),
                    "test_loss": float(test_metrics["loss"]),
                    "test_mse": float(test_metrics["mse"]),
                    "test_score": float(test_metrics["score"]),
                    "num_test_sample_losses": int(test_metrics["num_sample_losses"]),
                }
            )

        min_sample_loss_count = min((len(losses) for losses in repeat_sample_losses), default=0)
        if min_sample_loss_count:
            sample_loss_matrix = np.asarray(
                [losses[:min_sample_loss_count] for losses in repeat_sample_losses],
                dtype=np.float64,
            )
            test_sample_losses = sample_loss_matrix.mean(axis=0).tolist()
            test_sample_loss_std = sample_loss_matrix.std(axis=0).tolist()
        else:
            test_sample_losses = []
            test_sample_loss_std = []

        test_loss_array = np.asarray(repeat_test_losses, dtype=np.float64)
        test_score_array = np.asarray(repeat_test_scores, dtype=np.float64)
        final_train_loss_array = np.asarray(repeat_final_train_losses, dtype=np.float64)
        best_valid_loss_array = np.asarray(repeat_best_valid_losses, dtype=np.float64)

        run_payload = {
            "sample_index": int(offset),
            "subset_seed": int(subset_spec["subset_seed"]),
            "subset_artifact_relpath": str(subset_artifact_relpath),
            "train_repeat_seeds": [int(repeat_seed) for repeat_seed in args.train_repeat_seeds],
            "train_repeats": repeat_payloads,
            "group_ids": [int(group_id) for group_id in group_ids],
            "data_group_ids": [int(group_id) for group_id in data_group_ids],
            "data_indices": [int(index) for index in subset_indices.tolist()],
            "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
            "kept_atom_indices": kept_atom_indices,
            "groups_per_subset": int(groups_per_subset),
            "selected_group_count": int(len(group_ids)),
            "selected_sample_count": int(subset_indices.numel()),
            "effective_sample_count": int(effective_sample_count),
            "subset_size": int(subset_indices.numel()),
            "effective_sample_keep_ratio": float(
                effective_sample_count / max(len(train_loader.dataset), 1)
            ),
            "selected_sample_keep_ratio": float(
                subset_indices.numel() / max(len(train_loader.dataset), 1)
            ),
            "modality_keep_ratio_over_selected": float(sample_modality_mask.mean()),
            "modality_missing_ratio_over_selected": float(
                1.0 - sample_modality_mask.mean()
            ),
            "checkpoint_path": checkpoint_paths[0] if checkpoint_paths else "",
            "checkpoint_paths": checkpoint_paths,
            "final_train_loss": float(final_train_loss_array.mean()),
            "final_train_loss_std": float(final_train_loss_array.std()),
            "best_valid_loss": float(best_valid_loss_array.mean()),
            "best_valid_loss_std": float(best_valid_loss_array.std()),
            "test_loss": float(test_loss_array.mean()),
            "test_loss_std": float(test_loss_array.std()),
            "test_mse": float(test_loss_array.mean()),
            "test_mse_std": float(test_loss_array.std()),
            "test_score": float(test_score_array.mean()),
            "test_score_std": float(test_score_array.std()),
            "test_metric_name": "mse",
            "test_loader_name": test_loader_name,
            "max_test_loss_samples": int(args.max_test_loss_samples),
            "num_test_sample_losses": int(len(test_sample_losses)),
            "test_sample_indices": [int(index) for index in range(len(test_sample_losses))],
            "test_sample_losses": [float(loss) for loss in test_sample_losses],
            "test_sample_loss_std": [float(loss) for loss in test_sample_loss_std],
            "metric_json_path": str(metric_path),
            "result_json_path": str(metric_path),
            "group_index_path": str(group_index_path),
            "manifest_json_path": str(manifest_path),
        }
        _save_json(metric_path, run_payload)
        _save_json(
            manifest_path,
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "sample_index": int(offset),
                "subset_seed": int(subset_spec["subset_seed"]),
                "subset_artifact_relpath": str(subset_artifact_relpath),
                "group_ids": [int(group_id) for group_id in group_ids],
                "data_group_ids": [int(group_id) for group_id in data_group_ids],
                "group_index_path": str(group_index_path),
                "checkpoint_path": checkpoint_paths[0] if checkpoint_paths else "",
                "checkpoint_paths": checkpoint_paths,
                "result_json_path": str(metric_path),
                "test_loss": float(run_payload["test_loss"]),
                "test_mse": float(run_payload["test_mse"]),
                "test_score": float(run_payload["test_score"]),
                "test_metric_name": "mse",
            },
        )

        completed_runs.append(run_payload)
        payload["runs"] = completed_runs
        payload["summary"] = _summarize_runs(completed_runs)
        _save_json(result_path, payload)

    return payload


def _write_combined_results(result_dir: Path, all_results: list[dict[str, Any]]) -> None:
    _save_json(
        result_dir / "all_results.json",
        {
            "num_dataset_runs": int(len(all_results)),
            "results": all_results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LDS-style group/subset retraining for the gentle_push LF model. "
            "Training samples are kmeans-partitioned into groups, subsets are sampled "
            "from group ids, and each subset is retrained from scratch."
        )
    )
    Task.add_dataset_arguments(parser)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gentle-push-cache-dir",
        type=Path,
        default=DEFAULT_GENTLE_PUSH_CACHE,
        help="Directory containing the gentle_push hdf5 files.",
    )
    parser.add_argument("--subsequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-val", type=float, default=8.0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--num-groups", type=int, default=25)
    parser.add_argument("--group-method", choices=["random", "kmeans"], default="kmeans")
    parser.add_argument(
        "--intervention",
        choices=["subset_retrain", "auxiliary_acquisition"],
        default="subset_retrain",
        help=(
            "subset_retrain removes unselected groups (legacy LDS protocol); "
            "auxiliary_acquisition keeps every sample's main modality and only "
            "reveals auxiliary modalities for sampled groups."
        ),
    )
    parser.add_argument(
        "--main-modality",
        choices=list(MODALITY_NAMES),
        default="control",
        help="Modality retained for every sample in auxiliary_acquisition mode.",
    )
    parser.add_argument(
        "--aux-mask-granularity",
        choices=["group", "sample"],
        default="group",
        help=(
            "Whether auxiliary modality availability is sampled once per selected "
            "group or independently per sample. Used only by auxiliary_acquisition."
        ),
    )
    parser.add_argument("--data-keep-rate", type=float, default=0.9)
    parser.add_argument("--modality-keep-rate", type=float, default=0.9)
    parser.add_argument("--data-missing-rate", type=float, default=None)
    parser.add_argument("--modality-missing-rate", type=float, default=None)
    parser.add_argument("--num-subset-samples", type=int, default=100)
    parser.add_argument("--train-repeat-seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of subset retraining processes to run in parallel. "
            "Use >1 to keep batch_size fixed while using more GPU memory."
        ),
    )
    parser.add_argument("--max-test-loss-samples", type=int, default=1000)
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--subset-sampling-seed-base", type=int, default=17_000)
    parser.add_argument("--group-cache-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.data_missing_rate is not None:
        args.data_keep_rate = 1.0 - float(args.data_missing_rate)
    if args.modality_missing_rate is not None:
        args.modality_keep_rate = 1.0 - float(args.modality_missing_rate)
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    args.dataset_args = Task.get_dataset_args(args)
    _ensure_fannypack_cache_aliases(args.gentle_push_cache_dir)
    fannypack.data.set_cache_path(str(args.gentle_push_cache_dir))

    _validate_args(args)
    _resolve_output_dirs(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        result = run_single(seed, args)
        all_results.append(result)
        print(
            f"[{DATASET_NAME}] seed={seed}: completed "
            f"{result['summary']['num_completed_subsets']} subsets, "
            f"mean test MSE={result['summary']['mean_test_loss']:.6g}",
            flush=True,
        )
    _write_combined_results(args.result_dir, all_results)


if __name__ == "__main__":
    main()
