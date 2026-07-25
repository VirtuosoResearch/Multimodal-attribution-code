"""Pairwise loss readout for PID-similar source-group fine-tunes.

This is the loss-side companion to pid_matched_embedding_similarity_eval.py.
It uses existing base/source fine-tune checkpoints and evaluates the intended
unit of comparison: two different source groups with similar PID profiles,
each fine-tuned from the same base checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fisher_geometry.data import load_dataset, make_loader
from fisher_geometry.fisher_metrics import compute_loss
from fisher_geometry.verify_actual_loss_gap_sampled import resolve_eval_group_indices
from fisher_geometry.verify_humor_fisher_gpt_metrics import load_model, resolve_requested_device
from kernel_surrogate.real_data_cmis import make_model_args


DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT
    / "src"
    / "fisher_geometry"
    / "verification_results"
    / "real_motivating_example"
    / "pid_matched_group_finetune_base1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two PID-similar source fine-tunes by loss on the same eval group."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--eval-split", choices=["auto", "test", "val", "train"], default="auto")
    parser.add_argument("--eval-group", type=int, default=None)
    parser.add_argument("--eval-group-selection", choices=["largest_source", "largest_any"], default="largest_source")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-eval-group-size", type=int, default=5)
    parser.add_argument("--source-pair-top-k", type=int, default=20)
    parser.add_argument("--source-pair-max-pid-distance", type=float, default=None)
    parser.add_argument(
        "--pid-profile-space",
        choices=["raw_zscore", "raw_minmax", "clipped_norm"],
        default="raw_zscore",
        help=(
            "Feature space used to decide PID-near source pairs. "
            "raw_zscore uses signed LSMI [R,U1,U2,S] with dataset-level z-score; "
            "clipped_norm preserves the older clipped-positive normalized profile."
        ),
    )
    parser.add_argument("--output-suffix", default="source_pair_loss")
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    result_paths = discover_result_paths(args.experiment_dir, args.datasets, int(args.seed))
    if not result_paths:
        raise FileNotFoundError(f"no pid-matched finetune JSONs found in {args.experiment_dir}")

    summaries = []
    for result_path in result_paths:
        print(f"Reading {result_path}", flush=True)
        output = evaluate_one_dataset(
            result_path=result_path,
            device=str(device),
            feature_device=args.feature_device,
            eval_split=str(args.eval_split),
            eval_group=args.eval_group,
            eval_group_selection=str(args.eval_group_selection),
            batch_size=args.batch_size,
            min_eval_group_size=int(args.min_eval_group_size),
            source_pair_top_k=int(args.source_pair_top_k),
            source_pair_max_pid_distance=args.source_pair_max_pid_distance,
            pid_profile_space=str(args.pid_profile_space),
        )
        output_json = result_path.with_name(
            result_path.stem.replace("_pid_matched_group_finetune", f"_pid_matched_{args.output_suffix}") + ".json"
        )
        output_md = output_json.with_suffix(".md")
        output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
        output_md.write_text(dataset_markdown(output), encoding="utf-8")
        print(dataset_markdown(output), flush=True)
        print(f"Wrote {output_json}", flush=True)
        print(f"Wrote {output_md}", flush=True)
        summaries.append(dict(output["summary"]))

    summary_output = {
        "config": {
            "experiment_dir": str(args.experiment_dir),
            "seed": int(args.seed),
            "device": str(device),
            "eval_split": str(args.eval_split),
            "eval_group": args.eval_group,
            "eval_group_selection": str(args.eval_group_selection),
            "min_eval_group_size": int(args.min_eval_group_size),
            "source_pair_top_k": int(args.source_pair_top_k),
            "source_pair_max_pid_distance": args.source_pair_max_pid_distance,
            "pid_profile_space": str(args.pid_profile_space),
            "output_suffix": str(args.output_suffix),
        },
        "summaries": summaries,
    }
    summary_json = args.experiment_dir / f"summary_pid_matched_{args.output_suffix}_all_datasets.json"
    summary_md = summary_json.with_suffix(".md")
    summary_json.write_text(json.dumps(summary_output, indent=2, allow_nan=True), encoding="utf-8")
    summary_md.write_text(summary_markdown(summary_output), encoding="utf-8")
    print(summary_markdown(summary_output), flush=True)
    print(f"Wrote {summary_json}", flush=True)
    print(f"Wrote {summary_md}", flush=True)


def discover_result_paths(experiment_dir: Path, datasets: Sequence[str] | None, seed: int) -> list[Path]:
    if datasets:
        return [
            experiment_dir / f"{dataset}_seed{seed}_pid_matched_group_finetune.json"
            for dataset in datasets
        ]
    return sorted(experiment_dir.glob(f"*_seed{seed}_pid_matched_group_finetune.json"))


def evaluate_one_dataset(
    *,
    result_path: Path,
    device: str,
    feature_device: str | None,
    eval_split: str,
    eval_group: int | None,
    eval_group_selection: str,
    batch_size: int | None,
    min_eval_group_size: int,
    source_pair_top_k: int,
    source_pair_max_pid_distance: float | None,
    pid_profile_space: str,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = dict(result["config"])
    dataset = str(config["dataset"])
    seed = int(config["seed"])
    payload_path = Path(config["payload_path"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload_config = dict(payload["config"])
    effective_eval_split = resolve_eval_split(config, eval_split)

    data = load_dataset(
        dataset,
        seed,
        text_backbone=str(payload_config.get("text_backbone", "clip_cached")),
        vision_backbone=str(payload_config.get("vision_backbone", "clip_cached")),
        feature_device=str(feature_device or config.get("feature_device") or payload_config.get("feature_device", device)),
        backbone_batch_size=int(payload_config.get("backbone_batch_size", 32)),
        download_datasets=bool(payload_config.get("download_datasets", False)),
        backbone_cache_dir=payload_config.get("backbone_cache_dir"),
    )
    if f"{effective_eval_split}_x" not in data:
        raise KeyError(f"{dataset}: split {effective_eval_split!r} is unavailable")
    task = str(data["task"])

    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    eval_group_indices = resolve_eval_group_indices(
        data=data,
        group_indices=group_indices,
        eval_split=effective_eval_split,
    )
    source_info = collect_source_info(result["records"])
    selected_eval_group = select_eval_group(
        explicit=eval_group,
        source_groups=sorted(source_info),
        eval_group_indices=eval_group_indices,
        selection=eval_group_selection,
        min_size=min_eval_group_size,
    )
    eval_indices = eval_group_indices[selected_eval_group]
    if len(eval_indices) < min_eval_group_size:
        raise ValueError(
            f"{dataset}: eval group {selected_eval_group} has {len(eval_indices)} samples "
            f"(< {min_eval_group_size})"
        )

    model_args = make_model_args(payload_config, device=device)
    model_args.eval_batch_size = int(batch_size or config.get("eval_batch_size") or model_args.eval_batch_size)
    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    model_type = str(config.get("model", payload_config.get("model", "cross_attention")))

    all_source_pairs = build_source_pair_rows(source_info, pid_profile_space=pid_profile_space)
    add_pid_close_percentiles(all_source_pairs)
    source_pairs = select_pid_similar_source_pairs(
        all_pairs=all_source_pairs,
        top_k=source_pair_top_k,
        max_pid_distance=source_pair_max_pid_distance,
    )
    global_normalizers = build_normalizers(
        all_source_pairs,
        ["source_pair_pid_distance", "FIC_gap", "test_loss_gap"],
    )
    selected_normalizers = build_normalizers(
        source_pairs,
        ["source_pair_pid_distance", "FIC_gap", "test_loss_gap"],
    )
    add_normalized_values(
        pairs=source_pairs,
        normalizers=selected_normalizers,
    )
    if not source_pairs:
        raise ValueError(f"{dataset}: no source-source pairs available")

    source_finetunes = dict(result.get("source_finetunes", {}))
    needed_sources = sorted({int(pair["source_group_a"]) for pair in source_pairs} | {int(pair["source_group_b"]) for pair in source_pairs})
    fixed_eval_loss_by_source = {}
    for source_group in needed_sources:
        finetune_info = source_finetunes.get(str(source_group)) or source_finetunes.get(source_group)
        if not finetune_info or "checkpoint" not in finetune_info:
            raise KeyError(f"{dataset}: missing finetune checkpoint for source group {source_group}")
        model = load_model(
            Path(finetune_info["checkpoint"]),
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=device,
        )
        fixed_eval_loss_by_source[source_group] = group_loss(
            model,
            data=data,
            split=effective_eval_split,
            indices=eval_indices,
            task=task,
            batch_size=int(model_args.eval_batch_size),
            device=device,
        )

    records = []
    for pair in source_pairs:
        source_a = int(pair["source_group_a"])
        source_b = int(pair["source_group_b"])
        info_a = source_info[source_a]
        info_b = source_info[source_b]
        fic_a = float(info_a["FIC_base_source"])
        fic_b = float(info_b["FIC_base_source"])
        fixed_loss_a = float(fixed_eval_loss_by_source[source_a])
        fixed_loss_b = float(fixed_eval_loss_by_source[source_b])
        global_loss_a = info_a.get("finetuned_test_loss")
        global_loss_b = info_b.get("finetuned_test_loss")
        fic_gap = float(pair["FIC_gap"])
        heldout_group_loss_gap = float(abs(fixed_loss_a - fixed_loss_b))
        test_loss_gap = none_or_float(pair.get("test_loss_gap"))
        pid_gap_over_range = gap_over_global_range(
            float(pair["source_pair_pid_distance"]),
            global_normalizers,
            "source_pair_pid_distance",
        )
        fic_gap_over_range = gap_over_global_range(fic_gap, global_normalizers, "FIC_gap")
        test_loss_gap_over_range = (
            None
            if test_loss_gap is None
            else gap_over_global_range(test_loss_gap, global_normalizers, "test_loss_gap")
        )
        records.append(
            {
                "source_group_a": source_a,
                "source_group_b": source_b,
                "source_a_size": int(info_a["source_size"]),
                "source_b_size": int(info_b["source_size"]),
                "source_pair_pid_distance": float(pair["source_pair_pid_distance"]),
                "PID_close_percentile": float(pair["PID_close_percentile"]),
                "PID_gap_over_global_range": pid_gap_over_range,
                "PID_gap_norm": float(pair["source_pair_pid_distance_norm"]),
                "source_pair_pid_distance_norm": float(pair["source_pair_pid_distance_norm"]),
                "FIC_base_source_a": fic_a,
                "FIC_base_source_b": fic_b,
                "FIC_gap": fic_gap,
                "FIC_gap_over_global_range": fic_gap_over_range,
                "FIC_gap_norm": float(pair["FIC_gap_norm"]),
                "abs_delta_FIC_base_source": fic_gap,
                "signed_delta_FIC_b_minus_a": float(fic_b - fic_a),
                "fixed_eval_split": effective_eval_split,
                "fixed_eval_group": int(selected_eval_group),
                "fixed_eval_group_size": int(len(eval_indices)),
                "fixed_eval_loss_a": fixed_loss_a,
                "fixed_eval_loss_b": fixed_loss_b,
                "heldout_group_loss_gap": heldout_group_loss_gap,
                "abs_delta_fixed_eval_loss": heldout_group_loss_gap,
                "signed_delta_fixed_eval_loss_b_minus_a": float(fixed_loss_b - fixed_loss_a),
                "test_loss_a": none_or_float(global_loss_a),
                "test_loss_b": none_or_float(global_loss_b),
                "test_loss_gap": test_loss_gap,
                "test_loss_gap_over_global_range": test_loss_gap_over_range,
                "test_loss_gap_norm": none_or_float(pair.get("test_loss_gap_norm")),
                "global_test_loss_a": none_or_float(global_loss_a),
                "global_test_loss_b": none_or_float(global_loss_b),
                "abs_delta_global_test_loss": test_loss_gap,
            }
        )

    summary = summarize_pair_records(
        dataset=dataset,
        config={**config, "pid_profile_space": pid_profile_space},
        records=records,
        eval_split=effective_eval_split,
        eval_group=selected_eval_group,
        eval_group_size=len(eval_indices),
    )
    return {
        "config": {
            "dataset": dataset,
            "seed": seed,
            "source_result_path": str(result_path),
            "payload_path": str(payload_path),
            "base_checkpoint": str(config["base_checkpoint"]),
            "model": model_type,
            "device": device,
            "feature_device": str(feature_device or config.get("feature_device") or payload_config.get("feature_device", device)),
            "fixed_eval_split": effective_eval_split,
            "fixed_eval_group": int(selected_eval_group),
            "fixed_eval_group_size": int(len(eval_indices)),
            "eval_group_selection": eval_group_selection,
            "source_pair_top_k": int(source_pair_top_k),
            "source_pair_max_pid_distance": source_pair_max_pid_distance,
            "pid_profile_space": pid_profile_space,
            "normalization": {
                key: {"min": float(value[0]), "max": float(value[1]), "range": float(value[2])}
                for key, value in selected_normalizers.items()
            },
            "global_pair_normalization_reference": {
                key: {"min": float(value[0]), "max": float(value[1]), "range": float(value[2])}
                for key, value in global_normalizers.items()
            },
            "base_groups": [int(group) for group in config.get("base_groups", [])],
            "finetune_train_data": str(config.get("finetune_train_data", "")),
            "pair_unit": "source_group_to_source_group",
            "fic_contrast": "FIC_gap = abs(FIC(base_group_or_base_subset, source_a) - FIC(base_group_or_base_subset, source_b))",
            "core_readout": "test_loss_gap = abs(test_loss(source_a_finetune) - test_loss(source_b_finetune))",
        },
        "summary": summary,
        "records": records,
    }


def group_loss(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    split: str,
    indices: Sequence[int],
    task: str,
    batch_size: int,
    device: str,
) -> float:
    index = torch.as_tensor(indices, dtype=torch.long)
    loader = make_loader(
        data[f"{split}_x"][index],
        data[f"{split}_y"][index],
        data[f"{split}_labels"][index],
        int(batch_size),
        False,
    )
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.inference_mode():
        for bx, by, bl in loader:
            bx = bx.to(device)
            by = by.to(device)
            bl = bl.to(device)
            logits = model(bx, by, use_x=True, use_y=True)
            loss = compute_loss(task, logits, bl)
            batch = int(bl.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch
            total_samples += batch
    return float(total_loss / max(total_samples, 1))


def collect_source_info(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_source: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        by_source.setdefault(int(row["source_group"]), []).append(row)
    info = {}
    for source, rows in by_source.items():
        first = rows[0]
        fic_values = [float(row["FIC_base_source"]) for row in rows]
        info[source] = {
            "source_group": int(source),
            "source_size": int(first["source_size"]),
            "source_pid_profile": dict(first["source_pid_profile"]),
            "FIC_base_source": float(np.mean(fic_values)),
            "FIC_base_source_values": fic_values,
            "finetuned_test_loss": first.get("finetuned_test_loss"),
        }
    return info


def build_source_pair_rows(
    source_info: dict[int, dict[str, Any]],
    *,
    pid_profile_space: str,
) -> list[dict[str, Any]]:
    sources = sorted(source_info)
    pid_vectors = build_pid_vectors(source_info, profile_space=pid_profile_space)
    rows = []
    for i, source_a in enumerate(sources):
        vec_a = pid_vectors[source_a]
        for source_b in sources[i + 1 :]:
            vec_b = pid_vectors[source_b]
            dist = float(np.linalg.norm(vec_a - vec_b))
            fic_gap = float(abs(float(source_info[source_a]["FIC_base_source"]) - float(source_info[source_b]["FIC_base_source"])))
            test_loss_gap = none_or_abs_delta(
                source_info[source_a].get("finetuned_test_loss"),
                source_info[source_b].get("finetuned_test_loss"),
            )
            rows.append(
                {
                    "source_group_a": int(source_a),
                    "source_group_b": int(source_b),
                    "source_pair_pid_distance": dist,
                    "FIC_gap": fic_gap,
                    "test_loss_gap": test_loss_gap,
                }
            )
    return rows


def build_pid_vectors(
    source_info: dict[int, dict[str, Any]],
    *,
    profile_space: str,
) -> dict[int, np.ndarray]:
    sources = sorted(source_info)
    if profile_space == "clipped_norm":
        return {
            source: clipped_norm_pid_vector(source_info[source]["source_pid_profile"])
            for source in sources
        }

    raw = np.stack(
        [raw_signed_pid_vector(source_info[source]["source_pid_profile"]) for source in sources],
        axis=0,
    ).astype(np.float64)
    if profile_space == "raw_zscore":
        center = raw.mean(axis=0, keepdims=True)
        scale = raw.std(axis=0, keepdims=True)
        scale = np.where(scale <= 0.0, 1.0, scale)
        normalized = (raw - center) / scale
    elif profile_space == "raw_minmax":
        min_value = raw.min(axis=0, keepdims=True)
        max_value = raw.max(axis=0, keepdims=True)
        scale = max_value - min_value
        scale = np.where(scale <= 0.0, 1.0, scale)
        normalized = (raw - min_value) / scale
    else:
        raise ValueError(f"unsupported pid_profile_space: {profile_space!r}")
    return {
        source: normalized[index]
        for index, source in enumerate(sources)
    }


def select_pid_similar_source_pairs(
    *,
    all_pairs: Sequence[dict[str, Any]],
    top_k: int,
    max_pid_distance: float | None,
) -> list[dict[str, Any]]:
    rows = []
    for row in all_pairs:
        if max_pid_distance is not None and float(row["source_pair_pid_distance"]) > float(max_pid_distance):
            continue
        rows.append(dict(row))
    rows.sort(key=lambda row: (float(row["source_pair_pid_distance"]), int(row["source_group_a"]), int(row["source_group_b"])))
    return rows[: max(1, int(top_k))]


def add_pid_close_percentiles(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["source_pair_pid_distance"]),
            int(row["source_group_a"]),
            int(row["source_group_b"]),
        ),
    )
    total = float(len(ordered))
    for rank, row in enumerate(ordered, start=1):
        row["PID_close_percentile"] = float(rank / total)


def build_normalizers(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> dict[str, tuple[float, float, float]]:
    normalizers = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        if not values:
            normalizers[key] = (float("nan"), float("nan"), float("nan"))
            continue
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        range_value = float(max_value - min_value)
        normalizers[key] = (min_value, max_value, range_value)
    return normalizers


def add_normalized_values(
    *,
    pairs: Sequence[dict[str, Any]],
    normalizers: dict[str, tuple[float, float, float]],
) -> None:
    alias = {
        "source_pair_pid_distance": "source_pair_pid_distance_norm",
        "FIC_gap": "FIC_gap_norm",
        "test_loss_gap": "test_loss_gap_norm",
    }
    for row in pairs:
        for key, out_key in alias.items():
            value = row.get(key)
            if value is None:
                row[out_key] = None
                continue
            min_value, _, range_value = normalizers[key]
            if not math.isfinite(range_value) or range_value <= 0.0:
                row[out_key] = 0.0
            else:
                row[out_key] = float((float(value) - min_value) / range_value)


def gap_over_global_range(
    value: float,
    normalizers: dict[str, tuple[float, float, float]],
    key: str,
) -> float:
    _, _, range_value = normalizers[key]
    if not math.isfinite(range_value) or range_value <= 0.0:
        return 0.0
    return float(float(value) / range_value)


def clipped_norm_pid_vector(profile: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(profile.get("norm_R", 0.0)),
            float(profile.get("norm_U1", 0.0)),
            float(profile.get("norm_U2", 0.0)),
            float(profile.get("norm_S", 0.0)),
        ],
        dtype=np.float64,
    )


def raw_signed_pid_vector(profile: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(profile.get("R", 0.0)),
            float(profile.get("U1", 0.0)),
            float(profile.get("U2", 0.0)),
            float(profile.get("S", 0.0)),
        ],
        dtype=np.float64,
    )


def resolve_eval_split(config: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    return str(config.get("full_eval_split") or "test")


def select_eval_group(
    *,
    explicit: int | None,
    source_groups: Sequence[int],
    eval_group_indices: Sequence[Sequence[int]],
    selection: str,
    min_size: int,
) -> int:
    if explicit is not None:
        return int(explicit)
    candidates = list(source_groups) if selection == "largest_source" else list(range(len(eval_group_indices)))
    sized = [
        (len(eval_group_indices[int(group)]), int(group))
        for group in candidates
        if 0 <= int(group) < len(eval_group_indices) and len(eval_group_indices[int(group)]) >= min_size
    ]
    if not sized:
        sized = [
            (len(indices), group)
            for group, indices in enumerate(eval_group_indices)
            if len(indices) >= min_size
        ]
    if not sized:
        raise ValueError(f"no eval group has at least {min_size} samples")
    return int(max(sized, key=lambda item: (item[0], -item[1]))[1])


def summarize_pair_records(
    *,
    dataset: str,
    config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    eval_split: str,
    eval_group: int,
    eval_group_size: int,
) -> dict[str, Any]:
    if not records:
        return {
            "dataset": dataset,
            "source_pairs": 0,
            "fixed_eval_split": eval_split,
            "fixed_eval_group": int(eval_group),
            "fixed_eval_group_size": int(eval_group_size),
        }
    pid = np.asarray([float(row["source_pair_pid_distance"]) for row in records], dtype=np.float64)
    pid_close_percentile = np.asarray([float(row["PID_close_percentile"]) for row in records], dtype=np.float64)
    pid_over_range = np.asarray([float(row["PID_gap_over_global_range"]) for row in records], dtype=np.float64)
    pid_norm = np.asarray([float(row["source_pair_pid_distance_norm"]) for row in records], dtype=np.float64)
    fic_delta = np.asarray([float(row["FIC_gap"]) for row in records], dtype=np.float64)
    fic_delta_over_range = np.asarray([float(row["FIC_gap_over_global_range"]) for row in records], dtype=np.float64)
    fic_delta_norm = np.asarray([float(row["FIC_gap_norm"]) for row in records], dtype=np.float64)
    heldout_group_loss_gap = np.asarray([float(row["heldout_group_loss_gap"]) for row in records], dtype=np.float64)
    global_loss_delta_values = [
        float(row["test_loss_gap"])
        for row in records
        if row.get("test_loss_gap") is not None
    ]
    global_loss_delta_norm_values = [
        float(row["test_loss_gap_norm"])
        for row in records
        if row.get("test_loss_gap_norm") is not None
    ]
    global_loss_delta_over_range_values = [
        float(row["test_loss_gap_over_global_range"])
        for row in records
        if row.get("test_loss_gap_over_global_range") is not None
    ]
    ordered = sorted(records, key=lambda row: float(row["FIC_gap_norm"]))
    third = max(1, len(ordered) // 3)
    low = ordered[:third]
    high = ordered[-third:]
    low_heldout = mean_value(low, "heldout_group_loss_gap")
    high_heldout = mean_value(high, "heldout_group_loss_gap")
    low_test = mean_optional_value(low, "test_loss_gap")
    high_test = mean_optional_value(high, "test_loss_gap")
    low_test_norm = mean_optional_value(low, "test_loss_gap_norm")
    high_test_norm = mean_optional_value(high, "test_loss_gap_norm")
    return {
        "dataset": dataset,
        "source_pairs": int(len(records)),
        "base_groups": [int(group) for group in config.get("base_groups", [])],
        "pid_profile_space": str(config.get("pid_profile_space", "")),
        "fixed_eval_split": eval_split,
        "fixed_eval_group": int(eval_group),
        "fixed_eval_group_size": int(eval_group_size),
        "source_pair_pid_distance_mean": float(np.mean(pid)),
        "source_pair_pid_distance_median": float(np.median(pid)),
        "source_pair_pid_distance_max": float(np.max(pid)),
        "PID_close_percentile_mean": float(np.mean(pid_close_percentile)),
        "PID_close_percentile_median": float(np.median(pid_close_percentile)),
        "PID_close_percentile_max": float(np.max(pid_close_percentile)),
        "PID_gap_over_global_range_mean": float(np.mean(pid_over_range)),
        "PID_gap_over_global_range_median": float(np.median(pid_over_range)),
        "PID_gap_over_global_range_max": float(np.max(pid_over_range)),
        "PID_gap_norm_mean": float(np.mean(pid_norm)),
        "PID_gap_norm_median": float(np.median(pid_norm)),
        "PID_gap_norm_max": float(np.max(pid_norm)),
        "FIC_gap_mean": float(np.mean(fic_delta)),
        "FIC_gap_min": float(np.min(fic_delta)),
        "FIC_gap_max": float(np.max(fic_delta)),
        "FIC_gap_over_global_range_mean": float(np.mean(fic_delta_over_range)),
        "FIC_gap_over_global_range_min": float(np.min(fic_delta_over_range)),
        "FIC_gap_over_global_range_max": float(np.max(fic_delta_over_range)),
        "FIC_gap_norm_mean": float(np.mean(fic_delta_norm)),
        "FIC_gap_norm_min": float(np.min(fic_delta_norm)),
        "FIC_gap_norm_max": float(np.max(fic_delta_norm)),
        "abs_delta_FIC_mean": float(np.mean(fic_delta)),
        "abs_delta_FIC_min": float(np.min(fic_delta)),
        "abs_delta_FIC_max": float(np.max(fic_delta)),
        "heldout_group_loss_gap_mean": float(np.mean(heldout_group_loss_gap)),
        "test_loss_gap_mean": float(np.mean(global_loss_delta_values)) if global_loss_delta_values else None,
        "test_loss_gap_over_global_range_mean": (
            float(np.mean(global_loss_delta_over_range_values)) if global_loss_delta_over_range_values else None
        ),
        "test_loss_gap_over_global_range_median": (
            float(np.median(global_loss_delta_over_range_values)) if global_loss_delta_over_range_values else None
        ),
        "test_loss_gap_over_global_range_max": (
            float(np.max(global_loss_delta_over_range_values)) if global_loss_delta_over_range_values else None
        ),
        "test_loss_gap_norm_mean": float(np.mean(global_loss_delta_norm_values)) if global_loss_delta_norm_values else None,
        "abs_delta_fixed_eval_loss_mean": float(np.mean(heldout_group_loss_gap)),
        "abs_delta_global_test_loss_mean": float(np.mean(global_loss_delta_values)) if global_loss_delta_values else None,
        "low_FIC_gap_mean": mean_value(low, "FIC_gap"),
        "high_FIC_gap_mean": mean_value(high, "FIC_gap"),
        "low_FIC_gap_over_global_range_mean": mean_value(low, "FIC_gap_over_global_range"),
        "high_FIC_gap_over_global_range_mean": mean_value(high, "FIC_gap_over_global_range"),
        "low_FIC_gap_norm_mean": mean_value(low, "FIC_gap_norm"),
        "high_FIC_gap_norm_mean": mean_value(high, "FIC_gap_norm"),
        "low_FIC_gap_heldout_group_loss_gap_mean": low_heldout,
        "high_FIC_gap_heldout_group_loss_gap_mean": high_heldout,
        "high_minus_low_heldout_group_loss_gap": float(high_heldout - low_heldout),
        "low_FIC_gap_test_loss_gap_mean": low_test,
        "high_FIC_gap_test_loss_gap_mean": high_test,
        "high_minus_low_test_loss_gap": none_or_delta(high_test, low_test),
        "low_FIC_gap_test_loss_gap_norm_mean": low_test_norm,
        "high_FIC_gap_test_loss_gap_norm_mean": high_test_norm,
        "high_minus_low_test_loss_gap_norm": none_or_delta(high_test_norm, low_test_norm),
        "abs_high_minus_low_test_loss_gap": none_or_abs_delta(high_test, low_test),
        "abs_high_minus_low_test_loss_gap_norm": none_or_abs_delta(high_test_norm, low_test_norm),
        "low_delta_FIC_mean": mean_value(low, "FIC_gap"),
        "high_delta_FIC_mean": mean_value(high, "FIC_gap"),
        "low_delta_FIC_fixed_eval_loss_delta_mean": low_heldout,
        "high_delta_FIC_fixed_eval_loss_delta_mean": high_heldout,
        "high_minus_low_fixed_eval_loss_delta": float(high_heldout - low_heldout),
        "low_delta_FIC_global_test_loss_delta_mean": low_test,
        "high_delta_FIC_global_test_loss_delta_mean": high_test,
        "high_minus_low_global_test_loss_delta": none_or_delta(high_test, low_test),
        "abs_high_minus_low_global_test_loss_delta": none_or_abs_delta(high_test, low_test),
        "low_delta_FIC_pairs": source_pair_labels(low),
        "high_delta_FIC_pairs": source_pair_labels(high),
    }


def source_pair_labels(records: Sequence[dict[str, Any]]) -> list[str]:
    return [
        f"{int(row['source_group_a'])}-{int(row['source_group_b'])}"
        for row in records
    ]


def mean_value(records: Sequence[dict[str, Any]], key: str) -> float:
    if not records:
        return float("nan")
    return float(np.mean([float(row[key]) for row in records]))


def mean_optional_value(records: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def none_or_abs_delta(value_a: Any, value_b: Any) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(abs(float(value_a) - float(value_b)))


def none_or_delta(value_a: Any, value_b: Any) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(float(value_a) - float(value_b))


def dataset_markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        f"# {summary['dataset']} PID-Similar Source-Pair Test Loss",
        "",
        f"- test/eval split used by ckpt summary: `{summary['fixed_eval_split']}`",
        f"- optional heldout eval group: `{summary['fixed_eval_group']}` (n={summary['fixed_eval_group_size']})",
        f"- base groups: `{summary.get('base_groups', [])}`",
        f"- source pairs: `{summary['source_pairs']}`",
        f"- PID profile space: `{summary.get('pid_profile_space', '')}`",
        (
            f"- PID-close percentile mean/median/max among all source pairs: "
            f"`{fmt(summary.get('PID_close_percentile_mean'))}` / "
            f"`{fmt(summary.get('PID_close_percentile_median'))}` / "
            f"`{fmt(summary.get('PID_close_percentile_max'))}`"
        ),
        (
            f"- PID gap / global range mean/median/max: "
            f"`{fmt(summary.get('PID_gap_over_global_range_mean'))}` / "
            f"`{fmt(summary.get('PID_gap_over_global_range_median'))}` / "
            f"`{fmt(summary.get('PID_gap_over_global_range_max'))}`"
        ),
        (
            f"- selected-pair normalized PID distance mean/median/max: "
            f"`{fmt(summary.get('PID_gap_norm_mean'))}` / "
            f"`{fmt(summary.get('PID_gap_norm_median'))}` / "
            f"`{fmt(summary.get('PID_gap_norm_max'))}`"
        ),
        (
            f"- raw source-pair PID distance mean/median/max: "
            f"`{fmt(summary.get('source_pair_pid_distance_mean'))}` / "
            f"`{fmt(summary.get('source_pair_pid_distance_median'))}` / "
            f"`{fmt(summary.get('source_pair_pid_distance_max'))}`"
        ),
        "",
        "| FIC_gap bin | mean FIC_gap_norm | mean test_loss_gap_norm | raw mean FIC_gap | raw mean test_loss_gap | source pairs |",
        "|---|---:|---:|---:|---:|---|",
        (
            f"| low third | {fmt(summary.get('low_FIC_gap_norm_mean'))} | "
            f"{fmt(summary.get('low_FIC_gap_test_loss_gap_norm_mean'))} | "
            f"{fmt(summary.get('low_FIC_gap_mean'))} | "
            f"{fmt(summary.get('low_FIC_gap_test_loss_gap_mean'))} | "
            f"`{summary.get('low_delta_FIC_pairs', [])}` |"
        ),
        (
            f"| high third | {fmt(summary.get('high_FIC_gap_norm_mean'))} | "
            f"{fmt(summary.get('high_FIC_gap_test_loss_gap_norm_mean'))} | "
            f"{fmt(summary.get('high_FIC_gap_mean'))} | "
            f"{fmt(summary.get('high_FIC_gap_test_loss_gap_mean'))} | "
            f"`{summary.get('high_delta_FIC_pairs', [])}` |"
        ),
        "",
        f"- abs high-low normalized test_loss_gap: `{fmt(summary.get('abs_high_minus_low_test_loss_gap_norm'))}`",
        f"- abs high-low test_loss_gap: `{fmt(summary.get('abs_high_minus_low_test_loss_gap'))}`",
        f"- signed high-low normalized test_loss_gap: `{fmt(summary.get('high_minus_low_test_loss_gap_norm'))}`",
        f"- signed high-low test_loss_gap: `{fmt(summary.get('high_minus_low_test_loss_gap'))}`",
        f"- high-low heldout_group_loss_gap: `{fmt(summary.get('high_minus_low_heldout_group_loss_gap'))}`",
    ]
    return "\n".join(lines) + "\n"


def summary_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# PID-Similar Source-Pair Test Loss All Datasets",
        "",
        (
            "| dataset | source pairs | base groups | test/eval split | heldout group | heldout n | "
            "PID close pct max | PID_gap/range median | PID_gap/range max | FIC_gap/range mean | "
            "test_loss_gap/range mean | mean test_loss_gap_norm | low-FIC test_loss_gap_norm | high-FIC test_loss_gap_norm | abs high-low test_loss_gap_norm | "
            "raw abs high-low test_loss_gap |"
        ),
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in output["summaries"]:
        lines.append(
            "| {dataset} | {pairs} | `{base}` | {split} | {group} | {n} | {pid_pct_max} | "
            "{pid_range_med} | {pid_range_max} | {fic_range_mean} | {loss_range} | {loss} | {low} | {high} | {effect} | {global_effect} |".format(
                dataset=summary["dataset"],
                pairs=int(summary.get("source_pairs", 0)),
                base=summary.get("base_groups", []),
                split=summary.get("fixed_eval_split", ""),
                group=summary.get("fixed_eval_group", ""),
                n=summary.get("fixed_eval_group_size", ""),
                pid_pct_max=fmt(summary.get("PID_close_percentile_max")),
                pid_range_med=fmt(summary.get("PID_gap_over_global_range_median")),
                pid_range_max=fmt(summary.get("PID_gap_over_global_range_max")),
                fic_range_mean=fmt(summary.get("FIC_gap_over_global_range_mean")),
                loss_range=fmt(summary.get("test_loss_gap_over_global_range_mean")),
                loss=fmt(summary.get("test_loss_gap_norm_mean")),
                low=fmt(summary.get("low_FIC_gap_test_loss_gap_norm_mean")),
                high=fmt(summary.get("high_FIC_gap_test_loss_gap_norm_mean")),
                effect=fmt(summary.get("abs_high_minus_low_test_loss_gap_norm")),
                global_effect=fmt(summary.get("abs_high_minus_low_test_loss_gap")),
            )
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4g}"


if __name__ == "__main__":
    main()
