"""Pairwise embedding readout for PID-similar source-group fine-tunes.

This evaluates the intended contrast:
  two different source groups with similar PID profiles are fine-tuned from the
  same base checkpoint, then their shared/fusion-layer embeddings are compared
  on the same eval group. The explanatory variable is the FIC difference between
  the base group(s) and each source group.
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fisher_geometry.data import load_dataset, make_loader
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
        description="Compare two PID-similar source fine-tunes by shared-layer embedding cosine."
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
    parser.add_argument("--output-suffix", default="source_pair_embedding_similarity")
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

    source_pairs = select_pid_similar_source_pairs(
        source_info=source_info,
        top_k=source_pair_top_k,
        max_pid_distance=source_pair_max_pid_distance,
    )
    if not source_pairs:
        raise ValueError(f"{dataset}: no source-source pairs available")

    source_finetunes = dict(result.get("source_finetunes", {}))
    needed_sources = sorted({int(pair["source_group_a"]) for pair in source_pairs} | {int(pair["source_group_b"]) for pair in source_pairs})
    embeddings_by_source = {}
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
        embeddings_by_source[source_group] = collect_shared_embeddings(
            model,
            data=data,
            split=effective_eval_split,
            indices=eval_indices,
            batch_size=int(model_args.eval_batch_size),
            device=device,
        )

    records = []
    for pair in source_pairs:
        source_a = int(pair["source_group_a"])
        source_b = int(pair["source_group_b"])
        metrics = embedding_similarity(embeddings_by_source[source_a], embeddings_by_source[source_b])
        info_a = source_info[source_a]
        info_b = source_info[source_b]
        fic_a = float(info_a["FIC_base_source"])
        fic_b = float(info_b["FIC_base_source"])
        ft_loss_a = info_a.get("finetuned_test_loss")
        ft_loss_b = info_b.get("finetuned_test_loss")
        records.append(
            {
                "source_group_a": source_a,
                "source_group_b": source_b,
                "source_a_size": int(info_a["source_size"]),
                "source_b_size": int(info_b["source_size"]),
                "source_pair_pid_distance": float(pair["source_pair_pid_distance"]),
                "FIC_base_source_a": fic_a,
                "FIC_base_source_b": fic_b,
                "abs_delta_FIC_base_source": float(abs(fic_a - fic_b)),
                "signed_delta_FIC_b_minus_a": float(fic_b - fic_a),
                "finetuned_test_loss_a": none_or_float(ft_loss_a),
                "finetuned_test_loss_b": none_or_float(ft_loss_b),
                "abs_delta_finetuned_test_loss": none_or_abs_delta(ft_loss_a, ft_loss_b),
                "eval_embedding_split": effective_eval_split,
                "eval_embedding_group": int(selected_eval_group),
                "eval_embedding_group_size": int(len(eval_indices)),
                "embedding_cosine_mean_between_finetunes": float(metrics["cosine_mean"]),
                "embedding_cosine_std_between_finetunes": float(metrics["cosine_std"]),
                "embedding_drift_mean_between_finetunes": float(metrics["drift_mean"]),
                "embedding_drift_std_between_finetunes": float(metrics["drift_std"]),
                "embedding_l2_mean_between_finetunes": float(metrics["l2_mean"]),
            }
        )

    summary = summarize_pair_records(
        dataset=dataset,
        config=config,
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
            "eval_embedding_split": effective_eval_split,
            "eval_embedding_group": int(selected_eval_group),
            "eval_embedding_group_size": int(len(eval_indices)),
            "eval_group_selection": eval_group_selection,
            "source_pair_top_k": int(source_pair_top_k),
            "source_pair_max_pid_distance": source_pair_max_pid_distance,
            "embedding_layer": shared_layer_name(model_type),
            "base_groups": [int(group) for group in config.get("base_groups", [])],
            "finetune_train_data": str(config.get("finetune_train_data", "")),
            "pair_unit": "source_group_to_source_group",
            "fic_contrast": "abs(FIC(base_group_or_base_subset, source_a) - FIC(base_group_or_base_subset, source_b))",
        },
        "summary": summary,
        "records": records,
    }


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


def select_pid_similar_source_pairs(
    *,
    source_info: dict[int, dict[str, Any]],
    top_k: int,
    max_pid_distance: float | None,
) -> list[dict[str, Any]]:
    sources = sorted(source_info)
    rows = []
    for i, source_a in enumerate(sources):
        vec_a = pid_vector(source_info[source_a]["source_pid_profile"])
        for source_b in sources[i + 1 :]:
            vec_b = pid_vector(source_info[source_b]["source_pid_profile"])
            dist = float(np.linalg.norm(vec_a - vec_b))
            if max_pid_distance is not None and dist > float(max_pid_distance):
                continue
            rows.append(
                {
                    "source_group_a": int(source_a),
                    "source_group_b": int(source_b),
                    "source_pair_pid_distance": dist,
                }
            )
    rows.sort(key=lambda row: (float(row["source_pair_pid_distance"]), int(row["source_group_a"]), int(row["source_group_b"])))
    return rows[: max(1, int(top_k))]


def pid_vector(profile: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(profile.get("norm_R", 0.0)),
            float(profile.get("norm_U1", 0.0)),
            float(profile.get("norm_U2", 0.0)),
            float(profile.get("norm_S", 0.0)),
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


def collect_shared_embeddings(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    split: str,
    indices: Sequence[int],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    index = torch.as_tensor(indices, dtype=torch.long)
    loader = make_loader(
        data[f"{split}_x"][index],
        data[f"{split}_y"][index],
        data[f"{split}_labels"][index],
        int(batch_size),
        False,
    )
    chunks = []
    model.eval()
    with torch.inference_mode():
        for bx, by, _ in loader:
            bx = bx.to(device)
            by = by.to(device)
            chunks.append(shared_embedding(model, bx, by).detach().float().cpu())
    return torch.cat(chunks, dim=0)


def shared_embedding(model: torch.nn.Module, feat_x: torch.Tensor, feat_y: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "pool_norm") and hasattr(model, "layers") and hasattr(model, "cls_x"):
        return cross_attention_shared_embedding(model, feat_x, feat_y)
    if hasattr(model, "shared_mlp"):
        h = model.shared_mlp(model.proj_x(feat_x)) + model.shared_mlp(model.proj_y(feat_y))
        return h / 2.0
    if hasattr(model, "fusion_in") and hasattr(model, "fusion_hidden"):
        h = model.fusion_in(torch.cat([feat_x, feat_y], dim=1))
        return model.fusion_hidden(h)
    raise TypeError(f"unsupported model type for shared embedding extraction: {type(model).__name__}")


def cross_attention_shared_embedding(
    model: torch.nn.Module,
    feat_x: torch.Tensor,
    feat_y: torch.Tensor,
) -> torch.Tensor:
    batch_size = feat_x.shape[0]
    hx = model.proj_x(feat_x).unsqueeze(1)
    hy = model.proj_y(feat_y).unsqueeze(1)
    seq_x = torch.cat([model.cls_x.expand(batch_size, -1, -1), hx], dim=1)
    seq_y = torch.cat([model.cls_y.expand(batch_size, -1, -1), hy], dim=1)
    for layer in model.layers:
        seq_x, seq_y = layer(seq_x, seq_y, use_x=True, use_y=True)
    pooled = torch.cat([seq_x[:, 0], seq_y[:, 0]], dim=1)
    return model.pool_norm(pooled)


def embedding_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> dict[str, float]:
    cosine = F.cosine_similarity(emb_a, emb_b, dim=1)
    drift = 1.0 - cosine
    l2 = torch.linalg.vector_norm(emb_b - emb_a, dim=1)
    return {
        "cosine_mean": float(cosine.mean().item()),
        "cosine_std": float(cosine.std(unbiased=False).item()),
        "drift_mean": float(drift.mean().item()),
        "drift_std": float(drift.std(unbiased=False).item()),
        "l2_mean": float(l2.mean().item()),
    }


def shared_layer_name(model_type: str) -> str:
    if model_type == "cross_attention":
        return "pool_norm(concat(CLS_x, CLS_y))"
    if model_type == "shared_mlp":
        return "mean(shared_mlp(proj_x), shared_mlp(proj_y))"
    if model_type == "concat_mlp":
        return "fusion_hidden(fusion_in(concat(x, y)))"
    return "pre_classifier_shared_embedding"


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
            "eval_embedding_split": eval_split,
            "eval_embedding_group": int(eval_group),
            "eval_embedding_group_size": int(eval_group_size),
        }
    pid = np.asarray([float(row["source_pair_pid_distance"]) for row in records], dtype=np.float64)
    fic_delta = np.asarray([float(row["abs_delta_FIC_base_source"]) for row in records], dtype=np.float64)
    drift = np.asarray([float(row["embedding_drift_mean_between_finetunes"]) for row in records], dtype=np.float64)
    cosine = np.asarray([float(row["embedding_cosine_mean_between_finetunes"]) for row in records], dtype=np.float64)
    ordered = sorted(records, key=lambda row: float(row["abs_delta_FIC_base_source"]))
    third = max(1, len(ordered) // 3)
    low = ordered[:third]
    high = ordered[-third:]
    low_drift = mean_value(low, "embedding_drift_mean_between_finetunes")
    high_drift = mean_value(high, "embedding_drift_mean_between_finetunes")
    low_cos = mean_value(low, "embedding_cosine_mean_between_finetunes")
    high_cos = mean_value(high, "embedding_cosine_mean_between_finetunes")
    return {
        "dataset": dataset,
        "source_pairs": int(len(records)),
        "base_groups": [int(group) for group in config.get("base_groups", [])],
        "eval_embedding_split": eval_split,
        "eval_embedding_group": int(eval_group),
        "eval_embedding_group_size": int(eval_group_size),
        "source_pair_pid_distance_mean": float(np.mean(pid)),
        "source_pair_pid_distance_median": float(np.median(pid)),
        "source_pair_pid_distance_max": float(np.max(pid)),
        "abs_delta_FIC_mean": float(np.mean(fic_delta)),
        "abs_delta_FIC_min": float(np.min(fic_delta)),
        "abs_delta_FIC_max": float(np.max(fic_delta)),
        "embedding_cosine_mean_between_finetunes": float(np.mean(cosine)),
        "embedding_drift_mean_between_finetunes": float(np.mean(drift)),
        "low_delta_FIC_mean": mean_value(low, "abs_delta_FIC_base_source"),
        "high_delta_FIC_mean": mean_value(high, "abs_delta_FIC_base_source"),
        "low_delta_FIC_embedding_drift_mean": low_drift,
        "high_delta_FIC_embedding_drift_mean": high_drift,
        "high_minus_low_embedding_drift": float(high_drift - low_drift),
        "low_delta_FIC_embedding_cosine_mean": low_cos,
        "high_delta_FIC_embedding_cosine_mean": high_cos,
        "high_minus_low_embedding_cosine": float(high_cos - low_cos),
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


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def none_or_abs_delta(value_a: Any, value_b: Any) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(abs(float(value_a) - float(value_b)))


def dataset_markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        f"# {summary['dataset']} PID-Similar Source-Pair Embedding Similarity",
        "",
        f"- eval embedding split: `{summary['eval_embedding_split']}`",
        f"- eval embedding group: `{summary['eval_embedding_group']}` (n={summary['eval_embedding_group_size']})",
        f"- base groups: `{summary.get('base_groups', [])}`",
        f"- source pairs: `{summary['source_pairs']}`",
        (
            f"- source-pair PID distance mean/median/max: "
            f"`{fmt(summary.get('source_pair_pid_distance_mean'))}` / "
            f"`{fmt(summary.get('source_pair_pid_distance_median'))}` / "
            f"`{fmt(summary.get('source_pair_pid_distance_max'))}`"
        ),
        "",
        "| | mean | embedding cos(ft_a, ft_b) | embedding drift 1-cos | source pairs |",
        "|---|---:|---:|---:|---|",
        (
            f"| low `abs_delta_FIC` third | {fmt(summary.get('low_delta_FIC_mean'))} | "
            f"{fmt(summary.get('low_delta_FIC_embedding_cosine_mean'))} | "
            f"{fmt(summary.get('low_delta_FIC_embedding_drift_mean'))} | "
            f"`{summary.get('low_delta_FIC_pairs', [])}` |"
        ),
        (
            f"| high `abs_delta_FIC` third | {fmt(summary.get('high_delta_FIC_mean'))} | "
            f"{fmt(summary.get('high_delta_FIC_embedding_cosine_mean'))} | "
            f"{fmt(summary.get('high_delta_FIC_embedding_drift_mean'))} | "
            f"`{summary.get('high_delta_FIC_pairs', [])}` |"
        ),
        "",
        f"- high-low embedding drift: `{fmt(summary.get('high_minus_low_embedding_drift'))}`",
        f"- high-low embedding cosine: `{fmt(summary.get('high_minus_low_embedding_cosine'))}`",
    ]
    return "\n".join(lines) + "\n"


def summary_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# PID-Similar Source-Pair Embedding Similarity All Datasets",
        "",
        (
            "| dataset | source pairs | base groups | eval split | eval group | eval n | "
            "PID pair dist median | PID pair dist max | mean abs Delta FIC | mean drift | "
            "low Delta-FIC drift | high Delta-FIC drift | high-low drift |"
        ),
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in output["summaries"]:
        lines.append(
            "| {dataset} | {pairs} | `{base}` | {split} | {group} | {n} | {pid_med} | {pid_max} | "
            "{fic_delta} | {drift} | {low} | {high} | {effect} |".format(
                dataset=summary["dataset"],
                pairs=int(summary.get("source_pairs", 0)),
                base=summary.get("base_groups", []),
                split=summary.get("eval_embedding_split", ""),
                group=summary.get("eval_embedding_group", ""),
                n=summary.get("eval_embedding_group_size", ""),
                pid_med=fmt(summary.get("source_pair_pid_distance_median")),
                pid_max=fmt(summary.get("source_pair_pid_distance_max")),
                fic_delta=fmt(summary.get("abs_delta_FIC_mean")),
                drift=fmt(summary.get("embedding_drift_mean_between_finetunes")),
                low=fmt(summary.get("low_delta_FIC_embedding_drift_mean")),
                high=fmt(summary.get("high_delta_FIC_embedding_drift_mean")),
                effect=fmt(summary.get("high_minus_low_embedding_drift")),
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
