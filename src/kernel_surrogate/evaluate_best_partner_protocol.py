from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset
from kernel_surrogate.real_data_cmis import (
    DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT,
    analytical_second_derivative_at_zero,
    build_group_mask_blocks,
    build_metric,
    compute_group_modality_gradients,
    fit_surrogate_method,
    load_or_train_full_model,
    make_model_args,
    normalize_cmis_scores,
    resolve_device,
    resolve_full_checkpoint_path,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "grouped50_subset100_data50_mod50"
)
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "best_partner_protocol_humor.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "best_partner_protocol_humor.md"


@dataclass
class ProbeData:
    train_x: np.ndarray
    train_y: np.ndarray
    train_labels: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_labels: np.ndarray
    num_classes: int
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-partner evaluation with real linear probing retrains.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument("--pretrained-full-checkpoint-root", type=Path, default=DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--full-checkpoint-dir", type=Path, default=ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis")
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    parser.add_argument("--gamma-scale", type=float, default=1e-3)
    parser.add_argument("--structured-shared-weight", type=float, default=1.0)
    parser.add_argument("--cmis-normalization-eps", type=float, default=1e-12)
    parser.add_argument("--anchor-groups", nargs="+", type=int, default=[0, 10, 20, 30, 40])
    parser.add_argument("--background-groups", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--random-baseline-repeats", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    payload_path = args.result_dir / f"{args.dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    probe_data = load_probe_data(args.dataset, int(args.seed), feature_device=str(args.feature_device))
    mask_blocks, _ = build_group_mask_blocks(payload)
    num_groups = int(payload["group_partition"]["num_groups"])
    group_indices = [np.asarray(indices, dtype=np.int64) for indices in payload["group_partition"]["group_indices"]]

    model_metric, group_img_repr, group_txt_repr = compute_model_metric_and_group_embeddings(
        args,
        payload=payload,
        config=config,
        dataset_name=args.dataset,
        device=device,
        group_indices=group_indices,
    )
    fit = fit_surrogate_method(
        "metric_rbf",
        mask_blocks,
        np.asarray([float(run["test_loss"]) for run in payload["runs"]], dtype=np.float64),
        model_metric,
        ridge_lambda=float(args.ridge_lambda),
        gamma_scale=float(args.gamma_scale),
        structured_shared_weight=float(args.structured_shared_weight),
    )
    raw_scores = analytical_second_derivative_at_zero(
        fit["train_masks"],
        fit["metric"],
        alpha=fit["alpha"],
        gamma=float(fit["gamma"]),
        num_groups=num_groups,
        atom_offset=int(fit.get("atom_offset", 0)),
    )
    cmis_scores = normalize_cmis_scores(raw_scores, eps=float(args.cmis_normalization_eps))
    cosine_matrix = normalized_rows(group_img_repr) @ normalized_rows(group_txt_repr).T

    rows: list[dict[str, Any]] = []
    for anchor in args.anchor_groups:
        if anchor < 0 or anchor >= num_groups:
            raise ValueError(f"anchor {anchor} outside [0,{num_groups})")
        candidates = [group for group in range(num_groups) if group != anchor]
        if args.candidate_limit is not None:
            candidates = candidates[: int(args.candidate_limit)]
        background = choose_background_groups(
            num_groups=num_groups,
            anchor=anchor,
            candidates=candidates,
            count=int(args.background_groups),
        )
        gt_rows, cached_losses = compute_ground_truth_rows(
            probe_data,
            group_indices=group_indices,
            anchor=anchor,
            candidates=candidates,
            background=background,
        )
        relevance = gt_to_relevance(np.asarray([row["gt_synergy"] for row in gt_rows], dtype=np.float64))
        gt_order = np.argsort([-row["gt_synergy"] for row in gt_rows])  # more negative is better
        methods = {
            "cmis": rank_desc({cand: float(cmis_scores[anchor, cand]) for cand in candidates}),
            "embedding_cosine": rank_desc({cand: float(cosine_matrix[anchor, cand]) for cand in candidates}),
            "one_d_attribution": rank_desc({cand: float(-(cached_losses[cand]["L01"] - cached_losses[cand]["L00"])) for cand in candidates}),
        }
        rng = np.random.default_rng(int(args.seed) + int(anchor))
        random_scores = []
        for _ in range(int(args.random_baseline_repeats)):
            perm = candidates.copy()
            rng.shuffle(perm)
            random_scores.append(compute_metrics_for_order(perm, gt_rows, relevance, gt_order, top_k=10))
        random_mean = {
            key: float(np.mean([row[key] for row in random_scores]))
            for key in ("ndcg_at_10", "spearman", "top10_recall")
        }
        method_metrics = {
            name: compute_metrics_for_order(order, gt_rows, relevance, gt_order, top_k=10)
            for name, order in methods.items()
        }
        rows.append(
            {
                "anchor_group": int(anchor),
                "background_groups": [int(group) for group in background],
                "num_candidates": len(candidates),
                "methods": method_metrics,
                "random_baseline_mean": random_mean,
                "ground_truth_top10": [int(gt_rows[idx]["candidate_group"]) for idx in gt_order[:10]],
            }
        )

    summary = summarize_anchor_rows(rows)
    output = {
        "config": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "anchor_groups": [int(anchor) for anchor in args.anchor_groups],
            "background_groups_per_anchor": int(args.background_groups),
            "gradient_params": str(args.gradient_params),
            "metric_normalization": str(args.metric_normalization),
            "random_baseline_repeats": int(args.random_baseline_repeats),
            "top_k": 10,
        },
        "summary": summary,
        "anchors": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def load_probe_data(dataset: str, seed: int, *, feature_device: str) -> ProbeData:
    data = load_dataset(
        dataset,
        seed,
        text_backbone="clip_cached",
        vision_backbone="clip_cached",
        feature_device=feature_device,
        download_datasets=False,
    )
    return ProbeData(
        train_x=data["train_x"].cpu().numpy(),
        train_y=data["train_y"].cpu().numpy(),
        train_labels=data["train_labels"].cpu().numpy().reshape(-1),
        test_x=data["test_x"].cpu().numpy(),
        test_y=data["test_y"].cpu().numpy(),
        test_labels=data["test_labels"].cpu().numpy().reshape(-1),
        num_classes=int(data["num_classes"]),
        task=str(data["task"]),
    )


def compute_model_metric_and_group_embeddings(
    args: argparse.Namespace,
    *,
    payload: dict[str, Any],
    config: dict[str, Any],
    dataset_name: str,
    device: Any,
    group_indices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_dataset(
        dataset_name,
        int(args.seed),
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(args.feature_device or config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    model_args = make_model_args(config, device=str(device))
    full_checkpoint = resolve_full_checkpoint_path(
        dataset=dataset_name,
        seed=int(args.seed),
        pretrained_root=args.pretrained_full_checkpoint_root,
        fallback_dir=args.full_checkpoint_dir,
    )
    model, _ = load_or_train_full_model(
        full_checkpoint,
        data=data,
        task=str(data["task"]),
        dim_x=int(data["train_x"].shape[1]),
        dim_y=int(data["train_y"].shape[1]),
        num_classes=int(data["num_classes"]),
        model_args=model_args,
        model_type=str(config.get("model", "cross_attention")),
        seed=int(args.seed),
        force_train=False,
    )
    model.to(device)
    model.eval()
    print(f"[{args.dataset}] computing last-layer gradient metric on {device}...", flush=True)
    gradients = compute_group_modality_gradients(
        model,
        data=data,
        group_indices=payload["group_partition"]["group_indices"],
        task=str(data["task"]),
        device=str(device),
        gradient_params=str(args.gradient_params),
    )
    metric, _ = build_metric(gradients, normalization=str(args.metric_normalization))
    with torch.inference_mode():
        emb_x, emb_y = model.get_embedding(data["train_x"].to(device), data["train_y"].to(device))
    emb_x_np = emb_x.detach().cpu().numpy()
    emb_y_np = emb_y.detach().cpu().numpy()
    group_img_repr = np.stack([emb_x_np[idx].mean(axis=0) for idx in group_indices], axis=0)
    group_txt_repr = np.stack([emb_y_np[idx].mean(axis=0) for idx in group_indices], axis=0)
    return metric, group_img_repr, group_txt_repr


def choose_background_groups(*, num_groups: int, anchor: int, candidates: list[int], count: int) -> list[int]:
    pool = [group for group in range(num_groups) if group != anchor and group not in candidates[:0]]
    pool = [group for group in pool if group != anchor]
    if count >= len(pool):
        return pool
    step = max(1, len(pool) // count)
    chosen = pool[::step][:count]
    return chosen


def compute_ground_truth_rows(
    probe_data: ProbeData,
    *,
    group_indices: list[np.ndarray],
    anchor: int,
    candidates: list[int],
    background: list[int],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    base_loss = train_probe_loss(probe_data, group_indices=group_indices, active_modalities={group: (1, 1) for group in background})
    anchor_loss = train_probe_loss(
        probe_data,
        group_indices=group_indices,
        active_modalities={**{group: (1, 1) for group in background}, anchor: (1, 0)},
    )
    rows: list[dict[str, Any]] = []
    cache: dict[int, dict[str, float]] = {}
    total = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        partner_loss = train_probe_loss(
            probe_data,
            group_indices=group_indices,
            active_modalities={**{group: (1, 1) for group in background}, candidate: (0, 1)},
        )
        joint_loss = train_probe_loss(
            probe_data,
            group_indices=group_indices,
            active_modalities={**{group: (1, 1) for group in background}, anchor: (1, 0), candidate: (0, 1)},
        )
        gt = joint_loss - anchor_loss - partner_loss + base_loss
        cache[candidate] = {"L00": base_loss, "L10": anchor_loss, "L01": partner_loss, "L11": joint_loss}
        rows.append(
            {
                "candidate_group": int(candidate),
                "L00": float(base_loss),
                "L10": float(anchor_loss),
                "L01": float(partner_loss),
                "L11": float(joint_loss),
                "gt_synergy": float(gt),
            }
        )
        if idx % 10 == 0 or idx == total:
            print(f"    anchor {anchor}: ground-truth {idx}/{total}", flush=True)
    return rows, cache


def train_probe_loss(
    probe_data: ProbeData,
    *,
    group_indices: list[np.ndarray],
    active_modalities: dict[int, tuple[int, int]],
) -> float:
    active_indices = np.unique(
        np.concatenate([group_indices[int(group_id)] for group_id in active_modalities], axis=0)
    )
    train_x = np.zeros((active_indices.shape[0], probe_data.train_x.shape[1]), dtype=probe_data.train_x.dtype)
    train_y = np.zeros((active_indices.shape[0], probe_data.train_y.shape[1]), dtype=probe_data.train_y.dtype)
    train_labels = probe_data.train_labels[active_indices]
    local_index = {int(global_idx): local_idx for local_idx, global_idx in enumerate(active_indices.tolist())}
    for group_id, (use_x, use_y) in active_modalities.items():
        idx = group_indices[int(group_id)]
        local = np.asarray([local_index[int(i)] for i in idx.tolist()], dtype=np.int64)
        if int(use_x):
            train_x[local] = probe_data.train_x[idx]
        if int(use_y):
            train_y[local] = probe_data.train_y[idx]
    train_features = np.concatenate([train_x, train_y], axis=1)
    test_features = np.concatenate([probe_data.test_x, probe_data.test_y], axis=1)
    probe = make_probe(task=probe_data.task)
    probe.fit(train_features, train_labels)
    proba = probe.predict_proba(test_features)
    if probe_data.task == "binary":
        return float(log_loss(probe_data.test_labels.astype(int), proba[:, 1], labels=[0, 1]))
    return float(log_loss(probe_data.test_labels.astype(int), proba, labels=np.unique(train_labels.astype(int))))


def make_probe(*, task: str) -> Any:
    if task == "binary":
        return LogisticRegression(max_iter=200)
    return make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver="liblinear"))


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return matrix / norms


def rank_desc(scores: dict[int, float]) -> list[int]:
    return [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def gt_to_relevance(gt_values: np.ndarray) -> np.ndarray:
    shifted = np.max(gt_values) - gt_values
    shifted = shifted - np.min(shifted)
    return shifted


def compute_metrics_for_order(
    order: list[int],
    gt_rows: list[dict[str, Any]],
    relevance: np.ndarray,
    gt_order: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float]:
    candidate_to_index = {int(row["candidate_group"]): idx for idx, row in enumerate(gt_rows)}
    pred_indices = np.asarray([candidate_to_index[c] for c in order], dtype=np.int64)
    gt_rank = np.empty(len(gt_rows), dtype=np.int64)
    gt_rank[gt_order] = np.arange(len(gt_rows))
    pred_rank = np.empty(len(gt_rows), dtype=np.int64)
    pred_rank[pred_indices] = np.arange(len(gt_rows))
    top_gt = set(int(gt_rows[idx]["candidate_group"]) for idx in gt_order[:top_k])
    top_pred = set(order[:top_k])
    return {
        "ndcg_at_10": ndcg_at_k(relevance[pred_indices], relevance[gt_order], top_k),
        "spearman": spearman_from_ranks(pred_rank, gt_rank),
        "top10_recall": float(len(top_gt & top_pred) / max(len(top_gt), 1)),
    }


def dcg(values: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, values.shape[0] + 2))
    return float(np.sum(values * discounts))


def ndcg_at_k(pred_rel: np.ndarray, ideal_rel: np.ndarray, k: int) -> float:
    k = min(int(k), pred_rel.shape[0], ideal_rel.shape[0])
    ideal = dcg(ideal_rel[:k])
    if ideal <= 1e-12:
        return float("nan")
    return float(dcg(pred_rel[:k]) / ideal)


def spearman_from_ranks(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs.astype(np.float64)
    rhs = rhs.astype(np.float64)
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    denom = float(np.sqrt(np.sum(lhs**2) * np.sum(rhs**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(lhs * rhs) / denom)


def summarize_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method_names = ["cmis", "embedding_cosine", "one_d_attribution", "random"]
    summary = []
    for name in method_names:
        metrics = []
        for row in rows:
            source = row["random_baseline_mean"] if name == "random" else row["methods"][name]
            metrics.append(source)
        summary.append(
            {
                "method": name,
                "mean_ndcg_at_10": float(np.nanmean([m["ndcg_at_10"] for m in metrics])),
                "mean_spearman": float(np.nanmean([m["spearman"] for m in metrics])),
                "mean_top10_recall": float(np.nanmean([m["top10_recall"] for m in metrics])),
            }
        )
    return summary


def to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Best-Partner Evaluation",
        "",
        f"- dataset: `{config['dataset']}`",
        f"- anchor groups: `{config['anchor_groups']}`",
        f"- background groups per anchor: `{config['background_groups_per_anchor']}`",
        f"- gradient params: `{config['gradient_params']}`",
        "",
        "## Summary",
        "",
        "| method | mean NDCG@10 | mean Spearman | mean Top-10 recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['method']} | {row['mean_ndcg_at_10']:.6g} | "
            f"{row['mean_spearman']:.6g} | {row['mean_top10_recall']:.6g} |"
        )
    lines.extend(["", "## Per Anchor", "", "| anchor | method | NDCG@10 | Spearman | Top-10 recall |", "| --- | --- | ---: | ---: | ---: |"])
    for anchor in payload["anchors"]:
        for method, row in anchor["methods"].items():
            lines.append(
                f"| {anchor['anchor_group']} | {method} | {row['ndcg_at_10']:.6g} | {row['spearman']:.6g} | {row['top10_recall']:.6g} |"
            )
        random_row = anchor["random_baseline_mean"]
        lines.append(
            f"| {anchor['anchor_group']} | random | {random_row['ndcg_at_10']:.6g} | {random_row['spearman']:.6g} | {random_row['top10_recall']:.6g} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
