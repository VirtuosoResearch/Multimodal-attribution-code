from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fisher_geometry.data import load_dataset, make_loader
from src.kernel_surrogate.kernel_attribution import load_result_payload, rmse, spearmanr
from src.kernel_surrogate.lds_real_train_gnn import _evaluate_classifier, _train_model
from src.kernel_surrogate.real_data_cmis import build_group_mask_blocks, fit_surrogate_method, metric_sqdist, split_indices

DEFAULT_RESULT_DIR = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "downstream_structured_rbf_graph_cross_state.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "downstream_structured_rbf_graph_cross_state.md"


def _to_namespace(config: dict[str, Any], *, device: str) -> SimpleNamespace:
    batch_size = int(config.get("batch_size", 4096))
    return SimpleNamespace(
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
        proj_dim=int(config.get("proj_dim", 256)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        traffic_gnn_layers=int(config.get("traffic_gnn_layers", 2)),
        traffic_classifier_layers=int(config.get("traffic_classifier_layers", 2)),
        batch_size=batch_size,
        eval_batch_size=int(config.get("eval_batch_size", batch_size)),
        epochs=int(config.get("epochs", 15)),
        lr=float(config.get("lr", 5e-4)),
        device=device,
    )


def _resolve_model_kwargs(model_type: str, data: dict[str, Any]) -> dict[str, Any]:
    if model_type != "traffic_pair_gnn_mlp":
        return {}
    return {
        "x_node_dim": int(data["traffic_node_tabular_dim"]),
        "x_edge_dim": int(data["traffic_edge_tabular_dim"]),
        "y_node_dim": int(data["traffic_vision_node_dim"]),
    }


def _load_data_from_config(dataset: str, seed: int, config: dict[str, Any], device: str) -> dict[str, Any]:
    return load_dataset(
        dataset,
        int(seed),
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
        traffic_state_name=str(config.get("traffic_state_name", "DE")),
        traffic_data_dir=config.get("traffic_data_dir"),
        traffic_embeddings_dir=config.get("traffic_embeddings_dir"),
        traffic_node_feature_type=str(config.get("traffic_node_feature_type", "node2vec")),
        traffic_vision_feature_type=str(config.get("traffic_vision_feature_type", "clip_image")),
        traffic_train_years=config.get("traffic_train_years"),
        traffic_valid_years=config.get("traffic_valid_years"),
        traffic_test_years=config.get("traffic_test_years"),
        traffic_months=config.get("traffic_months"),
        traffic_include_dynamic_node_features=bool(config.get("traffic_include_dynamic_node_features", True)),
        traffic_include_static_edge_features=bool(config.get("traffic_include_static_edge_features", True)),
        traffic_include_dynamic_edge_features=bool(config.get("traffic_include_dynamic_edge_features", True)),
        traffic_negative_ratio=float(config.get("traffic_negative_ratio", 1.0)),
        traffic_label_threshold=float(config.get("traffic_label_threshold", 0.0)),
        traffic_normalize_modalities=bool(config.get("traffic_normalize_modalities", True)),
    )


def _predict_from_queries(queries: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    distances = metric_sqdist(np.asarray(queries, dtype=np.float64), fit["train_masks"], fit["metric"])
    kernel = np.exp(-float(fit["gamma"]) * distances)
    return float(fit["target_mean"]) + kernel @ fit["alpha"]


def _fit_structured_rbf(mask_blocks: dict[str, np.ndarray], targets: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if float(args.surrogate_train_fraction) >= 1.0:
        train_indices = np.arange(len(targets), dtype=np.int64)
        test_indices = np.asarray([], dtype=np.int64)
    else:
        train_indices, test_indices = split_indices(
            len(targets), train_fraction=float(args.surrogate_train_fraction), seed=int(args.surrogate_split_seed)
        )
    fit = fit_surrogate_method(
        "structured_rbf",
        mask_blocks,
        targets,
        metric=np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64),
        ridge_lambda=float(args.ridge_lambda),
        gamma_scale=float(args.gamma_scale),
        structured_shared_weight=float(args.structured_shared_weight),
        train_indices=train_indices,
    )
    diagnostics = {
        "num_train_subsets": int(train_indices.size),
        "num_test_subsets": int(test_indices.size),
        "train_rmse": rmse(fit["train_predictions"], targets[train_indices]),
        "train_lds": spearmanr(fit["train_predictions"], targets[train_indices]),
    }
    if test_indices.size > 0:
        sample = mask_blocks["sample_deleted"][test_indices]
        modality = mask_blocks["modality_deleted"][test_indices]
        test_queries = np.concatenate([sample, modality], axis=1)
        pred = _predict_from_queries(test_queries, fit)
        truth = targets[test_indices]
        diagnostics["test_rmse"] = rmse(pred, truth)
        diagnostics["test_lds"] = spearmanr(pred, truth)
    return {"fit": fit, "diagnostics": diagnostics}


def _build_group_mapping(source_num_groups: int, target_num_groups: int) -> np.ndarray:
    if source_num_groups <= 0 or target_num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if source_num_groups == target_num_groups:
        return np.arange(target_num_groups, dtype=np.int64)
    mapped = np.floor(np.arange(target_num_groups, dtype=np.float64) * source_num_groups / target_num_groups).astype(
        np.int64
    )
    return np.clip(mapped, 0, source_num_groups - 1)


def _group_scores_graph_cross_state(
    fit: dict[str, Any], source_num_groups: int, target_num_groups: int
) -> tuple[float, np.ndarray, np.ndarray]:
    mapping = _build_group_mapping(source_num_groups, target_num_groups)
    dim = int(source_num_groups + 2 * source_num_groups)
    anchor = np.zeros((1, dim), dtype=np.float64)
    anchor_pred = float(_predict_from_queries(anchor, fit)[0])
    scores = np.zeros(target_num_groups, dtype=np.float64)
    for target_group_id in range(target_num_groups):
        source_group_id = int(mapping[target_group_id])
        q = anchor.copy()
        q[0, source_num_groups + (2 * source_group_id)] = 1.0
        pred = float(_predict_from_queries(q, fit)[0])
        scores[target_group_id] = pred - anchor_pred
    return anchor_pred, scores, mapping


def _mask_graph_groups(train_x: torch.Tensor, group_indices: list[list[int]], groups_to_mask: list[int]) -> torch.Tensor:
    masked = train_x.clone()
    for group_id in groups_to_mask:
        masked[torch.as_tensor(group_indices[group_id], dtype=torch.long)] = 0.0
    return masked


def _train_and_eval(
    data: dict[str, Any],
    model_args: SimpleNamespace,
    model_type: str,
    model_kwargs: dict[str, Any],
    task: str,
    train_seed: int,
    use_x: bool,
    use_y: bool,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
) -> dict[str, float]:
    train_loader = make_loader(train_x, train_y, data["train_labels"], int(model_args.batch_size), True)
    test_loader = make_loader(data["test_x"], data["test_y"], data["test_labels"], int(model_args.eval_batch_size), False)
    model, _ = _train_model(
        train_loader,
        dim_x=int(data["train_x"].shape[1]),
        dim_y=int(data["train_y"].shape[1]),
        num_classes=int(data["num_classes"]),
        task=task,
        model_type=model_type,
        model_kwargs=model_kwargs,
        args=model_args,
        seed=int(train_seed),
        use_x=use_x,
        use_y=use_y,
    )
    metrics = _evaluate_classifier(
        model,
        test_loader,
        task=task,
        device=str(model_args.device),
        max_sample_losses=0,
        use_x=use_x,
        use_y=use_y,
    )
    return {"test_loss": float(metrics["loss"]), "test_acc": float(metrics["acc"])}


def _to_markdown(payload: dict[str, Any]) -> str:
    out = payload["result"]
    lines = [
        "# Downstream Structured RBF Graph Cross-State Transfer",
        "",
        f"- source run: `{payload['config']['source_run_name']}`",
        f"- target run: `{payload['config']['target_run_name']}`",
        f"- keep graph group ratio: `{payload['config']['keep_graph_group_ratio']:.2f}`",
        f"- keep groups: `{out['num_keep_groups']}/{out['target_num_groups']}`",
        "",
        f"- selected-mask target test loss: `{out['selected_mask_target_test_loss']:.6f}`",
        f"- selected-mask target test acc: `{out['selected_mask_target_test_acc']:.6f}`",
    ]
    if "full_baseline_target_test_acc" in out:
        lines.append(f"- full-data baseline target test acc: `{out['full_baseline_target_test_acc']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument("--target-run-name", required=True)
    parser.add_argument("--dataset", default="traffic_accident")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-graph-group-ratio", type=float, default=0.2)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--gamma-scale", type=float, default=1.0)
    parser.add_argument("--structured-shared-weight", type=float, default=0.0)
    parser.add_argument("--surrogate-train-fraction", type=float, default=1.0)
    parser.add_argument("--surrogate-split-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--run-full-baseline", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    source_payload = load_result_payload(args.result_dir / args.source_run_name, args.dataset, int(args.seed))
    target_payload = load_result_payload(args.result_dir / args.target_run_name, args.dataset, int(args.seed))
    source_config = source_payload["config"]
    target_config = target_payload["config"]

    source_mask_blocks, source_targets = build_group_mask_blocks(source_payload)
    fit_bundle = _fit_structured_rbf(source_mask_blocks, source_targets, args)
    source_num_groups = int(source_payload["group_partition"]["num_groups"])
    target_num_groups = int(target_payload["group_partition"]["num_groups"])
    anchor_pred, scores, mapping = _group_scores_graph_cross_state(
        fit_bundle["fit"], source_num_groups=source_num_groups, target_num_groups=target_num_groups
    )

    keep_count = max(1, int(math.ceil(float(args.keep_graph_group_ratio) * target_num_groups)))
    order = np.argsort(-scores, kind="mergesort")
    keep_groups = sorted(int(group_id) for group_id in order[:keep_count].tolist())
    mask_groups = sorted(int(group_id) for group_id in order[keep_count:].tolist())

    target_data = _load_data_from_config(args.dataset, int(args.seed), target_config, str(args.device))
    task = str(target_data["task"])
    model_type = str(target_config.get("model", "traffic_pair_gnn_mlp"))
    model_args = _to_namespace(target_config, device=str(args.device))
    model_kwargs = _resolve_model_kwargs(model_type, target_data)
    use_x = bool(target_config.get("use_x", True))
    use_y = bool(target_config.get("use_y", True))
    target_group_indices = [
        [int(index) for index in indices] for indices in target_payload["group_partition"]["group_indices"]
    ]

    masked_train_x = _mask_graph_groups(target_data["train_x"], target_group_indices, mask_groups)
    masked_metrics = _train_and_eval(
        target_data,
        model_args,
        model_type,
        model_kwargs,
        task,
        int(args.train_seed),
        use_x,
        use_y,
        masked_train_x,
        target_data["train_y"],
    )

    result: dict[str, Any] = {
        "source_num_groups": int(source_num_groups),
        "target_num_groups": int(target_num_groups),
        "num_keep_groups": int(len(keep_groups)),
        "keep_group_ids": keep_groups,
        "mask_group_ids": mask_groups,
        "source_anchor_pred_loss": float(anchor_pred),
        "selected_mask_target_test_loss": float(masked_metrics["test_loss"]),
        "selected_mask_target_test_acc": float(masked_metrics["test_acc"]),
        "surrogate_fit_source": fit_bundle["diagnostics"],
        "target_to_source_group_mapping": [int(v) for v in mapping.tolist()],
    }
    if args.run_full_baseline:
        full_metrics = _train_and_eval(
            target_data,
            model_args,
            model_type,
            model_kwargs,
            task,
            int(args.train_seed),
            use_x,
            use_y,
            target_data["train_x"],
            target_data["train_y"],
        )
        result["full_baseline_target_test_loss"] = float(full_metrics["test_loss"])
        result["full_baseline_target_test_acc"] = float(full_metrics["test_acc"])

    output = {
        "config": {
            "source_run_name": str(args.source_run_name),
            "target_run_name": str(args.target_run_name),
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "device": str(args.device),
            "source_traffic_state_name": str(source_config.get("traffic_state_name", "NA")),
            "target_traffic_state_name": str(target_config.get("traffic_state_name", "NA")),
            "keep_graph_group_ratio": float(args.keep_graph_group_ratio),
            "method": "structured_rbf_graph_cross_state",
        },
        "result": result,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(_to_markdown(output), encoding="utf-8")
    print("\n" + _to_markdown(output), flush=True)


if __name__ == "__main__":
    main()
