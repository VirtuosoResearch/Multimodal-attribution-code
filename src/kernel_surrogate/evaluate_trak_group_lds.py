from __future__ import annotations

import argparse
import json
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
ROOT = REPO_ROOT

from fisher_geometry.data import load_dataset
from kernel_surrogate.evaluate_influence_vs_kernel_lds import (
    DATASETS,
    DEFAULT_FULL_CHECKPOINT_DIR,
    DEFAULT_GROUPED25_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    KMEANS_SETTINGS,
    _batch_slices,
    _flatten_tensors,
    choose_full_checkpoint_path,
    ensure_test_split_aliases,
    group_tensors,
    load_or_train_full_model_current,
    pearsonr,
)
from kernel_surrogate.kernel_attribution import rmse, spearmanr
from kernel_surrogate.real_data_cmis import (
    build_group_mask_blocks,
    make_model_args,
    resolve_device,
    select_gradient_parameters,
    split_indices,
)


DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "trak_group_lds.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "trak_group_lds.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate group-level TRAK on kmeans group LDS subsets."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=[name for name, _ in KMEANS_SETTINGS],
        default=[name for name, _ in KMEANS_SETTINGS],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--pretrained-full-checkpoint-root", type=Path, default=DEFAULT_GROUPED25_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--alternate-full-checkpoint-root", type=Path, default=DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--full-checkpoint-dir", type=Path, default=DEFAULT_FULL_CHECKPOINT_DIR)
    parser.add_argument("--force-train-full", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--trak-ridge", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def model_output_and_q(task: str, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if task == "multiclass":
        labels = labels.long().view(-1)
        row = torch.arange(logits.shape[0], device=logits.device)
        correct = logits[row, labels]
        masked = logits.clone()
        masked[row, labels] = -torch.inf
        output = correct - torch.logsumexp(masked, dim=1)
        probs = torch.softmax(logits, dim=1)[row, labels]
        q = 1.0 - probs
        return output, q
    if task == "binary":
        logit = logits.view(-1)
        target = labels.float().view(-1)
        sign = target.mul(2.0).sub(1.0)
        output = sign * logit
        q = 1.0 - torch.sigmoid(output)
        return output, q

    target = labels.float()
    sign = target.mul(2.0).sub(1.0)
    output = (sign * logits).mean(dim=1)
    q = (1.0 - torch.sigmoid(sign * logits)).mean(dim=1)
    return output, q


def output_grad_and_q(
    model: torch.nn.Module,
    params: Sequence[torch.nn.Parameter],
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, float]:
    if int(labels.shape[0]) == 0:
        num_params = sum(int(param.numel()) for param in params)
        return torch.zeros(num_params, device=device), 0.0
    model.eval()
    total_items = int(labels.shape[0])
    grad_accum = torch.zeros(sum(int(param.numel()) for param in params), device=device)
    q_total = 0.0
    for sl in _batch_slices(total_items, batch_size):
        bx = x[sl].to(device)
        by = y[sl].to(device)
        bl = labels[sl].to(device)
        weight = float(int(bl.shape[0])) / float(total_items)
        logits = model(bx, by, use_x=True, use_y=True)
        outputs, q = model_output_and_q(task, logits, bl)
        grads = torch.autograd.grad(outputs.mean() * weight, params, allow_unused=True)
        grad_accum += _flatten_tensors(grads, params).detach()
        q_total += float(q.detach().mean().cpu().item()) * int(bl.shape[0])
    return grad_accum, q_total / float(total_items)


def trak_group_scores(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    payload: dict[str, Any],
    task: str,
    device: torch.device,
    gradient_params: str,
    batch_size: int,
    ridge: float,
) -> dict[str, Any]:
    params = select_gradient_parameters(model, gradient_params=gradient_params)
    num_params = sum(int(param.numel()) for param in params)
    print(f"  TRAK params={num_params}, ridge={ridge:g}", flush=True)

    test_grad, _ = output_grad_and_q(
        model,
        params,
        x=data["test_x"],
        y=data["test_y"],
        labels=data["test_labels"],
        task=task,
        device=device,
        batch_size=batch_size,
    )

    group_indices = payload["group_partition"]["group_indices"]
    num_groups = int(payload["group_partition"]["num_groups"])
    num_modalities = int(payload["group_partition"]["num_modalities"])
    if num_modalities != 2:
        raise ValueError(f"expected two modalities, got {num_modalities}")
    total_train = float(int(data["train_labels"].shape[0]))
    train_grads = []
    train_q = []
    train_weights = []
    item_kinds: list[tuple[str, int, int | None]] = []
    group_sizes = np.zeros(num_groups, dtype=np.int64)

    for group_id, indices in enumerate(group_indices):
        group_sizes[group_id] = int(len(indices))
        weight = float(group_sizes[group_id]) / max(total_train, 1.0)
        gx, gy, gl = group_tensors(data, indices)
        full_grad, full_q = output_grad_and_q(
            model,
            params,
            x=gx,
            y=gy,
            labels=gl,
            task=task,
            device=device,
            batch_size=batch_size,
        )
        train_grads.append(full_grad)
        train_q.append(full_q)
        train_weights.append(weight)
        item_kinds.append(("group", group_id, None))

        for modality in range(num_modalities):
            mx, my, ml = group_tensors(data, indices, masked_modality=modality)
            masked_grad, masked_q = output_grad_and_q(
                model,
                params,
                x=mx,
                y=my,
                labels=ml,
                task=task,
                device=device,
                batch_size=batch_size,
            )
            train_grads.append(masked_grad - full_grad)
            train_q.append(masked_q)
            train_weights.append(-weight)
            item_kinds.append(("modality", group_id, modality))
        if (group_id + 1) % 5 == 0 or group_id + 1 == num_groups:
            print(f"    TRAK group features {group_id + 1}/{num_groups}", flush=True)

    x_train = torch.stack(train_grads, dim=0).to(device=device, dtype=torch.float64)
    x_target = test_grad.to(device=device, dtype=torch.float64).view(1, -1)
    gram = x_train @ x_train.T
    gram = gram + float(ridge) * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    raw_scores = torch.linalg.solve(gram, x_train @ x_target.T).view(-1).detach().cpu().numpy()

    group_scores = np.zeros(num_groups, dtype=np.float64)
    modality_scores = np.zeros((num_groups, num_modalities), dtype=np.float64)
    q_values = np.asarray(train_q, dtype=np.float64)
    weights = np.asarray(train_weights, dtype=np.float64)
    for index, (kind, group_id, modality) in enumerate(item_kinds):
        value = float(raw_scores[index] * q_values[index] * weights[index])
        if kind == "group":
            group_scores[group_id] = value
        else:
            assert modality is not None
            modality_scores[group_id, modality] = value

    return {
        "num_params": int(num_params),
        "num_trak_items": int(len(item_kinds)),
        "ridge": float(ridge),
        "group_scores": group_scores,
        "modality_scores": modality_scores,
        "group_sizes": group_sizes,
        "q_mean": float(np.mean(q_values)),
        "q_std": float(np.std(q_values)),
    }


def predict_from_scores(
    mask_blocks: dict[str, np.ndarray],
    *,
    base_loss: float,
    group_scores: np.ndarray,
    modality_scores: np.ndarray,
) -> np.ndarray:
    sample_deleted = np.asarray(mask_blocks["sample_deleted"], dtype=np.float64)
    modality_deleted = np.asarray(mask_blocks["modality_deleted"], dtype=np.float64)
    return np.asarray(
        float(base_loss) + sample_deleted @ np.asarray(group_scores, dtype=np.float64)
        + modality_deleted @ np.asarray(modality_scores, dtype=np.float64).reshape(-1),
        dtype=np.float64,
    )


def make_record(
    *,
    dataset: str,
    setting: str,
    run_name: str,
    split_seed: int,
    predictions: np.ndarray,
    targets: np.ndarray,
    train_fraction: float,
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    pred = predictions[test_indices]
    truth = targets[test_indices]
    return {
        "dataset": dataset,
        "setting": setting,
        "run_name": run_name,
        "method": "trak",
        "split_seed": int(split_seed),
        "test_lds": spearmanr(pred, truth),
        "test_pearson": pearsonr(pred, truth),
        "test_rmse": rmse(pred, truth),
        "all_lds": spearmanr(predictions, targets),
        "all_pearson": pearsonr(predictions, targets),
        "all_rmse": rmse(predictions, targets),
        "num_train": int(train_indices.size),
        "num_test": int(test_indices.size),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["dataset"]), str(row["setting"]), str(row["method"])), []).append(row)
    rows = []
    for (dataset, setting, method), items in sorted(grouped.items()):
        lds = np.asarray([float(item["test_lds"]) for item in items], dtype=np.float64)
        pearson = np.asarray([float(item["test_pearson"]) for item in items], dtype=np.float64)
        rmses = np.asarray([float(item["test_rmse"]) for item in items], dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "mean_lds": float(np.nanmean(lds)),
                "std_lds": float(np.nanstd(lds)),
                "mean_pearson": float(np.nanmean(pearson)),
                "mean_rmse": float(np.nanmean(rmses)),
                "num_splits": int(len(items)),
            }
        )
    return rows


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TRAK Group LDS",
        "",
        f"- unit: `{payload['config']['unit']}`",
        f"- settings: `{payload['config']['settings']}`",
        f"- split: `{payload['config']['split']}`",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- gradient params: `{payload['config']['gradient_params']}`",
        f"- trak ridge: `{payload['config']['trak_ridge']}`",
        f"- score formula: `{payload['config']['prediction_formula']}`",
        "",
        "## Summary",
        "",
        "| dataset | setting | method | mean LDS | std LDS | mean Pearson | mean RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['method']} | "
            f"{row['mean_lds']:.6f} | {row['std_lds']:.6f} | "
            f"{row['mean_pearson']:.6f} | {row['mean_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    split_seeds = [int(seed) for seed in args.split_seeds]
    setting_lookup = dict(KMEANS_SETTINGS)
    selected_settings = [(name, setting_lookup[name]) for name in args.settings]

    records: list[dict[str, Any]] = []
    dataset_setting_details: list[dict[str, Any]] = []
    for dataset in [str(item) for item in args.datasets]:
        for setting, run_name in selected_settings:
            payload_path = args.result_root / run_name / f"{dataset}_seed{int(args.seed)}.json"
            if not payload_path.exists():
                print(f"[{dataset}/{setting}] missing {payload_path}, skipping", flush=True)
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            group_method = str(payload.get("group_partition", {}).get("group_method", ""))
            if group_method != "kmeans":
                raise ValueError(f"{payload_path} is not a kmeans group payload: group_method={group_method!r}")
            config = dict(payload["config"])
            print(f"[{dataset}/{setting}] loading data/checkpoint on {device}...", flush=True)
            data = load_dataset(
                dataset,
                int(args.seed),
                text_backbone=str(config.get("text_backbone", "clip_cached")),
                vision_backbone=str(config.get("vision_backbone", "clip_cached")),
                feature_device=str(args.feature_device or config.get("feature_device", device)),
                backbone_batch_size=int(config.get("backbone_batch_size", 32)),
                download_datasets=bool(config.get("download_datasets", False)),
                backbone_cache_dir=config.get("backbone_cache_dir"),
            )
            ensure_test_split_aliases(data)
            model_args = make_model_args(config, device=str(device))
            dim_x = int(data["train_x"].shape[1])
            dim_y = int(data["train_y"].shape[1])
            num_classes = int(data["num_classes"])
            full_checkpoint = choose_full_checkpoint_path(
                dataset=dataset,
                seed=int(args.seed),
                preferred_root=args.pretrained_full_checkpoint_root,
                alternate_root=args.alternate_full_checkpoint_root,
                fallback_dir=args.full_checkpoint_dir,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
            )
            model, full_metrics = load_or_train_full_model_current(
                full_checkpoint,
                data=data,
                task=str(data["task"]),
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                model_type=str(config.get("model", "cross_attention")),
                seed=int(args.seed),
                force_train=bool(args.force_train_full),
            )
            model.to(device)
            model.eval()
            mask_blocks, targets = build_group_mask_blocks(payload)
            base_loss = float(full_metrics.get("loss", np.mean(targets)))

            print(f"[{dataset}/{setting}] computing TRAK group scores...", flush=True)
            scores = trak_group_scores(
                model,
                data=data,
                payload=payload,
                task=str(data["task"]),
                device=device,
                gradient_params=str(args.gradient_params),
                batch_size=int(args.batch_size),
                ridge=float(args.trak_ridge),
            )
            predictions = predict_from_scores(
                mask_blocks,
                base_loss=base_loss,
                group_scores=scores["group_scores"],
                modality_scores=scores["modality_scores"],
            )
            setting_records: list[dict[str, Any]] = []
            for split_seed in split_seeds:
                row = make_record(
                    dataset=dataset,
                    setting=setting,
                    run_name=run_name,
                    split_seed=split_seed,
                    predictions=predictions,
                    targets=targets,
                    train_fraction=float(args.train_fraction),
                )
                records.append(row)
                setting_records.append(row)
                print(
                    f"  split_seed={split_seed} trak: "
                    f"LDS={row['test_lds']:.6f}, Pearson={row['test_pearson']:.6f}, RMSE={row['test_rmse']:.6f}",
                    flush=True,
                )

            dataset_setting_details.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "run_name": run_name,
                    "payload_path": str(payload_path),
                    "full_checkpoint": str(full_checkpoint),
                    "full_checkpoint_metrics": full_metrics,
                    "base_loss": base_loss,
                    "num_params": int(scores["num_params"]),
                    "num_trak_items": int(scores["num_trak_items"]),
                    "trak_ridge": float(scores["ridge"]),
                    "q_mean": float(scores["q_mean"]),
                    "q_std": float(scores["q_std"]),
                    "group_sizes": scores["group_sizes"].tolist(),
                    "group_scores": scores["group_scores"].tolist(),
                    "group_modality_scores": scores["modality_scores"].tolist(),
                    "predictions": predictions.tolist(),
                    "targets": targets.tolist(),
                    "records": setting_records,
                }
            )

    output = {
        "config": {
            "datasets": [str(item) for item in args.datasets],
            "settings": [name for name, _ in selected_settings],
            "run_names": [run_name for _, run_name in selected_settings],
            "seed": int(args.seed),
            "unit": "kmeans_group",
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "split_seeds": split_seeds,
            "gradient_params": str(args.gradient_params),
            "batch_size": int(args.batch_size),
            "trak_ridge": float(args.trak_ridge),
            "prediction_formula": (
                "full_loss + sample_deleted @ group_trak_scores + "
                "modality_deleted @ group_modality_trak_scores"
            ),
            "reference": "/data/zhenshuo/multimodal/trak: TRAK uses X(X^T X + lambda I)^-1 X_target^T times train out-to-loss weights.",
            "pretrained_full_checkpoint_root": str(args.pretrained_full_checkpoint_root),
            "alternate_full_checkpoint_root": str(args.alternate_full_checkpoint_root),
            "full_checkpoint_dir": str(args.full_checkpoint_dir),
        },
        "summary": summarize(records),
        "records": records,
        "dataset_setting_details": dataset_setting_details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
