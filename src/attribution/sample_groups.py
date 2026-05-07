from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from MultiBench_selection.datasets import load_dataset


def _flatten_features(values: torch.Tensor, *, max_dim: int | None) -> torch.Tensor:
    flat = values.detach().cpu().float().reshape(values.shape[0], -1)
    if max_dim is not None and flat.shape[1] > int(max_dim):
        flat = flat[:, : int(max_dim)]
    return F.normalize(flat, dim=1)


def build_feature_matrix(
    *,
    dataset: str,
    seed: int,
    full_modality: str,
    eval_split: str,
    standardize: bool,
    max_train_samples: int | None,
    max_seq_len: int,
    feature_modality: str,
    max_feature_dim: int | None,
) -> np.ndarray:
    data = load_dataset(
        dataset,
        seed=int(seed),
        full_modality=full_modality,
        eval_split=eval_split,
        standardize=standardize,
        max_train_samples=max_train_samples,
        max_eval_samples=None,
        max_seq_len=max_seq_len,
    )
    main = data.train.main
    aux = data.train.aux
    if feature_modality == "main":
        parts = [_flatten_features(main, max_dim=max_feature_dim)]
    elif feature_modality == "aux":
        parts = [_flatten_features(aux, max_dim=max_feature_dim)]
    else:
        parts = [
            _flatten_features(main, max_dim=max_feature_dim),
            _flatten_features(aux, max_dim=max_feature_dim),
        ]
    return torch.cat(parts, dim=1).numpy().astype(np.float32, copy=False)


def random_group_indices(num_samples: int, num_groups: int, *, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(num_samples))
    chunks = np.array_split(order, min(int(num_groups), int(num_samples)))
    return [sorted(int(index) for index in chunk.tolist()) for chunk in chunks if chunk.size]


def kmeans_group_indices(features: np.ndarray, num_groups: int, *, seed: int) -> list[list[int]]:
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    num_samples = int(features.shape[0])
    resolved_groups = min(int(num_groups), num_samples)
    if resolved_groups <= 0:
        raise ValueError("num_groups must be positive")
    if resolved_groups == num_samples:
        return [[index] for index in range(num_samples)]
    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("kmeans grouping requires scikit-learn") from exc

    assignments = KMeans(
        n_clusters=resolved_groups,
        random_state=int(seed),
        n_init=10,
    ).fit_predict(np.asarray(features, dtype=np.float32))
    groups: list[list[int]] = []
    for group_id in range(resolved_groups):
        members = np.flatnonzero(assignments == group_id)
        if members.size == 0:
            raise RuntimeError(f"kmeans produced an empty group: {group_id}")
        groups.append(sorted(int(index) for index in members.tolist()))
    return groups


def write_group_payload(
    *,
    output: Path,
    dataset: str,
    seed: int,
    group_method: str,
    group_indices: list[list[int]],
    metadata: dict[str, Any],
) -> None:
    num_samples = sum(len(group) for group in group_indices)
    payload = {
        "group_partition": {
            "dataset_name": dataset,
            "dataset_size": int(num_samples),
            "num_groups": int(len(group_indices)),
            "num_modalities": 2,
            "seed": int(seed),
            "group_method": group_method,
            "group_sizes": [int(len(group)) for group in group_indices],
            "group_indices": group_indices,
        },
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random or kmeans sample groups for multimodal attribution."
    )
    parser.add_argument("--dataset", default="avmnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-groups", type=int, default=25)
    parser.add_argument("--group-method", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--full-modality", choices=["x", "y"], default="x")
    parser.add_argument("--eval-split", choices=["auto", "valid", "test"], default="auto")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--feature-modality", choices=["both", "main", "aux"], default="both")
    parser.add_argument(
        "--max-feature-dim",
        type=int,
        default=2048,
        help="Optional per-modality feature truncation before kmeans.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/attribution/results/groups/{dataset}_seed{seed}.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(str(args.output).format(dataset=args.dataset, seed=int(args.seed)))
    if args.group_method == "random":
        features = build_feature_matrix(
            dataset=args.dataset,
            seed=int(args.seed),
            full_modality=args.full_modality,
            eval_split=args.eval_split,
            standardize=bool(args.standardize),
            max_train_samples=args.max_train_samples,
            max_seq_len=int(args.max_seq_len),
            feature_modality=args.feature_modality,
            max_feature_dim=args.max_feature_dim,
        )
        groups = random_group_indices(features.shape[0], int(args.num_groups), seed=int(args.seed))
    else:
        features = build_feature_matrix(
            dataset=args.dataset,
            seed=int(args.seed),
            full_modality=args.full_modality,
            eval_split=args.eval_split,
            standardize=bool(args.standardize),
            max_train_samples=args.max_train_samples,
            max_seq_len=int(args.max_seq_len),
            feature_modality=args.feature_modality,
            max_feature_dim=args.max_feature_dim,
        )
        groups = kmeans_group_indices(features, int(args.num_groups), seed=int(args.seed))

    write_group_payload(
        output=output,
        dataset=args.dataset,
        seed=int(args.seed),
        group_method=str(args.group_method),
        group_indices=groups,
        metadata={
            "feature_modality": str(args.feature_modality),
            "full_modality": str(args.full_modality),
            "standardize": bool(args.standardize),
            "max_train_samples": args.max_train_samples,
            "max_feature_dim": args.max_feature_dim,
        },
    )
    print(f"saved {len(groups)} groups to {output}", flush=True)


if __name__ == "__main__":
    main()

