from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain, combinations, permutations
from typing import Any

import numpy as np
import torch

from .data import MultimodalDataset, load_pickle_dataset, save_pickle_dataset

_BINARY_X1 = np.array([[0], [0], [1], [1]], dtype=np.int64)
_BINARY_X2 = np.array([[0], [1], [0], [1]], dtype=np.int64)


BINARY_LOGIC_CASES: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "and": (_BINARY_X1, _BINARY_X2, np.logical_and(_BINARY_X1, _BINARY_X2).astype(np.int64)),
    "or": (_BINARY_X1, _BINARY_X2, np.logical_or(_BINARY_X1, _BINARY_X2).astype(np.int64)),
    "xor": (_BINARY_X1, _BINARY_X2, np.logical_xor(_BINARY_X1, _BINARY_X2).astype(np.int64)),
    "unique0": (_BINARY_X1, _BINARY_X2, _BINARY_X1.copy()),
    "unique1": (_BINARY_X1, _BINARY_X2, _BINARY_X2.copy()),
    "redundant": (_BINARY_X1, _BINARY_X1.copy(), _BINARY_X1.copy()),
    "redundant_and_unique1": (
        np.concatenate([_BINARY_X1, _BINARY_X2], axis=1),
        _BINARY_X2,
        np.logical_and(_BINARY_X1, _BINARY_X2).astype(np.int64),
    ),
    "redundant_or_unique1": (
        np.concatenate([_BINARY_X1, _BINARY_X2], axis=1),
        _BINARY_X2,
        np.logical_or(_BINARY_X1, _BINARY_X2).astype(np.int64),
    ),
    "redundant_xor_unique1": (
        np.concatenate([_BINARY_X1, _BINARY_X2], axis=1),
        _BINARY_X2,
        np.logical_xor(_BINARY_X1, _BINARY_X2).astype(np.int64),
    ),
}


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    num_data: int = 10000
    modality_number: int = 2
    feature_dim: int = 5
    feature_sep: float = 1.5
    transform_dim: int = 32
    setting: str = "official"
    rho: float = 0.5
    noise_degree: float = 0.5
    mix_ratio: tuple[float, ...] | None = None
    num_classes: int = 32
    train_size: float = 0.7
    valid_size: float = 0.15
    seed: int = 0


def generate_synthetic_dataset(config: SyntheticGenerationConfig) -> dict[str, dict[str, np.ndarray]]:
    if config.modality_number != 2:
        raise ValueError("Official LSMI synthetic generator only supports modality_number=2.")
    if config.setting != "official":
        raise ValueError(
            f"Unsupported synthetic setting {config.setting!r}. "
            "Use setting='official' for the official LSMI Gaussian generator, "
            "or use make_binary_logic_dataset for logic PID cases."
        )

    total = int(config.num_data)
    train_count = max(1, int(round(total * config.train_size)))
    valid_count = max(1, int(round(total * config.valid_size)))
    if train_count + valid_count >= total:
        valid_count = max(1, total - train_count - 1)
    test_count = max(1, total - train_count - valid_count)

    rng_train = np.random.default_rng(config.seed)
    rng_valid = np.random.default_rng(config.seed + 1)
    rng_test = np.random.default_rng(config.seed + 2)

    train_modalities, train_labels = _generate_official_gaussian_split(
        num_samples=train_count,
        num_classes=config.num_classes,
        rho=config.rho,
        noise_degree=config.noise_degree,
        rng=rng_train,
    )
    valid_modalities, valid_labels = _generate_official_gaussian_split(
        num_samples=valid_count,
        num_classes=config.num_classes,
        rho=config.rho,
        noise_degree=config.noise_degree,
        rng=rng_valid,
    )
    test_modalities, test_labels = _generate_official_gaussian_split(
        num_samples=test_count,
        num_classes=config.num_classes,
        rho=config.rho,
        noise_degree=config.noise_degree,
        rng=rng_test,
    )

    return {
        "train": {
            "0": train_modalities[0],
            "1": train_modalities[1],
            "label": train_labels.reshape(-1, 1),
        },
        "valid": {
            "0": valid_modalities[0],
            "1": valid_modalities[1],
            "label": valid_labels.reshape(-1, 1),
        },
        "test": {
            "0": test_modalities[0],
            "1": test_modalities[1],
            "label": test_labels.reshape(-1, 1),
        },
    }


def _generate_official_gaussian_split(
    *,
    num_samples: int,
    num_classes: int,
    rho: float,
    noise_degree: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    labels = rng.integers(0, num_classes, size=num_samples, endpoint=False, dtype=np.int64)
    rows = np.arange(num_samples)
    onehot = np.zeros((num_samples, num_classes), dtype=np.float64)
    onehot[rows, labels] = 1.0

    u = rng.standard_normal((num_samples, num_classes), dtype=np.float64) * noise_degree
    v = rng.standard_normal((num_samples, num_classes), dtype=np.float64) * noise_degree
    x1 = onehot + u
    x2 = onehot + rho * u + np.sqrt(max(1.0 - rho**2, 0.0)) * v
    return [x1.astype(np.float32), x2.astype(np.float32)], labels.astype(np.int64)


def save_generated_dataset(
    path: str,
    config: SyntheticGenerationConfig,
    *,
    dataset: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    output = dataset if dataset is not None else generate_synthetic_dataset(config)
    save_pickle_dataset(output, path)
    return output


def make_binary_logic_dataset(
    name: str,
    *,
    repeat: int = 1024,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MultimodalDataset]:
    if name not in BINARY_LOGIC_CASES:
        raise KeyError(f"unknown binary synthetic case: {name}")
    x1, x2, y = BINARY_LOGIC_CASES[name]
    x1_tensor = torch.as_tensor(x1, dtype=dtype).repeat((repeat, 1))
    x2_tensor = torch.as_tensor(x2, dtype=dtype).repeat((repeat, 1))
    y_tensor = torch.as_tensor(y, dtype=torch.long).repeat((repeat, 1))
    dataset = MultimodalDataset([x1_tensor, x2_tensor], y_tensor)
    return x1_tensor, x2_tensor, y_tensor, dataset


def load_generated_dataset(path: str) -> dict[str, dict[str, np.ndarray]]:
    return load_pickle_dataset(path)


def prob_set(values: list[Any]) -> dict[Any, float]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values))
    return {key: count / total for key, count in counts.items()}


def get_min_mi(x1: list[Any], x2: list[Any], y: list[Any]) -> float:
    p1 = prob_set(x1)
    p1y = prob_set(list(zip(x1, y)))
    p2 = prob_set(x2)
    p2y = prob_set(list(zip(x2, y)))
    py = prob_set(y)

    values_by_label: dict[Any, list[list[float]]] = {}
    for value_1, value_2, label in zip(x1, x2, y):
        values_by_label.setdefault(label, [])
        info_1 = math.log2(p1y[value_1, label] / (p1[value_1] * py[label]))
        info_2 = math.log2(p2y[value_2, label] / (p2[value_2] * py[label]))
        values_by_label[label].append([info_1, info_2])

    result = 0.0
    for label, scores in values_by_label.items():
        result += (len(scores) / len(y)) * float(np.min(np.mean(np.array(scores), axis=0)))
    return result


def get_mi_terms(x1: list[Any], x2: list[Any], y: list[Any]) -> float:
    p1 = prob_set(x1)
    p1y = prob_set(list(zip(x1, y)))
    p2 = prob_set(x2)
    p2y = prob_set(list(zip(x2, y)))
    py = prob_set(y)
    p12 = prob_set(list(zip(x1, x2)))
    p12y = prob_set(list(zip(x1, x2, y)))

    total = 0.0
    for value_1, value_2, label in zip(x1, x2, y):
        total += (
            math.log2(p1y[value_1, label] / (p1[value_1] * py[label]))
            + math.log2(p2y[value_2, label] / (p2[value_2] * py[label]))
            - math.log2(p12y[value_1, value_2, label] / (p12[value_1, value_2] * py[label]))
        )
    return total / len(y)


def enumerate_alignments(x1: list[Any], x2: list[Any], y: list[int]) -> tuple[float, tuple[int, ...] | None, tuple[int, ...] | None]:
    x10 = [x1[index] for index in range(len(x1)) if not y[index]]
    x11 = [x1[index] for index in range(len(x1)) if y[index]]
    x20 = [x2[index] for index in range(len(x2)) if not y[index]]
    x21 = [x2[index] for index in range(len(x2)) if y[index]]

    perm0 = list(permutations(range(len(x10))))
    perm1 = list(permutations(range(len(x11))))

    max_mi = -float("inf")
    best_perm0 = None
    best_perm1 = None
    for current_perm0 in perm0:
        for current_perm1 in perm1:
            x20_perm = [x20[index] for index in current_perm0]
            x21_perm = [x21[index] for index in current_perm1]
            labels = [0] * len(x10) + [1] * len(x11)
            mi = get_mi_terms(x10 + x11, x20_perm + x21_perm, labels)
            if mi > max_mi:
                max_mi = mi
                best_perm0 = current_perm0
                best_perm1 = current_perm1
    return max_mi, best_perm0, best_perm1


def _q_x1_x2(x1: Any, x2: Any, p1y: dict[Any, float], p2y: dict[Any, float], py: dict[Any, float]) -> float:
    total = 0.0
    for label in py:
        total += py[label] * p1y.get((x1, label), 0.0) / py[label] * p2y.get((x2, label), 0.0) / py[label]
    return total


def _q_x1(x1: Any, p1y: dict[Any, float], p2y: dict[Any, float], py: dict[Any, float]) -> float:
    total = 0.0
    for x2, label in p2y:
        total += py[label] * p1y.get((x1, label), 0.0) / py[label] * p2y.get((x2, label), 0.0) / py[label]
    return total


def _q_x2(x2: Any, p1y: dict[Any, float], p2y: dict[Any, float], py: dict[Any, float]) -> float:
    total = 0.0
    for x1, label in p1y:
        total += py[label] * p1y.get((x1, label), 0.0) / py[label] * p2y.get((x2, label), 0.0) / py[label]
    return total


def get_analytic_upperbound(x1: list[Any], x2: list[Any], y: list[Any]) -> float:
    p1 = prob_set(x1)
    p2 = prob_set(x2)
    py = prob_set(y)
    p1y = prob_set(list(zip(x1, y)))
    p2y = prob_set(list(zip(x2, y)))

    total = 0.0
    for label in py:
        for value_1 in p1:
            for value_2 in p2:
                weight = py[label] * (p1y.get((value_1, label), 0.0) / py[label]) * (
                    p2y.get((value_2, label), 0.0) / py[label]
                )
                if weight == 0:
                    continue
                total += weight * math.log2(
                    _q_x1_x2(value_1, value_2, p1y, p2y, py)
                    / (_q_x1(value_1, p1y, p2y, py) * _q_x2(value_2, p1y, p2y, py))
                )
    return total


def get_mi_1_1(x1: list[Any], x2: list[Any]) -> float:
    p1 = prob_set(x1)
    p2 = prob_set(x2)
    p12 = prob_set(list(zip(x1, x2)))
    return sum(math.log2(p12[value_1, value_2] / (p1[value_1] * p2[value_2])) for value_1, value_2 in zip(x1, x2)) / len(x1)


def get_mi_2_1(x1: list[Any], x2: list[Any], y: list[Any]) -> float:
    p12 = prob_set(list(zip(x1, x2)))
    p12y = prob_set(list(zip(x1, x2, y)))
    py = prob_set(y)
    return sum(
        math.log2(p12y[value_1, value_2, label] / (p12[value_1, value_2] * py[label]))
        for value_1, value_2, label in zip(x1, x2, y)
    ) / len(y)


def get_cmi_1_1_1(x1: list[Any], x2: list[Any], y: list[Any]) -> float:
    p12y = prob_set(list(zip(x1, x2, y)))
    py = prob_set(y)
    p1y = prob_set(list(zip(x1, y)))
    p2y = prob_set(list(zip(x2, y)))
    return sum(
        math.log2(p12y[value_1, value_2, label] * py[label] / (p1y[value_1, label] * p2y[value_2, label]))
        for value_1, value_2, label in zip(x1, x2, y)
    ) / len(y)


def get_interaction_information(x1: list[Any], x2: list[Any], y: list[Any]) -> float:
    return get_mi_1_1(x1, x2) - get_cmi_1_1_1(x1, x2, y)


def analytic_pid_decomposition(
    x1: list[Any],
    x2: list[Any],
    y: list[Any],
    *,
    redundancy_fn=get_min_mi,
) -> dict[str, float]:
    redundancy = redundancy_fn(x1, x2, y)
    unique_1 = get_mi_1_1(x1, y) - redundancy
    unique_2 = get_mi_1_1(x2, y) - redundancy
    synergy = get_mi_2_1(x1, x2, y) - redundancy - unique_1 - unique_2
    synergy_from_interaction = redundancy - get_interaction_information(x1, x2, y)
    return {
        "redundancy": redundancy,
        "unique_1": unique_1,
        "unique_2": unique_2,
        "synergy": synergy,
        "synergy_from_interaction": synergy_from_interaction,
    }


def _intersection_keys(modality_number: int) -> list[str]:
    combinations_iter = chain.from_iterable(combinations(np.arange(modality_number), size) for size in range(1, modality_number + 1))
    return ["".join(str(index) for index in combo) for combo in combinations_iter]


__all__ = [
    "BINARY_LOGIC_CASES",
    "SyntheticGenerationConfig",
    "analytic_pid_decomposition",
    "enumerate_alignments",
    "generate_synthetic_dataset",
    "get_analytic_upperbound",
    "get_interaction_information",
    "get_min_mi",
    "get_mi_1_1",
    "get_mi_2_1",
    "get_mi_terms",
    "load_generated_dataset",
    "make_binary_logic_dataset",
    "prob_set",
    "save_generated_dataset",
]
