from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


def _as_bool_mask(mask: Sequence[bool] | np.ndarray, num_atoms: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape[0] != num_atoms:
        raise ValueError(f"expected mask with {num_atoms} atoms, got {values.shape[0]}")
    return values


@dataclass
class LocalMotif:
    atom_a: int
    atom_b: int
    motif_type: str
    strength: float = 1.0
    singleton_weight: float = 0.0
    pair_bonus: float = 0.0

    def value(self, a_selected: bool, b_selected: bool) -> float:
        if self.motif_type == "synergy":
            return float(self.strength) if (a_selected and b_selected) else 0.0
        if self.motif_type == "redundancy":
            return float(self.strength) if (a_selected or b_selected) else 0.0
        if self.motif_type == "unique_a":
            return float(self.strength) if a_selected else 0.0
        if self.motif_type == "unique_b":
            return float(self.strength) if b_selected else 0.0
        if self.motif_type == "coupling":
            return (
                float(self.singleton_weight) * float(a_selected)
                + float(self.singleton_weight) * float(b_selected)
                + float(self.pair_bonus) * float(a_selected and b_selected)
            )
        raise ValueError(f"unknown motif type: {self.motif_type}")

    def max_value(self) -> float:
        candidates = [
            self.value(False, False),
            self.value(True, False),
            self.value(False, True),
            self.value(True, True),
        ]
        return float(max(candidates))

    def true_pair_interaction(self) -> float:
        delta = (
            self.value(True, True)
            - self.value(True, False)
            - self.value(False, True)
            + self.value(False, False)
        )
        return float(delta)

    def evaluation_label(self) -> str:
        if self.motif_type == "synergy":
            return "S"
        if self.motif_type == "redundancy":
            return "R"
        if self.motif_type in {"unique_a", "unique_b"}:
            return "U"
        if self.motif_type == "coupling":
            if self.pair_bonus > 0:
                return "S"
            if self.pair_bonus < 0:
                return "R"
            return "U"
        raise ValueError(f"unknown motif type: {self.motif_type}")


@dataclass
class SyntheticSubsetAttributionGame:
    num_atoms: int
    atom_modalities: np.ndarray
    atom_samples: np.ndarray
    motifs: list[LocalMotif]
    candidate_edges: list[tuple[int, int]]
    name: str = "synthetic_subset_game"
    edge_labels: dict[tuple[int, int], str] | None = None

    def __post_init__(self) -> None:
        self.atom_modalities = np.asarray(self.atom_modalities, dtype=np.int64).reshape(-1)
        self.atom_samples = np.asarray(self.atom_samples, dtype=np.int64).reshape(-1)
        if self.atom_modalities.shape[0] != self.num_atoms or self.atom_samples.shape[0] != self.num_atoms:
            raise ValueError("atom metadata must match num_atoms")
        self.candidate_edges = [tuple(sorted(edge)) for edge in self.candidate_edges]
        if len(set(self.candidate_edges)) != len(self.candidate_edges):
            raise ValueError("candidate edges must be unique")
        self._motif_by_edge = {
            tuple(sorted((motif.atom_a, motif.atom_b))): motif for motif in self.motifs
        }
        if self.edge_labels is None:
            self.edge_labels = {
                edge: self._motif_by_edge.get(edge, LocalMotif(edge[0], edge[1], "coupling")).evaluation_label()
                for edge in self.candidate_edges
            }
        self.normalizer = float(sum(max(motif.max_value(), 1e-8) for motif in self.motifs))

    def utility(self, mask: Sequence[bool] | np.ndarray) -> float:
        selected = _as_bool_mask(mask, self.num_atoms)
        total = 0.0
        for motif in self.motifs:
            total += motif.value(bool(selected[motif.atom_a]), bool(selected[motif.atom_b]))
        return float(total / self.normalizer)

    def utility_batch(self, masks: Sequence[Sequence[bool]] | np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=bool)
        if values.ndim != 2 or values.shape[1] != self.num_atoms:
            raise ValueError(f"expected masks with shape [N, {self.num_atoms}]")
        return np.asarray([self.utility(row) for row in values], dtype=np.float64)

    def full_mask(self) -> np.ndarray:
        return np.ones(self.num_atoms, dtype=bool)

    def exact_pair_interactions(self, exhaustive: bool = False) -> dict[tuple[int, int], float]:
        if exhaustive:
            return {edge: self._exhaustive_pair_interaction(*edge) for edge in self.candidate_edges}
        values: dict[tuple[int, int], float] = {}
        for edge in self.candidate_edges:
            motif = self._motif_by_edge.get(edge)
            values[edge] = 0.0 if motif is None else float(motif.true_pair_interaction() / self.normalizer)
        return values

    def positive_interaction_mass(self, interactions: dict[tuple[int, int], float] | None = None) -> float:
        source = interactions if interactions is not None else self.exact_pair_interactions(exhaustive=False)
        return float(sum(max(value, 0.0) for value in source.values()))

    def _exhaustive_pair_interaction(self, atom_a: int, atom_b: int) -> float:
        other_atoms = [index for index in range(self.num_atoms) if index not in (atom_a, atom_b)]
        total = 0.0
        for background_bits in product([False, True], repeat=len(other_atoms)):
            base = np.zeros(self.num_atoms, dtype=bool)
            for index, bit in zip(other_atoms, background_bits):
                base[index] = bit
            b00 = base.copy()
            b10 = base.copy()
            b01 = base.copy()
            b11 = base.copy()
            b10[atom_a] = True
            b01[atom_b] = True
            b11[atom_a] = True
            b11[atom_b] = True
            total += self.utility(b11) - self.utility(b10) - self.utility(b01) + self.utility(b00)
        return float(total / (2 ** max(self.num_atoms - 2, 0)))


@dataclass(frozen=True)
class PairProbeRecord:
    edge: tuple[int, int]
    target_delta: float


@dataclass(frozen=True)
class StructuredSubsetSurrogateConfig:
    feature_bernoulli_p: float = 0.5
    ridge_lambda: float = 1e-4
    pair_probe_weight: float = 1.0


@dataclass(frozen=True)
class RBFSubsetSurrogateConfig:
    ridge_lambda: float = 1e-4
    gamma: float | None = None


@dataclass
class StructuredSubsetSurrogate:
    num_atoms: int
    candidate_edges: list[tuple[int, int]]
    intercept: float
    main_effects: np.ndarray
    pair_effects: np.ndarray
    feature_bernoulli_p: float = 0.5

    def __post_init__(self) -> None:
        self.main_effects = np.asarray(self.main_effects, dtype=np.float64).reshape(self.num_atoms)
        self.pair_effects = np.asarray(self.pair_effects, dtype=np.float64).reshape(len(self.candidate_edges))
        self._edge_to_index = {edge: index for index, edge in enumerate(self.candidate_edges)}

    def centered_mask(self, mask: Sequence[bool] | np.ndarray) -> np.ndarray:
        selected = _as_bool_mask(mask, self.num_atoms).astype(np.float64)
        p = float(self.feature_bernoulli_p)
        scale = np.sqrt(max(p * (1.0 - p), 1e-12))
        return (selected - p) / scale

    def predict(self, mask: Sequence[bool] | np.ndarray) -> float:
        centered = self.centered_mask(mask)
        pair_terms = np.asarray(
            [centered[left] * centered[right] for left, right in self.candidate_edges],
            dtype=np.float64,
        )
        value = self.intercept + float(centered @ self.main_effects) + float(pair_terms @ self.pair_effects)
        return float(value)

    def predict_batch(self, masks: Sequence[Sequence[bool]] | np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=bool)
        if values.ndim != 2 or values.shape[1] != self.num_atoms:
            raise ValueError(f"expected masks with shape [N, {self.num_atoms}]")
        centered = np.asarray([self.centered_mask(row) for row in values], dtype=np.float64)
        pair_terms = np.stack(
            [
                centered[:, left] * centered[:, right]
                for left, right in self.candidate_edges
            ],
            axis=1,
        )
        return self.intercept + centered @ self.main_effects + pair_terms @ self.pair_effects

    def estimate_pair_interactions(self) -> dict[tuple[int, int], float]:
        p = float(self.feature_bernoulli_p)
        scale = 1.0 / max(p * (1.0 - p), 1e-12)
        return {
            edge: float(scale * self.pair_effects[index])
            for index, edge in enumerate(self.candidate_edges)
        }

    def attribution_mass(self, positive_only: bool = True) -> float:
        values = self.estimate_pair_interactions().values()
        if positive_only:
            return float(sum(max(value, 0.0) for value in values))
        return float(sum(values))


@dataclass
class LinearSubsetSurrogate:
    num_atoms: int
    candidate_edges: list[tuple[int, int]]
    intercept: float
    main_effects: np.ndarray
    feature_bernoulli_p: float = 0.5

    def __post_init__(self) -> None:
        self.main_effects = np.asarray(self.main_effects, dtype=np.float64).reshape(self.num_atoms)

    def centered_mask(self, mask: Sequence[bool] | np.ndarray) -> np.ndarray:
        selected = _as_bool_mask(mask, self.num_atoms).astype(np.float64)
        p = float(self.feature_bernoulli_p)
        scale = np.sqrt(max(p * (1.0 - p), 1e-12))
        return (selected - p) / scale

    def predict(self, mask: Sequence[bool] | np.ndarray) -> float:
        centered = self.centered_mask(mask)
        return float(self.intercept + centered @ self.main_effects)

    def predict_batch(self, masks: Sequence[Sequence[bool]] | np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=bool)
        if values.ndim != 2 or values.shape[1] != self.num_atoms:
            raise ValueError(f"expected masks with shape [N, {self.num_atoms}]")
        centered = np.asarray([self.centered_mask(row) for row in values], dtype=np.float64)
        return self.intercept + centered @ self.main_effects

    def estimate_pair_interactions(self) -> dict[tuple[int, int], float]:
        return {edge: 0.0 for edge in self.candidate_edges}

    def attribution_mass(self, positive_only: bool = True) -> float:
        return 0.0


@dataclass
class RBFKernelSubsetSurrogate:
    num_atoms: int
    candidate_edges: list[tuple[int, int]]
    train_masks: np.ndarray
    dual_weights: np.ndarray
    gamma: float

    def __post_init__(self) -> None:
        self.train_masks = np.asarray(self.train_masks, dtype=np.float64).reshape(-1, self.num_atoms)
        self.dual_weights = np.asarray(self.dual_weights, dtype=np.float64).reshape(-1)

    def predict(self, mask: Sequence[bool] | np.ndarray) -> float:
        return float(self.predict_batch(np.asarray(mask, dtype=bool).reshape(1, -1))[0])

    def predict_batch(self, masks: Sequence[Sequence[bool]] | np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=bool)
        if values.ndim != 2 or values.shape[1] != self.num_atoms:
            raise ValueError(f"expected masks with shape [N, {self.num_atoms}]")
        kernel = _rbf_kernel(values.astype(np.float64), self.train_masks, self.gamma)
        return kernel @ self.dual_weights

    def estimate_pair_interactions(self) -> dict[tuple[int, int], float]:
        return estimate_pair_interactions_from_surrogate(
            self,
            num_atoms=self.num_atoms,
            candidate_edges=self.candidate_edges,
            exhaustive=self.num_atoms <= 16,
        )

    def attribution_mass(self, positive_only: bool = True) -> float:
        interactions = self.estimate_pair_interactions()
        values = interactions.values()
        if positive_only:
            return float(sum(max(value, 0.0) for value in values))
        return float(sum(values))

    def estimate_singleton_attributions(self) -> dict[int, float]:
        return estimate_singleton_attributions_from_surrogate(
            self,
            num_atoms=self.num_atoms,
            exhaustive=self.num_atoms <= 16,
        )


def build_cross_modal_candidate_edges(atom_modalities: Sequence[int], atom_samples: Sequence[int]) -> list[tuple[int, int]]:
    modalities = np.asarray(atom_modalities, dtype=np.int64).reshape(-1)
    samples = np.asarray(atom_samples, dtype=np.int64).reshape(-1)
    if modalities.shape != samples.shape:
        raise ValueError("atom metadata must have the same shape")
    edges: list[tuple[int, int]] = []
    for left in range(modalities.shape[0]):
        for right in range(left + 1, modalities.shape[0]):
            if samples[left] == samples[right]:
                continue
            if modalities[left] == modalities[right]:
                continue
            edges.append((left, right))
    return edges


def sample_global_subset_queries(
    game: SyntheticSubsetAttributionGame,
    *,
    num_queries: int,
    query_probabilities: Sequence[float] = (0.3, 0.5, 0.7),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = np.asarray(query_probabilities, dtype=np.float64)
    masks = np.zeros((num_queries, game.num_atoms), dtype=bool)
    for index in range(num_queries):
        probability = float(rng.choice(probs))
        masks[index] = rng.random(game.num_atoms) < probability
    return masks, game.utility_batch(masks)


def sample_pair_probes(
    game: SyntheticSubsetAttributionGame,
    *,
    num_backgrounds_per_edge: int,
    edges: Iterable[tuple[int, int]] | None = None,
    background_probability: float = 0.5,
    seed: int = 0,
) -> list[PairProbeRecord]:
    rng = np.random.default_rng(seed)
    selected_edges = list(edges if edges is not None else game.candidate_edges)
    records: list[PairProbeRecord] = []
    for edge in selected_edges:
        left, right = edge
        other_atoms = [index for index in range(game.num_atoms) if index not in edge]
        for _ in range(num_backgrounds_per_edge):
            background = np.zeros(game.num_atoms, dtype=bool)
            if other_atoms:
                background[other_atoms] = rng.random(len(other_atoms)) < background_probability
            b00 = background.copy()
            b10 = background.copy()
            b01 = background.copy()
            b11 = background.copy()
            b10[left] = True
            b01[right] = True
            b11[left] = True
            b11[right] = True
            delta = game.utility(b11) - game.utility(b10) - game.utility(b01) + game.utility(b00)
            records.append(PairProbeRecord(edge=edge, target_delta=float(delta)))
    return records


def fit_structured_subset_surrogate(
    subset_masks: np.ndarray,
    subset_values: np.ndarray,
    *,
    num_atoms: int,
    candidate_edges: Sequence[tuple[int, int]],
    pair_probes: Sequence[PairProbeRecord] | None = None,
    config: StructuredSubsetSurrogateConfig | None = None,
) -> StructuredSubsetSurrogate:
    config = config or StructuredSubsetSurrogateConfig()
    candidate_edges = [tuple(sorted(edge)) for edge in candidate_edges]
    subset_masks = np.asarray(subset_masks, dtype=bool)
    subset_values = np.asarray(subset_values, dtype=np.float64).reshape(-1)
    if subset_masks.ndim != 2 or subset_masks.shape != (subset_values.shape[0], num_atoms):
        raise ValueError("subset masks and values must align")

    centered = _center_masks(subset_masks, feature_bernoulli_p=config.feature_bernoulli_p)
    subset_design = _design_matrix_from_centered(centered, candidate_edges)
    design_rows = [subset_design]
    target_rows = [subset_values]

    if pair_probes:
        pair_design = np.zeros((len(pair_probes), 1 + num_atoms + len(candidate_edges)), dtype=np.float64)
        scale = 1.0 / max(config.feature_bernoulli_p * (1.0 - config.feature_bernoulli_p), 1e-12)
        edge_to_index = {edge: index for index, edge in enumerate(candidate_edges)}
        for row, record in enumerate(pair_probes):
            pair_column = 1 + num_atoms + edge_to_index[tuple(sorted(record.edge))]
            pair_design[row, pair_column] = scale
        design_rows.append(np.sqrt(config.pair_probe_weight) * pair_design)
        target_rows.append(
            np.sqrt(config.pair_probe_weight)
            * np.asarray([record.target_delta for record in pair_probes], dtype=np.float64)
        )

    design = np.concatenate(design_rows, axis=0)
    targets = np.concatenate(target_rows, axis=0)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + float(config.ridge_lambda) * penalty
    rhs = design.T @ targets
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.pinv(lhs) @ rhs
    return StructuredSubsetSurrogate(
        num_atoms=num_atoms,
        candidate_edges=candidate_edges,
        intercept=float(solution[0]),
        main_effects=solution[1 : 1 + num_atoms],
        pair_effects=solution[1 + num_atoms :],
        feature_bernoulli_p=config.feature_bernoulli_p,
    )


def fit_linear_subset_surrogate(
    subset_masks: np.ndarray,
    subset_values: np.ndarray,
    *,
    num_atoms: int,
    candidate_edges: Sequence[tuple[int, int]],
    config: StructuredSubsetSurrogateConfig | None = None,
) -> LinearSubsetSurrogate:
    config = config or StructuredSubsetSurrogateConfig()
    subset_masks = np.asarray(subset_masks, dtype=bool)
    subset_values = np.asarray(subset_values, dtype=np.float64).reshape(-1)
    if subset_masks.ndim != 2 or subset_masks.shape != (subset_values.shape[0], num_atoms):
        raise ValueError("subset masks and values must align")
    centered = _center_masks(subset_masks, feature_bernoulli_p=config.feature_bernoulli_p)
    design = np.concatenate([np.ones((centered.shape[0], 1), dtype=np.float64), centered], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + float(config.ridge_lambda) * penalty
    rhs = design.T @ subset_values
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.pinv(lhs) @ rhs
    return LinearSubsetSurrogate(
        num_atoms=num_atoms,
        candidate_edges=[tuple(sorted(edge)) for edge in candidate_edges],
        intercept=float(solution[0]),
        main_effects=solution[1:],
        feature_bernoulli_p=config.feature_bernoulli_p,
    )


def fit_rbf_subset_surrogate(
    subset_masks: np.ndarray,
    subset_values: np.ndarray,
    *,
    num_atoms: int,
    candidate_edges: Sequence[tuple[int, int]],
    config: RBFSubsetSurrogateConfig | None = None,
) -> RBFKernelSubsetSurrogate:
    config = config or RBFSubsetSurrogateConfig()
    subset_masks = np.asarray(subset_masks, dtype=bool)
    subset_values = np.asarray(subset_values, dtype=np.float64).reshape(-1)
    if subset_masks.ndim != 2 or subset_masks.shape != (subset_values.shape[0], num_atoms):
        raise ValueError("subset masks and values must align")
    train_inputs = subset_masks.astype(np.float64)
    gamma = _resolve_rbf_gamma(train_inputs, config.gamma)
    kernel = _rbf_kernel(train_inputs, train_inputs, gamma)
    regularized = kernel + float(config.ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64)
    try:
        dual_weights = np.linalg.solve(regularized, subset_values)
    except np.linalg.LinAlgError:
        dual_weights = np.linalg.pinv(regularized) @ subset_values
    return RBFKernelSubsetSurrogate(
        num_atoms=num_atoms,
        candidate_edges=[tuple(sorted(edge)) for edge in candidate_edges],
        train_masks=train_inputs,
        dual_weights=dual_weights,
        gamma=gamma,
    )


def make_exact_logic_bridge_game(
    *,
    num_concepts: int = 4,
    logic_case: str = "xor",
) -> SyntheticSubsetAttributionGame:
    if logic_case not in {"xor", "and"}:
        raise ValueError("logic_case must be 'xor' or 'and'")
    atom_modalities: list[int] = []
    atom_samples: list[int] = []
    motifs: list[LocalMotif] = []
    edge_labels: dict[tuple[int, int], str] = {}
    for concept_index in range(num_concepts):
        atom_a = 2 * concept_index
        atom_b = atom_a + 1
        atom_modalities.extend([0, 1])
        atom_samples.extend([2 * concept_index, 2 * concept_index + 1])
        motifs.append(
            LocalMotif(
                atom_a=atom_a,
                atom_b=atom_b,
                motif_type="synergy",
                strength=1.0,
            )
        )
    candidate_edges = build_cross_modal_candidate_edges(atom_modalities, atom_samples)
    for edge in candidate_edges:
        edge_labels[edge] = "S" if edge in {(motif.atom_a, motif.atom_b) for motif in motifs} else "U"
    return SyntheticSubsetAttributionGame(
        num_atoms=2 * num_concepts,
        atom_modalities=np.asarray(atom_modalities, dtype=np.int64),
        atom_samples=np.asarray(atom_samples, dtype=np.int64),
        motifs=motifs,
        candidate_edges=candidate_edges,
        name=f"exact_logic_bridge_{logic_case}",
        edge_labels=edge_labels,
    )


def make_motif_mixture_game(
    *,
    num_concepts: int = 8,
    pi_redundancy: float = 1 / 3,
    pi_unique: float = 1 / 3,
    pi_synergy: float = 1 / 3,
    seed: int = 0,
) -> SyntheticSubsetAttributionGame:
    probabilities = np.asarray([pi_redundancy, pi_unique, pi_synergy], dtype=np.float64)
    if np.any(probabilities < 0):
        raise ValueError("mixture probabilities must be non-negative")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("mixture probabilities must sum to 1")

    rng = np.random.default_rng(seed)
    atom_modalities: list[int] = []
    atom_samples: list[int] = []
    motifs: list[LocalMotif] = []
    edge_labels: dict[tuple[int, int], str] = {}
    motif_names = ["redundancy", "unique", "synergy"]

    for concept_index in range(num_concepts):
        atom_a = 2 * concept_index
        atom_b = atom_a + 1
        atom_modalities.extend([0, 1])
        atom_samples.extend([2 * concept_index, 2 * concept_index + 1])
        motif_name = str(rng.choice(motif_names, p=probabilities))
        strength = float(rng.uniform(0.8, 1.2))
        if motif_name == "redundancy":
            motifs.append(LocalMotif(atom_a=atom_a, atom_b=atom_b, motif_type="redundancy", strength=strength))
        elif motif_name == "synergy":
            motifs.append(LocalMotif(atom_a=atom_a, atom_b=atom_b, motif_type="synergy", strength=strength))
        else:
            informative_side = "unique_a" if bool(rng.integers(0, 2)) else "unique_b"
            motifs.append(LocalMotif(atom_a=atom_a, atom_b=atom_b, motif_type=informative_side, strength=strength))

    candidate_edges = build_cross_modal_candidate_edges(atom_modalities, atom_samples)
    motif_edges = {(motif.atom_a, motif.atom_b): motif for motif in motifs}
    for edge in candidate_edges:
        motif = motif_edges.get(edge)
        edge_labels[edge] = "U" if motif is None else motif.evaluation_label()
    return SyntheticSubsetAttributionGame(
        num_atoms=2 * num_concepts,
        atom_modalities=np.asarray(atom_modalities, dtype=np.int64),
        atom_samples=np.asarray(atom_samples, dtype=np.int64),
        motifs=motifs,
        candidate_edges=candidate_edges,
        name="motif_mixture",
        edge_labels=edge_labels,
    )


def make_coupling_game(
    *,
    num_concepts: int = 8,
    coupling_strength: float = 0.0,
    singleton_weight: float = 0.4,
) -> SyntheticSubsetAttributionGame:
    if coupling_strength < 0.0:
        raise ValueError("coupling_strength must be non-negative for the monotonicity sweep")
    atom_modalities: list[int] = []
    atom_samples: list[int] = []
    motifs: list[LocalMotif] = []
    edge_labels: dict[tuple[int, int], str] = {}
    for concept_index in range(num_concepts):
        atom_a = 2 * concept_index
        atom_b = atom_a + 1
        atom_modalities.extend([0, 1])
        atom_samples.extend([2 * concept_index, 2 * concept_index + 1])
        motifs.append(
            LocalMotif(
                atom_a=atom_a,
                atom_b=atom_b,
                motif_type="coupling",
                singleton_weight=singleton_weight,
                pair_bonus=coupling_strength,
            )
        )
    candidate_edges = build_cross_modal_candidate_edges(atom_modalities, atom_samples)
    for edge in candidate_edges:
        edge_labels[edge] = "S" if edge in {(motif.atom_a, motif.atom_b) for motif in motifs} else "U"
    return SyntheticSubsetAttributionGame(
        num_atoms=2 * num_concepts,
        atom_modalities=np.asarray(atom_modalities, dtype=np.int64),
        atom_samples=np.asarray(atom_samples, dtype=np.int64),
        motifs=motifs,
        candidate_edges=candidate_edges,
        name="coupling_sweep",
        edge_labels=edge_labels,
    )


def make_missing_modality_rescue_game(
    *,
    num_concepts: int = 8,
    missing_rate: float = 0.5,
    partner_purity: float = 0.8,
    seed: int = 0,
) -> SyntheticSubsetAttributionGame:
    if not (0.0 <= missing_rate <= 1.0):
        raise ValueError("missing_rate must lie in [0, 1]")
    if not (0.0 <= partner_purity <= 1.0):
        raise ValueError("partner_purity must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    atom_modalities = np.zeros(2 * num_concepts, dtype=np.int64)
    atom_samples = np.zeros(2 * num_concepts, dtype=np.int64)
    motifs: list[LocalMotif] = []
    edge_labels: dict[tuple[int, int], str] = {}
    rescue_edges: set[tuple[int, int]] = set()

    for concept_index in range(num_concepts):
        anchor_atom = concept_index
        partner_atom = num_concepts + concept_index
        atom_modalities[anchor_atom] = 0
        atom_samples[anchor_atom] = concept_index
        atom_modalities[partner_atom] = 1
        atom_samples[partner_atom] = 10_000 + concept_index
        is_missing = bool(rng.random() < missing_rate)
        is_true_partner = bool(rng.random() < partner_purity)
        if is_missing and is_true_partner:
            motifs.append(
                LocalMotif(
                    atom_a=anchor_atom,
                    atom_b=partner_atom,
                    motif_type="synergy",
                    strength=float(rng.uniform(0.8, 1.2)),
                )
            )
            rescue_edges.add((anchor_atom, partner_atom))
        elif not is_missing:
            motifs.append(
                LocalMotif(
                    atom_a=anchor_atom,
                    atom_b=partner_atom,
                    motif_type="unique_a",
                    strength=float(rng.uniform(0.5, 0.8)),
                )
            )

    candidate_edges = build_cross_modal_candidate_edges(atom_modalities, atom_samples)
    for edge in candidate_edges:
        edge_labels[edge] = "S" if edge in rescue_edges else "U"
    game = SyntheticSubsetAttributionGame(
        num_atoms=2 * num_concepts,
        atom_modalities=atom_modalities,
        atom_samples=atom_samples,
        motifs=motifs,
        candidate_edges=candidate_edges,
        name="missing_modality_rescue",
        edge_labels=edge_labels,
    )
    game.rescue_edges = rescue_edges  # type: ignore[attr-defined]
    game.missing_rate = float(missing_rate)  # type: ignore[attr-defined]
    game.partner_purity = float(partner_purity)  # type: ignore[attr-defined]
    return game


def _center_masks(masks: np.ndarray, *, feature_bernoulli_p: float) -> np.ndarray:
    probabilities = float(feature_bernoulli_p)
    scale = np.sqrt(max(probabilities * (1.0 - probabilities), 1e-12))
    return (masks.astype(np.float64) - probabilities) / scale


def _design_matrix_from_centered(centered: np.ndarray, candidate_edges: Sequence[tuple[int, int]]) -> np.ndarray:
    pair_terms = np.stack(
        [centered[:, left] * centered[:, right] for left, right in candidate_edges],
        axis=1,
    )
    intercept = np.ones((centered.shape[0], 1), dtype=np.float64)
    return np.concatenate([intercept, centered, pair_terms], axis=1)


def estimate_pair_interactions_from_surrogate(
    surrogate: object,
    *,
    num_atoms: int,
    candidate_edges: Sequence[tuple[int, int]],
    exhaustive: bool = False,
    num_mc_backgrounds: int = 512,
    seed: int = 0,
) -> dict[tuple[int, int], float]:
    if exhaustive:
        all_masks = np.zeros((2**num_atoms, num_atoms), dtype=bool)
        for index in range(2**num_atoms):
            for bit in range(num_atoms):
                all_masks[index, bit] = bool((index >> bit) & 1)
        predictions = np.asarray(surrogate.predict_batch(all_masks), dtype=np.float64)
        mask_to_index = np.arange(2**num_atoms, dtype=np.int64)
        values: dict[tuple[int, int], float] = {}
        for edge in candidate_edges:
            left_mask = 1 << edge[0]
            right_mask = 1 << edge[1]
            total = 0.0
            count = 0
            for index in range(2**num_atoms):
                if (index & left_mask) or (index & right_mask):
                    continue
                b00 = index
                b10 = index | left_mask
                b01 = index | right_mask
                b11 = index | left_mask | right_mask
                total += predictions[b11] - predictions[b10] - predictions[b01] + predictions[b00]
                count += 1
            values[tuple(sorted(edge))] = float(total / max(count, 1))
        return values

    rng = np.random.default_rng(seed)
    values = {}
    for edge in candidate_edges:
        other_atoms = [index for index in range(num_atoms) if index not in edge]
        batch_masks = np.zeros((4 * num_mc_backgrounds, num_atoms), dtype=bool)
        for sample_index in range(num_mc_backgrounds):
            background = np.zeros(num_atoms, dtype=bool)
            if other_atoms:
                background[other_atoms] = rng.random(len(other_atoms)) < 0.5
            base = 4 * sample_index
            batch_masks[base] = background
            batch_masks[base + 1] = background
            batch_masks[base + 2] = background
            batch_masks[base + 3] = background
            batch_masks[base + 1, edge[0]] = True
            batch_masks[base + 2, edge[1]] = True
            batch_masks[base + 3, edge[0]] = True
            batch_masks[base + 3, edge[1]] = True
        predictions = np.asarray(surrogate.predict_batch(batch_masks), dtype=np.float64).reshape(num_mc_backgrounds, 4)
        deltas = predictions[:, 3] - predictions[:, 1] - predictions[:, 2] + predictions[:, 0]
        values[tuple(sorted(edge))] = float(np.mean(deltas))
    return values


def estimate_singleton_attributions_from_surrogate(
    surrogate: object,
    *,
    num_atoms: int,
    exhaustive: bool = False,
    num_mc_backgrounds: int = 512,
    seed: int = 0,
) -> dict[int, float]:
    if exhaustive:
        all_masks = np.zeros((2**num_atoms, num_atoms), dtype=bool)
        for index in range(2**num_atoms):
            for bit in range(num_atoms):
                all_masks[index, bit] = bool((index >> bit) & 1)
        predictions = np.asarray(surrogate.predict_batch(all_masks), dtype=np.float64)
        values: dict[int, float] = {}
        for atom in range(num_atoms):
            atom_mask = 1 << atom
            total = 0.0
            count = 0
            for index in range(2**num_atoms):
                if index & atom_mask:
                    continue
                total += predictions[index | atom_mask] - predictions[index]
                count += 1
            values[atom] = float(total / max(count, 1))
        return values

    rng = np.random.default_rng(seed)
    values: dict[int, float] = {}
    for atom in range(num_atoms):
        other_atoms = [index for index in range(num_atoms) if index != atom]
        batch_masks = np.zeros((2 * num_mc_backgrounds, num_atoms), dtype=bool)
        for sample_index in range(num_mc_backgrounds):
            background = np.zeros(num_atoms, dtype=bool)
            if other_atoms:
                background[other_atoms] = rng.random(len(other_atoms)) < 0.5
            base = 2 * sample_index
            batch_masks[base] = background
            batch_masks[base + 1] = background
            batch_masks[base + 1, atom] = True
        predictions = np.asarray(surrogate.predict_batch(batch_masks), dtype=np.float64).reshape(num_mc_backgrounds, 2)
        deltas = predictions[:, 1] - predictions[:, 0]
        values[atom] = float(np.mean(deltas))
    return values


def estimate_cross_modal_pid_from_scores(
    scores: np.ndarray,
    *,
    num_atoms: int,
    atom_samples: Sequence[int],
    atom_modalities: Sequence[int],
    modality_pair: tuple[int, int] = (0, 1),
) -> dict[tuple[int, int], dict[str, float]]:
    atom_samples = np.asarray(atom_samples, dtype=np.int64).reshape(-1)
    atom_modalities = np.asarray(atom_modalities, dtype=np.int64).reshape(-1)
    if atom_samples.shape[0] != num_atoms or atom_modalities.shape[0] != num_atoms:
        raise ValueError("atom metadata must match num_atoms")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != 2**num_atoms:
        raise ValueError(f"expected scores for all {2**num_atoms} masks")

    left_modality, right_modality = modality_pair
    left_atoms = {
        int(sample): int(index)
        for index, (sample, modality) in enumerate(zip(atom_samples, atom_modalities, strict=True))
        if int(modality) == int(left_modality)
    }
    right_atoms = {
        int(sample): int(index)
        for index, (sample, modality) in enumerate(zip(atom_samples, atom_modalities, strict=True))
        if int(modality) == int(right_modality)
    }

    values: dict[tuple[int, int], dict[str, float]] = {}
    for left_sample, left_atom in sorted(left_atoms.items()):
        for right_sample, right_atom in sorted(right_atoms.items()):
            if left_sample == right_sample:
                continue
            left_mask = 1 << left_atom
            right_mask = 1 << right_atom
            gains: list[tuple[float, float, float, float, float, float, float]] = []
            for index in range(2**num_atoms):
                if (index & left_mask) or (index & right_mask):
                    continue
                g00 = scores[index]
                g10 = scores[index | left_mask]
                g01 = scores[index | right_mask]
                g11 = scores[index | left_mask | right_mask]
                i1 = float(g10 - g00)
                i2 = float(g01 - g00)
                i12 = float(g11 - g00)
                redundancy = _signed_min_redundancy(i1, i2)
                unique_left = i1 - redundancy
                unique_right = i2 - redundancy
                synergy = i12 - i1 - i2 + redundancy
                gains.append((i1, i2, i12, redundancy, unique_left, unique_right, synergy))
            component_means = np.mean(np.asarray(gains, dtype=np.float64), axis=0)
            values[(left_sample, right_sample)] = {
                "I1": float(component_means[0]),
                "I2": float(component_means[1]),
                "I12": float(component_means[2]),
                "R": float(component_means[3]),
                "U1": float(component_means[4]),
                "U2": float(component_means[5]),
                "S": float(component_means[6]),
            }
    return values


def estimate_cross_modal_pid_from_surrogate(
    surrogate: object,
    *,
    num_atoms: int,
    atom_samples: Sequence[int],
    atom_modalities: Sequence[int],
    modality_pair: tuple[int, int] = (0, 1),
    exhaustive: bool = False,
    num_mc_backgrounds: int = 512,
    seed: int = 0,
) -> dict[tuple[int, int], dict[str, float]]:
    atom_samples = np.asarray(atom_samples, dtype=np.int64).reshape(-1)
    atom_modalities = np.asarray(atom_modalities, dtype=np.int64).reshape(-1)
    if atom_samples.shape[0] != num_atoms or atom_modalities.shape[0] != num_atoms:
        raise ValueError("atom metadata must match num_atoms")

    if exhaustive:
        all_masks = np.zeros((2**num_atoms, num_atoms), dtype=bool)
        for index in range(2**num_atoms):
            for bit in range(num_atoms):
                all_masks[index, bit] = bool((index >> bit) & 1)
        scores = np.asarray(surrogate.predict_batch(all_masks), dtype=np.float64)
        return estimate_cross_modal_pid_from_scores(
            scores,
            num_atoms=num_atoms,
            atom_samples=atom_samples,
            atom_modalities=atom_modalities,
            modality_pair=modality_pair,
        )

    left_modality, right_modality = modality_pair
    left_atoms = {
        int(sample): int(index)
        for index, (sample, modality) in enumerate(zip(atom_samples, atom_modalities, strict=True))
        if int(modality) == int(left_modality)
    }
    right_atoms = {
        int(sample): int(index)
        for index, (sample, modality) in enumerate(zip(atom_samples, atom_modalities, strict=True))
        if int(modality) == int(right_modality)
    }
    rng = np.random.default_rng(seed)
    values: dict[tuple[int, int], dict[str, float]] = {}
    for left_sample, left_atom in sorted(left_atoms.items()):
        for right_sample, right_atom in sorted(right_atoms.items()):
            if left_sample == right_sample:
                continue
            other_atoms = [index for index in range(num_atoms) if index not in {left_atom, right_atom}]
            batch_masks = np.zeros((4 * num_mc_backgrounds, num_atoms), dtype=bool)
            for sample_index in range(num_mc_backgrounds):
                background = np.zeros(num_atoms, dtype=bool)
                if other_atoms:
                    background[other_atoms] = rng.random(len(other_atoms)) < 0.5
                base = 4 * sample_index
                batch_masks[base] = background
                batch_masks[base + 1] = background
                batch_masks[base + 2] = background
                batch_masks[base + 3] = background
                batch_masks[base + 1, left_atom] = True
                batch_masks[base + 2, right_atom] = True
                batch_masks[base + 3, left_atom] = True
                batch_masks[base + 3, right_atom] = True
            predictions = np.asarray(surrogate.predict_batch(batch_masks), dtype=np.float64).reshape(num_mc_backgrounds, 4)
            i1 = predictions[:, 1] - predictions[:, 0]
            i2 = predictions[:, 2] - predictions[:, 0]
            i12 = predictions[:, 3] - predictions[:, 0]
            redundancy = np.asarray([_signed_min_redundancy(left_gain, right_gain) for left_gain, right_gain in zip(i1, i2, strict=True)])
            unique_left = i1 - redundancy
            unique_right = i2 - redundancy
            synergy = i12 - i1 - i2 + redundancy
            values[(left_sample, right_sample)] = {
                "I1": float(np.mean(i1)),
                "I2": float(np.mean(i2)),
                "I12": float(np.mean(i12)),
                "R": float(np.mean(redundancy)),
                "U1": float(np.mean(unique_left)),
                "U2": float(np.mean(unique_right)),
                "S": float(np.mean(synergy)),
            }
    return values


def aggregate_cross_modal_pid_masses(
    pair_components: dict[tuple[int, int], dict[str, float]],
) -> dict[str, float]:
    redundancy = np.asarray([components["R"] for components in pair_components.values()], dtype=np.float64)
    synergy = np.asarray([components["S"] for components in pair_components.values()], dtype=np.float64)
    return {
        "positive_redundancy_mass": float(np.sum(np.maximum(redundancy, 0.0))),
        "negative_redundancy_mass": float(np.sum(np.maximum(-redundancy, 0.0))),
        "positive_synergy_mass": float(np.sum(np.maximum(synergy, 0.0))),
        "negative_synergy_mass": float(np.sum(np.maximum(-synergy, 0.0))),
    }


def _signed_min_redundancy(i1: float, i2: float) -> float:
    positive_shared = min(max(float(i1), 0.0), max(float(i2), 0.0))
    negative_shared = min(max(-float(i1), 0.0), max(-float(i2), 0.0))
    return float(positive_shared - negative_shared)


def _resolve_rbf_gamma(inputs: np.ndarray, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    if inputs.shape[0] <= 1:
        return 1.0
    diffs = inputs[:, None, :] - inputs[None, :, :]
    normalization = max(inputs.shape[1], 1)
    sq_dists = np.sum(diffs * diffs, axis=-1) / float(normalization)
    positives = sq_dists[sq_dists > 0]
    if positives.size == 0:
        return 1.0
    return float(1.0 / np.median(positives))


def _rbf_kernel(lhs: np.ndarray, rhs: np.ndarray, gamma: float) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs_sq = np.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs * rhs, axis=1, keepdims=True).T
    normalization = max(lhs.shape[1], 1)
    sq_dist = (lhs_sq + rhs_sq - 2.0 * lhs @ rhs.T) / float(normalization)
    return np.exp(-gamma * np.maximum(sq_dist, 0.0))
