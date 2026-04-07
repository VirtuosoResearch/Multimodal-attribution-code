# Pilot Experiment: Representational Alignment ≠ Optimization Alignment

## Core Hypothesis

**Representational similarity (CKA, mutual k-NN) between modalities does NOT predict optimization compatibility (Fisher subspace overlap, gradient cosine). Only optimization compatibility reliably predicts when multimodal learning succeeds.**

Concretely:

- Samples where modalities have high CKA but low Fisher overlap → model fails despite "aligned" representations
- Samples where modalities have low CKA but high Fisher overlap → model succeeds despite "unaligned" representations
- This decoupling is systematic and predicted by the sample's interaction type (PID)

---

## Model Architecture

### Choice: I2MoE-style small encoders + MulT fusion

Drawn from I2MoE (ICML 2025) and MERGE (ICLR 2026) settings. These are lightweight, trainable on a single GPU, and use cross-entropy loss.

**Modality encoders** (per dataset):

| Dataset  | Modality 1                    | Modality 2                      | Modality 3             |
| -------- | ----------------------------- | ------------------------------- | ---------------------- |
| CMU-MOSI | GRU (d=256) for text          | GRU (d=256) for audio           | GRU (d=256) for video  |
| MIMIC    | LSTM (d=256) for lab          | LSTM (d=256) for notes          | LSTM (d=256) for codes |
| MM-IMDb  | VGG-16 (pretrained) for image | Word2Vec + MLP (d=256) for text | —                      |

**Fusion**: MulT (Multimodal Transformer) — 2-layer cross-attention with 4 heads.

- This gives us explicit cross-modal interaction layers where we can measure Fisher overlap.
- Hidden dim: 128
- FFN dim: 256

**Classification head**: Linear layer → softmax over classes.

**Loss**: Cross-entropy (primary) + optional auxiliary interaction loss (λ=0.01, following I2MoE).

### Why this architecture?

1. **Lightweight**: GRU/LSTM encoders + 2-layer transformer ≈ 5-10M parameters. Trainable on single V100/A100.
2. **Cross-entropy loss**: Not contrastive. Matches modern multimodal practice.
3. **Explicit fusion layers**: MulT cross-attention gives us clear "interaction points" where Fisher analysis is meaningful.
4. **Compatible with PID/LSMI**: Same architecture family used in MERGE and I2MoE, so we can directly compare.

---

## Datasets

### Primary: CMU-MOSI (2,199 video segments)

**Why**: Has diverse interaction types (unlike COCO which was all redundancy). Sentiment prediction requires integrating audio tone, visual expression, and language — creating genuine synergy samples alongside redundant/unique ones.

- 3 modalities: text (word embeddings), audio (COVAREP), video (FACET)
- Task: binary sentiment classification
- Official train/val/test split
- Small enough for rapid iteration

### Secondary: MM-IMDb (25,959 movies)

**Why**: Image (poster) and text (plot) have asymmetric information — genre is mostly in text, poster is weak signal. Creates a natural uniqueness-heavy distribution, contrasting with MOSI's synergy.

- 2 modalities: image (poster), text (plot summary)
- Task: multi-label genre classification (23 genres)
- Standard split

### Validation: MIMIC (40K ICU patients)

**Why**: Clinical data where modality interactions have real consequences. Different from vision-language.

- 3 modalities: lab values, clinical notes, diagnostic codes
- Task: in-hospital mortality prediction (binary)
- Standard split

---

## Measurement Framework

### Step 1: Train Models

Train three model variants per dataset:

1. **Joint model**: All modalities → MulT fusion → classifier
2. **Unimodal models**: One per modality → classifier (no fusion)

Training details (following I2MoE):

- Optimizer: Adam, lr=1e-4
- Batch size: 32
- Epochs: 50 (early stopping, patience=10)
- 3 random seeds, report mean ± std

### Step 2: LSMI Interaction Estimation

For each sample x = (x₁, x₂, ..., y), compute:

```
conf_joint(x) = softmax probability of correct class from joint model
conf_i(x)     = softmax probability of correct class from unimodal model i
```

Sample-level interaction scores:

```
r(x) = min_i(conf_i(x))                           # Redundancy: both modalities confident
s(x) = conf_joint(x) - max_i(conf_i(x))           # Synergy: joint >> best unimodal
u_i(x) = conf_i(x) - mean_{j≠i}(conf_j(x))       # Uniqueness: one modality >> others
```

Bucket assignment (dominant regime):

- **RED**: r(x) > s(x) and r(x) > max_i(u_i(x))
- **SYN**: s(x) > r(x) and s(x) > max_i(u_i(x))
- **UNQ**: max_i(u_i(x)) > r(x) and max_i(u_i(x)) > s(x)

**Expected on MOSI**: SYN ≈ 20-30%, RED ≈ 30-40%, UNQ ≈ 30-40%
**Expected on IMDb**: UNQ ≈ 50-60%, RED ≈ 20-30%, SYN ≈ 10-20%

### Step 3: Representational Alignment Metrics

At each layer l (encoder output, each cross-attention layer, classifier input), compute:

#### 3a. Calibrated Mutual k-NN Overlap (MkNN)

For N samples with representations {v₁,...,v_N} (modality 1) and {t₁,...,t_N} (modality 2) at layer l:

1. k-NN in modality 1 space: NN_v(i) = k nearest neighbors of v_i
2. k-NN in modality 2 space: NN_t(i) = k nearest neighbors of t_i
3. Overlap: MkNN(i) = |NN_v(i) ∩ NN_t(i)| / k
4. Dataset-level: MkNN = (1/N) Σ_i MkNN(i)

Null calibration (100 permutations):

- Permute modality 2 indices → break pairing → recompute MkNN
- z-score: MkNN_cal = (MkNN - mean_null) / std_null

Hyperparameter: k = 10.

#### 3b. CKA (Centered Kernel Alignment)

RBF kernel with median bandwidth σ:

```
K_V(i,j) = exp(-||v_i - v_j||² / 2σ²)
K_T(i,j) = exp(-||t_i - t_j||² / 2σ²)
```

Center: K̃ = HKH where H = I - (1/N)11^T

```
CKA(V,T) = ||K̃_V ⊙ K̃_T||_F / (||K̃_V||_F · ||K̃_T||_F)
```

Null calibration: same permutation procedure as MkNN.

### Step 4: Optimization Alignment Metrics (THE NOVEL PART)

#### 4a. Per-Sample Gradient Cosine Similarity

For each sample x, compute the loss gradient w.r.t. the shared fusion representation h:

```
g_1(x) = ∂L(x) / ∂h |_{keeping modality 2 fixed}   # gradient from modality 1's contribution
g_2(x) = ∂L(x) / ∂h |_{keeping modality 1 fixed}   # gradient from modality 2's contribution
```

Implementation: Use partial forward passes. Feed one modality while zeroing the other's cross-attention input, compute gradients.

```
GradCos(x) = cos(g_1(x), g_2(x))
```

- GradCos > 0: modalities push the shared representation in the same direction → cooperative optimization
- GradCos < 0: modalities push in opposing directions → optimization conflict
- GradCos ≈ 0: modalities push in orthogonal directions → no cooperation

#### 4b. Fisher Subspace Overlap (CurvAgree) — Per Interaction Bucket

For a bucket B (e.g., all RED samples), compute the empirical Fisher information matrix at layer l:

```
F_1^(l) = (1/|B|) Σ_{x∈B} [∂L/∂θ_1^(l)]·[∂L/∂θ_1^(l)]^T    # Fisher from modality 1
F_2^(l) = (1/|B|) Σ_{x∈B} [∂L/∂θ_2^(l)]·[∂L/∂θ_2^(l)]^T    # Fisher from modality 2
```

where θ_i^(l) are the parameters at layer l relevant to modality i.

**CurvAgree**: Principal-angle overlap between top-k eigenspaces:

1. Compute top-k eigenvectors of F_1: U_1 ∈ R^{d×k}
2. Compute top-k eigenvectors of F_2: U_2 ∈ R^{d×k}
3. SVD of U_1^T · U_2 → singular values σ_1 ≥ ... ≥ σ_k
4. CurvAgree = (1/k) Σ_j σ_j   (average cosine of principal angles)

CurvAgree ∈ [0, 1]:

- CurvAgree ≈ 1: modalities curve the loss surface in similar directions → optimization aligned
- CurvAgree ≈ 0: modalities curve in orthogonal directions → optimization conflicting

Hyperparameter: k = 10 (top-10 Fisher eigenvectors).

#### 4c. Gradient Covariance Rank Overlap

Cheaper proxy for Fisher overlap:

1. Collect per-sample gradients for modality 1: G_1 ∈ R^{|B|×d}
2. Collect per-sample gradients for modality 2: G_2 ∈ R^{|B|×d}
3. PCA on each → top-k principal directions
4. Subspace overlap = same principal-angle calculation as CurvAgree

This is cheaper than Fisher (no eigendecomposition of d×d matrix) and can be done per-sample via batched operations.

### Step 5: The Core Analysis — Decoupling

For each sample, we now have:

- **Interaction type**: RED / SYN / UNQ (from Step 2)
- **Representational alignment**: calibrated MkNN, CKA (from Step 3)
- **Optimization alignment**: GradCos, CurvAgree (from Step 4)
- **Task performance**: correct/incorrect classification (from Step 1)

#### Analysis 5a: 2×2 Contingency Table

Divide samples into quadrants:

|                         | High Rep. Alignment  | Low Rep. Alignment   |
| ----------------------- | -------------------- | -------------------- |
| **High Opt. Alignment** | Q1: Both aligned     | Q2: Opt-only aligned |
| **Low Opt. Alignment**  | Q3: Rep-only aligned | Q4: Neither aligned  |

**Core prediction**:

- Q1 (both high): dominated by RED samples, high accuracy
- Q2 (opt high, rep low): dominated by SYN samples, moderate-high accuracy
- Q3 (rep high, opt low): model failures — these are the "learning failures" where the model sees the right thing but can't learn
- Q4 (both low): dominated by UNQ samples, accuracy depends on unimodal performance

**Key test**: Q3 should contain significantly more incorrect predictions than Q2. This proves representational alignment alone is insufficient.

#### Analysis 5b: Predictive Power Comparison

Logistic regression to predict per-sample correctness:

- Model A: correctness ~ CKA + MkNN (representational alignment only)
- Model B: correctness ~ GradCos + CurvAgree (optimization alignment only)
- Model C: correctness ~ CKA + MkNN + GradCos + CurvAgree (both)

**Core prediction**: Model B > Model A in AUC. Model C ≈ Model B (representational alignment adds nothing once optimization alignment is controlled).

#### Analysis 5c: Interaction Type Conditions the Decoupling

Within each PID bucket:

- **RED**: high rep alignment AND high opt alignment (no decoupling)
- **SYN**: low rep alignment BUT high opt alignment at fusion layers (decoupling!)
- **UNQ**: rep alignment varies BUT opt alignment is consistently low (alignment is noise)

This shows the interaction type predicts WHERE and WHETHER decoupling occurs.

---

## Statistical Testing

### Test 1: Decoupling exists

Spearman rank correlation between rep alignment (CKA) and opt alignment (GradCos) per sample.

- H0: ρ = 1 (perfect correlation, no decoupling)
- H1: ρ < 1 (decoupling exists)
- Expect: ρ ≈ 0.3-0.5 (moderate correlation — they measure different things)

### Test 2: Opt alignment predicts performance better than rep alignment

Compare AUC of logistic regression models A vs B (paired bootstrap, 1000 resamples).

- H0: AUC_A ≥ AUC_B
- H1: AUC_B > AUC_A

### Test 3: Interaction type conditions the decoupling

Per-bucket Spearman correlation:

- ρ_RED should be high (rep and opt alignment agree for redundancy)
- ρ_SYN should be low (rep and opt alignment decouple for synergy)
- ρ_UNQ should be near zero (both are noisy for uniqueness)

Kruskal-Wallis test on per-bucket correlations, with post-hoc pairwise comparisons.

### Test 4: Q3 (rep-only aligned) samples have worse performance

Fisher exact test on the 2×2 table (Q2 vs Q3 accuracy).

---

## Computational Requirements

Models are tiny (~5-10M params) and datasets are small (MOSI: 2,199 samples, IMDb: 25K).

| Component                      | Estimate                                                  |
| ------------------------------ | --------------------------------------------------------- |
| Model training (MOSI, 3 seeds) | ~10 min on A100                                           |
| Model training (IMDb, 3 seeds) | ~30 min on A100                                           |
| LSMI estimation                | < 5 min (reuse trained model predictions)                 |
| Mutual k-NN (k=10) per layer   | ~10 sec per layer (N≈2000 for MOSI)                       |
| CKA per layer                  | ~10 sec per layer                                         |
| Per-sample gradients (GradCos) | ~5 min per dataset (forward+backward per sample)          |
| Fisher eigendecomposition      | ~10 min per layer per bucket (approximate with Lanczos)   |
| 100 null permutations × layers | ~15 min total                                             |
| **Total estimate**             | **~1-2 hours on single A100 (or ~3-4 hours on RTX 3090)** |

---

## Implementation Plan

```
Phase 1: Model Training (Days 1-3)
├── Set up MOSI data loading (use MultiBench or CMU SDK)
├── Implement GRU encoders + MulT fusion
├── Train joint + unimodal models × 3 seeds
└── Checkpoint all intermediate representations

Phase 2: PID Estimation (Day 4)
├── Compute conf_joint, conf_1, conf_2, conf_3 per sample
├── Compute r(x), s(x), u_i(x)
├── Bucket into RED / SYN / UNQ
└── Verify: SYN bucket has ≥ 200 samples (critical check!)

Phase 3: Representational Alignment (Days 5-6)
├── Extract representations at each layer
├── Compute MkNN (k=10) + null calibration
├── Compute CKA (RBF kernel) + null calibration
└── Plot per-bucket alignment trajectories

Phase 4: Optimization Alignment (Days 7-9)
├── Compute per-sample gradient cosine (GradCos)
├── Compute Fisher matrices per bucket per layer
├── Eigendecompose → CurvAgree
└── Gradient covariance rank overlap (cheap proxy)

Phase 5: Core Analysis (Days 10-12)
├── Build 2×2 contingency table
├── Logistic regression comparison (AUC)
├── Per-bucket decoupling analysis
├── Statistical tests (all 4)
└── Generate figures

Decision Gate: If ρ(CKA, GradCos) < 0.7 AND AUC_B > AUC_A → proceed to full paper.
```

---

## Expected Figures

1. **Figure 1** (Hero figure): Scatter plot of rep alignment (x-axis) vs opt alignment (y-axis), colored by interaction type. Shows decoupling and that PID predicts quadrant membership.

2. **Figure 2**: Per-layer alignment trajectories, split by metric type (CKA vs GradCos) and by interaction bucket. Shows that CKA trajectories look similar across regimes but GradCos trajectories diverge.

3. **Figure 3**: 2×2 contingency table with accuracy in each quadrant. Shows Q3 (rep-only) has low accuracy.

4. **Figure 4**: ROC curves for logistic regression models A, B, C. Shows optimization alignment is the better predictor.

5. **Figure 5**: CurvAgree per layer per bucket. Shows RED has high CurvAgree at early layers, SYN has high CurvAgree only at fusion layers.