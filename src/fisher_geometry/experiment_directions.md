# Fisher Geometry — Next Experiment Directions

Based on results from the initial experiments (3 models × 7 datasets × 3 seeds, main + perturbation sweep). See `results/experiment_summary.md` for full data.

## Key Problems from Current Results

1. **Cross-attention Fisher is degenerate**: subspace overlap ≈ 0, residual energy = 1.0, cosine ≈ 0 across all datasets. The bidirectional cross-attention architecture creates orthogonal parameter subspaces per modality stream, making full-model Fisher metrics uninformative.
2. **Top-k spectrum mass does not reliably distinguish related vs unrelated modalities**: C_related and C_unrelated show similar values for most datasets.
3. **Concat/shared MLP show partial signal**: subspace overlap and cosine similarity separate related/unrelated on oxford_pets and imdb, but not consistently across all datasets.

---

## Direction 1: Interaction Fisher (F_XY − F_X − F_Y)

**Idea**: Current metrics compare F_X, F_Y, F_XY independently. Define the *interaction Fisher* Δ = F_XY − F_X − F_Y, which isolates curvature that exists only when both modalities are jointly present.

**Metrics on Δ**:
- trace(Δ): total interaction curvature
- rank(Δ): dimensionality of interaction subspace
- top-k eigenvalues of Δ: spectral structure of interaction
- sign structure: positive eigenvalues = synergistic directions, negative = redundant

**Hypothesis**: trace(Δ) > 0 and rank(Δ) > 0 for related modalities; trace(Δ) ≈ 0 for unrelated.

**Implementation**: Already have F_X, F_Y, F_XY eigendecompositions. Need to compute Δ = G_XY^T G_XY / n - G_X^T G_X / n - G_Y^T G_Y / n and eigendecompose it. Can be done in gradient space without forming the full d×d matrix.

**Effort**: Low — extend `fisher_metrics.py` with interaction computation.

---

## Direction 2: Per-Layer Fisher Decomposition

**Idea**: The cross-attention degeneracy arises because X and Y gradients hit disjoint parameter sets (separate projection, attention, FFN per stream). But the classifier head, LayerNorm, and null_embed parameters are shared. Computing Fisher per parameter group should reveal interaction even for cross-attention.

**Parameter groups**:
- Cross-attention: proj_x, proj_y, cross_attn_x, cross_attn_y, ffn_x, ffn_y, classifier, pool_norm, cls tokens, null_embed
- Concat MLP: fusion_in (split into X-half and Y-half of weight matrix), fusion_hidden, classifier
- Shared MLP: proj_x, proj_y, shared_mlp, classifier

**Key prediction**: The classifier-layer Fisher should show interaction signal for all architectures, since classification parameters are always shared across modalities.

**Effort**: Medium — need to modify `per_example_gradients` to return per-group gradient vectors.

---

## Direction 3: Hessian Spectral Analysis via SLQ

**Idea**: The empirical Fisher E[g g^T] is a rank-deficient outer product approximation. The true Hessian ∇²L captures full curvature including cross-modal interaction terms in the loss landscape. Use Stochastic Lanczos Quadrature (`src/slq/`) to estimate Hessian spectral properties without forming the full matrix.

**Metrics**:
- Hessian trace (total curvature)
- Hessian log-determinant
- Hessian spectral density (via SLQ Lanczos vectors)
- Hessian effective rank

**Comparison**: H_XY vs H_X vs H_Y, and the interaction Hessian H_XY − H_X − H_Y.

**Why it might work better than Fisher**: The Hessian captures second-order cross-terms ∂²L/∂θ_x∂θ_y that the outer-product Fisher misses entirely. For cross-attention, the Hessian through the attention mechanism creates coupling between X and Y parameter blocks even though the first-order gradients are orthogonal.

**Effort**: Medium-High — need to wire SLQ to the fusion models with Hessian-vector products via `torch.autograd.functional.hvp`.

---

## Direction 4: Eigenspectrum Shape Metrics

**Idea**: Top-k mass is too coarse — it only captures one number. Richer spectral shape metrics may better distinguish modal interaction.

**Metrics**:
- **Effective rank** = exp(H(λ̃)) where λ̃ = λ/Σλ and H is Shannon entropy
- **Participation ratio** = (Σλ)² / Σλ² — measures how many eigenvalues contribute
- **Spectral decay exponent** — fit λ_k ∝ k^{-α}, report α
- **Spectral gap ratio** — λ₁/λ₂ or λ_k/λ_{k+1}

**Hypothesis**: Related modalities concentrate Fisher information into fewer directions (lower effective rank, steeper spectral decay) because the model learns a shared low-dimensional structure. Unrelated modalities spread information more uniformly.

**Effort**: Low — just new analysis functions on existing eigendecomposition data.

---

## Direction 5: Fisher ↔ PID Correspondence

**Idea**: The repo has PID estimators (`src/pid_estimator/`). Run PID decomposition (redundancy, synergy, unique info) on the same datasets and test whether Fisher geometry metrics correlate with PID quantities.

**Hypothesized correspondences**:
- Fisher subspace overlap ↔ redundancy (shared information = shared gradient directions)
- Interaction Fisher trace ↔ synergy (curvature that only exists jointly)
- Residual energy ↔ unique information (Y-gradient energy outside X's eigenspace)

**Experiment**: Compute PID on validation features for each dataset. Scatter-plot Fisher metrics vs PID components across datasets. Report rank correlation.

**Why this matters**: If Fisher geometry and PID agree, it provides a geometric interpretation of information-theoretic decomposition — and Fisher is cheaper to compute than PID for large models.

**Effort**: Medium — need to run PID estimators on the same datasets and build comparison analysis.

---

## Direction 6: Per-Sample Gradient Analysis

**Idea**: Instead of aggregate Fisher (averaged over samples), analyze individual per-sample gradients to find *which samples* benefit from multimodal fusion.

**Analysis**:
- Compute per-sample cosine similarity cos(g_x^{(i)}, g_y^{(i)}) for each sample i
- Cluster samples by gradient direction (k-means on normalized gradients)
- Correlate per-sample gradient alignment with per-sample accuracy gain from adding Y

**Hypothesis**: Samples where both modalities are informative will have aligned per-sample gradients (high cosine). Samples where Y is noise will have orthogonal gradients.

**Connection to attribution**: This provides a sample-level "modal interaction score" derived from Fisher geometry, connecting to the attribution goal of the project.

**Effort**: Low-Medium — per-sample gradients are already computed, need analysis code.

---

## Direction 7: Gradient Alignment Dynamics During Training

**Idea**: Track cosine similarity between ∇L_X and ∇L_Y at every training step (not just final checkpoint). The trajectory of alignment during training may distinguish related vs unrelated modalities.

**Prediction**:
- Related: alignment increases during training as model learns to fuse
- Unrelated: alignment stays near zero or shows destructive interference
- The *rate* of alignment change may be more discriminative than the final value

**Effort**: Low — modify training loop to log cos(g_x, g_y) at each step.

---

## Direction 8: Shared-Parameter Fisher for Cross-Attention

**Idea**: Fix the cross-attention degeneracy by computing Fisher only on shared parameters (classifier, pool_norm, null_embed, cls tokens). This small parameter subset sees gradients from both modalities and should show interaction.

**Why it's interesting**: If shared-parameter Fisher works for cross-attention, it suggests a general principle — modal interaction in modular architectures is concentrated in the *interface* parameters, not the per-modality encoders.

**Effort**: Low — filter parameters in `per_example_gradients`.

---

## Suggested Priority Order

1. **Direction 1 (Interaction Fisher)** + **Direction 4 (Spectral shape)** — quick wins, extend existing code
2. **Direction 2 (Per-layer)** + **Direction 8 (Shared-param Fisher)** — fix cross-attention, moderate effort
3. **Direction 3 (Hessian via SLQ)** — the "big gun" if Fisher remains insufficient
4. **Direction 5 (Fisher ↔ PID)** — highest paper impact if correspondence holds
5. **Direction 6 (Per-sample)** + **Direction 7 (Training dynamics)** — deeper analysis, good for understanding
