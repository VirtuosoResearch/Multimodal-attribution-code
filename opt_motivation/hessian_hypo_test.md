# Experimental Protocol: Fisher/Hessian Analysis for Multimodal Interaction

## 1. Goal

We study multimodal interaction from two complementary perspectives:

\[
\text{interaction}
=
\underbrace{\text{statistical / geometric interaction}}_{\text{Fisher, spectrum, subspace}}
+
\underbrace{\text{optimization interaction}}_{\text{gradient, Hessian, local transfer}}
\]

The protocol aims to answer four questions:

1. Does auxiliary modality \(Y\) provide new information directions for target modality \(X\)?
2. Are these directions redundant or complementary?
3. During optimization, does \(Y\) help or hurt \(X\)?
4. Which Fisher/Hessian quantities best predict positive transfer, negative transfer, and robustness?

---

## 2. Model Setup

We partition parameters as

\[
\theta = (\phi, \psi_X, \psi_Y),
\]

where:

- \(\phi\): shared trunk / shared head
- \(\psi_X\): \(X\)-specific parameters
- \(\psi_Y\): \(Y\)-specific parameters

Primary analysis is performed on the **shared block** \(\phi\), since multimodal interaction is expected to occur mainly through shared parameters.

We also perform a **representation-level** analysis using a shared representation \(r\):

\[
r_X = h(f_X(x)), \qquad r_Y = h(f_Y(y)).
\]

This reduces sensitivity to parameterization and often gives cleaner geometric interpretation.

---

## 3. Data Conditions

We use four categories of data settings.

### 3.1 Related auxiliary modality
Use semantically related modalities, such as:

- image + text
- audio + image + text

These settings should reveal genuine multimodal complementarity.

### 3.2 Unrelated auxiliary modality control
Use semantically unrelated auxiliary modality as a negative control.

Example:
- target image classification on dataset A
- auxiliary text sampled from unrelated dataset B

Purpose:
- test whether Fisher/Hessian metrics distinguish useful vs irrelevant auxiliary information

### 3.3 Fixed-budget control
Keep total sample budget fixed and vary allocation between \(X\) and \(Y\).

Example:
- \(N\) total samples
- compare \(N\) samples of \(X\) vs \(N/2\) samples of \(X\) + \(N/2\) samples of \(Y\)

Purpose:
- separate “more data” from “better complementary data”

### 3.4 Synthetic Gaussian control
Use a linear-Gaussian synthetic setup with controllable shared/private factors.

Purpose:
- verify whether the proposed Fisher/Hessian metrics recover the known latent geometry
- validate blind-spot coverage and fixed-budget advantage in a controlled setting

---

## 4. Experimental Conditions

For each dataset, run the following training conditions:

1. **Unimodal-X**: only target modality \(X\)
2. **Unimodal-Y**: only auxiliary modality \(Y\)
3. **Joint-XY**: real auxiliary modality
4. **Joint-XY-shuffled**: preserve low-level statistics but break semantics
5. **Joint-XY-unrelated**: unrelated auxiliary modality
6. **Joint-XY-fixed-budget**: total sample budget fixed
7. **Joint-XY-vary-ratio**: vary batch ratio \(r \in \{0.25, 0.5, 1, 2, 4\}\)
8. **Frozen auxiliary encoder**
9. **Unfrozen auxiliary encoder**

---

## 5. Task Settings

At minimum, evaluate three regimes.

### 5.1 Few-shot linear probing
Best for studying low-data transfer and semantic complementarity.

### 5.2 Full fine-tuning
Best for studying optimization interaction and curvature effects.

### 5.3 Self-supervised / reconstruction
Best for matching Fisher-style shared-latent theory.

---

## 6. Checkpoint and Sampling Protocol

### 6.1 Random seeds
Use at least **5 seeds** per condition.

### 6.2 Checkpoints
Save model checkpoints at:

- initialization
- 10% training
- 25% training
- 50% training
- 75% training
- 100% training
- best validation checkpoint

### 6.3 Measurement batches
At each checkpoint, independently sample \(B\) mini-batches for each modality.

Recommended:

- small model: \(B = 200\)
- large model: \(B = 50\) to \(100\)

All Fisher / gradient / Hessian quantities should be measured on the **same checkpoint set**.

---

## 7. Parameter-Space Measurements

For modality \(m \in \{X,Y\}\), define the batch loss \(L_m^{(b)}\) and the shared-block gradient

\[
g_m^{(b)} := \nabla_\phi L_m^{(b)}.
\]

Define average gradient

\[
\bar g_m := \frac{1}{B} \sum_{b=1}^B g_m^{(b)}.
\]

---

## 8. Empirical Fisher

Define empirical Fisher on the shared block:

\[
\widehat F_m
=
\frac{1}{B} \sum_{b=1}^B g_m^{(b)} g_m^{(b)\top}.
\]

Joint Fisher:

\[
\widehat F_{X+Y} = \widehat F_X + \widehat F_Y.
\]

### 8.1 Implementation
For each batch:

1. compute loss \(L_m^{(b)}\)
2. backpropagate
3. extract gradient on shared parameters \(\phi\)
4. flatten to vector
5. accumulate outer product

### 8.2 Large-model approximation
For large models, do not explicitly store full Fisher matrix.

Use one of:

- top-\(k\) eigenspectrum via randomized SVD
- low-rank sketch
- Fisher-vector product oracle

---

## 9. Fisher Geometry Metrics

### 9.1 Total information
\[
T_m := \operatorname{tr}(\widehat F_m)
\]

Interpretation:
- overall curvature contributed by modality \(m\)

### 9.2 Maximum eigenvalue
\[
\lambda_{\max}(\widehat F_m)
\]

Interpretation:
- whether information is dominated by a few sharp directions

### 9.3 Condition number
\[
\kappa(\widehat F_m)
=
\frac{\lambda_{\max}(\widehat F_m)}{\lambda_{\min}(\widehat F_m)+\varepsilon}
\]

Interpretation:
- conditioning of shared space

### 9.4 Log-determinant gain
\[
\Delta_{\log\det}(Y \to X)
=
\log\det(\widehat F_X + \widehat F_Y + \varepsilon I)
-
\log\det(\widehat F_X + \varepsilon I)
\]

Interpretation:
- global information gain from adding \(Y\)

### 9.5 Weakest-direction lift
\[
\Delta_{\min}(Y \to X)
=
\lambda_{\min}(\widehat F_X + \widehat F_Y)
-
\lambda_{\min}(\widehat F_X)
\]

Interpretation:
- whether \(Y\) improves the weakest / blind-spot directions of \(X\)

---

## 10. Redundancy vs Complementarity

Perform eigendecomposition of \(\widehat F_X\):

\[
\widehat F_X = U_X \Lambda_X U_X^\top.
\]

Let \(P_X = U_{X,r} U_{X,r}^\top\) be the projector onto the effective subspace of \(X\).

### 10.1 Choice of \(r\)
Use either:

- the smallest \(r\) explaining 90% of Fisher energy
- or fixed \(r \in \{20, 50, 100\}\) for robustness analysis

### 10.2 Redundancy
\[
R_{Y \to X} := \operatorname{tr}(P_X \widehat F_Y)
\]

Interpretation:
- amount of \(Y\)-information that lies in the already-used subspace of \(X\)

### 10.3 Complementarity
\[
C_{Y \to X} := \operatorname{tr}((I - P_X)\widehat F_Y)
\]

Interpretation:
- amount of \(Y\)-information that fills the blind spots of \(X\)

### 10.4 Normalized versions
\[
\widetilde R_{Y \to X}
=
\frac{R_{Y \to X}}{\operatorname{tr}(\widehat F_Y)},
\qquad
\widetilde C_{Y \to X}
=
\frac{C_{Y \to X}}{\operatorname{tr}(\widehat F_Y)}.
\]

Interpretation:
- redundancy/complementarity as fractions of total \(Y\)-information

---

## 11. Directional Fisher Gain

### 11.1 Gain on principal directions of \(X\)
For top \(k\) eigenvectors \(u_i\) of \(\widehat F_X\), compute

\[
d_i^{(Y \to X)} := u_i^\top \widehat F_Y u_i.
\]

Interpretation:
- how much \(Y\) strengthens the dominant directions already present in \(X\)

### 11.2 Gain on weak directions of \(X\)
For bottom \(k\) eigenvectors \(w_i\) of \(\widehat F_X\), compute

\[
b_i^{(Y \to X)} := w_i^\top \widehat F_Y w_i.
\]

Interpretation:
- how much \(Y\) fills blind spots of \(X\)

### 11.3 Gain on new joint directions
Compute eigenspectrum of \(\widehat F_X + \widehat F_Y\) and compare it with \(\widehat F_X\).

Interpretation:
- whether new high-information directions emerge only after adding \(Y\)

---

## 12. Gradient Interaction Metrics

### 12.1 Mean cosine similarity
\[
\cos(g_X, g_Y)
=
\frac{\bar g_X^\top \bar g_Y}{\|\bar g_X\| \, \|\bar g_Y\|}
\]

Interpretation:
- first-order alignment between target and auxiliary gradients

### 12.2 Batch-level conflict rate
For independently sampled batches \(b_x, b_y\), compute

\[
s_{b_x,b_y}
=
\operatorname{sign}\left((g_X^{(b_x)})^\top g_Y^{(b_y)}\right).
\]

Estimate conflict probability:

\[
p_{\text{conflict}}
=
\Pr\left[g_X^\top g_Y < 0\right].
\]

Interpretation:
- frequency of gradient interference

### 12.3 Projected auxiliary gradient
Project \(\bar g_Y\) onto the target subspace and its orthogonal complement:

\[
\bar g_Y^\parallel = P_X \bar g_Y,
\qquad
\bar g_Y^\perp = (I - P_X)\bar g_Y.
\]

Measure:

\[
\cos(\bar g_X, \bar g_Y^\parallel),
\qquad
\|\bar g_Y^\perp\|.
\]

Interpretation:
- whether \(Y\) mainly helps in \(X\)-relevant directions or pushes off-subspace

---

## 13. Hessian Measurements

Define the Hessian of the target loss on the shared block:

\[
H_X := \nabla_{\phi\phi}^2 L_X.
\]

### 13.1 Curvature of \(Y\)-direction under \(X\)-loss
\[
q_{YY|X} := \bar g_Y^\top H_X \bar g_Y
\]

Interpretation:
- how much curvature penalty \(X\) assigns to updating along \(Y\)

### 13.2 Cross-curvature term
\[
q_{YX|X} := \bar g_Y^\top H_X \bar g_X
\]

Interpretation:
- how \(Y\)-direction interacts with the local curvature around the \(X\)-descent direction

### 13.3 Maximum eigenvalue of Hessian
\[
\lambda_{\max}(H_X)
\]

Interpretation:
- sharpness of target loss landscape

### 13.4 Hessian trace
Approximate via Hutchinson:

\[
\operatorname{tr}(H_X)
\approx
\frac{1}{M} \sum_{i=1}^M z_i^\top H_X z_i,
\qquad z_i \sim \{\pm 1\}^d.
\]

Interpretation:
- average curvature level

### 13.5 Practical computation
Use Hessian-vector products (HVPs) via Pearlmutter trick / double backward.

Do **not** form full Hessian explicitly.

---

## 14. Local Transfer Score

Consider joint update

\[
\delta = -\eta(\bar g_X + \beta \bar g_Y).
\]

Define local auxiliary benefit on target modality \(X\):

\[
B_{Y \to X}^{\text{opt}}(\beta)
\approx
\eta\beta \, \bar g_X^\top \bar g_Y
-
\eta^2 \beta \, \bar g_Y^\top H_X \bar g_X
-
\frac{\eta^2 \beta^2}{2} \bar g_Y^\top H_X \bar g_Y.
\]

Interpretation:

- first term: first-order alignment benefit
- second term: cross-curvature interaction
- third term: self-curvature penalty of \(Y\)-update under \(X\)

Decision rule:

- \(B_{Y \to X}^{\text{opt}} > 0\): local positive transfer
- \(B_{Y \to X}^{\text{opt}} < 0\): local negative transfer

This is the main optimization-level metric.

---

## 15. Fisher-Hessian Consistency

When NLL assumptions hold near optimum, Fisher and Hessian may align.

### 15.1 Subspace overlap
Let \(P_F\) and \(P_H\) denote projectors onto top-\(r\) eigenspaces of \(F_X\) and \(H_X\).

Measure:

\[
\text{Overlap}(F_X, H_X)
=
\frac{1}{r} \operatorname{tr}(P_F P_H).
\]

### 15.2 Quadratic-form correlation
Sample random vectors \(v_i\) and compare

\[
v_i^\top F_X v_i
\quad \text{vs} \quad
v_i^\top H_X v_i.
\]

Report:

- Pearson correlation
- Spearman correlation

Interpretation:
- whether Fisher can serve as a good Hessian proxy in the current regime

---

## 16. Representation-Level Version

Let \(r\) be the shared representation.

For modality \(m\), define

\[
u_m := \nabla_r \ell_m,
\qquad
K_m := \nabla_r^2 \ell_m,
\qquad
G_m := \mathbb E[u_m u_m^\top].
\]

Then repeat the full protocol in representation space:

- use \(G_m\) instead of \(F_m\)
- use \(K_X\) instead of \(H_X\)
- define representation-level redundancy/complementarity
- define representation-level local transfer score

Advantages:

- lower dimensional
- easier geometric interpretation
- closer to decision boundary and margin analysis

---

## 17. Downstream Performance Metrics

For each condition, report:

- top-1 accuracy / F1 / AUROC
- few-shot performance curve
- fixed-budget performance curve
- distribution-shift accuracy
- calibration (ECE)
- class margin statistics

### 17.1 Class margin
For classifier weights \(W\) and representation \(r\), define

\[
m(x) = w_y^\top r - \max_{c \neq y} w_c^\top r.
\]

Report:

- mean margin
- 5th percentile margin
- per-class margin variance

Interpretation:
- whether multimodal interaction widens inter-class separation

---

## 18. Statistical Analysis

### 18.1 Checkpoint-level regression
Regress downstream improvement on measured interaction metrics:

\[
\Delta \text{Acc}_{Y \to X}
\sim
\Delta_{\log\det}
+
\widetilde C_{Y \to X}
+
\cos(g_X, g_Y)
+
q_{YY|X}
+
B_{Y \to X}^{\text{opt}}.
\]

### 18.2 Main comparisons
Compare:

- related vs unrelated auxiliary modality
- fixed-budget vs extra-data
- frozen vs unfrozen auxiliary encoder
- different batch ratios
- aligned vs unaligned encoders

### 18.3 Uncertainty
Use seed-level bootstrap and report 95% confidence intervals.

---

## 19. Practical Recommendations

### 19.1 Small models
Measure full Fisher/Hessian on:

- shared head
- last shared transformer block

### 19.2 Large models
Restrict analysis to:

- shared head
- final shared block
- CLS / pooled projector
- shared representation \(r\)

Use:

- low-rank Fisher sketch
- HVP-based Hessian estimation

Do not attempt full dense Hessian over the entire backbone.

---

## 20. Minimal Publishable Version

If a smaller but still solid study is needed, use the following subset.

### Data
- one image-text benchmark
- one audio-image-text benchmark
- one synthetic Gaussian benchmark

### Conditions
- unimodal
- related auxiliary modality
- unrelated auxiliary modality
- fixed-budget
- frozen vs unfrozen

### Tasks
- few-shot linear probing
- full fine-tuning

### Metrics
- \(\widehat F_X, \widehat F_Y\)
- \(\Delta_{\log\det}(Y \to X)\)
- \(\widetilde C_{Y \to X}\)
- \(\cos(g_X, g_Y)\)
- \(q_{YY|X}, q_{YX|X}\)
- \(B_{Y \to X}^{\text{opt}}\)
- accuracy / robustness / margin

---

## 21. Final Definitions for the Paper

### Statistical interaction
\[
\mathcal I^{\text{stat}}_{Y \to X}
:=
\left(
\Delta_{\log\det}(Y \to X),
\widetilde R_{Y \to X},
\widetilde C_{Y \to X}
\right).
\]

### Optimization interaction
\[
\mathcal I^{\text{opt}}_{Y \to X}
:=
\left(
\cos(g_X, g_Y),
q_{YY|X},
q_{YX|X},
B_{Y \to X}^{\text{opt}}
\right).
\]

### Interpretation
- **Positive transfer**: high statistical interaction and \(B_{Y \to X}^{\text{opt}} > 0\)
- **Negative transfer**: nontrivial statistical interaction but \(B_{Y \to X}^{\text{opt}} < 0\)
- **Irrelevant auxiliary modality**: \(\widetilde C_{Y \to X} \approx 0\) and no downstream improvement

---

## 22. Summary

This protocol decomposes multimodal interaction into:

\[
\boxed{
\text{multimodal interaction}
=
\underbrace{\text{Fisher complementarity / redundancy}}_{\text{statistical layer}}
+
\underbrace{\text{Hessian-aware transfer effect}}_{\text{optimization layer}}
}
\]

In practice:

- use **Fisher** to measure what new directions the auxiliary modality contributes
- use **Hessian** to measure whether these directions actually help optimization
- use **downstream performance and margin** to validate whether the interaction metrics explain positive transfer, negative transfer, and robustness