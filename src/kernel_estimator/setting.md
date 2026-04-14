# Surrogate Attribution for Multimodal Interaction

## 1. The Alignment Principle

In task attribution, three objects target the same quantity:

| Object | Definition | Role |
|--------|-----------|------|
| Ground truth | $I_k^{\text{LOO}} = F(\mathbf{1}) - F(\mathbf{1} - e_k)$ | Defines attribution |
| Influence function | $\nabla_{s_k} F(s^\star)$ | First-order approximation of LOO |
| Surrogate coefficient | $\hat{\beta}_k$ from KRR on $\{(s^{(i)}, F(s^{(i)}))\}$ | Learned approximation of LOO |

All three are grounded in $F(s) = \mathbb{E}[\ell_{\text{test}}(f_{\hat{W}(s)})]$, a function from subset selection to test performance. The surrogate takes $s$ as input and $F(s)$ as output. The influence function is the gradient of the same $F$ at the anchor. The LOO score is a finite difference of the same $F$. This alignment is not incidental — it is the reason the surrogate works.

To transfer this framework to multimodal interaction, we need a function $G(s)$ that:
1. **Defines** multimodal interaction quantities (R, U, S) as contrasts of $G$.
2. **Takes subset selection** $s \in \{0,1\}^K$ as input, where $s$ selects modality components.
3. **Admits a gradient** $\nabla_s G(s^\star)$ that serves as the influence function for interaction.

## 2. The Target: $G(s) = I(X_s; Y)$

### 2.1 Definition

Let $s \in \{0,1\}^K$ select which modality components are active. Define:

$$G(s) = I(X_s; Y)$$

the mutual information between the selected components $X_s = \{X_k : s_k = 1\}$ and the target $Y$.

### 2.2 PID Quantities Are Contrasts of $G$

For two modalities ($s_1, s_2 \in \{0,1\}$):

$$G(1,0) = I(X_1; Y) = R + U_1$$
$$G(0,1) = I(X_2; Y) = R + U_2$$
$$G(1,1) = I(X_1, X_2; Y) = R + U_1 + U_2 + S$$

The LOO scores of $G$ are:

$$G(1,1) - G(0,1) = U_1 + S = I(X_1; Y \mid X_2)$$
$$G(1,1) - G(1,0) = U_2 + S = I(X_2; Y \mid X_1)$$

The discrete interaction term is:

$$G(1,1) - G(1,0) - G(0,1) + G(0,0) = S - R$$

These are not approximations. They are the definitions of PID consistency equations evaluated on $G$. The function $G(s) = I(X_s; Y)$ stands in exact analogy to $F(s) = \mathbb{E}[\ell_{\text{test}}]$ in task attribution: the target itself encodes the interaction structure.

### 2.3 Cross-Entropy Loss as a Realization of $G$

For a model trained with cross-entropy loss:

$$\ell_{\text{CE}}(f_{\hat{W}(s)}, X_s, Y) \xrightarrow{\text{well-calibrated}} H(Y \mid X_s)$$

Since $I(X_s; Y) = H(Y) - H(Y \mid X_s)$ and $H(Y)$ is constant across $s$:

$$G(s) = -\mathbb{E}[\ell_{\text{CE}}(f_{\hat{W}(s)}, X_s, Y)] + H(Y)$$

Up to the constant $H(Y)$, **the negative test loss under subset $s$ is $G(s)$**. This means the task-attribution target $F(s)$ already estimates the interaction target $G(s)$ (with sign flip and constant shift) when the loss is cross-entropy. The surrogate framework does not need a separate estimation pipeline for mutual information — it inherits the target from task attribution directly.

## 3. The Surrogate

### 3.1 Expanded Selection Space

With two whole modalities, $s \in \{0,1\}^2$ gives three non-trivial subsets: insufficient to separate $R$ from $S$ (one equation, two unknowns). The expansion breaks each modality into $K_m$ subcomponents:

$$s = [s^{(1)}_1, \ldots, s^{(1)}_{K_1}, s^{(2)}_1, \ldots, s^{(2)}_{K_2}] \in \{0,1\}^{K_1 + K_2}$$

The surrogate now has enough variation to resolve the interaction structure. Subcomponents can be:
- Feature groups (e.g., frequency bands in audio, spatial regions in images)
- Encoder layer outputs
- Representation dimensions (grouped by variance or Fisher eigenvalue)

### 3.2 Kernel Surrogate on $G(s)$

Fit the kernel surrogate:

$$g_\theta = \arg\min_{g \in \mathcal{H}_k} \sum_{i=1}^m \big(G(s^{(i)}) - g_\theta(s^{(i)})\big)^2 + \lambda \|g_\theta\|^2_{\mathcal{H}_k}$$

with RBF kernel $k(s^{(a)}, s^{(b)}) = \exp(-\gamma \|s^{(a)} - s^{(b)}\|^2)$.

In practice, $G(s^{(i)})$ is computed as the negative cross-entropy test loss of a model evaluated (or retrained) under mask $s^{(i)}$.

### 3.3 Three-Way Alignment (Transferred)

| Object | Multimodal Interaction Version |
|--------|-------------------------------|
| Ground truth | $G(\mathbf{1}) - G(\mathbf{1} - e_k)$: LOO of $I(X_s; Y)$ w.r.t. component $k$ |
| Influence function | $\nabla_{s_k} G(s^\star)$: marginal information contribution of component $k$ |
| Surrogate coefficient | $\hat{\beta}_k$ from KRR on $\{(s^{(i)}, G(s^{(i)}))\}$ |

The surrogate coefficient $\hat{\beta}_k$ estimates the marginal change in mutual information when component $k$ is included. This is the interaction-theoretic analog of the data attribution score.

## 4. From Surrogate Coefficients to PID

### 4.1 Linear Surrogate (First Order)

The linear surrogate $g(s) = \alpha + \beta^\top s$ gives:

$$\hat{\beta}_k \approx \nabla_{s_k} G(\mathbf{1}) = \frac{\partial I(X_s; Y)}{\partial s_k}\bigg|_{s = \mathbf{1}}$$

This is the marginal contribution of component $k$ to total mutual information. Summing within modalities:

$$\sum_{k \in \text{mod}_1} \hat{\beta}_k \approx U_1 + S, \qquad \sum_{k \in \text{mod}_2} \hat{\beta}_k \approx U_2 + S$$

The linear surrogate captures $U + S$ per modality but cannot separate them. This parallels the attribution paper's observation: the linear surrogate equals the influence function and captures only additive effects.

### 4.2 Kernel Surrogate (Higher Order)

The kernel surrogate captures the Hessian and higher-order terms. On $\{0,1\}^K$, any function decomposes via the Möbius (inclusion-exclusion) expansion:

$$g(s) = \sum_{T \subseteq [K]} c_T \prod_{k \in T} s_k$$

The coefficients $c_T$ for $|T| = 1$ are main effects; for $|T| = 2$ they are pairwise interactions; etc.

**Cross-modal pairwise terms**: $c_{\{i,j\}}$ for $i \in \text{mod}_1, j \in \text{mod}_2$ measure the mutual information gain from jointly including components $i$ and $j$ beyond their individual contributions. This is the operational definition of synergy at the subcomponent level:

$$c_{\{i,j\}} = G(s_{ij}^{11}) - G(s_{ij}^{10}) - G(s_{ij}^{01}) + G(s_{ij}^{00})$$

where $s_{ij}^{ab}$ sets $s_i = a, s_j = b$ and holds all others at $\mathbf{1}$.

**Aggregated PID estimates**:

$$\hat{S} = \sum_{\substack{i \in \text{mod}_1 \\ j \in \text{mod}_2}} c_{\{i,j\}} + \text{higher-order cross-modal terms}$$

$$\hat{R} = G(\mathbf{1}_{\text{mod}_1}, \mathbf{0}_{\text{mod}_2}) + G(\mathbf{0}_{\text{mod}_1}, \mathbf{1}_{\text{mod}_2}) - G(\mathbf{1}) + \hat{S}$$

The second equation uses the identity $R = I(X_1;Y) + I(X_2;Y) - I(X_1,X_2;Y) + S$, substituting the surrogate's estimate of $S$.

### 4.3 The Hessian Bias and Synergy

From Proposition 3.1 of the attribution paper, the linear surrogate's bias is:

$$\hat{\beta} \approx \nabla_s G(s^\star) + (p-1) H_s \mathbf{1}_K + \frac{1-2p}{2} \text{diag}[H_s]$$

The Hessian $H_s = \nabla_s^2 G(s^\star)$ encodes pairwise interaction in $G$. Cross-modal entries of $H_s$ correspond to synergy; the bias terms mix synergy into the linear attribution scores. The kernel surrogate resolves this by modeling $H_s$ explicitly through its non-linear capacity. The bias that causes the linear surrogate to fail at separating $R$ from $S$ is the same bias identified in Proposition 3.1.

## 5. Connection to Fisher Spectrum

When modality $Y$ contributes Fisher information $M_Y$ to the joint Fisher $F(\gamma) = I_X + \gamma M_Y$, the spectral concentration theorem gives:

$$\rho_k'(0) = \frac{1}{\text{tr}(I_X)} \big[\tau_R (1 - \rho_k(0)) - \tau_U \rho_k(0)\big]$$

where $\tau_R = \text{tr}(P_k M_Y P_k)$ and $\tau_U = \text{tr}(P_k^\perp M_Y P_k^\perp)$.

If we set the surrogate target to $G(s) = \rho_k(s)$ (spectral concentration under mask $s$), then the surrogate's first-order attribution for component $j$ of modality 2 becomes:

$$\hat{\beta}^{(2)}_j \propto \tau_{R,j}(1 - \rho_k(0)) - \tau_{U,j} \rho_k(0)$$

where $\tau_{R,j}$ and $\tau_{U,j}$ are the redundant and unique Fisher contributions of subcomponent $j$. The sign of $\hat{\beta}^{(2)}_j$ directly classifies the subcomponent:

- $\hat{\beta}^{(2)}_j > 0$: subcomponent $j$ is redundancy-dominated (reinforces top-$k$ eigenspace)
- $\hat{\beta}^{(2)}_j < 0$: subcomponent $j$ is uniqueness-dominated (contributes orthogonal directions)

This provides a per-subcomponent diagnostic that is unavailable from the global PID decomposition.

## 6. Algorithm

**Input**: Multimodal dataset $(X_1, X_2, Y)$; subcomponent partition $(K_1, K_2)$; sample budget $m$.

1. Sample $m$ masks $\{s^{(i)}\}_{i=1}^m$ from $\text{Bernoulli}(p)^{K_1 + K_2}$.
2. For each mask, evaluate $G(s^{(i)}) = -\ell_{\text{CE}}(f_{\hat{W}(s^{(i)})}, X_{s^{(i)}}, Y_{\text{test}})$.
3. Fit RBF kernel surrogate $g_\theta$ via kernel ridge regression on $\{(s^{(i)}, G(s^{(i)}))\}$.
4. Extract Möbius coefficients $\{c_T\}$ from the fitted surrogate.
5. Aggregate:
   - $\hat{S}$: sum of cross-modal interaction coefficients.
   - $\hat{U}_1, \hat{U}_2$: within-modality main effects minus redundant overlap.
   - $\hat{R}$: from consistency equation $R = I(X_1;Y) + I(X_2;Y) - I(X_1,X_2;Y) + S$.
6. (Optional) Repeat with target $G(s) = \rho_k(s)$ for spectral alignment diagnostics.

## 7. Summary of the Analogy

| Task Attribution | Multimodal Interaction Attribution |
|-----------------|-----------------------------------|
| $s \in \{0,1\}^K$ selects training tasks | $s \in \{0,1\}^{K_1+K_2}$ selects modality components |
| $F(s) = \mathbb{E}[\ell_{\text{test}}(f_{\hat{W}(s)})]$ | $G(s) = I(X_s; Y) \approx -\mathbb{E}[\ell_{\text{CE}}(f_{\hat{W}(s)})]$ + const |
| LOO $= F(\mathbf{1}) - F(\mathbf{1}-e_k)$ defines attribution | LOO $= G(\mathbf{1}) - G(\mathbf{1}-e_k)$ defines conditional MI contribution |
| $\nabla_s F$ = influence function | $\nabla_s G$ = marginal information contribution |
| Linear surrogate $\hat{\beta} \approx \nabla_s F$ | Linear surrogate $\hat{\beta} \approx \nabla_s G$, captures $U + S$ but not $R$ vs $S$ |
| Hessian bias from task interactions | Hessian bias from synergy |
| Kernel surrogate resolves Hessian | Kernel surrogate separates $R$ from $S$ via cross-modal interaction terms |

## 8. Open Questions

1. **Calibration requirement**: $G(s) \approx -\ell_{\text{CE}} + H(Y)$ holds for well-calibrated models. Miscalibration introduces a gap between the loss-based surrogate and the true MI target. How sensitive is PID extraction to this gap?

2. **Subcomponent granularity**: Finer partitions give more separation power but require more mask samples ($m \gg K$). What is the minimum $K$ needed to reliably separate $R$ from $S$?

3. **Consistency with existing PID**: On synthetic datasets where CVX gives ground-truth PID, does the surrogate-derived $(\hat{R}, \hat{U}_1, \hat{U}_2, \hat{S})$ converge to the same values?

4. **Beyond pairwise synergy**: The Möbius expansion gives higher-order interaction terms. Do these have a PID interpretation for $>2$ modalities, or do they require the partial information lattice?