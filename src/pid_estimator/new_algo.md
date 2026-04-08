# End-to-End Surrogate-Based PID-Like Proxy Estimation

## 1. Goal

We aim to estimate multimodal interaction quantities over many subset-weighted empirical distributions without rerunning a full sample-wise interaction estimator such as LSMI on every subset.

Let
\[
T^\star(P)=\bigl(R^\star(P),U_1^\star(P),U_2^\star(P),S^\star(P)\bigr)
\]
denote the distribution-level PID target, where redundancy, uniqueness, and synergy correspond to shared, modality-specific, and jointly emergent task-relevant information, respectively. In practice, exact PID estimation is difficult in high-dimensional or continuous settings. We therefore construct a **data-driven PID-like proxy** that preserves the same semantic roles while being much cheaper to evaluate repeatedly. :contentReference[oaicite:3]{index=3}

Our central idea is:

1. learn a **sample-wise interaction head**
   \[
   t_\phi(z)=\bigl(r_\phi(z),u_{1,\phi}(z),u_{2,\phi}(z),s_\phi(z)\bigr),
   \]
   where \(z=(x_1,x_2,y)\);
2. aggregate these sample-wise quantities over a subset-weighted empirical distribution;
3. optionally fit a surrogate model over the subset space to further compress the response map.

This yields a **PID-like interaction query engine**:
\[
s \longmapsto F_{c,\phi}(s),
\qquad c\in\{R,U_1,U_2,S\},
\]
where \(s\) indexes a data subset or block-weight vector.

---

## 2. Why This Is Related to PID

PID decomposes task-relevant multimodal information into four components:
- redundancy: information shared across modalities,
- uniqueness in modality 1,
- uniqueness in modality 2,
- synergy: information that emerges only jointly. :contentReference[oaicite:4]{index=4}

At the sample level, a pointwise decomposition of multimodal information can be written as
\[
i(x_1;y)=r+u_1,\qquad
i(x_2;y)=r+u_2,\qquad
i(x_1,x_2;y)=r+u_1+u_2+s.
\]
This gives an operational basis for sample-wise interaction quantification. Our method does not claim to compute the exact distribution-level PID on every subset. Instead, it learns a **PID-like proxy** with the same semantic roles:
- \(r_\phi\): shared task-relevant information,
- \(u_{1,\phi},u_{2,\phi}\): modality-specific information,
- \(s_\phi\): jointly emergent information. :contentReference[oaicite:5]{index=5}

Thus, the method should be viewed as a **learned operational proxy** aligned with PID semantics, rather than an exact PID solver.

---

## 3. Data Partition and Subset-Weighted Empirical Distributions

Let the dataset be
\[
D=\{z_i\}_{i=1}^n,\qquad z_i=(x_{1i},x_{2i},y_i).
\]
Partition the dataset into \(K\) disjoint blocks
\[
B_1,\dots,B_K,
\qquad
|B_k|=n_k,
\qquad
\sum_{k=1}^K n_k=n.
\]

Define the block empirical distributions
\[
\hat P_k=\frac{1}{n_k}\sum_{i\in B_k}\delta_{z_i},
\qquad
\pi_k=\frac{n_k}{n}.
\]

For a subset-weight vector
\[
s=(s_1,\dots,s_K)\in[0,\infty)^K,
\]
define the subset-weighted empirical distribution
\[
\hat P_s
=
\frac{\sum_{k=1}^K \pi_k s_k \hat P_k}
{\sum_{k=1}^K \pi_k s_k}.
\]

Special cases:
- \(s_k=1\): keep block \(k\),
- \(s_k=0\): remove block \(k\),
- \(s=\mathbf 1\): use the full empirical distribution.

---

## 4. End-to-End Interaction Head

### 4.1 Predictive information surrogates

Train three predictive models only once:
\[
q_1(y\mid x_1),\qquad
q_2(y\mid x_2),\qquad
q_{12}(y\mid x_1,x_2).
\]

For each sample \(z=(x_1,x_2,y)\), define pointwise predictive-information surrogates
\[
m_1(z)=\log q_1(y\mid x_1)-\log \hat p(y),
\]
\[
m_2(z)=\log q_2(y\mid x_2)-\log \hat p(y),
\]
\[
m_{12}(z)=\log q_{12}(y\mid x_1,x_2)-\log \hat p(y),
\]
where \(\hat p(y)\) is the empirical label prior.

Interpretation:
- \(m_1\): information provided by modality \(X_1\),
- \(m_2\): information provided by modality \(X_2\),
- \(m_{12}\): joint multimodal information.

These are predictive surrogates of pointwise mutual-information-like quantities.

### 4.2 Learned redundancy head

Instead of recomputing the original LSMI decomposition for every subset, we learn a redundancy head
\[
r_\phi(z)=\rho_\phi\bigl(m_1(z),m_2(z),m_{12}(z),h(z)\bigr),
\]
where \(h(z)\) denotes optional backbone features.

Then define the remaining interaction components algebraically:
\[
u_{1,\phi}(z)=m_1(z)-r_\phi(z),
\]
\[
u_{2,\phi}(z)=m_2(z)-r_\phi(z),
\]
\[
s_\phi(z)=m_{12}(z)-m_1(z)-m_2(z)+r_\phi(z).
\]

Thus the interaction head is
\[
t_\phi(z)=\bigl(r_\phi(z),u_{1,\phi}(z),u_{2,\phi}(z),s_\phi(z)\bigr).
\]

This preserves the same algebraic role decomposition as sample-wise PID-like interaction measures, while avoiding repeated entropy-style decomposition on each subset. :contentReference[oaicite:6]{index=6}

---

## 5. Subset-Level PID-Like Proxy

For each component \(c\in\{R,U_1,U_2,S\}\), define the subset-level proxy by averaging the corresponding sample-wise interaction score over \(\hat P_s\):
\[
F_{c,\phi}(s)
=
\mathbb E_{\hat P_s}[t_{\phi,c}(Z)].
\]

Since \(\hat P_s\) is an empirical weighted distribution, this becomes
\[
F_{c,\phi}(s)
=
\frac{\sum_{k=1}^K s_k \sum_{i\in B_k} t_{\phi,c}(z_i)}
{\sum_{k=1}^K n_k s_k}.
\]

This is the key computational simplification:

- once \(t_\phi(z_i)\) has been computed for all samples,
- any subset-level interaction quantity is obtained by **one weighted aggregation**,
- no subset-specific rerun of LSMI is needed.

Interpretation:
- \(F_{R,\phi}(s)\): subset-level shared-information proxy,
- \(F_{U_1,\phi}(s)\): subset-level modality-1-specific proxy,
- \(F_{U_2,\phi}(s)\): subset-level modality-2-specific proxy,
- \(F_{S,\phi}(s)\): subset-level joint-emergent proxy.

---

## 6. Training Objectives

We train the end-to-end interaction head using three types of objectives.

### 6.1 Predictive objective

Train the three predictive models with standard supervised losses:
\[
\mathcal L_{\mathrm{pred}}
=
\mathcal L_1+\mathcal L_2+\mathcal L_{12},
\]
where
\[
\mathcal L_1=-\sum_i \log q_1(y_i\mid x_{1i}),
\qquad
\mathcal L_2=-\sum_i \log q_2(y_i\mid x_{2i}),
\]
\[
\mathcal L_{12}=-\sum_i \log q_{12}(y_i\mid x_{1i},x_{2i}).
\]

### 6.2 Semantic regularization

To keep the learned decomposition PID-like, add semantic constraints such as:
- redundancy should not exceed single-modality information,
- synergy should capture the non-additive remainder,
- uniqueness should be modality-specific residual information.

A simple regularizer is
\[
\mathcal L_{\mathrm{sem}}
=
\sum_i
\Big[
\operatorname{ReLU}(r_\phi(z_i)-m_1(z_i))^2
+
\operatorname{ReLU}(r_\phi(z_i)-m_2(z_i))^2
\Big].
\]

Additional optional constraints may be added to stabilize the signs or magnitudes of \(u_{1,\phi},u_{2,\phi},s_\phi\).

### 6.3 Calibration loss

Select a small calibration set of subsets
\[
\mathcal S_{\mathrm{cal}}=\{s^{(1)},\dots,s^{(M)}\}.
\]
For each calibration subset, compute higher-fidelity labels
\[
y_c^{\mathrm{cal}}(s)\approx \widetilde T_c^{\mathrm{ref}}(\hat P_s),
\]
where \(\widetilde T^{\mathrm{ref}}\) may come from LSMI or another stronger interaction estimator.

Then fit the subset-level proxy to these labels:
\[
\mathcal L_{\mathrm{cal}}
=
\sum_{s\in\mathcal S_{\mathrm{cal}}}\sum_{c}
\Big(F_{c,\phi}(s)-y_c^{\mathrm{cal}}(s)\Big)^2.
\]

### 6.4 Total objective

The full objective is
\[
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{sem}}
+
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}.
\]

This trains the interaction head to be:
- predictive,
- semantically PID-like,
- calibrated to a stronger reference estimator.

---

## 7. Optional Surrogate Regression over Subsets

Once \(F_{c,\phi}(s)\) is available, we may optionally fit a surrogate model
\[
G_{c,\psi}(s)\approx F_{c,\phi}(s)
\]
to compress or smooth the subset-response map.

### 7.1 Linear surrogate

Fit
\[
(\hat\alpha_c,\hat\beta_c)
=
\arg\min_{\alpha,\beta}
\frac1m\sum_{j=1}^m
\Big(F_{c,\phi}(s^{(j)})-\alpha-\beta^\top s^{(j)}\Big)^2.
\]

Interpretation:
- \(\hat\beta_c\) captures first-order local sensitivity of the subset-level interaction map.

### 7.2 Kernel surrogate

Fit
\[
\hat g_c
=
\arg\min_{g\in\mathcal H_\kappa}
\frac1m\sum_{j=1}^m
\Big(F_{c,\phi}(s^{(j)})-g(s^{(j)})\Big)^2
+
\lambda\|g\|_{\mathcal H_\kappa}^2.
\]

Interpretation:
- the kernel surrogate captures local non-additive structure in the subset-response map,
- its second derivatives serve as curvature statistics rather than exact second-order PID functionals. :contentReference[oaicite:7]{index=7}

---

## 8. Why This Has an Advantage over LSMI

The proposed method does not primarily aim to improve the fidelity of **single-dataset** interaction estimation relative to LSMI. Instead, its main advantage is **amortization across many subset queries**.

### LSMI-style workflow
For each subset \(s\):
\[
s
\to
\hat P_s
\to
\text{run a sample-wise interaction estimator}
\to
\widetilde T(\hat P_s).
\]

### Proposed workflow
Train once:
\[
z_i \to t_\phi(z_i).
\]
Then answer any subset query by direct aggregation:
\[
F_{c,\phi}(s)
=
\frac{\sum_{k=1}^K s_k \sum_{i\in B_k} t_{\phi,c}(z_i)}
{\sum_{k=1}^K n_k s_k}.
\]

Therefore, compared with repeatedly applying LSMI-like decomposition, the proposed method:
1. avoids recomputing interaction decomposition for every subset,
2. produces an explicit, differentiable subset-response function,
3. integrates naturally with surrogate regression,
4. supports large-scale subset attribution and interaction querying.

---

## 9. Algorithm

### Algorithm 1: End-to-End Surrogate-Based PID-Like Proxy

**Input:** dataset \(D=\{z_i\}_{i=1}^n\), block partition \(\{B_k\}_{k=1}^K\), calibration subsets \(\mathcal S_{\mathrm{cal}}\), hyperparameters \(\lambda_{\mathrm{sem}},\lambda_{\mathrm{cal}}\).

**Step 1: Train predictive models.**  
Train
\[
q_1(y\mid x_1),\qquad q_2(y\mid x_2),\qquad q_{12}(y\mid x_1,x_2)
\]
with supervised losses.

**Step 2: Build the interaction head.**  
For each sample \(z_i\), compute
\[
m_1(z_i),\quad m_2(z_i),\quad m_{12}(z_i),
\]
then compute
\[
r_\phi(z_i),\quad
u_{1,\phi}(z_i)=m_1(z_i)-r_\phi(z_i),\quad
u_{2,\phi}(z_i)=m_2(z_i)-r_\phi(z_i),
\]
\[
s_\phi(z_i)=m_{12}(z_i)-m_1(z_i)-m_2(z_i)+r_\phi(z_i).
\]

**Step 3: Calibrate subset-level proxy.**  
For each calibration subset \(s\in\mathcal S_{\mathrm{cal}}\), compute
\[
F_{c,\phi}(s)=\frac{\sum_{k=1}^K s_k \sum_{i\in B_k} t_{\phi,c}(z_i)}{\sum_{k=1}^K n_k s_k},
\]
and minimize
\[
\mathcal L_{\mathrm{cal}}=
\sum_{s\in\mathcal S_{\mathrm{cal}}}\sum_c
\big(F_{c,\phi}(s)-y_c^{\mathrm{cal}}(s)\big)^2.
\]

**Step 4: Optimize the full objective.**  
Minimize
\[
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{sem}}
+
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}.
\]

**Step 5: Query subset-level interactions.**  
For any new subset \(s\), output
\[
F_{R,\phi}(s),\qquad
F_{U_1,\phi}(s),\qquad
F_{U_2,\phi}(s),\qquad
F_{S,\phi}(s).
\]

**Step 6 (optional): Fit a surrogate model.**  
Fit
\[
G_{c,\psi}(s)\approx F_{c,\phi}(s)
\]
using linear or kernel regression for faster querying or for local Taylor-style analysis.

---

## 10. Final Interpretation

The method should be interpreted as follows:
\[
F_{c,\phi}(s)
\approx
\text{PID-like interaction component of subset }s.
\]

More precisely:
- it is **not** the exact distribution-level PID in general,
- it is a **learned operational proxy** aligned with PID semantics,
- it is calibrated against a higher-fidelity reference estimator,
- it supports efficient repeated subset-level interaction estimation.

In this sense, the method provides an **end-to-end, amortized, surrogate-compatible estimator of PID-like multimodal interaction quantities**.

## References

[1] Distribution-level PID semantics and scalable PID estimation.  
[2] Sample-wise interaction decomposition and LSMI-style operational quantification.  
[3] Linear/kernel surrogate modeling of subset-response functions.