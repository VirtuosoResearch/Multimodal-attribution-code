# Stochastic Lanczos Quadrature (SLQ)

## 1. Problem Setting

Given a real symmetric matrix $A \in \mathbb{R}^{d \times d}$ (or an implicit matrix-vector product oracle $v \mapsto Av$), we want to efficiently estimate:

- **Spectral density**: the distribution of eigenvalues $\{\lambda_i\}$
- **Trace functionals**: $\mathrm{tr}(f(A)) = \sum_{i=1}^d f(\lambda_i)$ for scalar functions $f$
  - $f(x) = x$ gives the trace
  - $f(x) = \log x$ gives the log-determinant
- **Top-$k$ eigenpairs**: the leading eigenvalues and eigenvectors

Direct eigendecomposition costs $O(d^3)$. SLQ reduces this to $O(d \cdot m \cdot n_v)$ where $m \ll d$ is the number of Lanczos steps and $n_v$ is the number of random probes.

---

## 2. Building Block: Lanczos Tridiagonalization

### Algorithm

Given a matrix-vector oracle $v \mapsto Av$ and a starting vector $q_1$ with $\|q_1\| = 1$:

$$
\boxed{
\begin{aligned}
&\textbf{for } j = 1, \dots, m: \\
&\quad v = A q_j - \beta_{j-1} q_{j-1} \quad (\beta_0 = 0,\; q_0 = 0) \\
&\quad \alpha_j = q_j^\top v \\
&\quad v = v - \alpha_j q_j \\
&\quad \beta_j = \|v\| \\
&\quad q_{j+1} = v / \beta_j
\end{aligned}
}
$$

After $m$ steps, this produces:

- A **tridiagonal matrix** $T_m \in \mathbb{R}^{m \times m}$:

$$
T_m = \begin{pmatrix}
\alpha_1 & \beta_1 & & \\
\beta_1 & \alpha_2 & \beta_2 & \\
& \ddots & \ddots & \ddots \\
& & \beta_{m-1} & \alpha_m
\end{pmatrix}
$$

- An **orthonormal basis** $Q_m = [q_1, \dots, q_m] \in \mathbb{R}^{d \times m}$ such that:

$$
A \approx Q_m T_m Q_m^\top
$$

The eigenvalues of $T_m$ (called **Ritz values**) approximate the extreme eigenvalues of $A$. The cost is $m$ matrix-vector products.

### Reorthogonalization

In exact arithmetic, the Lanczos vectors $\{q_j\}$ are orthonormal by construction. In finite precision, orthogonality is gradually lost, causing:

- **Ghost eigenvalues**: spurious duplicate Ritz values near true eigenvalues
- **Corrupted Ritz vectors**: the eigenvectors of $T_m$ no longer map to meaningful directions in $\mathbb{R}^d$

**Full reorthogonalization** fixes this by explicitly projecting out all previous Lanczos vectors at each step:

$$
v \leftarrow v - Q_{1:j}(Q_{1:j}^\top v)
$$

This increases cost from $O(md)$ to $O(m^2 d)$ but is **essential for eigenvector computation**.

| Use case | Reorthogonalization needed? |
|---|---|
| Trace estimation | No (probe averaging dilutes ghosts) |
| Log-determinant | Helps for ill-conditioned spectra |
| Spectral density shape | Moderate help |
| **Top-$k$ eigenvectors** | **Yes, critical** |

---

## 3. SLQ for Trace Functionals

### Key Identity (Stochastic Trace Estimator)

For any function $f$ and random vector $z$ with $\mathbb{E}[zz^\top] = I$:

$$
\mathrm{tr}(f(A)) = \mathbb{E}_z\left[z^\top f(A) z\right]
$$

This is the Hutchinson identity. In the implementation here we use the normalized probe
$q = z / \|z\|$, which is uniform on the unit sphere and satisfies
$\mathbb{E}[qq^\top] = I / d$. Therefore:

$$
\mathrm{tr}(f(A)) = d \cdot \mathbb{E}_q\left[q^\top f(A) q\right]
$$

The inner quantity is a **quadratic form** that SLQ approximates via Lanczos.

### Algorithm: SLQ

$$
\boxed{
\begin{aligned}
&\textbf{Input: } \text{matvec oracle } v \mapsto Av, \;\text{dimension } d, \;\text{Lanczos steps } m, \;\text{probes } n_v \\
&\textbf{for } i = 1, \dots, n_v: \\
&\quad 1.\;\text{Draw } z_i \sim \mathcal{N}(0, I) \text{ and set } q_1 = z_i / \|z_i\| \\
&\quad 2.\;\text{Run } m\text{-step Lanczos from } q_1 \;\to\; T_m^{(i)} \\
&\quad 3.\;\text{Eigendecompose } T_m^{(i)} = S \,\Theta\, S^\top \\
&\quad 4.\;\text{Quadrature nodes: } \theta_j \quad\text{(eigenvalues of } T_m) \\
&\quad\;\;\; \text{Quadrature weights: } \omega_j = (S_{1j})^2 \quad\text{(squared first component)} \\
&\quad 5.\;\text{Estimate: } q_i^\top f(A) q_i \approx \sum_{j=1}^m \omega_j \, f(\theta_j) \\
&\textbf{Return: } \widehat{\mathrm{tr}(f(A))} = d \cdot \frac{1}{n_v} \sum_{i=1}^{n_v} \sum_{j=1}^m \omega_j \, f(\theta_j)
\end{aligned}
}
$$

### Why the Quadrature Weights are $(S_{1j})^2$

Starting from $q_1 = z/\|z\|$, the Lanczos relation gives $A \approx Q T Q^\top$. Then:

$$
q_1^\top f(A) q_1 \approx q_1^\top Q f(T) Q^\top q_1 = e_1^\top f(T) e_1
$$

where $e_1 = (1, 0, \dots, 0)^\top$ because $q_1$ is the first column of $Q$. Expanding in the eigenbasis $T = S\Theta S^\top$:

$$
e_1^\top f(T) e_1 = \sum_{j=1}^m (S_{1j})^2 \, f(\theta_j)
$$

This is a **Gauss quadrature rule** for the spectral measure of $A$ — it is exact for polynomials of degree $\leq 2m - 1$.

### Spectral Density

The spectral density is the distribution $\mu(\lambda) = \frac{1}{d}\sum_{i=1}^d \delta(\lambda - \lambda_i)$. SLQ approximates it as:

$$
\hat{\mu}(\lambda) = \frac{1}{n_v} \sum_{i=1}^{n_v} \sum_{j=1}^m \omega_j^{(i)} \, \delta(\lambda - \theta_j^{(i)})
$$

The quadrature nodes $\theta_j$ cluster near the true eigenvalues with high spectral weight.

---

## 4. SLQ for Top-$k$ Eigenvectors (Ritz Vectors)

### Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{Input: } \text{matvec oracle } v \mapsto Av, \;\text{Lanczos steps } m \gg k, \;\text{reorthogonalize}=\text{True} \\
&1.\;\text{Run } m\text{-step Lanczos with reorthogonalization} \;\to\; T_m,\; Q_m \\
&2.\;\text{Eigendecompose } T_m = S\,\Theta\,S^\top \\
&3.\;\text{Sort eigenvalues descending: } \theta_{\pi(1)} \geq \theta_{\pi(2)} \geq \cdots \\
&4.\;\text{Top-}k\text{ Ritz values: } \hat\lambda_i = \theta_{\pi(i)}, \quad i = 1,\dots,k \\
&5.\;\text{Top-}k\text{ Ritz vectors: } \hat{v}_i = Q_m \, s_{\pi(i)}, \quad i = 1,\dots,k \\
&\textbf{Return: } (\hat\lambda_1, \hat{v}_1), \dots, (\hat\lambda_k, \hat{v}_k)
\end{aligned}
}
$$

The Ritz vectors $\hat{v}_i = Q_m s_{\pi(i)}$ are the projections of the tridiagonal eigenvectors back into the original $\mathbb{R}^d$ space via the Lanczos basis.

### Convergence

The Ritz values converge to the **extreme** eigenvalues of $A$ first. For the $i$-th eigenvalue, after $m$ Lanczos steps:

$$
|\hat\lambda_i - \lambda_i| \leq (\lambda_1 - \lambda_d) \left(\frac{C_i}{T_{m-i}(\gamma_i)}\right)^2
$$

where $T_{m-i}$ is the Chebyshev polynomial and $\gamma_i$ measures the gap between $\lambda_i$ and the rest of the spectrum. Larger spectral gaps yield faster convergence.

### Why Reorthogonalization is Essential

Without reorthogonalization on the spiked Wishart test matrix ($n = 2000$, spikes at 5, 4, 3):

| Config | $|\cos(\hat{v}_1, v_1)|$ | $|\cos(\hat{v}_2, v_2)|$ | $|\cos(\hat{v}_3, v_3)|$ | Subspace overlap |
|---|---|---|---|---|
| $m = 50$, no reorth | 1.0000 | 0.0000 | 0.0000 | 0.60 |
| $m = 50$, **reorth** | **1.0000** | **1.0000** | **1.0000** | **1.00** |

Without reorthogonalization, ghost copies of $\lambda_1$ consume the Ritz spectrum, leaving no room for $v_2, v_3, \dots$ to converge. With reorthogonalization, $m = 50$ steps suffice for 6-digit accuracy on all 5 eigenpairs.

---

## 5. Complexity Summary

| Operation | Cost | Storage |
|---|---|---|
| Lanczos ($m$ steps, no reorth) | $O(md) + m \times$ matvec | $O(d)$ |
| Lanczos ($m$ steps, with reorth) | $O(m^2 d) + m \times$ matvec | $O(md)$ |
| Tridiagonal eigendecomposition | $O(m^2)$ | $O(m^2)$ |
| SLQ trace/logdet ($n_v$ probes) | $n_v \times$ Lanczos cost | same |
| Hutchinson trace only | $n_v \times$ one matvec = $O(n_v d)$ | $O(d)$ |
| Top-$k$ Ritz vectors | 1 Lanczos (with reorth) + $O(mk)$ | $O(md)$ |

For neural networks, each matvec is a Hessian-vector product (two backward passes), so minimizing $m$ and $n_v$ is important.

---

## 6. Usage

### Trace and Log-determinant

```python
from slq import SLQ

# From an explicit matrix
slq = SLQ(matrix=A)
tr = slq.trace(num_lanczos=50, num_probes=20)
ld = slq.logdet(num_lanczos=50, num_probes=20)

# From a neural network's Fisher/GGN
slq = SLQ(model=model, data_loader=loader,
          loss_fn=F.cross_entropy, device='cuda',
          loss_reduction='mean')
tr = slq.trace(num_lanczos=30, num_probes=10)
```

### Top-$k$ Eigenvectors

```python
slq = SLQ(matrix=A)

# reorthogonalize=True is the default and strongly recommended
eigenvalues, eigenvectors = slq.top_k_eigen(
    k=5, num_lanczos=100, reorthogonalize=True)

# eigenvalues:  shape [k], descending
# eigenvectors: shape [d, k], columns are approximate eigenvectors
```

### Spectral Density

```python
nodes, weights = slq.spectral_density(num_lanczos=100, num_probes=30)
# nodes:   quadrature points (approximate eigenvalue locations)
# weights: corresponding masses (sum to 1)
```

### Custom Matrix-Vector Operator

```python
# Diagonal matrix without forming it explicitly
diag = torch.linspace(1, 1000, 10000)
slq = SLQ(matvec=lambda x: diag * x, dim=10000)
tr = slq.trace()
```

### Hutchinson (Trace Only)

```python
from slq import hutchinson_trace_matrix

# More efficient than SLQ when you only need the trace
tr = hutchinson_trace_matrix(A, num_probes=100, distribution='rademacher')
```
