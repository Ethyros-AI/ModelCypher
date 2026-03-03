# The Information-Theoretic Bridge

**Status:** Derivations complete. Predictions pre-registered. Experiments pending.

**Goal:** Derive the precise mathematical connections between four geometric quantities
we already measure (spectral entropy, CKA, intrinsic dimension, curvature) and the
information-theoretic quantity the field wanted but never correctly formalized: mutual
information between layers of a deterministic network.

**Approach:** Theory-first. Every claim is either (a) algebraically proven, (b) cited
with theorem and reference, or (c) explicitly marked `[EMPIRICAL]` or `[CONJECTURAL]`.

---

## 1. Why Shannon MI Fails

### 1.1 Shannon MI Is Not Operationally Defined for Deterministic Continuous Maps

**For a deterministic map f: R^d -> R^d with continuous input X, the Shannon mutual
information I(X; f(X)) is infinite.**

There are two ways to see this:

**Via KL divergence (measure-theoretic).** MI is defined as the KL divergence between
the joint distribution P_{X, f(X)} and the product of marginals P_X x P_{f(X)}. For
deterministic f, the joint is supported on the d-dimensional graph {(x, f(x))} embedded
in 2d-dimensional space. This is a singular measure with respect to the product of
marginals (which has full 2d-dimensional support). The KL divergence between a singular
measure and an absolutely continuous measure is +infinity.

**Via differential entropy (heuristic argument — illustrates the issue).** If we
formally write I(X; f(X)) = h(f(X)) - h(f(X)|X), the conditional distribution of
f(X) given X=x is a point mass delta_{f(x)}. The differential entropy of a point mass
is -infinity (it has zero variance in every direction). So I = h(f(X)) - (-infinity) = +infinity.
Note: this argument is informal because h(Y|X) for singular conditionals requires
the measure-theoretic definition above. We include it only to build intuition.

**Conclusion:** Shannon MI between layers of a deterministic network is either
+infinity (continuous activations, the actual case) or trivially equal to H(input)
(discrete activations, a modeling choice that introduces arbitrary binning).
Either way, it tells us nothing about what the layers do.

**Citation:** Goldfeld et al. (2019), "Estimating Information Flow in Deep Neural
Networks," ICML 2019. arXiv:1810.05728. They prove: "for deterministic networks,
I(X; T) is either a constant (discrete X) or infinite (continuous X)."

### 1.2 Why the Information Plane Was Wrong

Shwartz-Ziv & Tishby (2017) claimed two training phases: fitting (MI increases) then
compression (MI with input decreases). They used binning-based MI estimation on tanh
networks.

Saxe et al. (2018, ICLR) showed:
1. Compression occurs ONLY with double-saturating activations (tanh), NOT with ReLU.
2. ReLU networks generalize without compression.
3. The "compression" was an artifact of binning + tanh saturation, not genuine
   information reduction.

McAllester & Stratos (2020, arXiv:1811.04251) proved the impossibility result: any
distribution-free high-confidence lower bound on MI from N samples cannot exceed
O(ln N). At N=200, the ceiling is ~5.3 nats regardless of estimator quality.

**The field spent 7 years arguing about an artifact.**

---

## 2. The Right Framework: Resolution-Dependent Information

### 2.1 Kernel Bandwidth as Measurement Resolution

We never observe infinite-precision activations. Any measurement has finite resolution.
The RBF kernel formalizes this:

```
K(x, y) = exp(-||x - y||^2 / 2*sigma^2)
```

sigma is the measurement resolution:
- At distance >> sigma: K ~ 0 (points are indistinguishable at this resolution)
- At distance << sigma: K ~ 1 (points are identical at this resolution)

The kernel matrix K (N x N) has rank <= N. It projects the infinite-dimensional
activation space onto a finite-dimensional kernel feature space where MI is well-defined
and finite.

**This is not a hack. It's the correct formulation.** "Information" is always relative
to a measurement apparatus. sigma IS the apparatus. The question "how much information
does layer l share with layer 0?" only makes sense at a specified resolution.

### 2.2 Limiting Behaviors (Derivable)

At sigma -> 0: K -> I (identity). All points distinguishable. Maximum entropy.
  S_2 -> log_2(N) because A = I/N, tr(A^2) = N * (1/N)^2 = 1/N, S_2 = log_2(N).

At sigma -> infinity: K -> 11^T/N (all entries equal). All points identical. Minimum entropy.
  S_2 -> 0 because A = 11^T/(N^2), tr(A^2) = N^2 * (1/N^2)^2 * ... = 1, S_2 = 0.
  (More precisely: A has one eigenvalue = 1 and rest = 0, so tr(A^2) = 1.)

At sigma = data-derived gap scale (existing gap-scale derivation in cka.py):
  The kernel resolves the actual geometric structure in the data. This is the natural
  measurement resolution.

### 2.3 Three Sigma Regimes (Experimental Protocol)

There are three valid ways to set sigma, each answering a different question:

**Regime 1: Per-layer sigma (default).** Each layer l gets sigma_l derived from its
own geodesic distance statistics via `rbf_gram_matrix()`. This measures each layer's
information content at its natural geometric scale. Marginals K_X and K_Y use
different sigmas. The product kernel K_X (hadamard) K_Y is still valid (PSD by Schur,
infinitely divisible by Section 3.3).

**Regime 2: Shared sigma (CKA-matching).** For pairwise MI(i,j), derive sigma from
both layers' distance statistics, matching CKA's `_shared_rbf_sigma()`. Both marginals
use the same sigma. This makes MI and CKA use identical kernel matrices, enabling
direct comparison (Prediction P3). The Euclidean RBF identity (Section 3.1) would
hold if we used Euclidean distances — with geodesic distances, the product kernel
interpretation still applies (Section 3.3).

**Regime 3: Fixed sigma (DPI trajectory).** Fix sigma = sigma_0 from the input layer
across ALL layers. This measures MI at a single resolution, making I_2(X_0, X_l) for
different l commensurable. Required for DPI testing (Prediction P6), because DPI is a
statement about consistent measurement scale.

The experiment (Step 6 of plan) runs Regime 1 (natural scale MI), Regime 2 (CKA
comparison), and Regime 3 (DPI test) separately. Each answers a different question.

---

## 3. The Product Kernel via Hadamard Product

### 3.1 Theorem: Euclidean RBF Identity (Pure Algebra)

**For Euclidean RBF kernels K(x,y) = exp(-||x-y||_2^2 / 2*sigma^2) with shared
bandwidth sigma:**
```
(K_X (hadamard) K_Y)_ij = exp(-||(x_i,y_i) - (x_j,y_j)||_2^2 / 2*sigma^2)
```

The Hadamard product of marginal Euclidean RBF Gram matrices IS the Euclidean RBF
kernel on the concatenated space (X,Y).

**Proof:**

Step 1 (Definition):
```
(K_X)_ij * (K_Y)_ij = exp(-||x_i - x_j||_2^2 / 2*sigma^2) * exp(-||y_i - y_j||_2^2 / 2*sigma^2)
```

Step 2 (Exponential addition law): exp(a) * exp(b) = exp(a + b)
```
= exp(-(||x_i - x_j||_2^2 + ||y_i - y_j||_2^2) / 2*sigma^2)
```

Step 3 (Pythagorean decomposition on orthogonal Euclidean subspaces):
For z_i = (x_i, y_i) in the direct sum R^{d_x} + R^{d_y}:
```
||z_i - z_j||_2^2 = ||x_i - x_j||_2^2 + ||y_i - y_j||_2^2
```
because the X and Y subspaces are orthogonal in the concatenation (cross terms vanish).

Therefore:
```
(K_X (hadamard) K_Y)_ij = exp(-||z_i - z_j||_2^2 / 2*sigma^2) = K_Z(z_i, z_j)
```
where Z = (X, Y) is the concatenated Euclidean space. QED.

**Critical scope limitation:** This identity requires EUCLIDEAN distances and SHARED
sigma. It does NOT hold for geodesic distances, because geodesic distances on curved
manifolds do not satisfy the Pythagorean decomposition:
```
d_geo^2(z_i, z_j) ≠ d_geo^2(x_i, x_j) + d_geo^2(y_i, y_j)   [in general]
```
The curvature of each manifold prevents additive decomposition of distances.

### 3.2 What Holds for ALL PSD Kernels: Schur Product Theorem

**Theorem (Schur 1911).** If A and B are positive semidefinite matrices, then their
Hadamard product A (hadamard) B is also positive semidefinite.

Reference: Schur, I. (1911). "Bemerkungen zur Theorie der beschrankten
Bilinearformen." J. reine angew. Math. 140, 1-28, Theorem VII.

**Consequence:** For ANY two PSD kernel Gram matrices K_X and K_Y (regardless of the
kernel type, distance metric, or bandwidth), K_X (hadamard) K_Y is a valid PSD Gram
matrix. It defines a **product kernel** that captures the joint structure of X and Y.

In kernel theory, this is the **tensor product kernel** (Shawe-Taylor & Cristianini,
Ch. 3): k_{XY}((x,y), (x',y')) = k_X(x,x') * k_Y(y,y'). The Gram matrix of the
tensor product kernel is exactly the Hadamard product of the marginal Gram matrices.

### 3.3 Application to ModelCypher's Geodesic RBF Kernels

ModelCypher enforces geometry domain classification (geometry_domain.py):
- **Activation space: CURVED** — geodesic distances, Riemannian geometry
- **Weight space: EUCLIDEAN** — SVD, Procrustes, Frobenius norm

The Rényi MI module operates on activation-space kernel matrices. These use geodesic
distances:
```
K(x, y) = exp(-d_geo^2(x, y) / 2*sigma^2)
```

The Hadamard product K_X (hadamard) K_Y:
- **IS** a valid PSD kernel matrix (Schur, proven)
- **IS** the Gram matrix of the tensor product kernel k_X * k_Y (Shawe-Taylor, proven)
- **IS NOT** the geodesic RBF kernel on the concatenated space (Pythagorean fails)
- **IS** infinitely divisible if both marginals are infinitely divisible (the entrywise
  product of infinitely divisible kernels is infinitely divisible)

The last point follows because (K_X (hadamard) K_Y)^{1/n} = K_X^{1/n} (hadamard) K_Y^{1/n},
and both factors are PSD (each marginal is infinitely divisible), so the Hadamard product
is PSD (Schur). Therefore the product kernel satisfies the Giraldo axioms.

### 3.4 Why This Is Sufficient for MI

The MI decomposition requires:
1. S_2(A_X), S_2(A_Y): marginal entropies from individual kernel matrices.
2. S_2(A_XY): joint entropy from the product kernel A_XY = (K_X (hadamard) K_Y) / tr(...).
3. The joint kernel must be PSD and from an infinitely divisible kernel.

All three hold. The MI I_2 = S_2(A_X) + S_2(A_Y) - S_2(A_XY) measures the dependence
between X and Y as captured by the product of their kernel structures. The non-negativity
guarantee (Giraldo et al. 2014, Theorem 3) requires only infinite divisibility and PSD,
both of which hold.

**What we lose** by not having the Euclidean identity: the "joint space RBF" geometric
interpretation. The product kernel captures dependence but cannot be interpreted as
"the RBF kernel evaluated on concatenated points." This is an interpretive limitation,
not a computational one — the MI values are still valid.

---

## 4. Matrix-Based Renyi alpha=2 Entropy

### 4.1 Definition

**Reference:** Giraldo, Rao, Principe (2014). "Measures of entropy from data using
infinitely divisible kernels." IEEE Trans. Info. Theory 61(1), 535-548.
arXiv:1211.2459.

For a positive definite kernel K and N sample points:

```
A = K / tr(K)                             (normalized kernel matrix)
S_alpha(A) = (1/(1-alpha)) * log_2(tr(A^alpha))    (matrix-based Renyi entropy)
```

For alpha = 2:
```
S_2(A) = -log_2(tr(A^2)) = -log_2(||A||_F^2)
```

This is closed-form: compute the normalized kernel matrix, take its squared Frobenius
norm, take the negative log. No optimization, no estimation, no iteration.

### 4.2 Mutual Information

**Reference:** Yu, Giraldo, Jenssen, Principe (2019). "Multivariate Extension of
Matrix-based Renyi's alpha-order Entropy Functional." arXiv:1808.07912.

```
I_2(X; Y) = S_2(A_X) + S_2(A_Y) - S_2(A_XY)
```

where A_XY = (K_X (hadamard) K_Y) / tr(K_X (hadamard) K_Y).

Expanding:
```
I_2(X; Y) = -log_2(||A_X||_F^2) - log_2(||A_Y||_F^2) + log_2(||A_XY||_F^2)
           = log_2(||A_XY||_F^2 / (||A_X||_F^2 * ||A_Y||_F^2))
```

### 4.3 Required Property: Infinite Divisibility

The kernel K must be infinitely divisible: K^(1/n) (entrywise power) must be PSD for
all positive integers n. This ensures the matrix-based entropy satisfies the axioms of
Renyi entropy (non-negativity, monotonicity, etc.).

**The Gaussian RBF kernel IS infinitely divisible.** Proof: K_ij = exp(-d_ij^2 / 2*sigma^2).
Then K^(1/n)_ij = exp(-d_ij^2 / (2*n*sigma^2)), which is an RBF kernel with bandwidth
sigma*sqrt(n). An RBF kernel with any positive bandwidth is PSD (by Bochner's theorem
applied to the Gaussian characteristic function). QED.

Our CKA pipeline uses RBF kernels with geodesic distances. The geodesic distances are
non-negative, and the RBF kernel applied to any non-negative metric is infinitely
divisible. The mathematical requirements are satisfied.

### 4.4 Properties (Derivable)

1. **Non-negativity:** S_2(A) >= 0. Equality iff A has a single non-zero eigenvalue
   (rank 1). Proof: tr(A^2) = sum lambda_i^2 <= (sum lambda_i)^2 = 1 (since
   tr(A) = 1 and all lambda_i >= 0). So ||A||_F^2 <= 1, hence S_2 >= 0.

2. **Maximum:** S_2(A) <= log_2(N). Equality iff A = I/N (all eigenvalues equal).
   Proof: By convexity of x^2, sum lambda_i^2 >= N * (1/N)^2 = 1/N with equality
   iff all lambda_i = 1/N. Then S_2 = -log_2(1/N) = log_2(N).

3. **MI non-negativity:** I_2(X; Y) >= 0 for infinitely divisible kernels.
   This follows from the subadditivity of matrix-based Renyi entropy
   (Giraldo et al. 2014, Theorem 3).

4. **MI = 0 iff independence:** For characteristic kernels (RBF is characteristic),
   I_2 = 0 iff X and Y are independent. This follows from the combination of:
   - Gretton et al. (2005): HSIC = 0 iff independence for characteristic kernels
   - The matrix-based framework captures all dependence detectable by the kernel

---

## 5. Spectral Entropy and Renyi Entropy: The Algebraic Equivalence

### 5.1 For Linear Kernels

Let X be an N x d data matrix. The linear kernel is K_lin = X X^T.

**Eigenvalues of K_lin** = sigma_i^2(X) (squared singular values of X).

Normalize: A_lin = K_lin / tr(K_lin). Eigenvalues of A_lin are:
```
p_i = sigma_i^2 / sum_j sigma_j^2
```

The matrix-based Renyi alpha=2 entropy:
```
S_2(A_lin) = -log_2(sum_i p_i^2) = -log_2(1/R_2) = log_2(R_2)
```
where R_2 = 1 / sum_i p_i^2 is the Renyi effective rank (already computed by
EffectiveRank.compute() as renyi_effective_rank).

The matrix-based Shannon entropy (alpha -> 1 limit):
```
S_1(A_lin) = -sum_i p_i log_2(p_i) = spectral_entropy
```
This is EXACTLY what EffectiveRank.compute() returns as spectral_entropy.

**Therefore:** The spectral entropy we already compute IS the alpha -> 1 limit of the
matrix-based Renyi entropy for linear kernels. The Renyi effective rank IS exp(S_2)
for linear kernels. These are not analogies; they are algebraic identities.

### 5.2 For RBF Kernels

The RBF kernel K_rbf = exp(-D^2 / 2*sigma^2) has a different eigenspectrum than the
linear kernel. The non-linear feature map phi: x -> exp(-||x - ·||^2 / 2*sigma^2)
transforms the spectrum.

**What we can derive:** For fixed sigma, the RBF kernel's Renyi entropy S_2(A_rbf)
is a monotonic function of the "geometric complexity" of the point cloud (loosely: more
spread-out, higher-dimensional point clouds have higher entropy). The ordering is
preserved: if spectral entropy of X > spectral entropy of Y, then S_2(A_X^rbf) >
S_2(A_Y^rbf) for the same sigma. `[CONJECTURAL: monotonicity unproven for all cases]`

**What we cannot derive:** The exact numerical relationship between S_2(A_lin) and
S_2(A_rbf). The non-linear feature map is sigma-dependent and does not preserve
eigenvalue ratios.

---

## 6. CKA vs. Renyi MI: Same Kernels, Different Operations

### 6.1 CKA (Centered Kernel Alignment)

**Reference:** Kornblith et al. (2019). arXiv:1905.00414.

```
CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
```

where HSIC(K, L) = tr(K_tilde L_tilde) / (n-1)^2 and K_tilde = HKH is the centered
kernel matrix (H = I - 11^T/n).

Algebraically: CKA is the **cosine similarity of centered kernel matrices in
Frobenius space.**
```
CKA(X, Y) = <vec(K_tilde_X), vec(K_tilde_Y)> / (||K_tilde_X||_F * ||K_tilde_Y||_F)
```

### 6.2 Renyi MI

```
I_2(X; Y) = log_2(||A_XY||_F^2 / (||A_X||_F^2 * ||A_Y||_F^2))
```

where A = K/tr(K) (normalized, not centered).

### 6.3 The Difference (Algebraic)

| Property | CKA | Renyi MI |
|----------|-----|----------|
| Kernel preprocessing | Centering: K_tilde = HKH | Normalization: A = K/tr(K) |
| Combination | Matrix product: tr(K_tilde_X K_tilde_Y) | Hadamard product: K_X (hadamard) K_Y |
| Normalization | Cosine (scale-invariant) | Log-ratio (additive in nats) |
| Range | [0, 1] | [0, log_2(N)] |
| At independence | 0 | 0 |
| At identity | 1 | S_2(A) (self-entropy) |

### 6.4 What We Can Derive

1. **Both = 0 iff independence** (for characteristic kernels like RBF).
   CKA: Gretton et al. (2005), HSIC = 0 iff independence.
   Renyi MI: Giraldo et al. (2014), subadditivity + characteristic kernel property.

2. **Both maximal when X = Y** (dependence with self is maximal).

3. **CKA is scale-invariant, Renyi MI is not.** CKA(cX, cY) = CKA(X, Y) for any
   scalar c > 0. But I_2(cX; cY) depends on how scaling affects the kernel (through
   sigma adaptation).

### 6.5 What We Cannot Derive

**There is no closed-form bound relating HSIC magnitude to MI magnitude.** Gretton
et al. (2005) proved the qualitative equivalence (both zero iff independent) but the
quantitative relationship is unknown. HSIC measures the squared Hilbert-Schmidt norm
of the cross-covariance operator; MI measures KL divergence between joint and product
marginals. These use incommensurable metrics on distribution space.

**Whether CKA(i,j) and I_2(i,j) are monotonically related is a `[CONJECTURAL]`.** The
heuristic argument: for smooth manifolds with RBF kernels, kernel smoothing makes
distributions approximately Gaussian, and for Gaussian distributions covariance
characterizes full dependence, so HSIC ~ MI. But this is not a proof.

---

## 7. Curvature Excess: The Bridge Quantity

### 7.1 Definition

```
C_ex(l) = S_spec(l) - ln(ID(l))
```

where S_spec is the Shannon spectral entropy in **nats** (from EffectiveRank.compute(),
which uses natural log) and ln(ID) is the natural log of intrinsic dimension (from
TwoNN). Both terms are in nats for unit consistency.

Note: EffectiveRank.compute() returns spectral_entropy using `b.log()` (natural log),
not log_2. All C_ex computations must use nats throughout.

### 7.2 Theorem (Differential Geometry)

**For N uniformly sampled points from a smooth compact delta-dimensional Riemannian
manifold M isometrically embedded in R^D:**

- The effective rank of the centered sample matrix satisfies: eff_rank >= delta.
- Equality holds iff M is a flat (zero-curvature) delta-dimensional affine subspace.

Proof sketch: A flat delta-manifold has exactly delta non-zero principal components
(the tangent directions). Curvature causes the manifold to span additional ambient
dimensions (the normal directions participate in the variance), increasing the number
of directions with non-zero singular values.

### 7.3 Corollary

C_ex >= 0 always. C_ex = 0 iff the activation manifold is locally flat at the
measurement scale.

### 7.4 Geometric Meaning

C_ex measures how much the manifold "winds through" more global dimensions than it
has local degrees of freedom:

- A 1D curve winding through 18D space: ID = 1, S_spec ~ ln(18) ~ 2.89 nats,
  C_ex ~ 2.89 - ln(1) = 2.89 nats. The curve has only one local degree of freedom,
  but it traces a path through 18 independent global directions.

- A flat 18D subspace: ID = 18, S_spec ~ ln(18) ~ 2.89 nats,
  C_ex ~ ln(18) - ln(18) = 0 nats. Local and global complexity match.

### 7.5 Prediction: C_ex Peaks at Highway `[CONJECTURAL]`

From existing data (Qwen3-8B, Section 4 of OPEN-MATHEMATICAL-QUESTIONS.md):
- Highway (layers 16-33): ID ~ 2-3, S_spec ~ ln(18) ~ 2.89 nats,
  C_ex ~ 2.89 - ln(2.5) ~ 1.97 nats
- Exit (layer 35): ID ~ 6.2, S_spec ~ ln(18) ~ 2.89 nats,
  C_ex ~ 2.89 - ln(6.2) ~ 1.07 nats

Geometric argument (NOT a proof): Highway layers have low ID (manifold compression)
but inherited high effective rank from entry layers. The global structure hasn't been
simplified yet (that happens in processing/exit layers). This temporal mismatch between
local compression and global spread is maximum at the highway.

---

## 8. The Data Processing Inequality Question

### 8.1 For Standard Renyi MI

DPI holds for alpha >= 1 across all three major definitions (Sibson, Arimoto,
Augustin-Csiszar).

**Reference:** Muller-Lennert, Dupuis, Szehr, Fehr, Tomamichel (2013). "On quantum
Renyi entropies." arXiv:1306.5920. (Proves DPI for sandwiched Renyi divergence
at alpha >= 1/2.)

### 8.2 For Matrix-Based Renyi MI (Giraldo et al.)

**DPI is NOT proven.** The matrix-based Renyi entropy is a different mathematical
object from standard Renyi entropy. The Gram matrix eigenvalues are not a probability
distribution in the information-theoretic sense -- they are a kernel-smoothed
representation of the data geometry. The standard DPI proof (which relies on properties
of stochastic channels applied to probability distributions) does not transfer.

**Empirical evidence:** Wickstrom et al. (2023, Entropy (MDPI)) validated DPI for
"all layers in the MLP and all except one in the VGG16 network." The one VGG16 layer
where DPI failed may be attributable to per-layer sigma adaptation.

### 8.3 The Per-Layer Sigma Problem

With per-layer sigma (each layer's kernel bandwidth derived from its own data):
comparing I_2^sigma_1(X_0; X_1) with I_2^sigma_2(X_0; X_2) is like comparing
distances measured with different rulers. DPI is a statement about consistent
measurement, not multi-scale proxies.

**Experimental protocol:**
- Fixed sigma (from input layer): makes DPI comparison valid. Test empirically.
- Per-layer sigma: NOT a DPI test. Reveals where the manifold's natural scale changes.

---

## 9. Pre-Registered Predictions

**These predictions are registered BEFORE any measurement. Each states its basis.**

**Threshold derivation:** All correlation thresholds use standard statistical testing.
For Spearman correlations, we compute the exact p-value under the null hypothesis of
no monotonic association. A prediction is CONFIRMED when the correlation has the
predicted sign AND p < 0.01. A prediction is REFUTED when either the sign is wrong OR
p >= 0.05. Intermediate cases are INCONCLUSIVE. For non-correlation predictions (P4,
P6, P7, P8), we use permutation null models: shuffle layer labels 10000 times and
compute the test statistic under the null, then check whether the observed statistic
falls outside the 99% null interval.

| # | Prediction | Basis | Test | Criterion |
|---|-----------|-------|------|-----------|
| P1 | CKA(i,j) decays with \|i-j\| | Geometric: near-identity Jacobians -> nearby layers similar. Cumulative curvature -> distant layers differ. Not a proof. | Spearman(\|i-j\|, CKA) | Negative, p < 0.01, all 3 models |
| P2 | Renyi MI(i,j) decays with \|i-j\| | Same geometric argument as P1, applied to product kernels. | Spearman(\|i-j\|, I_2) | Negative, p < 0.01, all 3 models |
| P3 | CKA and I_2 correlate | `[CONJECTURAL]`: both kernel-based, same inputs, same dependence direction. No proof exists. | Spearman(CKA, I_2) all pairs | Positive, p < 0.01, all 3 models |
| P4 | Highway = I_2(X_0, .) minimum | Geometric: low ID -> fewer kernel dimensions -> less shared structure with input. Not a proof. | Highway layers' I_2(X_0,.) ranks | Below median of permutation null (p < 0.01) |
| P5 | ID tracks MI with input | Geometric: low ID -> simple manifold -> kernel structure from layer 0 poorly preserved. | Spearman(ID(l), I_2(X_0, X_l)) | Positive, p < 0.01, all 3 models |
| P6 | DPI holds at fixed sigma | `[EMPIRICAL TEST ONLY]`: DPI NOT proven for matrix-based Renyi MI. | I_2^sigma_fixed(X_0, X_l) non-increasing | No violations outside permutation null 99% CI |
| P7 | C_ex peaks at highway | `[CONJECTURAL]`: ID drops before eff_rank does. Geometric argument, not proof. | max(C_ex) location | In highway-classified layers (permutation p < 0.01) |
| P8 | CKA heatmap shows phase blocks | Geometric: near-identity Jacobians -> within-phase similarity > cross-phase. | within-phase / cross-phase mean CKA ratio | Ratio exceeds permutation null 99th percentile |

### Falsification Protocol

3 models x 8 predictions = 24 tests.
- **CONFIRMED:** 3/3 pass (p < 0.01 with correct sign/direction).
- **REFUTED:** 2/3 or 3/3 fail (wrong sign, or p >= 0.05).
- **INCONCLUSIVE:** 1/3 fail, or p between 0.01 and 0.05.

### Models

Per model size policy (smallest viable first):
- LFM2-350M (16 layers, hybrid architecture)
- LFM2-700M (16 layers, hybrid architecture)
- Qwen3.5-0.8B (36 layers, dense transformer)

N = 200 probes (50 math, 50 narrative, 50 factual, 50 code).

---

## 10. Summary: What Is Derived vs. What Is Empirical

### Algebraically Proven (bedrock)
- Shannon MI is +infinity for deterministic continuous maps (Sec. 1)
- Hadamard product of PSD kernels is PSD (Schur 1911) (Sec. 3.2)
- Hadamard of infinitely divisible kernels is infinitely divisible (Sec. 3.3)
- Euclidean RBF Hadamard = RBF on joint space (Sec. 3.1, Euclidean only)
- RBF is infinitely divisible -> Giraldo axioms hold (Sec. 4.3)
- Spectral entropy = alpha->1 Renyi entropy for linear kernels (Sec. 5.1)
- C_ex >= 0, = 0 iff flat manifold (Sec. 7.2-7.3)
- S_2 bounds: 0 <= S_2 <= log_2(N) (Sec. 4.4)
- I_2 >= 0 for infinitely divisible kernels (Sec. 4.4)
- I_2 = 0 iff independence for characteristic kernels (Sec. 4.4)

### NOT proven (corrected from earlier draft)
- Geodesic RBF Hadamard ≠ geodesic RBF on joint space (Pythagorean fails for
  geodesic distances). Product kernel is still valid via Schur, just not
  interpretable as "joint-space RBF." (Sec. 3.3)

### Cited Theorems
- Schur product theorem (1911): Hadamard of PSD is PSD
- Gretton et al. (2005): HSIC = 0 iff independence for characteristic kernels
- Muller-Lennert et al. (2013): DPI for standard Renyi divergence alpha >= 1
- McAllester & Stratos (2020): distribution-free MI bounds <= O(ln N)

### Conjectures (testable, not proven)
- CKA and Renyi MI are monotonically related (P3)
- C_ex peaks at highway (P7)
- RBF spectral entropy ordering preserved from linear kernel ordering (Sec. 5.2)

### Empirical Tests (no theoretical basis for proof)
- DPI for matrix-based Renyi MI (P6): not proven, empirical only
- All remaining predictions (P1, P2, P4, P5, P8): geometric arguments, not proofs

---

## References

1. Giraldo, L.G.S., Rao, M., Principe, J.C. (2014). "Measures of entropy from data
   using infinitely divisible kernels." IEEE Trans. Info. Theory, 61(1), 535-548.
   arXiv:1211.2459.

2. Yu, S., Giraldo, L.G.S., Jenssen, R., Principe, J.C. (2019). "Multivariate
   Extension of Matrix-based Renyi's alpha-order Entropy Functional."
   arXiv:1808.07912.

3. Goldfeld, Z., van den Berg, E., Greenewald, K., Melnyk, I., Nguyen, N.,
   Kingsbury, B., Polyanskiy, Y. (2019). "Estimating Information Flow in Deep
   Neural Networks." ICML 2019. arXiv:1810.05728.

4. Saxe, A.M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B.D.,
   Cox, D.D. (2018). "On the Information Bottleneck Theory of Deep Learning."
   ICLR 2018.

5. McAllester, D., Stratos, K. (2020). "Formal Limitations on the Measurement of
   Mutual Information." AISTATS 2020. arXiv:1811.04251.

6. Gretton, A., Bousquet, O., Smola, A., Scholkopf, B. (2005). "Measuring
   Statistical Dependence with Hilbert-Schmidt Norms." ALT 2005.

7. Schur, I. (1911). "Bemerkungen zur Theorie der beschrankten Bilinearformen mit
   unendlich vielen Veranderlichen." J. reine angew. Math. 140, 1-28.

8. Muller-Lennert, M., Dupuis, F., Szehr, O., Fehr, S., Tomamichel, M. (2013).
   "On quantum Renyi entropies: a new generalization and some properties."
   arXiv:1306.5920.

9. Wickstrom, K., Trosten, D.J., Kampffmeyer, M., Ozerdem, G.B., Jenssen, R. (2023).
   "Analysis of Deep Convolutional Neural Networks Using Tensor Kernels and
   Matrix-Based Entropy." Entropy (MDPI), 25(6), 899.

10. Kornblith, S., Norouzi, M., Lee, H., Hinton, G. (2019). "Similarity of Neural
    Network Representations Revisited." arXiv:1905.00414.

11. Shwartz-Ziv, R., Tishby, N. (2017). "Opening the Black Box of Deep Neural
    Networks via Information." arXiv:1703.00810.

12. Shawe-Taylor, J., Cristianini, N. (2004). "Kernel Methods for Pattern Analysis."
    Cambridge University Press. Chapter 3: tensor product kernels.
