# Sigma Calibration via Constraint Satisfaction

**Status:** Design document for Regime 5 sigma selection.
**Depends on:** `information_bridge_derivation.md` (Sections 4.4, 5.1)

---

## 1. Problem Statement

The RBF kernel bandwidth sigma controls measurement resolution for matrix-based
Renyi MI (Giraldo et al. 2014). For cross-layer MI comparisons, all layers must
share a single sigma (Regime 4). The existing gap-based heuristic
(`_derive_rbf_sigma_from_values`) selects sigma from the largest relative gap in
sorted squared geodesic distances. This produces sigmas spanning 100x across models:

| Model | Gap-heuristic sigma | Outcome |
|-------|-------------------|---------|
| LFM2-350M | 3.515 | Non-degenerate |
| Qwen3.5-0.8B | 2.749 | Non-degenerate |
| LFM2-700M | 0.037 | Saturated (K ~ I for all layers) |

The heuristic has no mechanism to guarantee non-degeneracy.

## 2. Solution: Constraint Satisfaction

Instead of optimizing a proxy, define sigma as the solution to a constraint
satisfaction problem with a falsifiable empty-set outcome.

### 2.1 Renyi-2 Entropy as Degeneracy Measure

Given N probes, the RBF Gram matrix K_l(sigma) for layer l, and the normalized
matrix A_l = K_l / tr(K_l):

    S_2(K_l(sigma)) = -log_2(||A_l||_F^2)

Bounds (proven in information_bridge_derivation.md Section 4.4):
- S_2 = 0 iff rank(K) = 1 (sigma too large: all points indistinguishable)
- S_2 = log_2(N) iff K = I (sigma too small: all points maximally separated)

### 2.2 Monotonicity of S_2 in Sigma

**Claim:** S_2(K_l(sigma)) is strictly monotonically decreasing in sigma for any
fixed set of N distinct points.

**Proof:**

Let d_ij > 0 be the geodesic distance between probes i and j (i != j).

1. K_ij(sigma) = exp(-d_ij^2 / (2 sigma^2)). For d_ij > 0, K_ij is strictly
   increasing in sigma.

2. As sigma increases from 0:
   - K transitions from I (all off-diagonal entries 0) to 11^T (all entries 1)
   - A = K/tr(K) transitions from I/N to 11^T/N

3. ||A||_F^2 = sum_ij A_ij^2 measures concentration of A.
   - At A = I/N: ||A||_F^2 = N * (1/N)^2 = 1/N (minimum)
   - At A = 11^T/N: ||A||_F^2 = N^2 * (1/N)^2 = 1 (maximum)

4. As off-diagonal entries of K increase continuously (step 1), A becomes
   continuously more uniform, ||A||_F^2 increases strictly, and
   S_2 = -log_2(||A||_F^2) decreases strictly.

**Consequence:** Each layer's feasible sigma set is a connected interval [sigma_l^lo,
sigma_l^hi]. The intersection of finitely many intervals is an interval (possibly
empty). Binary search for each boundary is valid.

### 2.3 Non-Degeneracy Constraints

Define eps_mach = machine epsilon of the working dtype (bf16: 2^-7, float32: 2^-23).

Layer l is **non-degenerate** at sigma iff:
1. S_2(K_l(sigma)) > sqrt(eps_mach)        [not collapsed to rank 1]
2. log_2(N) - S_2(K_l(sigma)) > sqrt(eps_mach)   [not saturated to identity]

With bootstrap CI [L_l(sigma), U_l(sigma)] at confidence 1-alpha:
1. L_l(sigma) > sqrt(eps_mach)
2. log_2(N) - U_l(sigma) > sqrt(eps_mach)

### 2.4 Feasible Set and Sigma Selection

**Feasible set:** F = { sigma : all layers satisfy both constraints }

By Section 2.2, F is either empty or a connected interval [sigma_lower, sigma_upper].

**If F = empty:** Report "single-sigma measurement is insufficient; model is
intrinsically multi-scale for Renyi MI." This is a falsifiable negative result.

**If F non-empty:** sigma* = exp((ln(sigma_lower) + ln(sigma_upper)) / 2)
= sqrt(sigma_lower * sigma_upper) — geometric midpoint of the feasible interval.

Geometric midpoint maximizes margin to both boundaries in log-sigma space, which
is the natural parameterization for a scale parameter.

## 3. Algorithm

### 3.1 Binary Search for Layer Boundaries

For each layer l:

1. Compute squared geodesic distance matrix (pre-computed, shared with existing pipeline)
2. Determine search bounds:
   - sigma_lo = d_min * sqrt(eps_mach), where d_min = min positive distance
   - sigma_hi = d_max / sqrt(2 * eps_mach), where d_max = max distance
3. Binary search for sigma_l^lo: find sigma where S_2_l(sigma) = log_2(N) - sqrt(eps_mach)
   - At sigma_lo: S_2 ~ log_2(N) (too high → constraint 2 fails)
   - At sigma_hi: S_2 ~ 0 (too low → constraint 1 fails)
   - sigma_l^lo is where S_2 first drops below log_2(N) - sqrt(eps_mach)
4. Binary search for sigma_l^hi: find sigma where S_2_l(sigma) = sqrt(eps_mach)

### 3.2 Global Feasible Interval

sigma_lower = max_l(sigma_l^lo)
sigma_upper = min_l(sigma_l^hi)

If sigma_lower > sigma_upper: F is empty.

### 3.3 Bootstrap Verification

At sigma*, for each layer l:
1. Resample probe indices {1, ..., N} with replacement -> I_b
2. Extract sub-Gram: K_b[i,j] = K[I_b[i], I_b[j]]
3. Compute S_2(K_b)
4. Repeat B = ceil(2/alpha) times
5. CI = [percentile(alpha/2), percentile(1 - alpha/2)]

B = ceil(2/alpha) ensures >= 1 sample at each tail of the percentile CI.
At alpha = 0.01: B = 200.

If any layer's CI violates the constraints, report the violation (sigma* is
near a boundary; the feasible interval may be too narrow for statistical robustness).

## 4. Derived Constants

Every constant in this algorithm traces to one of:

| Constant | Source |
|----------|--------|
| sqrt(eps_mach) | IEEE 754 machine epsilon of working dtype |
| log_2(N) | Number of probes (data) |
| d_min, d_max | Computed from activation data |
| alpha = 0.01 | User-specified confidence level |
| B = ceil(2/alpha) | Derived from alpha |
| Binary search iterations | ceil(log_2((ln sigma_hi - ln sigma_lo) / eps_mach)) ~ 50 |

No magic numbers. No heuristics.
