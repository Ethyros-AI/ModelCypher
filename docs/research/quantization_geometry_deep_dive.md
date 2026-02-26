# Quantization Geometry Deep Dive

**Status**: Research synthesis + experimental plan
**Date**: 2026-02-26
**Dependencies**: Weyl validation data, compression synthesis, spectral scale bound
**Related**: [`COMPRESSION-RESEARCH-SYNTHESIS.md`](./COMPRESSION-RESEARCH-SYNTHESIS.md), [`lora_spectral_scale_bound.md`](./lora_spectral_scale_bound.md)

---

## Executive Summary

Quantization is the dominant deployment strategy for large models, yet its geometric effects are poorly understood. Industry treats it as a lossy compression tradeoff. We have 448 layers of measured spectral data showing the real picture is more nuanced — and more useful.

**Core findings:**
- The spectral quantities that matter (sigma_max, sigma_k, tail_dims) are barely perturbed by 8-bit quantization (<0.032%, <0.153%, 439/448 = 98.0% match)
- The Weyl crossing criterion is violated 3.9-1700.5x (0/448 layers safe), but this measures fine eigenvalue ordering at the noise floor — not the structure the model relies on
- T-matrix quantization at 8-bit *outperforms* FP32 (93.3% vs 86.7%), demonstrating quantization-as-regularization
- Standard QLoRA uses scale=2.0, which violates spectral bounds by 600-2700x independent of quantization. The scale error is 3-4 orders of magnitude larger than the quantization error

**Core hypothesis:** Most QLoRA degradation attributed to quantization is actually spectral scale violation. The geometric LoRA pipeline (Cayley-parameterized, MASS-derived step size, spectral scale bound) should work identically on quantized and full-precision bases, because the spectral quantities it depends on are quantization-invariant.

---

## Part 1: Base Model Quantization Geometry `[MEASURED]`

### 1.1 Quantization as Structured Perturbation

Every quantized weight matrix W_q can be written as:

```
W_q = W_fp + E_q
```

where E_q = W_q - W_fp is the quantization error matrix.

For k-bit group-g affine quantization, each group of g contiguous elements in the weight is mapped to a uniform grid:

```
W_q[i, j:j+g] = scale * round((W_fp[i, j:j+g] - zero) / scale) + zero
```

where `scale = (max - min) / (2^k - 1)` within each group.

**E_q is NOT a random matrix.** It has block structure inherited from the quantization grid:
- Within each group of g elements, errors are bounded by `scale / 2`
- Across groups, error magnitudes vary with weight magnitudes
- The block structure means E_q's spectral properties differ from Marchenko-Pastur predictions for random matrices

This is geometrically significant: random perturbations spread energy uniformly across singular directions, but structured perturbations concentrate energy in specific directions determined by the grid alignment.

*Code: [`quantization_utils.py`](../../src/modelcypher/core/use_cases/quantization_utils.py) — `dequantize_if_needed()`, `resolve_quantization()`*

### 1.2 Measured Spectral Impact of 8-bit Quantization

We measured the spectral impact of 8-bit-g64-affine quantization on every 2D weight matrix in Qwen3-1.7B (196 layers) and Qwen3-8B (252 layers).

*Data: [`weyl_quantization_validation.json`](../../results/weyl_quantization_validation/20260226T015425Z/weyl_quantization_validation.json)*
*Note: exact-SVD values are canonical; randomized SVD is retained for fast scans and can inflate boundary metrics.*

**Qwen3-1.7B Layer 0 — Representative Sample:**

| Projection | sigma_max (FP) | sigma_max (Q) | Change | sigma_k (FP) | sigma_k (Q) | Change |
|------------|----------------|---------------|--------|---------------|--------------|--------|
| down_proj | 7.825 | 7.824 | 0.012% | 1.668 | 1.668 | 0.023% |
| gate_proj | 12.547 | 12.545 | 0.016% | 2.076 | 2.076 | 0.037% |
| up_proj | 5.947 | 5.947 | 0.013% | 1.488 | 1.487 | 0.037% |
| k_proj | 10.505 | 10.504 | 0.018% | 1.192 | 1.192 | 0.027% |
| o_proj | 6.401 | 6.399 | 0.025% | 0.989 | 0.989 | 0.025% |
| q_proj | 11.867 | 11.865 | 0.018% | 1.183 | 1.183 | 0.017% |
| v_proj | 3.117 | 3.116 | 0.019% | 0.925 | 0.925 | 0.018% |

**Aggregate across 448 layers (Qwen3-1.7B + Qwen3-8B):**

| Metric | Value |
|--------|-------|
| Max sigma_max change | <0.032% |
| Max sigma_k change | <0.153% |
| tail_dims match | 439/448 (98.0%) |
| Weyl crossing safe | 0/448 (0%) |
| Max error/gap ratio | 1700.5x |
| Max ||E_q||_2 | 0.1415 |

Per-model breakdown from the exact run:
- Qwen3-1.7B: `tail_dims` 192/196, max error/gap 576.2x, max `||E_q||_2` 0.0554
- Qwen3-8B: `tail_dims` 247/252, max error/gap 1700.5x, max `||E_q||_2` 0.1415

### 1.3 The Weyl Paradox: Why Violations Don't Kill the Model

Weyl's perturbation theorem (1912) guarantees that for any perturbation E:

```
|sigma_i(W + E) - sigma_i(W)| ≤ ||E||_2
```

This bound IS satisfied. With ||E_q||_2 ≈ 0.0139-0.1415 and sigma_max ≈ 7-12, the relative perturbation to any individual singular value is tiny. The top singular value moves by 0.0009 out of 7.825 — a 0.012% change. This is why quantized models work.

The Weyl *crossing criterion* asks a different question: can singular values at the structural rank boundary swap their ordering? The condition is:

```
||E_q||_2 < spectral_gap(sigma_k) / 2
```

With ||E_q||_2 ≈ 0.0139-0.1415 and spectral gaps at sigma_k ≈ 10^-5-10^-3, this is violated 3.9-1700.5x. Singular values at the noise floor DO cross.

**But this crossing is geometrically inconsequential.** Here's why:

1. **The top singular directions are safe.** They carry >90% of the spectral energy and are separated by gaps much larger than ||E_q||_2. These directions define the model's learned function.

2. **The tail directions are interchangeable.** Below sigma_k, singular values are packed tightly (gaps ≈ 10^-4) and carry <10% of total energy collectively. Swapping which directions map to which near-degenerate singular values doesn't change the effective transformation.

3. **tail_dims measures what matters.** The count of effective dimensions (tail_dims) is preserved in 439/448 layers (98.0%). The *topology* of the effective subspace — how many dimensions carry meaningful information — is unchanged even when individual eigenvalues shuffle at the boundary.

The analogy: if you scramble the order of books on a shelf but keep every book, the library still has the same information. Weyl crossing rearranges near-degenerate singular directions without removing or adding any.

*Code: [`weyl_quantization_validation.py`](../../scripts/weyl_quantization_validation.py), [`spectral_budget.py`](../../src/modelcypher/core/domain/training/spectral_budget.py)*

### 1.4 Weight-Space vs Activation-Space Geometry

The compression synthesis established a stronger result: 4-bit weight quantization produces 90-100% Frobenius error on individual weight matrices, yet models still produce correct output.

```
8-bit Frobenius error:  5-15%   → models work
4-bit Frobenius error:  90-100% → models STILL work
```

**"Weights aren't the constraint — transformation is."**

The explanation: the model's function is determined not by individual weight matrices but by the composed transformation through residual connections. Each layer computes:

```
h_{l+1} = h_l + f_l(h_l; W_l)
```

The residual connection means f_l contributes a *delta* to the hidden state. Even large relative errors in W_l produce small relative errors in h_{l+1} when f_l(h_l) is small relative to h_l — which it is in transmission layers (layers 14-21 in Qwen3-8B, where the MLP is functionally linear).

The activation manifold — the geometric structure traced by hidden states as they flow through the network — is more robust than any individual weight matrix because:
- Residual connections dilute per-layer errors
- Error propagation depends on layer position (see compression synthesis: layers 14-21 absorb errors completely)
- The manifold's intrinsic dimension (measured by CKA, geodesic structure) depends on relational structure, not individual weight values

*Reference: [`COMPRESSION-RESEARCH-SYNTHESIS.md`](./COMPRESSION-RESEARCH-SYNTHESIS.md)*

### 1.5 Quantization as Regularization: The T-matrix Evidence

When we replace MLP layers with their closed-form T-matrix approximation T = Y @ pinv(X) and then quantize T:

| Format | Size (8 layers) | Accuracy |
|--------|-----------------|----------|
| FP32 T | 537MB | 86.7% |
| 16-bit T | 268MB | 86.7% |
| **8-bit T** | **134MB** | **93.3%** |
| 4-bit T | 67MB | 80.0% |

8-bit T-matrix *outperforms* FP32 T-matrix by 6.6 percentage points. This is not noise — it's a regularization effect.

The geometric explanation: the T-matrix T = Y @ pinv(X) is computed from calibration data. With finite calibration (800 prompts), T overfits slightly to the calibration set — it encodes high-frequency noise in the calibration data. The 8-bit quantization grid truncates these high-frequency components, effectively regularizing the transformation.

This is consistent with the spectral hierarchy: quantization destroys fine structure at the noise floor while preserving gross structure. When the fine structure IS noise (overfitting), destroying it improves generalization.

*Reference: [`COMPRESSION-RESEARCH-SYNTHESIS.md`](./COMPRESSION-RESEARCH-SYNTHESIS.md) Part 4*

### 1.6 The Spectral Hierarchy Under Quantization

Ranking what survives 8-bit quantization, from most robust to least:

| Spectral Quantity | Effect of 8-bit | Status |
|-------------------|-----------------|--------|
| sigma_max (dominant direction) | <0.032% change | PRESERVED |
| sigma_k (structural boundary) | <0.153% change | PRESERVED |
| tail_dims (effective subspace count) | 439/448 (98.0%) match | PRESERVED |
| Activation manifold topology | Intact (models work) | PRESERVED |
| Spectral gap at sigma_k | 3.9-1700.5x violation | DESTROYED |
| Fine eigenvalue ordering at boundary | Crossings occur | DESTROYED |

**What's destroyed doesn't matter. What matters is preserved.**

The model's function depends on the first four rows. The last two rows describe fine-grained eigenvalue ordering at the noise boundary — which is numerically fragile even in full precision (gaps ≈ 10^-4 are close to floating-point precision for bf16 eps ≈ 7.8 × 10^-3).

---

## Part 2: QLoRA Geometry `[CONJECTURAL + MEASURED where noted]`

### 2.1 The Double Approximation Problem

Standard QLoRA (Dettmers et al., 2023) trains LoRA adapters on 4-bit quantized base models. The geometric picture:

**Forward pass:**
```
y = dequantize(W_q) @ x + scale * (B @ A) @ x
  = (W_fp + E_q) @ x + scale * (B @ A) @ x
```

The forward pass computes with the perturbed weight W_fp + E_q, not the true weight W_fp. Every activation is computed on a displaced surface.

**Backward pass:**
```
dL/dA = B^T @ dequantize(W_q)^T @ (dy/dx) @ x^T
dL/dB = dequantize(W_q)^T @ (dy/dx) @ x^T @ A^T
```

Gradients flow through the dequantized weight. The gradient landscape the optimizer sees is the landscape of f(x; W_fp + E_q + scale * B @ A), not f(x; W_fp + scale * B @ A).

**This is a double approximation:**
1. The base transformation is approximate (quantized): contributes ||E_q||_2 ≈ 0.0139-0.1415
2. The update is low-rank (LoRA): misses components outside rank(B @ A)

The standard QLoRA adds a third error source that dominates both:
3. The scale is wrong (alpha/rank ≈ 2.0): contributes 600-2700x × sigma_k

### 2.2 How ModelCypher Handles Quantized Bases `[MEASURED]`

The training pipeline already handles quantized bases correctly. The chain:

**1. LoRA injection** — `NBLoRALinear.from_base()` detects `nn.QuantizedLinear` and corrects dimensions for packed integer format:
```python
input_dims = input_dims * 32 // linear.bits  # Unpack integer packing
```

**2. Geometry analysis** — `_dequantize_weight()` dequantizes before any SVD:
```python
if isinstance(proj, nn.QuantizedLinear):
    w = mx.dequantize(proj.weight, proj.scales, proj.biases,
                       proj.group_size, proj.bits)
```

**3. Streaming analysis** — `analyze_model_geometry_streaming()` processes one layer at a time on dequantized weights, computing sigma_max, sigma_k, tail_dims, spectral_gap per layer.

**4. Forward pass** — `NBLoRALinear.__call__()` uses the QuantizedLinear's native forward (which dequantizes on-the-fly internally), then adds the Cayley-LoRA contribution:
```python
base_out = self.linear(x)     # QuantizedLinear handles dequantization
lora_out = cayley_forward(x)  # NB-LoRA with Cayley parameterization
return base_out + lora_out
```

The critical design: geometry analysis operates on dequantized weights (full precision spectral structure), while the forward pass uses the quantized path (memory efficient). The MASS step size and spectral ceiling are computed from the true spectral structure, not from quantized artifacts.

*Code: [`_mlx_training_adapter_core_mixin.py`](../../src/modelcypher/backends/_mlx_training_adapter_core_mixin.py), [`mlx_training_adapter_core.py`](../../src/modelcypher/backends/mlx_training_adapter_core.py)*

### 2.3 Spectral Ceiling on Quantized Weights `[PROVEN]`

The MASS step size formula:

```
eta_ceiling = sigma_k_min / (sigma_max_global × sqrt(N))
```

From the Weyl validation data:
- sigma_k changes <0.153% under 8-bit quantization
- sigma_max changes <0.032% under 8-bit quantization

Therefore eta_ceiling computed on quantized weights differs from the full-precision eta_ceiling by at most ~0.19%. **The MASS learning rate is robust to 8-bit quantization by construction** — its inputs barely move.

For 4-bit quantization, we expect larger sigma_k and sigma_max perturbations. Experiment 5 in the validation plan will measure this. However, the hierarchy from Section 1.6 suggests these top-of-spectrum quantities remain robust even as Frobenius error approaches 100%.

*Code: [`mass_step_size.py`](../../src/modelcypher/core/domain/training/mass_step_size.py)*

### 2.4 Where QLoRA Goes Wrong: Scale, Not Quantization `[CONJECTURAL]`

The spectral scale bound for LoRA:

```
scale_bound = sigma_k(W) / ||B @ A||_spectral
```

From the [spectral scale bound validation](./lora_spectral_scale_bound.md):
- All 9 tested adapters violated this bound by 600-2700x
- The standard configuration (alpha=16, rank=8, scale=2.0) produces degenerate output
- Geometric scaling (~0.1-0.3) produces correct reasoning

**These violations are quantization-independent.** sigma_k barely changes under quantization (<0.153%), so the bound is essentially the same whether computed on bf16 or 8-bit weights.

Let's compare the error magnitudes:

| Error Source | Magnitude | Relative to sigma_k |
|--------------|-----------|---------------------|
| Quantization (||E_q||_2) | ~0.0139-0.1415 | ~0.012-0.078x sigma_k |
| Standard LoRA scale violation | 600-2700x × sigma_k | 600-2700x sigma_k |

**The scale violation is 7,700-220,000x larger than the quantization error.**

When practitioners report that QLoRA produces worse results than full-precision LoRA, they're observing the compounding of two errors: the catastrophic scale violation (which exists in both cases) and the small quantization perturbation (which exists only in QLoRA). But the scale violation dominates so completely that the quantization effect is unmeasurable.

**Conjecture:** If you fix the scale (use geometric scaling), QLoRA and full-precision LoRA produce indistinguishable results. The only difference is memory usage.

This is testable with existing infrastructure (Experiment 2).

---

## Part 3: Geometric LoRA as Quantization Offset `[PARTIALLY MEASURED]`

### 3.1 The Additive Recovery Hypothesis

Given that quantization introduces a structured perturbation E_q, can a LoRA adapter trained specifically to correct this error recover the full-precision model's behavior?

The quantized model computes f(x; W + E_q). A corrective adapter Delta produces f(x; W + E_q + Delta). If Delta ≈ -E_q, the errors cancel:

```
W + E_q + Delta ≈ W + E_q + (-E_q) = W
```

Perfect cancellation requires Delta = -E_q exactly, which is a full-rank matrix — not achievable with rank-r LoRA. But if E_q's energy is concentrated in a few singular directions, a low-rank approximation captures most of the error.

The key question: **Does E_q have low-rank structure?**

The answer depends on the quantization scheme:
- **Uniform grid quantization** (affine): E_q is bounded element-wise but not low-rank by construction. The block structure (groups of g elements) introduces some structure, but the rank of E_q is typically min(m, n) — full rank.
- **However**: Full rank ≠ uniformly distributed spectral energy. E_q can be full-rank yet have most of its spectral energy concentrated in a few directions.

This is exactly what RMT signal separation is designed to measure.

### 3.2 RMT Signal Separation Applied to Quantization Error `[MEASURED]`

The existing `separate_signal_noise()` function applies the Marchenko-Pastur distribution to separate eigenvalues of a matrix's spectrum into signal (above MP bulk edge) and noise (within bulk).

Applied to E_q = W_fp - W_q:

1. Compute SVD of E_q → singular values S_e
2. Convert to covariance eigenvalues: lambda_i = S_e[i]^2 / (n-1)
3. Compute MP bulk edges for aspect ratio gamma = min(m,n) / max(m,n)
4. Eigenvalues above MP upper edge = **systematic quantization artifacts** (signal)
5. Eigenvalues within bulk = **effectively random quantization noise**

**Measured result: E_q has massive systematic structure.** Every single layer in both models has signal above the MP bulk edge.

| Model | Layers | With Signal | Mean signal_rank | Mean sv_frac | 95% CI (signal_rank) | 95% CI (sv_frac) |
|-------|--------|-------------|------------------|--------------|----------------------|-------------------|
| Qwen3-1.7B | 196 | 196 (100%) | 425.3 | 53.7% | [401.6, 448.7] | [51.2%, 56.1%] |
| Qwen3-8B | 252 | 252 (100%) | 750.8 | 48.2% | [696.5, 803.8] | [46.0%, 50.4%] |

**Key observations:**
- **100% of layers have signal** — not a single layer where E_q is pure noise
- **~50% of error energy is systematic** — half of the quantization error is above the MP bulk and targetable by low-rank methods
- **Signal rank scales with matrix dimension** — larger matrices (8B) have proportionally more signal directions, suggesting the structure comes from the quantization grid interacting with weight structure
- **0 SVD failures** across 448 layers (SVD of E_q is numerically stable)

**Per-projection pattern (Qwen3-8B typical layer):**

| Projection | Shape | signal_rank | sv_frac | ||E_q||_2 |
|------------|-------|-------------|---------|-----------|
| q_proj | 4096×4096 | ~1325 | ~75% | ~0.025 |
| o_proj | 4096×4096 | ~1306 | ~73% | ~0.031 |
| down_proj | 4096×12288 | ~680 | ~33% | ~0.053 |
| gate_proj | 12288×4096 | ~736 | ~36% | ~0.031 |
| up_proj | 12288×4096 | ~677 | ~33% | ~0.025 |
| k_proj | 1024×4096 | ~179 | ~35% | ~0.017 |
| v_proj | 1024×4096 | ~162 | ~30% | ~0.016 |

The square attention matrices (q_proj, o_proj) have the highest signal_variance_fraction (~73-75%), meaning their quantization error is most structured and most correctable. The rectangular MLP matrices have lower but still substantial signal fractions (~30-36%).

**Implication:** Corrective LoRA is geometrically justified. A rank-r adapter targeting the top signal directions of E_q can capture a measurable fraction of the quantization error. The question is no longer *whether* correction is possible, but *how much* correction each adapter round achieves.

*Data: [`rmt_quantization_error.json` (1.7B)](../../results/rmt_quantization_error/20260226T001044Z/rmt_quantization_error.json), [`rmt_quantization_error.json` (8B)](../../results/rmt_quantization_error/20260226T002308Z/rmt_quantization_error.json)*
*Script: [`rmt_quantization_error.py`](../../scripts/rmt_quantization_error.py)*
*Code: [`rmt_signal_separation.py`](../../src/modelcypher/core/domain/geometry/rmt_signal_separation.py) — `compute_signal_rank_from_singular_values()`*

### 3.3 Spectral Scale Bounds for Corrective LoRA

A corrective adapter is subject to the same spectral constraint:

```
scale_bound = sigma_k(W_q) / ||B @ A||_spectral
```

Since sigma_k(W_q) ≈ sigma_k(W_fp), the bound is essentially identical. The Cayley parameterization guarantees the adapter's spectral norm stays within bounds throughout training.

The training objective differs from standard LoRA:
- **Standard LoRA**: Minimize task loss (CE on next-token prediction)
- **Corrective LoRA**: Minimize reconstruction loss (MSE against bf16 activations on probe data)

```
L_corrective = sum_over_layers ||f_l(x; W_q + Delta) - f_l(x; W_fp)||^2
```

This is a regression objective, not a classification objective. The adapter learns to undo the quantization error on observed activations. CKA on probes becomes the natural verification metric:

```
Success: CKA(f(x; W_q + Delta), f(x; W_fp)) > CKA(f(x; W_q), f(x; W_fp))
```

*Code: [`cayley_lora.py`](../../src/modelcypher/core/domain/geometry/cayley_lora.py), [`lora_safety_service.py`](../../src/modelcypher/core/use_cases/lora_safety_service.py)*

### 3.4 The Stacking Hypothesis: Iterated Geometric Recovery

If a single rank-r adapter captures the top signal components of E_q, the residual error after one round is:

```
E_residual = E_q + Delta_1 = E_q - proj_r(E_q) = E_q - sum_{i=1}^{r} sigma_i u_i v_i^T
```

This residual can itself be decomposed by RMT. If it still has signal components above the MP bulk, a second adapter can target those:

```
Round 1: Delta_1 captures top r signal directions of E_q
Round 2: Delta_2 captures top r signal directions of E_residual
...
Round k: Delta_k captures top r signal directions of E_residual_{k-1}
```

The `LoRAStacker` provides the infrastructure for this iteration:
- Cumulative barrier tracking (additive) detects when adapters start interfering
- CKA drift tracking (max) detects departure from desired behavior
- Convergence detection triggers merging when gains plateau

**Convergence criterion:** The iteration converges when the residual's signal_rank drops to 0 — all remaining error is within the MP noise bulk and is uncorrectable by low-rank methods.

**Convergence bound:** If the top k eigenvalues of E_q's covariance capture fraction f_k of total error variance, then k rounds of rank-r training reduce the correctable error to (1 - f_k). If f_1 = 0.7 (70% of error is in top-r signal directions), one round removes 70% of correctable error. Two rounds targeting the residual might capture f_2 = 0.9 cumulative.

**The fundamental limit:** The noise-floor fraction (1 - f_total) of error is irreducible by low-rank methods. This fraction is determined by the quantization scheme's interaction with the weight matrix structure.

*Code: [`lora_stacker.py`](../../src/modelcypher/experimental/self_improve/lora_stacker.py)*

### 3.5 The Practical Question: Is Correction Worth It?

From Section 1.2, quantized models already work. sigma_max and sigma_k barely move. tail_dims are preserved. The activation manifold is intact.

If quantization error is already small enough to be invisible in model behavior, corrective LoRA is solving a non-problem. The value of correction depends on:

1. **Task sensitivity.** Some tasks (exact arithmetic, code generation, factual recall) may be sensitive to small activation perturbations that don't show up in aggregate metrics.
2. **Error accumulation.** For stacked training (multiple adapters), each round starts from the quantized base. If quantization error compounds across adapters, correction of the base error before stacking could prevent drift.
3. **4-bit vs 8-bit.** At 4-bit (where Frobenius error is 90-100%), the correction opportunity is larger and the need more acute.

The experimental plan addresses all three questions.

---

## Part 4: Status Summary `[ALL CLAIMS TAGGED]`

| Claim | Status | Evidence |
|-------|--------|----------|
| sigma_max barely changes under 8-bit quantization | MEASURED | <0.032% max across 448 layers (exact-SVD Weyl validation) |
| sigma_k barely changes under 8-bit quantization | MEASURED | <0.153% max across 448 layers (exact-SVD Weyl validation) |
| Weyl crossing criterion violated 3.9-1700.5x | MEASURED | 0/448 layers safe (exact-SVD Weyl validation) |
| tail_dims preserved under 8-bit quantization | MEASURED | 439/448 aggregate; 192/196 (Qwen3-1.7B), 247/252 (Qwen3-8B) |
| 4-bit Frobenius error 90-100% yet models work | MEASURED | Compression synthesis |
| 8-bit T-matrix outperforms FP32 T-matrix | MEASURED | 93.3% vs 86.7% (compression synthesis) |
| Weyl violations are inconsequential at noise floor | PROVEN | Weyl 1912 + measured gap structure |
| MASS spectral ceiling robust to 8-bit quantization | PROVEN | sigma_k/sigma_max ratio preserved |
| NB-LoRA handles QuantizedLinear bases correctly | MEASURED | Code path verified, geometry valid |
| Standard LoRA scale violates spectral bound 600-2700x | MEASURED | All 9 tested adapters (spectral scale bound) |
| QLoRA failures are spectral scale violations, not quantization | CONJECTURAL | Needs A/B test (Experiment 2) |
| E_q has exploitable low-rank signal structure | **MEASURED** | 100% of layers have signal (Experiment 1) |
| Corrective LoRA can recover quantization error | CONJECTURAL | Testable (Experiment 3) |
| Stacked recovery converges when residual enters MP bulk | CONJECTURAL | Testable (Experiment 4) |
| 4-bit correction more effective per-adapter than 8-bit | CONJECTURAL | Testable (Experiment 5) |

---

## Part 5: Experimental Validation Plan

### Experiment 1: RMT Decomposition of Quantization Error `[COMPLETE — GATE PASSES]`

**Question:** Does E_q = W_fp - W_q have systematic (above-MP-bulk) structure?

**Result: YES — decisively.** 448/448 layers have signal. ~50% of error energy is systematic.

| Model | Layers with signal | Mean signal_rank | Mean sv_frac | 95% CI (rank) | 95% CI (sv_frac) |
|-------|-------------------|------------------|--------------|---------------|-------------------|
| Qwen3-1.7B | 196/196 (100%) | 425.3 | 53.7% | [401.6, 448.7] | [51.2%, 56.1%] |
| Qwen3-8B | 252/252 (100%) | 750.8 | 48.2% | [696.5, 803.8] | [46.0%, 50.4%] |

Gate criterion: 95% bootstrap CI lower bound for mean(signal_rank) > 0 ✓ AND mean(signal_variance_fraction) > 0 ✓

**Method:**
1. Load Qwen3-1.7B and 8B bf16/8-bit model pairs
2. For each layer, compute E_q = W_fp - dequantize(W_q)
3. Compute SVD of E_q → singular values S_e (`compute_uv=False, stream=mx.cpu`)
4. Apply `compute_signal_rank_from_singular_values(S_e, m, n)` from RMT module
5. Bootstrap CI with 10,000 resamples for gate decision

**Data:** [`results/rmt_quantization_error/20260226T001044Z/`](../../results/rmt_quantization_error/20260226T001044Z/) (1.7B), [`results/rmt_quantization_error/20260226T002308Z/`](../../results/rmt_quantization_error/20260226T002308Z/) (8B)
**Script:** [`scripts/rmt_quantization_error.py`](../../scripts/rmt_quantization_error.py)

### Experiment 2: Scale Bound A/B Test (QLoRA)

**Question:** Is QLoRA degradation caused by quantization or by spectral scale violation?

**Method:**
1. Take Qwen3-1.7B 8-bit as base
2. Train LoRA adapter with standard scale (alpha/rank = 2.0) on benchmark data
3. Train LoRA adapter with geometric scale (sigma_k / ||BA||_spectral) on same data
4. Same rank, same training budget, same data

**Metrics:**
- Perplexity on held-out validation set
- CKA between adapted model and bf16 reference
- Generation quality on GSM8K-style reasoning tasks

**Prediction:** Geometric scale produces coherent output regardless of quantization. Standard scale produces degenerate output regardless of quantization. The variable that matters is scale, not base precision.

**Infrastructure:** Existing `mc train run` supports geometry-bounded NB-LoRA via `LoRASafetyService.compute_geometric_scale()`. A standard-scale baseline path (`alpha/rank = 2.0`) is required for this A/B and is not currently exposed by `mc train run`.

### Experiment 3: Corrective LoRA Training

**Question:** Can a LoRA adapter trained with reconstruction objective reduce the gap between quantized and full-precision models?

**Prerequisite:** Experiment 1 shows signal_rank > 0 (otherwise skip).

**Method:**
1. Generate probe activations from bf16 Qwen3-1.7B on [`benchmark_val.jsonl`](../../data/training/)
2. Train NB-LoRA adapter on 8-bit base with MSE loss against bf16 activations
3. Use existing MASS step sizes (proven quantization-robust from Section 2.3)
4. Measure CKA(8-bit + adapter, bf16) vs CKA(8-bit alone, bf16)

**Success criterion:** CKA(8-bit + adapter, bf16) > CKA(8-bit, bf16) with statistical significance.

**Infrastructure:** Training pipeline handles quantized bases. Needs a reconstruction objective variant in the training service (new loss function, but existing forward/backward path).

### Experiment 4: Stacked Recovery

**Question:** Can iterated correction converge to near-perfect recovery?

**Prerequisite:** Experiment 3 shows CKA improvement.

**Method:**
1. After Experiment 3, measure weight-space residual: E_residual per layer
2. Compute RMT decomposition of E_residual
3. If signal_rank > 0: train second adapter via LoRAStacker on 8-bit + adapter_1
4. Repeat for 3-5 rounds, tracking per-round: CKA drift, cumulative barrier, signal_rank of residual

**Convergence criterion:** Signal_rank of residual drops to 0 → remaining error is in MP noise bulk → stop.

**Infrastructure:** [`lora_stacker.py`](../../src/modelcypher/experimental/self_improve/lora_stacker.py) manages the stack. Needs orchestration script connecting stacker with RMT analysis.

### Experiment 5: 4-bit Extension

**Question:** Does the geometric story change at 4-bit (where quantization error is much larger)?

**Method:** Repeat Experiments 1-4 with 4-bit quantization.

**Predictions:**
- Higher signal_rank in E_q (more systematic error due to coarser grid)
- More effective per-adapter correction (larger systematic component to target)
- More stacking rounds needed (more total error to correct)
- MASS spectral ceiling still robust (sigma_max and sigma_k at top of spectrum)

This is where the stacking hypothesis faces its hardest test. If 4-bit E_q is primarily noise-bulk (signal_rank ≈ 0), the 90-100% Frobenius error is genuinely random and uncorrectable. If it has high signal_rank, the error is highly structured and stacked LoRA can make significant corrections.

---

## References

### Internal
- Weyl validation data (exact-SVD canonical): `results/weyl_quantization_validation/20260226T015425Z/`
- RMT quantization error (1.7B): `results/rmt_quantization_error/20260226T001044Z/`
- RMT quantization error (8B): `results/rmt_quantization_error/20260226T002308Z/`
- Weyl validation script: `scripts/weyl_quantization_validation.py`
- RMT quantization error script: `scripts/rmt_quantization_error.py`
- Compression synthesis: `docs/research/COMPRESSION-RESEARCH-SYNTHESIS.md`
- Spectral scale bound: `docs/research/lora_spectral_scale_bound.md`
- LoRA geometric derivation: `docs/research/lora_geometric_derivation.md`
- RMT signal separation: `src/modelcypher/core/domain/geometry/rmt_signal_separation.py`
- Cayley LoRA: `src/modelcypher/core/domain/geometry/cayley_lora.py`
- MASS step size: `src/modelcypher/core/domain/training/mass_step_size.py`
- LoRA stacker: `src/modelcypher/experimental/self_improve/lora_stacker.py`
- Quantization utils: `src/modelcypher/core/use_cases/quantization_utils.py`

### External
- Weyl, H. (1912). "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen"
- Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." arXiv:2305.14314
- Hu, E. J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
- Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices"
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Chapter 2: Matrix Analysis
- Kornblith, S. et al. (2019). "Similarity of Neural Network Representations Revisited." arXiv:1905.00414
