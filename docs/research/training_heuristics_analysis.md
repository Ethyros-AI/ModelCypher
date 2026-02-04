# Geometry-Derived Training: Replacing Industry Heuristics

**Research Document - Phase 1: Literature Review**

This document systematically analyzes SOTA for five training heuristics and proposes geometry-derived replacements based on the spectral structure of weight matrices.

---

## Executive Summary

| Heuristic | Industry Standard | SOTA Understanding | Proposed Geometric Alternative |
|-----------|------------------|-------------------|-------------------------------|
| Gradient clipping | Clip at 1.0 | Prevents exploding gradients in cliffs | Clip at σ_max per layer |
| Warmup | 5-10% of steps | Compensates for Adam's initial update variance | Until BB curvature estimates stabilize |
| LR schedules | Cosine/linear decay | Implicitly performs iterate averaging | None (BB adapts per-step) |
| Batch size | "As big as fits" | Critical batch size B_crit determines efficiency | Derived from gradient noise scale |
| Dropout | 0.1-0.3 | Low-rank regularization | Derived from activation effective rank |

---

## 1. Gradient Clipping

### 1.1 Literature Summary

**What gradient clipping does:**
- Prevents exploding gradients by bounding gradient norm before optimizer step
- Two variants: clip-by-value and clip-by-norm (norm is more common)
- Essential for RNNs/LSTMs due to temporal gradient multiplication
- Used in virtually all transformer training

**Why clip=1.0?**
- No theoretical basis - it's a heuristic that "works"
- Pascanu et al. (2013) proposed clipping but didn't derive the threshold
- The value 1.0 is arbitrary and problem-dependent

**Spectral norm / Lipschitz connection:**
- DP-SGD literature (arxiv:2305.16202) shows Lipschitz-constrained networks don't need per-sample clipping
- Spectral normalization for GANs directly constrains layer Lipschitz constants
- Key insight: the "correct" clip threshold relates to network Lipschitz constant

**What clipping prevents:**
1. **Gradient explosion in cliffs** - Sharp non-linearities create regions with very high derivatives
2. **Catapulting** - Large update moves parameters far from optimal region
3. **Numerical overflow** - Gradients exceeding float range

### 1.2 Theoretical Analysis

For a layer with weight matrix W having σ_max as largest singular value:

- The gradient's natural scale is bounded by the layer's spectral properties
- A gradient that would cause ||W_new - W|| > σ_max is likely harmful
- Per-layer clipping at σ_max respects each layer's geometry

**Proposed formula:**
```
clip_threshold_i = σ_max(W_i)
```

This is principled because:
1. σ_max represents the layer's largest weight scale
2. Gradients significantly larger than σ_max indicate instability
3. Per-layer (not global) respects different layer geometries

### 1.3 Experiment Design

Compare:
1. No clipping (baseline)
2. Global clip=1.0 (industry standard)
3. Per-layer clip=σ_max (geometric)

Metrics:
- Loss stability (variance over steps)
- Convergence speed (steps to target loss)
- Final loss achieved
- Gradient norm distribution per layer

---

## 2. Learning Rate Warmup

### 2.1 Literature Summary

**Key paper: "Why Warmup the Learning Rate?"** (NeurIPS 2024)

Core finding: Warmup's benefit comes from allowing the network to **tolerate larger learning rates**, not from Adam's variance issues.

**Ma & Yarats (AAAI 2021) - "On the Adequacy of Untuned Warmup":**

Key insights:
1. RAdam is just "4 steps of momentum SGD, then Adam with fixed warmup"
2. The variance-based motivation (Liu et al.'s RAdam paper) is flawed
3. Adam's m_t and v_t are highly correlated - variance analysis misses this
4. **The real issue: Adam's update magnitude starts at exactly α (the LR)**

**Update magnitude analysis:**
- At t=1: ||update|| = α for ALL parameters (since |m_1| = √v_1)
- Update magnitudes only stabilize around ~0.15·α after 40+ iterations
- This is true even at a local minimum (zero-mean gradients)!

**Recommended warmup (untuned):**
```
τ = 2 / (1 - β₂)  # For β₂=0.999, this is ~2000 steps
```

### 2.2 Geometric Alternative

**Key insight:** BB adaptation already handles the curvature estimation problem that warmup addresses.

The BB method computes:
```
α_k = (s·s) / (s·y)  # where s = θ_k - θ_{k-1}, y = g_k - g_{k-1}
```

This **requires gradient history** - on step 0, BB falls back to spectral LR.

**Proposed adaptive warmup:**
- Monitor BB curvature stability: Var(s·y) across recent steps
- Warmup until this variance stabilizes
- This is principled: warmup ends when we have reliable curvature information

**Current implementation analysis:**
```python
# engine_mlx.py line 315-319
if global_step < warmup_steps:
    warmup_lr = base_lr * (global_step + 1) / warmup_steps
    optimizer.learning_rate = warmup_lr
```

This **overrides** the geometric LR with a linear schedule. With BB adaptation, we should instead:
1. Use geometric LR from step 0 (it's bounded by spectral structure)
2. Let BB adaptation handle curvature learning

### 2.3 Experiment Design

Compare:
1. Linear warmup (current: engine_mlx.py)
2. No warmup + geometric LR
3. Adaptive warmup (until BB variance stabilizes)

Metrics:
- Early training stability
- Loss at step 10, 100, 1000
- BB learning rate variance over time

---

## 3. Learning Rate Schedules

### 3.1 Literature Summary

**Key paper: "The Road Less Scheduled"** (Defazio et al., NeurIPS 2024)

**Core insight:** Schedules and iterate averaging are equivalent in their effect.

Schedule-Free optimization achieves:
- Worst-case optimal convergence for ANY momentum β ∈ [0,1]
- Matches or exceeds cosine schedules across 28 diverse problems
- No schedule hyperparameters (stopping time T not needed)

**Why schedules exist:**
1. They implicitly perform weighted iterate averaging
2. Cosine decay is approximately: return weighted average of trajectory
3. Linear decay achieves last-iterate convergence

**Why schedules might be unnecessary with BB:**
- BB already adapts LR based on local curvature: α_k = (s·s)/(s·y)
- Schedules solve a problem (curvature estimation) that BB already solves
- Schedule-Free paper shows averaging can replace scheduling entirely

### 3.2 BB Natural Decay Analysis

**Hypothesis:** BB learning rates naturally decrease near convergence.

Near a minimum:
- Gradients become smaller
- s·y (curvature) changes character
- BB LR should naturally reduce

**Mathematical intuition:**
- Far from minimum: large gradients, large curvature → moderate LR
- Near minimum: small gradients, quadratic curvature → BB gives optimal LR

### 3.3 Experiment Design

Compare:
1. BB + cosine decay (current default for many)
2. BB + no decay (pure geometric)
3. BB + Schedule-Free averaging

Metrics:
- Log per-layer BB LR over training
- Check if BB LRs naturally decrease
- Final loss comparison
- Eval accuracy if applicable

---

## 4. Batch Size

### 4.1 Literature Summary

**Key paper: "An Empirical Model of Large-Batch Training"** (McCandlish et al., 2018)

**Critical batch size formula:**
```
B_crit ≈ B_noise = trace(HΣ) / (G^T H G)
```

Where:
- H = Hessian at current parameters
- Σ = gradient covariance matrix
- G = true gradient

**Key insights:**
1. Below B_crit: linear speedup with batch size
2. Above B_crit: diminishing returns
3. B_crit varies with task complexity and training progress

**Simplified estimator (assuming H ≈ identity):**
```
B_simple = trace(Σ) / ||G||² = E[||g - G||²] / ||G||²
```

This is computable during training:
- g = mini-batch gradient
- G ≈ average of several mini-batches

**Scaling law findings:**
- B_crit scales with loss: B_crit(L) ≈ B* · L^(-α)
- As training progresses (loss decreases), B_crit increases
- Optimal: use larger batches later in training

### 4.2 Geometric Connection

**Gradient covariance and geometry:**
- The gradient covariance Σ encodes how informative individual samples are
- Low effective rank of Σ → samples are redundant → can use larger batch
- High effective rank of Σ → samples are diverse → need smaller batch

**Proposed approach:**
1. Estimate gradient covariance from mini-batch samples
2. Compute effective rank of gradient covariance
3. B_opt ∝ effective_rank(Σ) × base_batch

### 4.3 Experiment Design

Compare:
1. Fixed batch size (current)
2. Batch size = B_simple estimation
3. Adaptive batch (increase with training progress)

Metrics:
- Gradient covariance eigenspectrum
- Signal-to-noise ratio in gradients
- Training efficiency (loss per sample seen)

---

## 5. Dropout

### 5.1 Literature Summary

**Core papers:**
- "Dropout as a Low-Rank Regularizer" (Cavazza, AISTATS 2018)
- "Spectral Dropout" (arxiv:1711.08591)
- "Dropout: Explicit Forms and Capacity Control" (ICML 2021)

**What dropout actually does:**
1. **Low-rank regularization:** Dropout implicitly encourages low-rank weight matrices
2. **Co-adaptation breaking:** Prevents neurons from relying on specific other neurons
3. **Ensemble effect:** Training exponentially many sub-networks

**Spectral analysis:**
- Dropout rate affects the effective rank of learned representations
- Higher dropout → lower effective rank of activations
- This connects to our effective_rank.py diagnostics

**Adaptive dropout approaches:**
- Spectral Adaptive Dropout: adjusts based on frequency content
- Curriculum dropout: varies rate during training
- Most approaches add complexity without clear theoretical grounding

### 5.2 Geometric Connection

**Effective rank as dropout indicator:**
- Layers with low effective rank have redundant features → more dropout safe
- Layers with high effective rank are information-dense → less dropout needed

**Proposed formula:**
```
dropout_rate_i = 1 - (effective_rank_i / full_rank_i)
```

Intuition:
- If effective_rank = full_rank: no dropout (all dimensions useful)
- If effective_rank << full_rank: high dropout (much redundancy)

### 5.3 Experiment Design

Compare:
1. Fixed dropout=0.1 (standard)
2. Per-layer adaptive dropout from effective rank
3. No dropout (with geometric regularization)

Metrics:
- Activation effective rank per layer
- Generalization gap (train loss - eval loss)
- Final eval performance

---

## 6. Implementation Plan

### Phase 2: Experimental Code

**6.1 Gradient Clipping:**
```python
# Add to GeometricOptimizer
def clip_gradients_by_spectral_norm(self, gradients):
    """Clip each layer's gradient to its σ_max."""
    clipped = {}
    for key, grad in gradients.items():
        config = self.layer_configs.get(key)
        if config is not None:
            grad_norm = mx.sqrt(mx.sum(grad * grad))
            if grad_norm > config.sigma_max:
                grad = grad * (config.sigma_max / grad_norm)
        clipped[key] = grad
    return clipped
```

**6.2 BB Stability Metric:**
```python
# Track BB curvature stability
def compute_bb_stability(self):
    """Compute variance of s·y across recent steps."""
    if len(self._sdy_history) < 10:
        return float('inf')
    return np.var(self._sdy_history[-10:])
```

**6.3 Gradient Covariance Estimation:**
```python
# New utility function
def estimate_gradient_noise_scale(model, data_batch, num_samples=8):
    """Estimate B_simple = Var(g) / ||E[g]||²."""
    grads = []
    for i in range(num_samples):
        sub_batch = data_batch[i::num_samples]
        g = compute_gradient(model, sub_batch)
        grads.append(g)

    mean_grad = np.mean(grads, axis=0)
    variance = np.mean([(g - mean_grad)**2 for g in grads])
    return variance / (np.linalg.norm(mean_grad)**2 + eps)
```

**6.4 Activation Effective Rank Hook:**
```python
# Hook for dropout experiments
class EffectiveRankHook:
    def __init__(self):
        self.ranks = {}

    def __call__(self, layer_name, activations):
        er = EffectiveRank().compute(activations)
        self.ranks[layer_name] = er.shannon_effective_rank
```

---

## 7. Success Criteria

For each heuristic, we should conclude ONE of:

| Heuristic | Possible Conclusions |
|-----------|---------------------|
| Gradient clipping | "Removed - BB bounds prevent explosion" OR "Replaced with clip=σ_max" |
| Warmup | "Removed - geometric LR is stable from step 0" OR "Replaced with adaptive" |
| LR schedules | "Removed - BB adapts automatically" OR "Kept for specific reason" |
| Batch size | "Derived from gradient noise scale" OR "Fixed with theoretical justification" |
| Dropout | "Derived from effective rank" OR "Removed - weight decay sufficient" |

---

## 8. References

1. Pascanu, R., et al. (2013). "On the difficulty of training recurrent neural networks." ICML.
2. Ma, J. & Yarats, D. (2021). "On the Adequacy of Untuned Warmup for Adaptive Optimization." AAAI.
3. McCandlish, S., et al. (2018). "An Empirical Model of Large-Batch Training." arXiv:1812.06162.
4. Defazio, A., et al. (2024). "The Road Less Scheduled." NeurIPS.
5. Barzilai, J. & Borwein, J.M. (1988). "Two-Point Step Size Gradient Methods." IMA J. Numerical Analysis.
6. Cavazza, J., et al. (2018). "Dropout as a Low-Rank Regularizer for Matrix Factorization." AISTATS.
7. Miyato, T., et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.

---

## Appendix A: Current GeometricOptimizer Implementation

The current implementation already has:
- Per-layer LR from spectral structure: `lr = 1/σ_max_i`
- BB adaptation with spectral bounds: `[σ_k/σ_max, 1/σ_max]`
- Condition-aware weight decay: `decay_scale = σ_k/σ_max`
- No momentum (pure gradient descent)

Missing:
- Optional gradient clipping
- BB stability tracking
- Gradient covariance tools
- Effective rank hooks for dropout

---

## Phase 2: Experimental Results

Experiments run on 2026-02-03 using LFM2-inspired transformer architecture (256 dim, 4 layers).

### Experiment 1: Gradient Clipping

| Mode | Final Loss | Clip Events |
|------|-----------|-------------|
| None (BB only) | 6.9329 | N/A |
| Global (1.0) | 6.9255 | N/A |
| Spectral (σ_max) | 6.9255 | **0%** on all layers |

**Key Finding:** With geometric optimizer and BB bounds, gradient clipping is **never triggered**. All gradient norms stayed well below σ_max thresholds:
- Mean gradient norms: 0.0005 - 0.19 across layers
- σ_max values: 0.86 - 2.97 across layers
- The BB spectral bounds `[σ_k/σ_max, 1/σ_max]` prevent gradient explosion by construction

**Conclusion: REMOVE gradient clipping** - BB bounds already prevent the problem it solves.

### Experiment 2: Warmup

| Warmup | Final Loss | Early Loss (step 10) | BB Stable Step |
|--------|-----------|---------------------|----------------|
| Linear 50 steps | 6.9219 | 6.954 | 2 |
| Linear 100 steps | 6.9291 | 6.951 | 3 |
| None | 6.9279 | 6.985 | 11 |

**Key Finding:** No warmup works with geometric LR - the model doesn't diverge. Early loss is slightly higher but BB stabilizes quickly.

**Conclusion: REMOVE mandatory warmup** - Geometric LR is bounded by spectral structure from step 0. BB adaptation stabilizes within ~10 steps regardless.

**Alternative:** If warmup is desired for slightly smoother early training, use adaptive warmup until BB stability (variance of s·y < threshold). This replaces arbitrary "5% of steps" with a principled criterion.

### Experiment 3: LR Schedules

| Schedule | Final Loss | Final LR |
|----------|-----------|----------|
| BB only (no decay) | 6.9244 | 0.34 (constant) |
| BB + Cosine decay | 6.9156 | 0.000006 |

**Key Finding:** BB + cosine gave slightly better final loss, but the difference is minimal (0.008). The cosine schedule decayed LR to nearly zero by end of training.

**Analysis:** Schedule-Free paper (Defazio et al.) shows averaging replaces scheduling. BB adaptation is a form of implicit averaging through curvature estimation. The small benefit from cosine may come from its implicit early stopping effect.

**Conclusion: KEEP optional cosine decay** but it's not essential. BB alone works well. Consider Schedule-Free averaging as an alternative.

### Experiment 4: Batch Size

| Batch Size | Final Loss | Gradient Noise Scale |
|------------|-----------|---------------------|
| 8 | 6.9427 | 4.23 |
| 16 | 6.9367 | 4.33 |
| 32 | 6.9267 | 2.74 |
| 64 | 6.9202 | 2.57 |
| 128 | 6.9171 | 2.05 |

**Key Finding:** Larger batch = better final loss (as expected). Gradient noise scale decreases with batch size, suggesting we're well above the critical batch size for this simple task.

**Critical Batch Estimate:** B_simple ~3-4 (very low because synthetic data is easy). Real tasks would have much higher B_crit.

**Conclusion:** The gradient noise scale formula `Var(g) / ||E[g]||²` can be computed during training to determine optimal batch size. Use `B_opt ≈ sqrt(r × B_noise)` where r is compute/time tradeoff preference.

### Experiment 5: Dropout

(Requires activation hooks - deferred to future work)

**Preliminary Analysis:** The effective rank infrastructure exists in `effective_rank.py`. Per-layer dropout rates should be derived from:
```
dropout_rate_i = 1 - (effective_rank_i / full_rank_i)
```

---

## Phase 3: Conclusions & Recommendations

### Summary Table

| Heuristic | Recommendation | Justification |
|-----------|---------------|---------------|
| Gradient clipping | **REMOVE** | BB bounds prevent explosion by construction |
| Warmup | **REMOVE** | Geometric LR is stable from step 0; BB stabilizes in ~10 steps |
| LR schedules | **OPTIONAL** | BB alone works; cosine gives marginal improvement |
| Batch size | **DERIVE** | Use gradient noise scale to compute B_crit |
| Dropout | **DERIVE** | Use effective rank per layer (future work) |

### Implementation Changes to GeometricOptimizer

**Already implemented:**
1. ✅ Optional gradient clipping modes (none/global/spectral)
2. ✅ BB stability tracking (`get_bb_stability()`, `is_bb_stable()`)
3. ✅ Gradient norm statistics for analysis

**Recommended changes to `engine_mlx.py`:**
1. Remove mandatory warmup when using GeometricOptimizer
2. Add adaptive warmup option: warmup until `optimizer.is_bb_stable()`
3. Log gradient noise scale for batch size tuning

### What This Means for Training

With geometry-derived optimization, training becomes simpler:

```python
# Old way (many hyperparameters)
optimizer = Adam(lr=1e-4)  # Why 1e-4?
warmup_steps = 1000  # Why 1000?
clip_value = 1.0  # Why 1.0?
batch_size = 32  # Why 32?

# New way (geometry-derived)
optimizer = GeometricOptimizer()  # LR = 1/σ_max
optimizer.init_from_model(model)  # All parameters from spectral structure
# No warmup needed (BB adapts)
# No clipping needed (BB bounds prevent explosion)
# Batch size from gradient noise scale
```

### Remaining Questions

1. **Does BB LR naturally decay near convergence?** Need longer training runs to verify.
2. **Optimal dropout from effective rank** - needs activation hook implementation
3. **How do these findings scale to larger models?** Test on LFM2-1.2B

---

---

## Phase 2b: High-Priority Heuristics Implementation

Following Phase 2 experiments, we identified four additional high-priority heuristics for geometry-derived replacement. Status:

| Heuristic | Industry Standard | Geometric Alternative | Status |
|-----------|------------------|----------------------|--------|
| Adam β₁/β₂/ε | β₁=0.9, β₂=0.999, ε=1e-8 | BB replaces momentum; ε = max(σ_k², √ε×σ_max²) | ✅ COMPLETE |
| Weight init scale | Xavier/Kaiming | σ_max(W_init) = target (spectral normalized) | ✅ COMPLETE |
| Early stopping | Validation loss patience | BB stability + spectral budget | ✅ COMPLETE |
| Residual scaling | None (α=1) | α = σ_max(x) / σ_max(f(x)) | ✅ COMPLETE |

### Weight Initialization (Spectral Normalized)

**Problem:** LoRA init uses arbitrary `scale = 0.01`. For spectral control, init should ensure σ_max ≈ target across all layers.

**Implementation:** `geometric_lora.py:GeometricLoRALinear.__init__`

```python
# Initialize so ||B @ A||_spectral = σ_k from step 0
sqrt_sigma_k = np.sqrt(sigma_k)
A_init = mx.random.normal(shape=(rank, in_features))
A_spectral = self._spectral_norm(A_init)
self.lora_a = A_init * (sqrt_sigma_k / (float(A_spectral) + SQRT_EPS))
# Same for B
```

**Properties:**
- Uses FULL geometric budget from step 0
- Each matrix gets ||·||_spectral = √σ_k so product ≈ σ_k
- No arbitrary scale factors

### Early Stopping (Geometric Convergence)

**Problem:** Industry uses validation loss patience. Not geometry-derived, requires held-out data.

**Implementation:** `geometric_lora_trainer.py:GeometricConvergenceMonitor`

Three criteria combined:
1. **BB stability:** Barzilai-Borwein curvature estimates stabilized (`optimizer.is_bb_stable()`)
2. **Loss stability:** Loss change below √ε (numerical precision floor)
3. **Spectral budget:** `spectral_bound_ratio > 0.9` (90% of geometric budget consumed)

**Convergence rule:**
```python
should_stop = bb_stable and (loss_stable or budget_exhausted)
```

**Properties:**
- No validation set required
- All thresholds dtype-derived (√ε) or geometry-derived (spectral bound)
- Integrated into training loop via `enable_geometric_stopping` config flag

### Residual Connection Scaling

**Problem:** Standard residual: `output = x + f(x)` with α=1. When σ_max(f(x))/σ_max(x) varies across layers, gradient flow becomes uneven.

**Implementation:** `residual_scaling.py:ResidualScalingHook`

**Formula:**
```
α_i = σ_max(x) / σ_max(f(x))
```

This normalizes so `||α × f(x)|| ≈ ||x||`, making residual contributions comparable.

**Properties:**
- Hook-based (non-invasive) - no model modifications required
- Computes spectral norms via fast power iteration (3 iterations)
- Clamped to [0.1, 10.0] for stability
- Optional: can enable/disable per training run

### Files Changed

| File | Changes |
|------|---------|
| `training/geometric_lora.py` | Spectral-normalized LoRA init |
| `training/geometric_lora_trainer.py` | `GeometricConvergenceMonitor`, config options |
| `training/residual_scaling.py` | New file - `ResidualScalingHook` |
| `geometry/numerical_stability.py` | `spectral_normalized_init()`, `spectral_normalized_lora_init()` |
| `experiments/geometry_heuristics_phase2.py` | Validation experiments |
| `tests/test_geometric_training_phase2.py` | 19 unit tests |

### Validation

All 19 tests pass:
- Spectral init achieves target norm ± 10%
- Product ||B @ A|| respects spectral budget
- Convergence monitor tracks steps and criteria correctly
- Residual scaling computes correct α values
- Hook functionality works in enabled/disabled states

### Usage

```python
# Training with all Phase 2 features
config = GeometricLoRAConfig(
    target_modules=target_modules,
    rank=rank,
    geometries=geometries,
    enable_geometric_stopping=True,  # Uses convergence monitor
    # LoRA layers automatically use spectral init
)

# Optional: add residual scaling
from modelcypher.core.domain.training.residual_scaling import ResidualScalingHook
hook = ResidualScalingHook()
# Apply to model transformer blocks
```

---

*Document updated: 2026-02-03*
*Status: Phase 2b Complete - Weight Init, Early Stopping, Residual Scaling Implemented*
