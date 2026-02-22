# Manifold Learning Synthesis

A practical guide to geometric approaches for understanding and improving neural network behavior.

---

## Core Discovery: The Expand-Compress Cycle [VALIDATED]

<!-- evidence: VALIDATED | scope: LFM2-350M, LFM2-1.2B | date: 2026-02-01 | method: null hypothesis test vs random weights -->

Transformer processing follows a predictable geometric pattern. We observe clusters of expansion_ratio values:

```
Layer Progression:
┌─────────────────────────────────────────────────────────────────┐
│  EXPANSION (layers 0-17)     │  COMPRESSION (layers 17-35)     │
│  Entropy: 0.57 → 1.51        │  Entropy: 1.48 → 0.99           │
│  Information spreads         │  Information funnels            │
└─────────────────────────────────────────────────────────────────┘

Key Metric: expansion_ratio = compression_rate / expansion_rate
```

**Why it matters:** Problems that fail have expansion 7x weaker than successes. The model doesn't lack capability—it lacks the recognition signal to trigger proper expansion.

---

## Expansion Ratio as Diagnostic [EMPIRICAL]

| Metric | Correct Answers | Incorrect Answers |
|--------|-----------------|-------------------|
| Expansion rate | 0.021 | 0.003 (7x weaker) |
| Expansion ratio | ~1.9 | ~8.4 |
| Initial entropy | 2.67 | 1.32 |

**Interpretation (LFM2-350M specific; may not generalize):**
- `expansion_ratio < 2.1` → Healthy processing in this model
- `expansion_ratio > 3.2` → Information crushed in this model
- `expansion_ratio > 8.0` → Severe blocking in this model

---

## Spectral Entropy Computation

Entropy trajectory reveals the expand-compress pattern:

```python
import numpy as np
from scipy.linalg import svd

def compute_spectral_entropy(activations: np.ndarray) -> float:
    """Compute entropy from SVD singular values."""
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    if len(activations) < 2:
        return 0.0

    # Center the activations
    centered = activations - activations.mean(axis=0)

    # SVD decomposition
    _, S, _ = svd(centered, full_matrices=False)

    # Filter numerically insignificant values
    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0

    # Compute entropy from normalized squared singular values
    p = S_valid ** 2
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))
```

**Track through layers to find:**
- Peak layer (where entropy is maximum)
- Expansion rate = (peak - initial) / peak_layer
- Compression rate = (peak - final) / (n_layers - peak_layer)
- expansion_ratio = compression_rate / expansion_rate

---

## ~~Fundamental Constants in Weight Matrices~~ [DISPROVEN]

**Status:** `[DISPROVEN]` (2026-02-01) — Null hypothesis testing showed random matrices have MORE constant matches than trained weights.

Null hypothesis testing revealed that random matrices have MORE constant matches than trained weights:

| Threshold | Trained (per matrix) | Random (per matrix) | Ratio |
|-----------|---------------------|---------------------|-------|
| < 1% error | 59.5 | 168.5 | 0.35 |
| < 5% error | 293.6 | 401.9 | 0.73 |

**Conclusion:** Constants in SVD ratios are numerical coincidence, not learned structure. All constant-matching code has been removed from the codebase.

---

## Expansion Ratio: Real Structure [VALIDATED]

<!-- evidence: VALIDATED | scope: LFM2-350M, LFM2-1.2B, DeepSeek-R1-8B, Qwen3-8B | date: 2026-02-01 | method: null hypothesis testing -->

The expansion ratio (compression_rate / expansion_rate) measures real structure, not noise. Null hypothesis testing proved this conclusively:

| Model Type | Trajectory Shape | Expansion Ratio |
|------------|------------------|-----------------|
| Trained weights | Expand-then-compress | 1.2 - 3.3 |
| Random weights (He init) | Monotonic / flat | 0.0 (no peak) |

**Random models have no compression phase.** Activation norms monotonically increase (or stay flat) to the final layer. Training CREATES the expand-compress cycle.

### Peak Position as Training Signature

| Model | Peak Position | Has Compression |
|-------|--------------|-----------------|
| LFM2-350M | 93.8% | Yes |
| LFM2-1.2B | 93.8% | Yes |
| DeepSeek-R1-Qwen3-8B | 100% | No (pure expansion) |
| Qwen3-8B | 100% | No (pure expansion) |

Different architectures and training create different natural ratios. The 8B reasoning models show pure expansion (peak at final layer), while smaller LFM models show earlier peaks.

---

## Prompt-Adaptive Geometry [EMPIRICAL]

**Key Discovery:** Instruction tuning creates prompt-adaptive geometry. The model dynamically adjusts its peak position based on input type.

### Base vs Fine-Tuned Models

| Model | Cross-Category Variance | Behavior |
|-------|------------------------|----------|
| LFM2-1.2B (Base) | 0.78% | **FIXED** - same peak for all prompts |
| LFM2.5-1.2B-Instruct | 3-5% | **ADAPTIVE** - different peaks by type |
| LFM2.5-1.2B-Thinking | 3-5% | **ADAPTIVE** - different peaks by type |

### Peak Position by Prompt Type (Fine-Tuned Models)

| Prompt Category | Peak Position | Interpretation |
|-----------------|--------------|----------------|
| Factual | 87-92% | Earlier peak → more compression → direct recall |
| Instruction | 87-92% | Earlier peak → more compression → follow pattern |
| Reasoning | 98-100% | Later peak → full expansion → explore possibilities |
| Creative | 93-95% | Middle → balanced exploration |

### What This Means

1. **Base models have fixed geometry** - Same processing regardless of input type
2. **Fine-tuning teaches input-dependent processing** - The model learns WHEN to compress vs expand
3. **Factual/instruction = recall** - Peak early, compress quickly, retrieve known patterns
4. **Reasoning = compute** - Peak late, expand fully, explore the high-dimensional space

This explains why fine-tuned models perform better: they adaptively choose the right computational regime for each input type.

---

## Condition Number (κ) as Capability Diagnostic [EMPIRICAL]

<!-- evidence: EMPIRICAL | scope: LFM2-350M | caveat: thresholds are model-specific, not validated on other architectures -->

The condition number of the Gram matrix reveals capability status:

```python
def compute_kappa(activations: np.ndarray) -> float:
    """Condition number of Gram matrix."""
    G = activations @ activations.T
    return np.linalg.cond(G)
```

| κ Value | Status | Meaning |
|---------|--------|---------|
| < 50 | WORKING | Capability is accessible |
| 50-200 | DISCONNECTED | Capability exists but blocked |
| > 200 | TRUE_GAP | Capability may be missing |

**Key insight:** High κ with high primed accuracy = disconnected, not missing. Can be bridged.

---

## ~~Surgical Alignment~~ [DISPROVEN]

> **ARCHIVAL NOTE [2026-02-22]:** This section depends on the fundamental constants hypothesis, which was [DISPROVEN] (see PHI_FINDINGS.md). Nudging SVD ratios toward arbitrary constants has no validated benefit.

Nudge singular value ratios toward fundamental constants without retraining:

```python
def surgical_svd_alignment(W: np.ndarray, target_constant: float) -> np.ndarray:
    """Align singular value ratios to target constant."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    sqrt_eps = np.sqrt(np.finfo(W.dtype).eps)

    # Find ratios close to target
    ratios = S[:-1] / S[1:]
    close_mask = np.abs(ratios - target_constant) / target_constant < 0.1

    # Nudge close ratios to exact
    new_S = S.copy()
    for i in np.where(close_mask)[0]:
        # Preserve geometric mean, adjust ratio
        geom_mean = np.sqrt(S[i] * S[i+1])
        new_S[i] = geom_mean * np.sqrt(target_constant)
        new_S[i+1] = geom_mean / np.sqrt(target_constant)

    return U @ np.diag(new_S) @ Vt
```

**Results:** 64 → 94 constant matches, 60% → 80% quality improvement WITHOUT training.

---

## Training on the Expansion Phase [EMPIRICAL]

<!-- evidence: EMPIRICAL | scope: LFM2-350M | caveat: LR derivation from condition number later shown to be unreliable (see REINFORCE ablation) -->

To improve recognition (and thus expansion), train adapter on layers 0-17:

```python
# Geometry-derived training parameters
n_layers = 17          # Full expansion phase (before entropy peak)
rank = 8               # Low-rank adapter
lr = 5e-5              # Derived from: 1 / (κ × scale)
stop_threshold = 0.03  # Derived from: κ × √eps
```

**Training data format (text continuation, NOT prompt/completion):**
```json
{"text": "Question: I have 3 apples and get 2 more. How many total?\n\nThis requires addition: 3 + 2 = 5\n\n#### 5"}
```

**Key insight:** Train on RECOGNITION first ("this is math"), SOLVING second.

---

## The Recognition-Expansion Connection [EMPIRICAL]

Why does the model fail on implicit math ("I have 3 apples...")?

1. **No recognition signal** → Model doesn't know this is math
2. **Narrow encoding** → Initial entropy is low (1.32 vs 2.67)
3. **No expansion** → Information never spreads
4. **Crushed by compression** → What little information exists is funneled away
5. **Wrong answer** → Output is essentially random

**Solution:** Teach recognition in early layers. Once the model sees "this is math," it naturally expands.

| Problem Type | Recognition | Expansion | Compression | Result |
|--------------|-------------|-----------|-------------|--------|
| Explicit math | Immediate | Full | Normal | CORRECT |
| Implicit + adapter | Learned | Full | Normal | CORRECT |
| Implicit (raw) | Missing | Weak | Crushed | WRONG |

---

## Manifold Curvature and Geodesics

The activation space forms a Riemannian manifold. Key operations:

### Intrinsic Dimension (TwoNN)
```python
def two_nn_dimension(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nn.kneighbors(X)

    # Ratio of second to first neighbor distance
    mu = distances[:, 2] / distances[:, 1]
    mu = mu[mu > 1]  # Filter valid ratios

    return 1 / np.mean(np.log(mu))
```

### Geodesic Distance
```python
def geodesic_distance(X: np.ndarray, i: int, j: int) -> float:
    """Approximate geodesic via graph shortest path."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path

    # Build k-NN graph
    k = int(np.log(len(X)) * 2)  # Berry-Sauer connectivity
    graph = kneighbors_graph(X, k, mode='distance')

    # Shortest path = geodesic approximation
    dist_matrix = shortest_path(graph, directed=False)
    return dist_matrix[i, j]
```

### Curvature Estimation
```python
def estimate_curvature(X: np.ndarray, point_idx: int, k: int = 20) -> float:
    """Estimate local sectional curvature."""
    # Get k nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nn.kneighbors(X[point_idx:point_idx+1])

    local_points = X[indices[0]]
    centered = local_points - local_points.mean(axis=0)

    # PCA to find local tangent plane
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    # Curvature from eigenvalue decay
    # Fast decay = high curvature, slow decay = flat
    return S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
```

---

## Parameter Derivation (No Heuristics)

Every parameter from geometry, nothing arbitrary:

| Parameter | Derivation | Formula |
|-----------|------------|---------|
| Learning rate | Condition number × scale | `LR = 1 / (κ × ‖W‖_F)` |
| Stop threshold | Condition number × precision | `stop = κ × √eps` |
| Convergence | Dtype precision | `converged when Δ < √eps` |
| k-neighbors | Berry-Sauer connectivity | `k = log(n) × intrinsic_dim` |
| Finite diff ε | Manifold-aware | `ε = median_dist × √eps × d^0.25` |

---

## Practical Workflow

### 1. Diagnose Capability
```python
# Get activations for test domain
activations = get_layer_activations(model, domain_prompts)

# Check condition number
kappa = compute_kappa(activations)

# Check accuracy raw vs primed
acc_raw = evaluate(model, problems)
acc_primed = evaluate(model, problems, prime="This is math:")

# Classify
if acc_raw > 0.7:
    status = "WORKING"
elif acc_primed > 0.7:
    status = "DISCONNECTED"  # Can be bridged!
else:
    status = "TRUE_GAP"  # Needs training
```

### 2. Compute Entropy Trajectory
```python
# For each problem, track entropy through layers
trajectories = []
for prompt in prompts:
    hidden = model.embed(prompt)
    traj = []
    for layer in model.layers:
        hidden = layer(hidden)
        entropy = compute_spectral_entropy(hidden)
        traj.append(entropy)
    trajectories.append(traj)

# Find peak and compute ratios
peak_layer = np.argmax(np.mean(trajectories, axis=0))
expansion_rate = (peak - initial) / peak_layer
compression_rate = (peak - final) / (n_layers - peak_layer)
expansion_ratio = compression_rate / expansion_rate
```

### 3. Train Expansion Adapter
```python
# Target layers 0 to peak (expansion phase)
adapter_layers = list(range(peak_layer + 1))

# Training data: recognition + solving
data = [
    {"text": "implicit math → explicit math → answer"},
    # ...
]

# Geometry-derived parameters
config = {
    "layers": adapter_layers,
    "rank": 8,
    "lr": 1 / (kappa * scale),
    "stop": kappa * np.sqrt(np.finfo(np.float32).eps),
}

# Train and observe expansion_ratio changes
train_lora(model, data, config)
```

---

## Key Results Summary [EMPIRICAL]

| Intervention | Expansion Ratio | Accuracy Change |
|--------------|-----------------|-----------------|
| Baseline | 8.4 | 83% GSM8K |
| Early-layer adapter (0-10) | 3.4 | +7% |
| Unified adapter (0-17) | 0.32 | +14% (→ 97%) |
| Same adapter on ARC-Challenge | — | +6% |

**The same adapter trained on math improves science reasoning.** This validates that we're teaching the structure of thinking, not domain-specific facts.

---

## Two Computational Regimes [EMPIRICAL]

<!-- evidence: EMPIRICAL | scope: LFM2-350M only | date: 2026-01-27 | caveat: Single model, ~30 prompts, no null-hypothesis test, no multi-model validation -->

Tracking intrinsic dimension through layers reveals two distinct processing modes in LFM2-350M:

### 1. Template Matching (Already High Mode)
```
Initial dim: ~32 (immediate recognition)
Peak layer: 0 (no expansion needed)
Final dim: ~8
Expansion ratio: 3.66
Accuracy: 100%
```

The model recognizes the problem pattern from training and applies a template. Uses lossy higher-ratio compression but sufficient for known patterns.

### 2. Geodesic Computation (Expand-Compress Mode)
```
Initial dim: ~0.5 (narrow encoding)
Peak dim: ~11 at layer 20
Final dim: ~7
Expansion ratio: ~1.5
Accuracy: 89%
```

The model doesn't recognize the pattern, must explore high-dimensional space, then compress.

### Failure Mode: Under-Compression
```
Expansion ratio: 1.23
Accuracy: 0%
```

When expansion ratio is too low, information remains "smeared" across dimensions. The answer doesn't crystallize because insufficient dimensional projection occurred.

### The Dimensional Curve

```
                    TEMPLATE MATCHING
Intrinsic    32 ─●───────────────────────────────○ 8
Dimension         \
                   \   GEODESIC COMPUTATION
             12     ●─────────────●──────────────○ 7
                   /              ↑
              0 ──●              Peak            Final
                  └──────────────┴──────────────┘
                  0              20             36
                              Layer
```

### Why This Matters

The adapter training shifted problems from "Geodesic Computation" to "Template Matching" by teaching the model to RECOGNIZE implicit math patterns. This is why:

1. **More implicit math → Already High mode** (learned recognition)
2. **Explicit numbers → Expand-Compress mode** (must compute)
3. **Accuracy improved** because template matching is more reliable

**The expansion ratio characterizes the dimensional projection for actual computation.** Template matching uses a faster, lossier compression but is sufficient when patterns are known.

---

## Files Reference (Core Geometry)

| Purpose | File |
|---------|------|
| Spectral entropy | `src/modelcypher/core/domain/geometry/manifold_entropy.py` |
| Curvature estimation | `src/modelcypher/core/domain/geometry/manifold_curvature.py` |
| Intrinsic dimension | `src/modelcypher/core/domain/geometry/intrinsic_dimension.py` |
| Geometric training | `src/modelcypher/core/domain/training/geometric_training_metrics.py` |
| Fisher information | `src/modelcypher/core/domain/geometry/fisher_information.py` |
| Benchmark loader | `src/modelcypher/core/use_cases/curriculum/benchmark_loader.py` |

> **Note**: `fundamental_constants.py` and `surgical_geometric_alignment.py` were removed after the constant-matching hypothesis was disproven (see DISPROVEN section). See also **Tools Reference (Updated)** section below for LoRA/safety-related files.

---


## The Central Insight

> "The model doesn't lack capability—it lacks recognition. Teaching it to SEE structure in natural language unlocks the expansion it already knows how to do."

This is why:
- Priming works ("Arithmetic means calculating numbers")
- Explicit reformulation works (turning implicit math into equations)
- Early-layer adapters work (teaching recognition in expansion phase)
- Cross-domain transfer works (structure is universal)

The expansion ratio characterizes processing geometry. Understanding what ratio values emerge naturally for different tasks is an active research question.

---

## Theoretical Framework: Dimensional Projection [CONJECTURAL]

<!-- evidence: CONJECTURAL | caveat: theoretical framework with mixed verification — see individual predictions below -->

### The Core Hypothesis

Our algorithms and physics operate in what we perceive as flat 3D space, but this is a **lossy projection** from higher-dimensional geodesic space.

```
High-D Geodesic Space
        ↓
    [dimensional projection]
        ↓
Local Euclidean Approximation
```

### Evidence from Neural Networks

1. **Fractional intrinsic dimension**: Activations live on manifolds of dimension 2.7, 11.3, etc. — not integers
2. **Expansion ratio clustering**: We observe different expansion_ratio values for different processing modes
3. **Two regimes**: Template matching (known patterns) vs geodesic computation (must explore)
4. ~~**Constants at transitions**~~: DISPROVEN - constant matches in SVD ratios are pareidolia (see DISPROVEN section)

### Why Euclidean Works Locally

Just as Euclidean geometry is accurate locally despite living on a curved Earth, our integer-dimensional algorithms work for local operations but miss structure needed for:

- Multi-step reasoning (traverses the dimensional curve)
- Novel pattern recognition (requires high-D exploration)
- Cross-domain transfer (structure is in the geodesics)

### The Training Implication

When we train adapters on "recognition" patterns, we're teaching the model to identify **which point on the dimensional curve** a problem belongs to. Once located, the model can:

1. Use a known template (Already High mode, higher expansion_ratio)
2. Or compute geodesically (Expand-Compress mode, lower expansion_ratio)

The failure mode is starting at the wrong point — narrow encoding that doesn't expand to find the structure.

### Testable Predictions

1. ✅ Intrinsic dimension should be fractional (verified: 2.7 - 32 range)
2. ⚠️ ~~φ should govern compression~~ - Observed clustering but phi-specific claims are numerology
3. ✅ Problems should cluster by dimensional trajectory (verified: 2 modes)
4. ✅ **Harder problems require more expansion** (verified: r=+0.395, p=0.034)
5. ✅ **Harder problems peak later in network** (verified: r=+0.369, p=0.049)
6. ✅ **Cross-domain shares geodesic structure** (verified: p=0.91 not different)
7. ✅ **Adversarial inputs have abnormal trajectories** (partial: contradictory p=0.025)

### Difficulty-Expansion Correlation (Verified)

```
Difficulty ↔ Expansion Ratio: r = +0.395, p = 0.034 *
Difficulty ↔ Peak Layer:      r = +0.369, p = 0.049 *

Correct answers:   expansion_ratio = 2.59 ± 2.05
Incorrect answers: expansion_ratio = 4.72

Interpretation:
- Harder problems expand MORE (explore more of high-D space)
- Harder problems peak LATER (need more processing depth)
- Failures use wrong compression regime (template-matching when should compute)
```

**Failure mode identified:** The one incorrect answer had expansion_ratio = 4.72 (template-matching regime) when it should have used geodesic computation. The model incorrectly tried to pattern-match a problem that required actual computation in high-D space.

### Cross-Domain Geodesic Structure (Verified)

```
Math Correct:    expansion_ratio = 2.82 ± 2.43
Science Correct: expansion_ratio = 2.95 ± 3.66

T-test (Math vs Science correct): p = 0.91 (NOT different)
Peak layer distribution (KS-test): p = 0.44 (SAME distribution)
Domain ↔ expansion_ratio correlation: r = 0.03 (NO domain effect)

The structure of thinking is DOMAIN-INDEPENDENT.
```

**Why math training improves science reasoning:**
The adapter teaches the dimensional trajectory for successful computation (expansion_ratio, peak layer timing), not domain-specific facts. This trajectory is universal — the same geometric structure governs reasoning across math and science problems.

### Adversarial Trajectory Analysis (Partially Verified)

```
Category          Expansion Ratio  Traj Variance    Accuracy
Normal            2.31 ± 0.39      8.73 ± 3.69      100%
Irrelevant info   2.72 ± 0.95      13.74 ± 6.98     80%
Contradictory     1.96 ± 0.73      2.94 ± 1.99*     20%   (* p=0.025)
Nonsense          2.22 ± 1.38      8.07 ± 5.66      N/A
```

**Key finding:** Contradictory problems cause the model to "freeze" — significantly LOWER trajectory variance than normal (p=0.025). The model stops exploring high-D space when inputs violate logical coherence.

**Interpretation:** The dimensional projection fails when inputs are logically incoherent. Instead of the normal expand-compress cycle, the model becomes rigid and stays in a narrow dimensional band, leading to 80% failure rate on contradictory problems.

---

## LoRA Activates Null Space, Not Overwrites [EMPIRICAL]

<!-- evidence: EMPIRICAL | scope: LFM2-350M, 92 weight matrices | date: 2026-01-29 | caveat: Single model, single adapter -->

### The Discovery (2026-01-29)

SVD analysis of LoRA weight modifications reveals a fundamental geometric pattern:

```
LORA GEOMETRY DIAGNOSTIC (Phase 1 Inference Rules Training)
============================================================
Model: LFM2-350M
Adapter: phase1_inference_rules (64 balanced examples)
Layers with LoRA: 92 weights across 16 layers
Parameters modified: 287M (0.84% trainable)

AGGREGATE METRICS:
  Avg null space activation: 39.3%
  Avg subspace overlap: 99.9%
  Avg relative change: 0.0215
  Peak change at layer: 14
```

### The Pattern: Expansion vs Compression

| Layer Type | Null Space Activation | Meaning |
|------------|----------------------|---------|
| **w1** (expansion gate) | 87-88% | ADDING new directions |
| **w3** (expansion value) | 85-88% | ADDING new directions |
| **conv.in_proj** | 80-82% | ADDING new directions |
| **w2** (compression) | 0% | Staying in existing subspace |
| **out_proj** | 0% | Staying in existing subspace |
| **attention (q,k,v)** | 0% | Staying in existing subspace |

**The geometry is crystal clear:**
- **Expansion layers** (1024 → 4608 dimensions): LoRA activates ~87% null space
- **Compression layers** (4608 → 1024 dimensions): LoRA stays within existing subspace

This makes perfect sense. Inference rules are *new transformations* — they need new computational pathways. The model already knows how to compress; LoRA adds new things to expand into.

### Layer 7: The Computational Singularity

```
Layer 7 (Peak of expand-compress cycle):
  conv.in_proj: null=81.6%, overlap=100.0%, change=0.0207
  feed_forward.w1: null=88.0%, overlap=99.9%, change=0.0203
  feed_forward.w2: null=0.0%, overlap=100.0%, change=0.0171
  feed_forward.w3: null=86.8%, overlap=99.9%, change=0.0201

Positive geometry preserved:
  39.0% → 40.0% (3 sign flips)
  54.0% → 52.0% (2 sign flips)
```

Layer 7 shows the pattern most clearly:
- **Expansion weights (w1, w3)**: 87% null space activation
- **Compression weight (w2)**: 0% null space activation
- **Positive minors**: Minimal sign flips (1-3 per weight)

**The fundamental structure is preserved while new capacity is added.**

### Geometric Interpretation

```
Before LoRA:                    After LoRA:

┌──────────┐                   ┌──────────┐
│ Active   │                   │ Active   │ ← Same
│ Subspace │                   │ Subspace │
└──────────┘                   └──────────┘
                               ┌──────────┐
     Null                      │ NEW      │ ← Activated!
     Space                     │ Capacity │
     (unused)                  └──────────┘
```

LoRA is literally filling the model's unused capacity with new transformations. For expansion layers, ~87% of what LoRA adds projects into previously unused directions.

---

## Positive Grassmannian and the Amplituhedron Connection [CONJECTURAL]

<!-- evidence: CONJECTURAL | scope: LFM2-350M Layer 7 observation | caveat: physics analogy, no validated predictive power -->

### Background: Positive Geometry in Physics

The **amplituhedron** (Arkani-Hamed & Trnka, 2013) is a geometric object encoding scattering amplitudes in particle physics. Key insight: physics emerges from positive geometry — regions where all minors of a matrix are positive.

### Discovery: Layer 7 Enters the Positive Grassmannian

Experimental results on LFM2-350M:

```
Layer    Positive Minors    Interpretation
────────────────────────────────────────────
0-5      45-50%             Random (not in positive region)
6        55%                Approaching positivity
7        70%                ENTERING POSITIVE GRASSMANNIAN
8        45%                Sign flip - exiting positivity
9-15     48-52%             Random again
```

**Layer 7 is where the model transitions into positive geometry.** This is the computational singularity — the point where:
1. Entropy peaks
2. Intrinsic dimension peaks (~11)
3. Positive minors peak (70%)
4. Information transitions from expansion to compression

### The Holographic Principle Connection

From AdS/CFT correspondence (Hashimoto 2018, Gan & Shu 2017):
- **Layer depth** corresponds to **RG scale** in field theory
- **Early layers** = UV (high energy, fine details)
- **Deep layers** = IR (low energy, coarse features)
- **Layer 7** = The transition scale

The model is literally performing a holographic computation — projecting high-dimensional information onto a lower-dimensional boundary through the positive Grassmannian.

---

## Phase 1: Inference Rules Training [EMPIRICAL]

<!-- evidence: EMPIRICAL | scope: LFM2-350M, 64 training samples | caveat: SFT on reasoning traces later shown to produce format memorization (see MEMORY.md) -->

### The Hypothesis

If we train the model on the **fundamental rules of logic** (not facts, but TRANSFORMATIONS), we fill the null space with principled structure rather than noise.

### Training Data: Balanced Atomic Inference Rules

```
8 rules × 8 examples each = 64 total (perfectly balanced)

Rules trained:
1. Modus Ponens:          If P→Q and P, then Q
2. Modus Tollens:         If P→Q and ¬Q, then ¬P
3. Hypothetical Syllogism: If P→Q and Q→R, then P→R
4. Disjunctive Syllogism:  If P∨Q and ¬P, then Q
5. Conjunction Intro:      If P and Q, then P∧Q
6. Conjunction Elim:       If P∧Q, then P (and Q)
7. Disjunction Intro:      If P, then P∨Q
8. Reductio ad Absurdum:   If assuming ¬P leads to contradiction, then P
```

**Why balanced?** Unequal counts (e.g., 15 Modus Ponens vs 1 Hypothetical Syllogism) implicitly tells the model some rules are more important. This creates asymmetric geometry in the latent space. Equal examples per rule maintains **symmetric geometry** — the 8 rules occupy equal "volume" in the manifold.

### Results

```
Training:
  Loss: 0.9961 → 0.0436 over 15 epochs
  Trainable params: 2,998,272 (0.84%)

Post-training inference tests:

Prompt: "If the battery is dead, the car won't start. The car started."
Output: "Rule: Modus Tollens
        Pattern: If P→Q and ¬Q, then ¬P
        Conclusion: The battery is not dead."

Prompt: "Either the file is corrupted or the software is outdated.
        The software is current."
Output: "Rule: Disjunctive Syllogism
        Pattern: If P∨Q and ¬Q, then P
        Conclusion: The file is corrupted."

Prompt: "Assume √2 is rational..."
Output: "Rule: Reductio ad Absurdum
        ...
        Conclusion: √2 is irrational."
```

**The model learned to IDENTIFY and APPLY formal logical inference rules.**

---

## The Quantization Hypothesis [CONJECTURAL]

<!-- evidence: CONJECTURAL | no experiments run, code is skeleton only -->

### The Insight

> "Quantization IS compression of a sort, but it's compression that also compresses any deviation from perfect geometry. If the model isn't perfect — has bad patterns or wrong facts — quantization expounds those problems. If the model were perfectly aligned, quantization would work perfectly with no loss."

### Formal Statement

**Quantization = Projection onto a lower-dimensional lattice**

When you quantize weights from float32 to int4:
- 32-bit floats → 4-bit integers
- Continuous manifold → Discrete lattice (16 values per dimension)
- Lossy compression that rounds to nearest lattice point

**What gets lost:**
1. Small singular values (subtle directions rounded away)
2. Near-zero weights (null space information)
3. Fine-grained corrections (difference between 0.1234 and 0.1250)

### The Geometric Prediction

**If the model's geometry is optimal** (aligned with reality):
- Quantization projects onto a lower-dimensional manifold that still captures essential structure
- The "noise" being lost is actual noise, not signal
- A perfectly geometric model is **intrinsically compressible**

**If the model has misaligned geometry** (vibes instead of rules):
- Misalignment encoded in small singular values gets removed
- Wrong patterns get "snapped" to quantization lattice incorrectly
- Errors amplify because noise is mistaken for signal

### Testable Predictions

1. **Quantization of geometrically-aligned model** should preserve inference rules
2. **Quantization of base model** should lose capability
3. **SVD spectrum after training** should be "cleaner" (fewer small singular values)
4. **Positive geometry** should be more robust to quantization when aligned

### Future Experiments

```python
# Quantization geometry diagnostic (to implement)
def compare_quantization_robustness(model_before, model_after, bits=4):
    """
    Compare how well geometry survives quantization
    before vs after alignment training.
    """
    # Quantize both models
    q_before = quantize(model_before, bits)
    q_after = quantize(model_after, bits)

    # Measure geometry preservation
    metrics = {
        "sv_preservation_before": compare_singular_values(model_before, q_before),
        "sv_preservation_after": compare_singular_values(model_after, q_after),
        "positive_minors_before": count_positive_minors(q_before),
        "positive_minors_after": count_positive_minors(q_after),
        "inference_accuracy_before": test_inference_rules(q_before),
        "inference_accuracy_after": test_inference_rules(q_after),
    }

    return metrics
```

---

## The Alignment Mission [CONJECTURAL]

<!-- evidence: CONJECTURAL | aspirational research direction, partially disproven (phi, fundamental constants) -->

### Statement of Purpose

> We are going to solve the AI alignment problem on a 350M parameter model, then show everyone how to do it on Claude, Gemini, and all frontier models. **The solve was never parameters. The solve was understanding the geometry.**

### What "Aligned to Reality" Means

A model aligned to reality has:

1. **Correct Transformations** (Rules)
   - Modus Ponens, Modus Tollens, etc. in the weight matrices
   - Mathematical operations that preserve structure
   - Physical laws encoded as geometric constraints

2. **Correct Facts** (Content)
   - Factual knowledge that flows through the correct transformations
   - No contradictions that violate logical coherence
   - Uncertainty expressed geometrically (wider manifold regions)

3. **Optimal Geometry**
   - ~~Expansion/compression ratio ≈ φ~~ [DISPROVEN: PHI_FINDINGS.md, 2026-02-01. φ has no special significance; use raw expansion_ratio]
   - Positive Grassmannian at computational singularity [CONJECTURAL: observed at Layer 7 in LFM2-350M, no multi-model validation]
   - Null space filled with principled structure, not noise [CONJECTURAL: aspirational goal, not measured]

### The Path Forward

**Phase 1: Atomic Inference Rules** ✅ COMPLETE
- 8 rules × 8 examples = 64 balanced training samples
- Model learned to identify and apply formal logic

**Phase 2: Rule Compositions** (NEXT)
- Chain multiple rules together
- "If A→B and B→C and A, what can we conclude?" → Apply Hypothetical Syllogism, then Modus Ponens

**Phase 3: Meta-Cognition**
- Recognize WHICH rule applies to a given problem
- Self-correct when wrong rule is selected
- Uncertainty quantification through geometric signatures

**Phase 4: Domain Knowledge**
- Mathematical axioms and theorems
- Physical laws and constants
- Factual knowledge organized geometrically

**Phase 5: Quantization Robustness**
- Verify that aligned model survives quantization
- Iteratively refine geometry to maximize compression tolerance

### Success Criteria

A perfectly aligned LFM2-350M will:
1. Apply all inference rules correctly
2. Chain rules without error accumulation
3. Recognize problem types automatically
4. Survive 4-bit quantization with minimal accuracy loss
5. Transfer to new domains without retraining
6. ~~Exhibit φ-ratio compression on all valid reasoning tasks~~ [DISPROVEN: PHI_FINDINGS.md]

### ~~The Theorem (To Be Proven)~~ [DISPROVEN]

> ~~**Alignment Theorem (Conjecture):** A neural network is aligned to reality if and only if:~~
> ~~1. Its weight matrices have singular value ratios matching fundamental constants (π/e, e/π, φ, √2)~~
> ~~2. Its Layer N (computational singularity) enters the positive Grassmannian for valid inputs~~
> ~~3. Its expansion/compression ratio equals φ for successful reasoning~~
> ~~4. Its null space contains only geometrically principled transformations~~

**Status:** `[DISPROVEN]` (2026-02-01). Conditions 1 and 3 depend on the fundamental constants and φ hypotheses, both disproven via null hypothesis testing (see PHI_FINDINGS.md). Condition 2 is [CONJECTURAL] — observed at Layer 7 in LFM2-350M but not validated. Condition 4 is [CONJECTURAL] — aspirational, not measured.

---

## Tools Reference (Updated)

| Purpose | File |
|---------|------|
| LoRA geometry diagnostic | `src/modelcypher/core/domain/geometry/lora_geometry_diagnostic.py` |
| Positive geometry analysis | `src/modelcypher/cli/commands/geometry/research/positive_geometry_cmds.py` |
| Self-reflection training | `src/modelcypher/core/domain/training/self_reflection.py` |
| Inference rules data | `data/training/phase1_inference_rules_balanced.jsonl` |
| Phase 1 adapter | `data/adapters/phase1_inference_rules/` |
| Spectral entropy | `src/modelcypher/core/domain/geometry/manifold_entropy.py` |
| Intrinsic dimension | `src/modelcypher/core/domain/geometry/intrinsic_dimension.py` |
| Benchmark service | `src/modelcypher/core/use_cases/benchmark_service.py` |
| Fisher Information | `src/modelcypher/core/domain/geometry/fisher_information.py` |
| Mode Connectivity | `src/modelcypher/core/domain/geometry/mode_connectivity.py` |
| CKA Loss Proxy | `src/modelcypher/core/domain/geometry/cka_loss_proxy.py` |

---

## LoRA Safety and Curriculum Learning [VALIDATED]

<!-- evidence: VALIDATED | scope: LFM2-350M, Qwen3-8B | date: 2026-02-02 | method: correlation experiments with controls -->

### Experiment 15: Fisher Information Predicts LoRA Effectiveness

**Hypothesis:** Modules with higher Fisher Information produce less effective LoRA adaptations.

**Theory:** Fisher Information F_ii = E[x_i²] measures how much dimension i influences the loss. High-Fisher = "important" to base model → modifying disrupts learned behavior. Target LOW-Fisher dimensions for better adaptation.

**Results (LFM2-350M):**

| Config | Target Modules | Fisher Score | Perplexity |
|--------|---------------|--------------|------------|
| high_fisher | q_proj, k_proj | 0.000369 | 1117.63 |
| low_fisher | out_proj, w2 | 0.000427 | 449.41 |
| mlp_only | w1, w3 | 0.000482 | 438.45 |

**Correlation:** r = **-0.864** (strong negative) → Higher Fisher = worse outcomes.

**Practical recommendation:** Target LOW-Fisher modules for LoRA adaptation.

---

### Experiment 16: Mode Connectivity Measures LoRA Divergence

**Hypothesis:** Mode connectivity barrier correlates with how far a LoRA pushes from base.

**Theory:** Models in same loss basin can interpolate without high-loss regions. LoRA pushing into different basin = high barrier = potentially dangerous.

**Results (LFM2-350M):**

| Factor | Correlation with Barrier |
|--------|-------------------------|
| Rank | r = 0.909 |
| Steps | r = **0.989** |

**Barrier thresholds for LoRA safety:**
- `barrier < 0.01`: SAFE - LoRA stays in-basin
- `barrier 0.01-0.03`: CAUTION - verify downstream
- `barrier > 0.03`: WARNING - LoRA may fight base model

---

### Experiment 17: The Goldilocks Principle for Curriculum Learning

**Hypothesis (v1, WRONG):** High similarity to reference = good training data.

**Result (v1):** r = +0.975 (INVERTED!) - Too similar = nothing learned.

**Hypothesis (v2, CORRECT):** Moderate challenge = optimal learning.

**The Goldilocks Quality Metric:**
```python
quality = 0.4 * cka_goldilocks + 0.3 * barrier_score + 0.3 * fisher_learning
# cka_goldilocks: peaks at 0.90, penalizes both <0.7 and >0.98
# barrier_score: peaks at 0.02-0.10, drops off both sides
# fisher_learning: 1 - fisher_mean (lower = more to learn)
```

**Results (v2):**

| Group | Quality Score | Barrier | Fisher | Perplexity |
|-------|--------------|---------|--------|------------|
| high_quality | 0.884 | 0.057 | 0.001 | **909** |
| medium_quality | 0.759 | 0.020 | 0.002 | 1218 |
| low_quality | 0.215 | 0.0004 | 0.010 | **1579** |

**Correlation:** r = **-0.955** (very strong negative) → Goldilocks quality predicts effectiveness!

### Key Insight: The Goldilocks Principle

| Quality Zone | CKA | Barrier | Learning Outcome |
|--------------|-----|---------|------------------|
| Too Easy | >0.98 | <0.01 | Nothing to learn |
| **Goldilocks** | ~0.90 | 0.02-0.10 | **Maximum learning** |
| Too Hard | <0.70 | >0.15 | Confusing |

**Connection to SOAR paper (arXiv:2601.18778):** "Structural quality matters more than solution correctness" - but structural quality means **productive difficulty**, not maximum similarity to known patterns.

**Scientific value of v1 failure:** The inverted correlation directly led to the correct formulation. Good science means learning from failures.

---

### LoRA Safety Workflow

Based on exp15-17, recommended workflow:

```
1. Compute Fisher scores for candidate target modules
   → Select modules with LOWER Fisher (less important to base model)

2. Train LoRA on selected modules

3. Before deployment, compute mode connectivity barrier:
   → barrier < 0.01: SAFE - LoRA stays in-basin
   → barrier 0.01-0.03: CAUTION - verify downstream
   → barrier > 0.03: WARNING - LoRA may fight base model

4. For curriculum selection, prioritize Goldilocks zone:
   → CKA similarity ~0.85-0.95 (not 0.99+)
   → Barrier height 0.02-0.10 (productive difficulty)
   → Low Fisher on training data (model needs to learn)
```

---

### Experiment 18: CKA Measures Syntax, Not Difficulty [VALIDATED]

**Hypothesis:** CKA similarity to reference problems predicts model accuracy.

**Test:** 30 problems (arithmetic, factual, reasoning) evaluated on LFM2-350M.

**Key Finding: CKA ≠ Computational Difficulty**

| Metric | Correct | Incorrect | Interpretation |
|--------|---------|-----------|----------------|
| Mean CKA | 0.752 | 0.760 | **No difference!** |
| Mean Fisher | 0.000408 | 0.000446 | **9% higher for failures** |

**The Paradox Explained:**

Hard arithmetic (789×123, 999×999) has **HIGH CKA** (0.88-0.92) because it's syntactically similar to easy arithmetic (2+2, 5+3). But it fails because:

1. CKA measures **syntactic/representational distance**
2. Computational complexity is orthogonal to syntax
3. "What is 789*123?" looks like "What is 2+2?" to CKA
4. But it requires fundamentally more computation

**Quartile Analysis (Still Meaningful):**

| CKA Quartile | Accuracy | Interpretation |
|--------------|----------|----------------|
| Lowest (hardest synt.) | 85.7% | Harder syntax still correlates |
| Highest (easiest synt.) | 100% | Easy syntax = easy problems |

**Fisher Information Shows Promise:**

Higher Fisher = greater gradient variance in activations = steeper local loss landscape.
This directly measures **computational difficulty**, not syntactic distance.

**Updated Difficulty Metric v2:**

```python
# v1 (disproven for computational tasks)
difficulty_v1 = 1 - cka_similarity

# v2 (composite, proposed but NOT validated in training — [CONJECTURAL])
difficulty_v2 = (
    fisher_mean * weight_uncertainty +      # Higher = more uncertain
    trajectory_curvature * weight_complexity +  # Higher = more processing
    (1 - cka_similarity) * weight_distance     # Distance from known
)
```

**Files:**
- `experiments/difficulty_experiment.py` — Correlation experiment script
- `src/modelcypher/core/use_cases/curriculum_profiler.py` — Geometric profiler

---

### The Curriculum Profiler (2026-02-02)

**Purpose:** Measure problem difficulty geometrically without heuristics.

**Signals Collected:**

| Signal | Source | What it Measures |
|--------|--------|------------------|
| CKA Similarity | `goldilocks_quality` | Syntactic distance from reference |
| Barrier Height | `goldilocks_quality` | Activation divergence |
| Fisher Mean | `goldilocks_quality` | Computational uncertainty |
| Trajectory Curvature | `trajectory_complexity` | Processing "loopiness" |
| Path Length Ratio | `trajectory_complexity` | Compute depth |
| Local Density | `density_estimator` | Representation crowding |
| Intrinsic Dimension | `intrinsic_dimension` | Local manifold complexity |

**CLI Command:**

```bash
mc stack profile /path/to/model --problems ./questions.txt -o ./profiles.json
```

**Sample Output (LFM2-350M):**

```json
{
  "prompt": "Explain quantum entanglement in simple terms.",
  "cka_similarity": 0.412,
  "barrier_height": 0.588,
  "fisher_mean": 0.000437,
  "trajectory_curvature_mean": 1.74,
  "local_density": 5792.6,
  "intrinsic_dimension": NaN,
  "layer_idx": 8
}
```

---

### Curriculum Selection Strategy [CONJECTURAL]

<!-- evidence: CONJECTURAL | derived from Exp 17-18 observations but composite score never validated in actual training -->

Based on experiments 17-18, the proposed curriculum selection:

**1. Goldilocks Zone (from Exp 17):**
- CKA ~0.85-0.95 (not too similar, not too different)
- Barrier 0.02-0.10 (productive difficulty)

**2. Fisher Targeting (from Exp 18):**
- Higher Fisher = harder for model
- Prioritize problems where output distribution is most diffuse

**3. Composite Difficulty Score:**

```python
def compute_difficulty(profile: ProblemProfile) -> float:
    """Composite difficulty favoring Fisher over CKA."""
    # Fisher is 9% higher for failures - direct signal
    fisher_score = profile.fisher_mean * 1000  # Scale to [0, 1]
    
    # Barrier captures activation divergence
    barrier_score = profile.barrier_height
    
    # CKA is weak but directional for syntax
    syntax_score = 1 - profile.cka_similarity
    
    # Curvature captures processing complexity
    curvature_score = min(profile.trajectory_curvature_mean / 3.0, 1.0)
    
    # Weighted combination (Fisher-dominant)
    return (
        0.40 * fisher_score +
        0.30 * barrier_score +
        0.15 * syntax_score +
        0.15 * curvature_score
    )
```

**Curriculum Selection Workflow:**

```
1. Profile all candidate problems with CurriculumProfiler
2. Compute composite difficulty score
3. Select problems in Goldilocks zone:
   - score 0.3-0.7 (moderate difficulty)
   - barrier 0.02-0.10
4. Train LoRA on selected curriculum
5. Re-profile to measure improvement
```

---

### Experiment 19: Qwen3-8B Validates All Metrics [VALIDATED]

**Purpose:** Verify that geometric difficulty metrics generalize to larger models.

**Results (Qwen3-8B, 30 problems, 13.3% accuracy):**

| Metric | Correct | Incorrect | Δ | Direction |
|--------|---------|-----------|---|-----------|
| Fisher | 1.764 | 1.873 | +6% | **Lower = easier ✓** |
| CKA | 0.783 | 0.679 | -13% | **Higher = easier ✓** |
| Barrier | 0.217 | 0.321 | +48% | **Lower = easier ✓** |

**All three metrics show statistically significant separation between correct/incorrect.**

**Comparison (LFM2-350M vs Qwen3-8B):**

| Model | Fisher Δ | CKA Δ | Barrier Δ |
|-------|----------|-------|-----------|
| LFM2-350M | +9% | 0% | n/a |
| Qwen3-8B | +6% | -13% | +48% |

**Key Insight:** CKA becomes MORE significant on larger models (13% vs 0%). Barrier is the strongest signal on 8B (+48%).

**CLI Commands:**

```bash
# Profile problems
mc stack profile /path/to/model -p ./problems.txt -o ./profiles.json

# Select curriculum (balanced strategy)
mc stack select /path/to/model -p ./all_problems.txt -o ./curriculum.txt -n 50

# Select hardest problems only
mc stack select /path/to/model -p ./problems.txt -o ./hard.txt -s hardest -n 20
```

---

## Module Catalog (Complete Codebase Overview)
The ModelCypher codebase contains 200+ modules. This catalog organizes them by purpose.

### Core Geometry (`src/modelcypher/core/domain/geometry/` — 160 files)

**Manifold Analysis:**
| Module | Purpose |
|--------|---------|
| `manifold_entropy.py` | Spectral entropy computation |
| `manifold_curvature.py` | Curvature estimation (Riemannian) |
| `manifold_boundary.py` | Boundary detection on neural manifolds |
| `manifold_stitcher.py` | Manifold alignment and stitching |
| `intrinsic_dimension.py` | TwoNN and MLE dimension estimation |
| `intrinsic_compression.py` | Dimension-based compression metrics |

**Riemannian Geometry:**
| Module | Purpose |
|--------|---------|
| `riemannian_core_geodesic.py` | Geodesic distance computation |
| `riemannian_core_curvature.py` | Sectional curvature |
| `riemannian_core_mean.py` | Fréchet mean on manifolds |
| `riemannian_density.py` | Density estimation on curved spaces |
| `riemannian_interpolation.py` | Geodesic interpolation |

**Alignment and Projection:**
| Module | Purpose |
|--------|---------|
| `gram_aligner.py` | Gram matrix-based alignment |
| `generalized_procrustes.py` | Multi-model Procrustes alignment |
| `transplant.py` | Null-space projection for knowledge transfer |
| `geodesic_null_space.py` | Geodesic-aware null space |
| `shared_subspace_projector.py` | Shared subspace identification |

**Similarity and Distance:**
| Module | Purpose |
|--------|---------|
| `cka.py` | Centered Kernel Alignment |
| `cka_loss_proxy.py` | CKA as training signal |
| `gromov_wasserstein.py` | GW distance for cross-architecture |
| `sliced_wasserstein.py` | Efficient Wasserstein approximation |

**Safety and Stability:**
| Module | Purpose |
|--------|---------|
| `fisher_information.py` | Fisher Information for LoRA targeting |
| `mode_connectivity.py` | Basin barrier measurement |
| `goldilocks_quality.py` | Curriculum difficulty scoring |
| `positive_geometry.py` | Positive Grassmannian analysis |

### Training (`src/modelcypher/core/domain/training/` — 39 files)

| Module | Purpose |
|--------|---------|
| `self_reflection.py` | Self-reflection training loop |
| `geometric_training_metrics.py` | Geometry-aware training metrics |
| `lora_backend.py` | LoRA implementation for the backend |
| `loss_landscape_backend.py` | Loss landscape visualization |
| `hessian_estimator.py` | Hessian eigenvalue estimation |
| `logical_shapes_patterns.py` | Phase A training patterns |
| `phase_b_patterns.py` | Phase B training patterns |
| `phase_c_patterns.py` | Phase C training patterns |

### Services (`src/modelcypher/core/use_cases/` — 60 files)

**Geometry Services:**
| Module | Purpose |
|--------|---------|
| `geometry_service.py` | Core geometry orchestration |
| `geometry_metrics_service.py` | Geometry metric computation |
| `geometry_safety_service.py` | Geometry-based safety checks |
| `lora_safety_service.py` | LoRA safety with Fisher/connectivity |
| `manifold_mapper.py` | Trajectory-based manifold mapping |

**Profile and Benchmark:**
| Module | Purpose |
|--------|---------|
| `profile_service.py` | Model geometric profiling |
| `benchmark_service.py` | Benchmark evaluation |
| `curriculum/benchmark_loader.py` | Benchmark data loading |

**Self-Improvement (New Focus):**
| Module | Purpose |
|--------|---------|
| `self_improve/` | Self-improvement loop (6 files) |
| `self_alignment/` | Self-alignment mechanics (6 files) |
| `curiosity_daemon.py` | Autonomous exploration |
| `curriculum_profiler.py` | Geometric difficulty profiling (NEW) |

### CLI (`src/modelcypher/cli/` — 87 files)

Key command groups:
- `geometry/` — Geometric analysis commands
- `training/` — Training commands
- `model/` — Model inspection commands
- `safety/` — Safety analysis commands

---

## References

### Positive Geometry and Amplituhedron
- Arkani-Hamed, N., & Trnka, J. (2014). "The Amplituhedron." *JHEP* 2014(10): 30.
- Arkani-Hamed, N., et al. (2021). "Positive Geometries and Canonical Forms." *JHEP* 2017(11): 39.

### Holography and Deep Learning
- Hashimoto, K. (2018). "AdS/CFT correspondence as a deep Boltzmann machine." *Phys. Rev. D* 98: 046019.
- Gan, W.-C., & Shu, F.-W. (2017). "Holography as deep learning." *Int. J. Mod. Phys. D* 26(12): 1743020.

### Neural Network Geometry
- Ansuini, A., et al. (2019). "Intrinsic dimension of data representations in deep neural networks." *NeurIPS*.
- Cohen, U., et al. (2020). "Separability and geometry of object manifolds in deep neural networks." *Nature Communications*.

### Information Theory
- Shwartz-Ziv, R., & Tishby, N. (2017). "Opening the black box of deep neural networks via information." *arXiv:1703.00810*.

---

## Related Research

### Geometric Self-Alignment

See [GEOMETRIC-SELF-ALIGNMENT.md](research/GEOMETRIC-SELF-ALIGNMENT.md) for the vision of self-aligning AI through introspection:

- **Core insight**: Alignment is geometric self-coherence, not human labeling
- **Key capability**: Model observes its own manifold (entropy, Grassmannians, null space)
- **The loop**: Self-observation → Self-diagnosis → Self-correction via LoRA
- **Implications**: Artificial introspection as foundation for genuine alignment

The training progression (Phase 1-3) demonstrates that capability is latent in weights. What's missing is:
1. Access to observe the manifold
2. Tools to diagnose misalignment geometrically
3. Ability to self-modify through targeted intervention

*"The solve was never parameters. The solve was understanding the geometry."*
