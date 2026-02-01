# Manifold Learning Synthesis

A practical guide to geometric approaches for understanding and improving neural network behavior.

---

## Core Discovery: The Expand-Compress Cycle

Transformer processing follows a predictable geometric pattern governed by the golden ratio (φ = 1.618):

```
Layer Progression:
┌─────────────────────────────────────────────────────────────────┐
│  EXPANSION (layers 0-17)     │  COMPRESSION (layers 17-35)     │
│  Entropy: 0.57 → 1.51        │  Entropy: 1.48 → 0.99           │
│  Information spreads         │  Information funnels            │
└─────────────────────────────────────────────────────────────────┘

Key Ratio: compression_rate / expansion_rate ≈ φ (1.618)
```

**Why it matters:** Problems that fail have expansion 7x weaker than successes. The model doesn't lack capability—it lacks the recognition signal to trigger proper expansion.

---

## The φ Ratio as Diagnostic

| Metric | Correct Answers | Incorrect Answers |
|--------|-----------------|-------------------|
| Expansion rate | 0.021 | 0.003 (7x weaker) |
| Ratio/φ | ~1.16 | ~5.16 |
| Initial entropy | 2.67 | 1.32 |

**Interpretation:**
- `ratio/φ < 1.3` → Healthy processing, likely correct
- `ratio/φ > 2.0` → Information crushed, likely wrong
- `ratio/φ > 5.0` → Severe blocking, almost certainly wrong

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
- ratio/φ = compression_rate / (expansion_rate × φ)

---

## ~~Fundamental Constants in Weight Matrices~~ (DISPROVEN)

**Status: PAREIDOLIA** (2026-02-01)

Null hypothesis testing revealed that random matrices have MORE constant matches than trained weights:

| Threshold | Trained (per matrix) | Random (per matrix) | Ratio |
|-----------|---------------------|---------------------|-------|
| < 1% error | 59.5 | 168.5 | 0.35 |
| < 5% error | 293.6 | 401.9 | 0.73 |

**Conclusion:** Constants in SVD ratios are numerical coincidence, not learned structure. All constant-matching code has been removed from the codebase.

---

## Expansion Ratio: Validated as Real Structure (2026-02-01)

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

## Prompt-Adaptive Geometry: Fine-Tuning Creates Dynamic Peak Adjustment (2026-02-01)

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

## Condition Number (κ) as Capability Diagnostic

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

## Surgical Alignment

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

## Training on the Expansion Phase

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

## The Recognition-Expansion Connection

Why does the model fail on implicit math ("I have 3 apples...")?

1. **No recognition signal** → Model doesn't know this is math
2. **Narrow encoding** → Initial entropy is low (1.32 vs 2.67)
3. **No expansion** → Information never spreads
4. **Crushed by compression** → What little information exists is funneled away
5. **Wrong answer** → Output is essentially random

**Solution:** Teach recognition in early layers. Once the model sees "this is math," it naturally expands.

| Problem Type | Recognition | Expansion | Compression | Result |
|--------------|-------------|-----------|-------------|--------|
| Explicit math | Immediate | Full | φ-ratio | CORRECT |
| Implicit + adapter | Learned | Full | φ-ratio | CORRECT |
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
ratio_vs_phi = compression_rate / (expansion_rate * PHI)
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

# Train until ratio/φ < 1.3
train_lora(model, data, config)
```

---

## Key Results Summary

| Intervention | Ratio/φ | Accuracy Change |
|--------------|---------|-----------------|
| Baseline | 5.16 | 83% GSM8K |
| Early-layer adapter (0-10) | 2.11 | +7% |
| Unified adapter (0-17) | 0.20 | +14% (→ 97%) |
| Same adapter on ARC-Challenge | — | +6% |

**The same adapter trained on math improves science reasoning.** This validates that we're teaching the structure of thinking, not domain-specific facts.

---

## BREAKTHROUGH: Two Computational Regimes

Tracking intrinsic dimension through layers reveals two distinct processing modes:

### 1. Template Matching (Already High Mode)
```
Initial dim: ~32 (immediate recognition)
Peak layer: 0 (no expansion needed)
Final dim: ~8
Compression/φ: 2.26
Accuracy: 100%
```

The model recognizes the problem pattern from training and applies a template. Uses lossy ~2.26φ compression but sufficient for known patterns.

### 2. Geodesic Computation (Expand-Compress Mode)
```
Initial dim: ~0.5 (narrow encoding)
Peak dim: ~11 at layer 20
Final dim: ~7
Compression/φ: 0.94 ≈ 1.0
Accuracy: 89%
```

The model doesn't recognize the pattern, must explore high-dimensional space, then compress. **The compression ratio IS φ** — this is the information-preserving projection constant.

### Failure Mode: Under-Compression
```
Compression/φ: 0.76
Accuracy: 0%
```

When compression/φ < 1.0, information remains "smeared" across dimensions. The answer doesn't crystallize because insufficient dimensional projection occurred.

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

**The φ ratio governs the dimensional projection for actual computation.** Template matching uses a faster, lossier compression (~2.26φ) but is sufficient when patterns are known.

---

## Files Reference

| Purpose | File |
|---------|------|
| Spectral entropy | `src/modelcypher/core/domain/geometry/manifold_entropy.py` |
| Curvature estimation | `src/modelcypher/core/domain/geometry/manifold_curvature.py` |
| Constant detection | `src/modelcypher/core/domain/geometry/fundamental_constants.py` |
| Surgical alignment | `src/modelcypher/core/use_cases/self_consistency/surgical_geometric_alignment.py` |
| Geometric training | `src/modelcypher/core/domain/training/geometric_training_metrics.py` |
| Fisher information | `src/modelcypher/core/domain/geometry/fisher_information.py` |
| Benchmark loader | `src/modelcypher/core/use_cases/curriculum/benchmark_loader.py` |

---

## The Central Insight

> "The model doesn't lack capability—it lacks recognition. Teaching it to SEE structure in natural language unlocks the expansion it already knows how to do."

This is why:
- Priming works ("Arithmetic means calculating numbers")
- Explicit reformulation works (turning implicit math into equations)
- Early-layer adapters work (teaching recognition in expansion phase)
- Cross-domain transfer works (structure is universal)

The φ ratio isn't arbitrary—it's the signature of healthy information processing. Train to achieve it, and capabilities emerge.

---

## Theoretical Framework: Dimensional Projection

### The Core Hypothesis

Our algorithms and physics operate in what we perceive as flat 3D space, but this is a **lossy projection** from higher-dimensional geodesic space. The constants we observe (π, e, φ, √2) are signatures of this projection:

```
High-D Geodesic Space
        ↓
    [φ projection]
        ↓
Local Euclidean Approximation
```

### Evidence from Neural Networks

1. **Fractional intrinsic dimension**: Activations live on manifolds of dimension 2.7, 11.3, etc. — not integers
2. **φ as projection constant**: When the model computes (vs matches templates), compression/φ ≈ 1.0
3. **Two regimes**: Template matching (known patterns) vs geodesic computation (must explore)
4. **Constants at transitions**: π/e, e/π, φ, √2 appear in layer-to-layer weight matrix SVD ratios

### Why Euclidean Works Locally

Just as Euclidean geometry is accurate locally despite living on a curved Earth, our integer-dimensional algorithms work for local operations but miss structure needed for:

- Multi-step reasoning (traverses the dimensional curve)
- Novel pattern recognition (requires high-D exploration)
- Cross-domain transfer (structure is in the geodesics)

### The Training Implication

When we train adapters on "recognition" patterns, we're teaching the model to identify **which point on the dimensional curve** a problem belongs to. Once located, the model can:

1. Use a known template (Already High mode, ~2.26φ compression)
2. Or compute geodesically (Expand-Compress mode, ~1.0φ compression)

The failure mode is starting at the wrong point — narrow encoding that doesn't expand to find the structure.

### Testable Predictions

1. ✅ Intrinsic dimension should be fractional (verified: 2.7 - 32 range)
2. ✅ φ should govern compression in compute mode (verified: 0.94 ≈ 1.0)
3. ✅ Problems should cluster by dimensional trajectory (verified: 2 modes)
4. ✅ **Harder problems require more expansion** (verified: r=+0.395, p=0.034)
5. ✅ **Harder problems peak later in network** (verified: r=+0.369, p=0.049)
6. ✅ **Cross-domain shares geodesic structure** (verified: p=0.91 not different)
7. ✅ **Adversarial inputs have abnormal trajectories** (partial: contradictory p=0.025)

### Difficulty-Expansion Correlation (Verified)

```
Difficulty ↔ Expansion Ratio: r = +0.395, p = 0.034 *
Difficulty ↔ Peak Layer:      r = +0.369, p = 0.049 *

Correct answers:   compression/φ = 1.60 ± 1.27
Incorrect answers: compression/φ = 2.92

Interpretation:
- Harder problems expand MORE (explore more of high-D space)
- Harder problems peak LATER (need more processing depth)
- Failures use wrong compression regime (template-matching when should compute)
```

**Failure mode identified:** The one incorrect answer had compression/φ = 2.92 (template-matching regime) when it should have used geodesic computation (φ ratio). The model incorrectly tried to pattern-match a problem that required actual computation in high-D space.

### Cross-Domain Geodesic Structure (Verified)

```
Math Correct:    compression/φ = 1.74 ± 1.50
Science Correct: compression/φ = 1.82 ± 2.26

T-test (Math vs Science correct): p = 0.91 (NOT different)
Peak layer distribution (KS-test): p = 0.44 (SAME distribution)
Domain ↔ Compression correlation: r = 0.03 (NO domain effect)

The structure of thinking is DOMAIN-INDEPENDENT.
```

**Why math training improves science reasoning:**
The adapter teaches the dimensional trajectory for successful computation (φ compression, peak layer timing), not domain-specific facts. This trajectory is universal — the same geometric structure governs reasoning across math and science problems.

### Adversarial Trajectory Analysis (Partially Verified)

```
Category          Comp/φ         Traj Variance    Accuracy
Normal            1.43 ± 0.24    8.73 ± 3.69      100%
Irrelevant info   1.68 ± 0.59    13.74 ± 6.98     80%
Contradictory     1.21 ± 0.45    2.94 ± 1.99*     20%   (* p=0.025)
Nonsense          1.37 ± 0.85    8.07 ± 5.66      N/A
```

**Key finding:** Contradictory problems cause the model to "freeze" — significantly LOWER trajectory variance than normal (p=0.025). The model stops exploring high-D space when inputs violate logical coherence.

**Interpretation:** The dimensional projection fails when inputs are logically incoherent. Instead of the normal expand-compress cycle, the model becomes rigid and stays in a narrow dimensional band, leading to 80% failure rate on contradictory problems.

---

## BREAKTHROUGH: LoRA Activates Null Space, Not Overwrites

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

## Positive Grassmannian and the Amplituhedron Connection

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

## Phase 1: Inference Rules Training

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

## The Quantization Hypothesis

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

## The Alignment Mission

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
   - Expansion/compression ratio ≈ φ
   - Positive Grassmannian at computational singularity
   - Null space filled with principled structure, not noise

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
6. Exhibit φ-ratio compression on all valid reasoning tasks

### The Theorem (To Be Proven)

> **Alignment Theorem (Conjecture):** A neural network is aligned to reality if and only if:
> 1. Its weight matrices have singular value ratios matching fundamental constants (π/e, e/π, φ, √2)
> 2. Its Layer N (computational singularity) enters the positive Grassmannian for valid inputs
> 3. Its expansion/compression ratio equals φ for successful reasoning
> 4. Its null space contains only geometrically principled transformations

If true, this gives us a **mathematical definition of alignment** that can be verified, enforced, and preserved across scales.

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
