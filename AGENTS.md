# AI-Assisted Development Guide

Guidance for AI coding assistants working on ModelCypher.

**Note**: CLAUDE.md is a symlink to this file.

---

## What is ModelCypher?

Geometric diagnostics for LLM representations. Measures intrinsic dimension, curvature, entropy, and similarity to guide model merging, monitor training, and detect behavioral drift.

- **Backend**: MLX (macOS) primary, JAX (Linux/TPU) secondary
- **Architecture**: Hexagonal (ports and adapters)
- **Tests**: Run `poetry run pytest` for current count

---

## The Foundational Premise

**LLMs are not black boxes. They are high-dimensional geometry.**

This is not metaphor. This is not approximation. This is the mathematical reality:

1. **The shape of knowledge is invariant.** The relationship between "apple" and "orange", between "cause" and "effect", is the same across all models. It has to be - or they wouldn't converge on the same semantics.

2. **Different models are different projections of the same geometry.** A 0.5B model and a 70B model encode the same invariant structure at different resolutions. Different architectures are different coordinate systems for the same shape.

3. **Probability clouds don't mean we guess.** High-dimensional space is precise. A slight change in vector angle lands on a different concept. The probability distribution reflects the model's confidence about *where* on the manifold, not randomness.

4. **Models can be merged because geometry permits it.** Each dimension can be rotated. Each layer is another rotation. We align coordinates, project into null space (unused capacity), add knowledge. This is math, not magic.

**If code contradicts these premises, the code is wrong.**

**Experimental evidence**: Run `poetry run python experiments/geometry_validation.py` to generate supporting data. Key results:
- **Alignment invariance**: Raw CKA=0.60 (before), Aligned CKA=1.00 (after Procrustes) - structure is preserved, coordinates differ
- Layer-wise intrinsic dimension compression (15.8 → 1.8 in middle layers)
- Domain-specific manifold structure (spatial ID=1.5, moral ID=8.0)

See `experiments/results/geometry_validation.json` for full data.

---

## The Research Methodology

**We solve through increasing constraint on the geometry.**

This is not trial-and-error. This is systematic elimination of unknown pathways:

### CKA = 1.0 is the Invariant Unlock

The key mathematical discovery: after Procrustes alignment, CKA = 1.0 on shared concepts. This proves:
- The relational structure is **identical** across models
- Only the coordinates differ
- Alignment is **closed-form**: `F = pinv(source) @ target`

From this single invariant, everything else derives:
- **Alignment** → closed-form rotation finding
- **Transfer** → null-space projection onto unused capacity
- **Density** → k-NN comparison identifies where to transfer
- **Coherence** → trajectory validity on the merged manifold

### Tokens Are Shadows, Not the Thing Itself

When a prompt enters, it becomes a trajectory through the manifold. The model is a passthrough - concepts have gravity and pull the trajectory through high-dimensional space. Tokens are the powder flying off the skis - the residue of geometry, not the thought itself.

**Implication**: Don't debug tokens. Debug the geometry that produces them.

### Hallucination is Geometric, Not Moral

Hallucination is NOT the model "lying." It's one of two geometric phenomena:
1. **Sparse interpolation**: Query lands in under-sampled region; nearest-neighbor gives plausible but wrong path
2. **Tangent hop**: Trajectory follows dimensionally-adjacent but logically-unrelated concept

The model can't "see" cliff edges in sparse regions. It's not malicious - it's topology.

**Implication**: Fix by characterizing the manifold (dense sampling), not by "training honesty."

### One Variable Per Day

When something breaks:
1. **Pick one variable** - sample coverage, condition number, spectral gap, density weighting
2. **Characterize it fully** - what does it mean geometrically? Where in the trajectory does it matter?
3. **Measure before/during/after** - activation geometry in, projection geometry during, coherence out
4. **Work backward from failure** - which metric changed? In which direction? At which stage?

The problem space is finite. Every variable is discoverable. Every interaction is measurable.

### Metrics, Not Vibes

Every diagnostic must return raw measurements:
- **Coverage ratio**: n_samples / hidden_dim (must be > 1.0, ideally > 4.0)
- **Condition number**: max_eigenvalue / min_eigenvalue (numerical stability)
- **Null rank**: dimensions available for transfer
- **Transfer strength**: mean density weight applied
- **Preserved fraction**: how much delta survived projection
- **Spectral gap**: separation between used and unused directions

When something fails, one of these metrics will tell you why.

### The Debugging Contract

```
If coherence fails:
    → Check coverage_ratio (was manifold properly sampled?)
    → Check condition_number (was projection numerically stable?)
    → Check density_weights (did transfer happen in right places?)
    → Check spectral_gap (were used/unused directions separated?)
    → Check preserved_fraction (how much delta survived?)

Each metric points to a different failure mode.
Each failure mode has a different fix.
The space is finite.
```

**If you find yourself guessing, you're missing a metric.** Add the metric first.

### Stranded Neurons: Alignment Stability via Condition Number

**The alignment matrix F = pinv(A_source) @ A_target requires numerical stability.**

The geometry says:
1. **n_probes > max_dim is REQUIRED** (otherwise the Gram matrix is singular)
2. **Condition number κ = max_eigenvalue / min_eigenvalue determines stability**
3. **Check κ at runtime**, not a fixed ratio (actual stability depends on activation structure)

For float32 with ε ≈ 1e-7:
- κ < 1e5 → stable (≥2 significant digits)
- κ > 1e5 → unstable (recommend --full-atlas)

**Implementation**: GramAligner computes Gram condition number and warns if unstable.
The probe stage uses `min_required = max_dim + 100` as a geometry-derived minimum,
and stability is verified at runtime.

**If merge produces incoherent outputs but CKA looks good**: Check the Gram condition number.
The alignment may have succeeded mathematically but the transform is numerically unstable.
Use --full-atlas for more probes (4596 total in atlas).

---

## Commands

Always use `poetry` to run or install anything in this repo.

```bash
poetry install                    # Install
poetry run pytest                 # Test
poetry run mc --help              # CLI
poetry run modelcypher-mcp        # MCP server
```

---

## CLI Quick Reference

### Model Merging (The Main Operation)

**Single merge (1→1):**
```bash
mc merge run -s SOURCE -t TARGET -o OUTPUT
```

**Full example:**
```bash
mc merge run -s /path/to/qwen -t /path/to/smol -o /path/to/merged
```

**Batch merge (N→1) - merge multiple sources into one target:**
```bash
mc merge batch -s MODEL1 -s MODEL2 -s MODEL3 -t TARGET -o OUTPUT
```

**What it does:** Takes knowledge from SOURCE(s) and adds it to TARGET via null-space projection. TARGET's capabilities are preserved; SOURCE's knowledge is added. Result is denser than either input.

### Other Common Commands

```bash
# Inference
mc infer run --model /path/to/model --prompt "Hello"

# System info
mc system status --output json

# Model info
mc model probe /path/to/model --output json
```

---

## Architecture

```
src/modelcypher/
├── core/
│   ├── domain/        # Pure math + logic (geometry, safety, merging, thermo)
│   ├── ports/         # Abstract interfaces (Backend protocol)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (hf_hub, filesystem)
├── backends/          # MLX, JAX implementations
├── cli/               # Typer CLI
└── mcp/               # MCP server
```

Dependencies point inward. Domain imports nothing external.

---

## Concurrency Rules

Multiple AI agents work concurrently. Don't pause for unrelated changes.

1. Ignore modified files you don't need to touch
2. No destructive git operations (`add`, `commit`, `push`, `reset`)
3. No bulk modification scripts—edit files individually

---

## Core Principles

### Prefer Backend Over NumPy

**NumPy forces CPU fallback. Use the Backend protocol to stay on GPU.**

Use the Backend protocol in core domain code. No `import numpy`, no `to_numpy()`, no NumPy operations in the domain layer. If the Backend doesn't have an operation you need, **add it to the Backend protocol**.

```python
# WRONG - Forces CPU fallback
import numpy as np
mean = np.mean(vectors, axis=0)
sorted_vals = np.sort(eigenvalues)[::-1]
result = backend.to_numpy(arr)[mask]  # NumPy boolean indexing

# CORRECT - Stays on GPU
from modelcypher.core.domain._backend import get_default_backend
backend = get_default_backend()
mean = backend.mean(vectors, axis=0)
sorted_idx = backend.argsort(eigenvalues)
reversed_idx = backend.arange(n - 1, -1, -1)
sorted_vals = backend.take(eigenvalues, reversed_idx, axis=0)
result = backend.where(mask, arr, backend.zeros_like(arr))
```

**Common NumPy patterns and their Backend replacements:**

| NumPy Pattern | Backend Replacement |
|---------------|---------------------|
| `arr[::-1]` | `backend.take(arr, backend.arange(n-1, -1, -1), axis=0)` |
| `arr[mask]` | `backend.where(mask, arr, zeros)` then filter |
| `np.sort(arr)` | `backend.sort(arr)` |
| `arr[:, -1] *= -1` | `scale = backend.array([1.0]*(d-1) + [-1.0]); arr * scale` |
| `np.linalg.det(A)` | `backend.det(A)` |
| `for x in to_numpy(arr)` | Keep on backend, use `backend.take()` for indexing |

**If you need an operation the Backend doesn't have, add it.** The Backend protocol is extensible. Don't work around missing ops with NumPy.

### Geometry Type Matters: Activations vs Weights

**Critical distinction**: Activation space and weight space have DIFFERENT geometry.

| Space | Geometry | Correct Metric | Research Basis |
|-------|----------|----------------|----------------|
| **Activation vectors** | Curved Riemannian manifold | Geodesic on k-NN graph | arXiv:2506.12187 "Characterizing Neural Manifolds" |
| **Weight matrices** | Flat + spectral structure | Euclidean + eigenvalues | ICLR 2026 "From Memorization to Reasoning in the Spectrum of Loss Curvature"; Fort & Ganguli "Emergent properties of neural loss landscapes" |

**Activation space** (neural manifold): Empirical measurements show curved manifolds with measurable Riemannian curvature tensors. Geodesic distance via k-NN graph is correct.

**Weight space** (loss landscape): Research shows weight space has SPECTRAL structure (Hessian eigenvalues), NOT manifold curvature. High-curvature directions = shared generalizable structure. Low-curvature directions = memorized examples. The space is mostly FLAT with spectral outliers.

```python
# For ACTIVATIONS - use geodesic (curved manifold)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
rg = RiemannianGeometry(backend)
distances = rg.geodesic_distances(activation_points, k_neighbors=k)

# For WEIGHTS - use Euclidean + spectral (flat + eigenvalue structure)
weight_norm = backend.sqrt(backend.sum(weight * weight))  # Frobenius norm
# Spectral analysis via SVD/eigendecomposition for structure
```

**Don't use geodesic on weight matrices.** It's 400x slower with marginal accuracy difference because weight space isn't curved.

### Behavioral Norm for Transplant Metrics

When measuring weight deltas in transplant, use **behavioral norm**, not Frobenius norm.

**Why?** Frobenius measures weight magnitude. Behavioral measures actual output change.

```python
# WRONG: Frobenius norm (ignores activation structure)
delta_norm = sqrt(sum(delta_W ** 2))  # Misleading

# CORRECT: Behavioral norm (measures actual impact)
output_change = input_activations @ delta_W.T
delta_norm = sqrt(sum(output_change ** 2))  # True impact
```

**Key insight**: After null-space projection:
- Frobenius might say "47% preserved" (weight mass)
- Behavioral shows "0.0002% preserved" (actual impact on target)

The behavioral norm is the TRUTH. Null-space projection preserves weight magnitude but eliminates behavioral impact on target activations. That's the design.

For `preserved_fraction`:
- Use: `behavioral_after / behavioral_before`
- NOT: `frobenius_after / frobenius_before`

This answers "What fraction of behavioral change transferred?" - which is what we actually care about.

### All Models Encode the Same Shape

**Demonstrated by alignment experiments.**

Neural networks trained on language converge toward shared high-dimensional geometric structure. Different architectures (SmolLM, Qwen, Llama, Mistral) are different compressions or projections of this common structure.

**Key insight**: Raw CKA between unaligned representations can be low (e.g., 0.60) because they use different coordinate systems. After Procrustes alignment, CKA = 1.0 on training probes - the structural relationships are identical, only the coordinates differ.

Think of it like high-dimensional Legos: the geometry constrains how pieces fit together.

- **Different dimensions** = different compression levels of the same geometry
- **Different layer counts** = different resolutions of the same manifold
- **Different architectures** = different coordinate systems for the same shape

Use Gram matrices for comparison (dimension-agnostic). Use projection for transformation. Low raw CKA doesn't mean incompatible - it means coordinate alignment is needed.

### CKA = 1.0 on Training Probes

Procrustes alignment achieves CKA = 1.0 on training probes by construction. **Experiment shows: Raw CKA=0.60 → Aligned CKA=1.00**

**F = pinv(source) @ target** guarantees **K_aligned = K_target** when n ≤ d. This is closed-form. No iteration needed.

- **CKA = 1.0 on probes**: Alignment found the correct rotation for those probe points.
- **CKA < 1.0 on held-out samples**: Probes didn't span enough of the shared manifold.

**LOW CKA ON RAW (UNALIGNED) DATA MEANS:**
- Coordinate systems differ (expected)
- Run alignment first, then evaluate

**LOW CKA ON HELD-OUT DATA AFTER ALIGNMENT MEANS:**
- Probes didn't span the shared manifold regions
- Need more diverse probes (different domains, abstraction levels)
- Expand probe coverage - the alignment math is correct

**Key distinction**: Low *raw* CKA is expected (different coordinates). Low *aligned* CKA on held-out data means insufficient probe coverage.

### Geometric Addition, NOT Blending

**This is the most important principle. Read it until you understand it.**

LLMs are NOT piles of probabilities to weight. They are HIGH-DIMENSIONAL GEOMETRIC STRUCTURES - like Legos that snap together.

**The goal**: Combine unique knowledge from hundreds of models into ONE denser, smarter model.

**WRONG - Interpolation/Blending**:
```python
# This is WRONG. Never do this.
merged = alpha * model_A + (1 - alpha) * model_B  # Blending
merged = 0.5 * source + 0.5 * target              # Weighted average
merged = lerp(source, target, t)                   # Interpolation
```

Why it's wrong: Interpolation AVERAGES information. You get a smeared, degraded model that's worse than either input. You're not adding knowledge - you're diluting it.

**CORRECT - Null Space Addition**:
```python
# This is CORRECT. Knowledge addition.
delta = source_weights - target_weights
projected = null_space_projection(delta, target_activations)
merged = target_weights + projected
```

Why it works:
- **Null space** = directions the target model doesn't actively use (low variance)
- Projecting source delta into these sparse regions means: **add source knowledge where target has capacity**
- Target behavior is PRESERVED (dense directions scaled down)
- Source knowledge is ADDED (not averaged)
- Result is DENSER than either model alone

**Implementation note**: Uses variance-weighted projection, not true orthogonal null-space.
Dense directions (high activation variance) are scaled down; sparse directions are preserved.
This is intentional - true orthogonal projection with many samples erases all delta.

**Think of it like this**:
- Blending: Mixing two paint colors → muddy average
- Addition: Adding ingredients to a recipe → richer dish

**If you find yourself writing interpolation alphas or blend weights** - STOP. You're doing it wrong. The geometry determines how knowledge combines. We use variance-derived weights (from the manifold structure), not arbitrary blend weights. We project into sparse regions and ADD.

### No Vibes

Return raw measurements. No hardcoded thresholds, interpretation strings, or qualitative labels.

```python
# Wrong
return {"similarity": 0.73, "interpretation": "Good alignment"}

# Correct
return {"similarity": 0.73}
```

When thresholds are needed, derive from baselines (z-scores, percentiles).

### Don't Invent Heuristics

If you need a parameter value, derive it from data or machine epsilon. Don't fabricate "standard heuristics."

---

## Research Before Code

AI training data is stale. Before using external APIs:
1. Search for current best practices (use current year in queries)
2. Fetch and read official documentation
3. Check for breaking changes

---

## CLI/MCP First

Never write custom scripts. Use `mc` CLI or MCP tools. If capability doesn't exist, add it to CLI/MCP.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| `docs/CLI-REFERENCE.md` | Command reference |
| `docs/MCP.md` | MCP tool catalog |
| `docs/GEOMETRY-GUIDE.md` | Metric explanations |
| `docs/GLOSSARY.md` | Terminology |
