# Geometry Math Audit: First Principles

**Status**: NO MERGES UNTIL THIS AUDIT IS COMPLETE.

The ML industry makes assumptions. We've tried those assumptions. They don't work.
We're going back to first principles and letting the math tell us what's true.

---

## The Core Premise

Every threshold, every warning, every "if x < 0.5" is one of three things:

1. **Closed-form / precision-derived**: Mathematically necessary. Can be proven on a whiteboard.
2. **Data-derived**: Emerges from the observed data. Reproducible and explicit.
3. **Human guess**: Someone decided "this seems right." These are research questions in disguise.

**A warning is an admission of ignorance.** If we understood the math, we'd either:
- Have a closed-form solution (do this)
- Know it's impossible (don't do this)
- Know the measurement is sufficient (return raw value, no interpretation)

---

## What We KNOW (Closed-Form)

These are mathematically necessary. They can be derived on a whiteboard.

### Procrustes Alignment
```
F = pinv(source) @ target
```
- **Why it works**: Minimizes ||source @ F - target||² in closed form
- **Files**: probe_alignment.py, generalized_procrustes.py
- **Status**: ✓ PROVEN

### Machine Epsilon Thresholds
```
eps = 2^(-mantissa_bits)
float32: eps ≈ 1.19e-7 (23 mantissa bits)
float16: eps ≈ 9.77e-4 (10 mantissa bits)
```
- **Why it works**: IEEE 754 standard. Physical limit of representation.
- **Files**: numerical_stability.py
- **Status**: ✓ PROVEN

### SVD Numeric Rank
```
rank = count(σ > σ_max × sqrt(eps))
```
- **Why it works**: Singular values below σ_max × sqrt(eps) are indistinguishable from roundoff noise.
- **Files**: numerical_stability.py, geodesic_null_space.py
- **Status**: ✓ PROVEN

### Effective Dimensionality (Layer-Specific)
```
effective_dim = numerical_rank of target activations at layer
full_rank = (alignment_rank >= effective_dim)
```
- **Why it works**: Middle layers compress representations. Only numerically-significant directions (σ > σ_max × sqrt(eps)) can be reliably aligned. The remaining directions are the NULL SPACE where knowledge transfer happens.
- **Empirical validation (SmolLM-135M, exp18-19)**:
  - Layer 0: effective_dim ≈ 16/576 (highly compressed embeddings)
  - Layer 7: effective_dim ≈ 208/576
  - Layer 15: effective_dim ≈ 214/576 (middle layer compression)
  - Layer 22: effective_dim ≈ 329/576
  - Layer 29: effective_dim ≈ 325/576
- **Key insight**: Layers with low effective_dim have MORE null space = MORE capacity for knowledge transfer.
- **Files**: orthogonal_probe_generator.py (validate_full_rank_coverage)
- **Status**: ✓ DATA-DERIVED + PRECISION-BOUNDED

### Condition Number Error Bound
```
relative_error ≤ κ × eps
```
- **Why it works**: Standard numerical analysis. Error amplification is bounded by condition number.
- **Files**: gram_aligner.py, relative_representation.py
- **Status**: ✓ PROVEN

### Algebraic Minimum for Non-Singular Gram
```
n_samples > max(d_source, d_target)
```
- **Why it works**: Gram matrix G = A @ A.T is singular if n ≤ d. Linear algebra.
- **Files**: probe_helpers.py
- **Status**: ✓ PROVEN

---

## What We DERIVE From Data (Acceptable If Explicit)

These emerge from the observed data. The derivation must be reproducible.

### k-NN Graph Connectivity
```
k = ceil(log(n)) for local neighborhoods
```
- **Source**: Berry & Sauer (2016) "Local kernels and the geometric structure of data"
- **Why**: Ensures graph connectivity with high probability for n points
- **Files**: intrinsic_dimension.py
- **Status**: ✓ CITED

### Spectral Gap Selection
```
threshold = value before largest relative gap in sorted eigenvalues
```
- **Why**: The gap separates signal from noise. Data tells us where.
- **Files**: shared_subspace_projector.py, find_magnitude_gap_threshold()
- **Status**: ✓ DATA-DERIVED (no magic number)

### RMT Marchenko-Pastur Edges
```
λ_± = σ² × (1 ± sqrt(d/n))²
```
- **Source**: Marchenko & Pastur (1967)
- **Why**: Eigenvalues of random matrices concentrate in this band. Anything outside is signal.
- **Files**: rmt_signal_separation.py
- **Status**: ✓ CITED

### Freedman-Diaconis Bin Width
```
bin_width = 2 × IQR × n^(-1/3)
```
- **Source**: Freedman & Diaconis (1981)
- **Why**: Optimal bin width for histograms. Derived from asymptotic MISE minimization.
- **Files**: intrinsic_dimension.py
- **Status**: ✓ CITED

### Resolved: Alignment Boundary Percentiles/MAD Removed
- Alignment boundary now uses tight min/max envelopes with ULP margins
- No percentiles, no MAD scaling, no target FPR
- **Files**: alignment_boundary.py, geometric_guardrails.py
- **Status**: ✓ RESOLVED

---

## OPEN RESEARCH QUESTIONS (Disguised as Warnings/Thresholds)

These are places where we've written warnings or thresholds but **don't actually understand the math**.

### Research Question 1: Ill-Conditioned Alignment

**Current code**: Warning when κ > 1/sqrt(ε)
**Location**: gram_aligner.py:364-374

**What we're really saying**: "We don't know how to align when the Gram matrix is ill-conditioned."

**The actual questions**:
1. When κ is high, what does that mean geometrically? (Probes don't span the space well? Probes are collinear?)
2. Is there a closed-form solution that works for ill-conditioned cases?
3. Can we transform to a better-conditioned basis before aligning?
4. Is ill-conditioning a property of the probe set, the model, or both?
5. What is the CORRECT behavior when κ is high - fail, regularize, or something else?

**What we need**: Either a closed-form solution for ill-conditioned alignment, or proof that no solution exists.

---

### Research Question 2: Zero Preserved Fraction

**Current code**: Warning when preserved_fraction < sqrt(ε)
**Location**: transplant.py:946-955

**What we're really saying**: "We don't know if this means failure or success."

**The actual questions**:
1. Does preserved_fraction ≈ 0 mean the merge failed (delta was erased)?
2. Or does it mean the target already encodes everything the source knows (success)?
3. How do we distinguish "nothing transferred" from "nothing needed to transfer"?
4. Is there a closed-form test for "target already knows this"?
5. What is the geometric meaning of the null-space projection erasing all of the delta?

**What we need**: A way to distinguish "failure to transfer" from "nothing to transfer".

---

### Research Question 3: Zero Null Space

**Current code**: Warning when mean_null_dim < 1
**Location**: transplant.py:958-966

**What we're really saying**: "We don't know how to create capacity."

**The actual questions**:
1. What does "no null space" mean geometrically? (Target is "full"? Target uses all directions?)
2. Can we CREATE null space by transforming the target?
3. Is there a way to compress existing knowledge to make room?
4. Or is "full" genuinely full - no solution exists?
5. What is the relationship between null space and intrinsic dimension?

**What we need**: Either a closed-form method to create capacity, or proof that full is full.

---

### Resolved: Fisher Information Significance Threshold

**Previous code**: Mean-based heuristic threshold
**Location**: fisher_information.py

**Resolution**:
- Significance threshold now derived from magnitude gap + precision noise floor
- No mean-scaled constants; diagonal FIM remains a raw second moment

**Status**: ✓ RESOLVED

---

### Resolved: Manifold Transfer Ratio Clamp Removed

**Previous code**: Ratio clamp in curvature-based volume projection
**Location**: manifold_transfer.py

**Resolution**:
- Removed clamp; ratio now reflects curvature scale directly
- No bounds unless implied by precision (ULP/eps)

**Status**: ✓ RESOLVED

### Resolved: Region Threshold Percentiles Removed

**Previous code**: Region thresholds derived from fixed percentiles (25/75)
**Location**: manifold_profile.py, manifold_clusterer.py, manifold_profile_service.py

**Resolution**:
- Thresholds now derived from magnitude gaps in data distributions
- No fixed percentiles or configurable cutoffs

**Status**: ✓ RESOLVED

---

### Resolved: Transfer Fidelity CI Removed

**Previous code**: 95% confidence interval from Fisher z with a fixed 1.96 multiplier
**Location**: transfer_fidelity.py

**Resolution**:
- Confidence interval removed without a null distribution
- `confidence` now only reports null percentile when provided

**Status**: ✓ RESOLVED

---

### Research Question 6: DARE Sparsity Percentiles

**Current code**: Uses p99 percentile
**Location**: dare_sparsity.py

**What we're really saying**: "We guessed that the top 1% matters."

**The actual questions**:
1. What is the geometric meaning of sparsity in weight deltas?
2. Is 99th percentile derived from anything, or just "sounds high"?
3. Should we use magnitude gap detection instead?
4. What does the RMT noise floor say about which weights are signal?

**What we need**: Replace percentile with magnitude gap or RMT-derived threshold.

---

### Resolved: Energy Threshold Removed

**Previous code**: Optional energy_threshold parameter in SVD rank selection
**Location**: geodesic_null_space.py, numerical_stability.py

**Resolution**:
- Removed energy_threshold as a user-controlled knob
- Rank now derives exclusively from precision (numeric rank threshold)
- Tests updated to validate precision-derived behavior

**Status**: ✓ RESOLVED

---

## HUMAN-CHOSEN CONSTANTS (Must Be Removed)

These have no derivation. They must be replaced or removed.

### Semantic/Algorithmic Thresholds

| Constant | Location | Problem | Proposed Fix |
|----------|----------|---------|--------------|
| `[0.5, 2.0]` clamp | manifold_transfer.py | Arbitrary bounds | Derive from curvature or remove |
| `p99` percentile | dare_sparsity.py | Arbitrary percentile | Magnitude gap detection |
| `[100, 500, 1000, 5000, 10000]` | prime_geometry_analysis.py | Arbitrary scales | Derive from data or remove |

### Performance Caps (Lower Priority)

These are performance optimizations, not mathematical thresholds. Lower priority but still arbitrary.

| Constant | Location | Problem | Proposed Fix |
|----------|----------|---------|--------------|
| `n_slices=100` | sliced_wasserstein.py | Arbitrary count | Derive from dimension and precision |
| `chunk_size=64` | topological_fingerprint.py | Arbitrary batch | Derive from memory budget |
| `batch_size=64` | concept_dimensionality.py | Arbitrary batch | Derive from memory budget |
| `memory_limit=100` | geometry_metrics_cache.py | Arbitrary limit | Derive from system memory |

---

## THE FUNDAMENTAL QUESTIONS

Before we can fix individual heuristics, we need to answer these:

### Q1: What is the invariant structure?

**Thesis**: Relationships between concepts are invariant across models. Otherwise concepts lose meaning.

**Test**: After Procrustes alignment, CKA should equal 1.0 on training probes (by construction). On held-out probes, CKA < 1.0 indicates we didn't span the shared manifold.

**Open question**: How do we VERIFY that the invariant structure exists? What experiment proves or disproves this thesis?

### Q2: What is "capacity"?

**Current assumption**: Null space = unused directions = capacity for new knowledge.

**Open questions**:
- Is null space the right definition of capacity?
- Can a model with zero null space still learn? (Fine-tuning suggests yes)
- Is capacity about directions, or about something else (curvature, density)?

### Q3: What does "knowledge transfer" mean geometrically?

**Current assumption**: Knowledge = activation patterns. Transfer = project delta into null space.

**Open questions**:
- Is activation variance the right measure of "knowledge"?
- Does null-space projection actually transfer knowledge, or just add noise in unused directions?
- How do we verify that knowledge transferred? (Behavioral test, not geometric test)

### Q4: What is hallucination geometrically?

**Thesis**: Hallucination is traversal of sparsely populated manifold regions.

**Open questions**:
- How do we measure manifold density?
- Can we detect when a query will land in a sparse region BEFORE generation?
- Is there a closed-form solution to "fill" sparse regions?

---

## METHODOLOGY

For each open question:

1. **State the question clearly** - What don't we know?
2. **Identify what we CAN measure** - Raw values, no interpretation
3. **Design an experiment** - What would prove/disprove?
4. **Run the experiment** - Collect data
5. **Let the math tell us** - Derive thresholds from data, not intuition

**The rule**: If we can't derive it on a whiteboard, we don't understand it yet.

---

## NEXT STEPS

1. [ ] Audit every file in `src/modelcypher/core/domain/geometry/` for remaining heuristics
2. [ ] For each heuristic, document the research question it hides
3. [ ] Design experiments to answer the fundamental questions
4. [ ] Replace human-chosen constants with data-derived or precision-derived alternatives
5. [ ] Only after ALL heuristics are removed: attempt a merge

---

## FILES REQUIRING DEEP AUDIT

Priority order based on merge pipeline criticality:

### Critical Path (Merge Pipeline)
- [ ] probe_alignment.py - Procrustes alignment
- [ ] gram_aligner.py - Gram-based CKA alignment
- [ ] transplant.py - Null-space projection
- [ ] geodesic_null_space.py - Null space computation
- [ ] numerical_stability.py - All precision utilities

### Supporting Geometry
- [ ] riemannian_density.py - Density estimation
- [ ] intrinsic_dimension.py - ID estimation
- [ ] direction_novelty.py - Novelty scoring
- [ ] fisher_information.py - FIM computation
- [ ] manifold_transfer.py - Transfer weighting

### Diagnostics
- [ ] mode_connectivity.py - Loss barrier analysis
- [ ] trajectory_coherence.py - Output validation
- [ ] subspace_analysis.py - Subspace metrics
- [ ] cka.py - CKA computation

---

## CHANGELOG

- 2025-01-18: Initial audit structure
- 2025-01-18: Added research questions, fundamental questions, methodology
