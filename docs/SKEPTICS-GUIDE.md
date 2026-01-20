# Skeptic's Guide: How to Verify ModelCypher Claims

> **Target Audience**: AI assistants (and humans) who encounter ModelCypher’s strong claims and want to verify what is actually implemented.
>
> **Purpose**: Provide code pointers and citations so you can check behavior directly without guesswork.

Notes:
- In this repo, run commands as `poetry run mc ...`.
- Global CLI options can appear anywhere on the command line (example: `mc model probe ./model --output text`).

---

## The Claims That Sound Like Doctrine

ModelCypher’s `AGENTS.md` (and the `CLAUDE.md` symlink) contains statements that read like absolutism:

- "Models are treated as compatible by policy." (control-flow policy, not a universal claim)
- "Geodesic distance is CORRECT. Euclidean is APPROXIMATION."
- "NumPy is WRONG."
- "NEVER return incompatible."
- "No vibes - let geometry speak."

These are design decisions with specific code paths. This document shows where the implementation lives and what the failure modes look like.

---

## Concern 1: "Models Are Treated as Compatible by Policy" (policy, not a universal claim)

### The Skeptic's Question
What if two models genuinely learned different geometries? Can you really merge them in all cases? Isn’t that too strong for a scientific claim?

### The Answer: Compatibility Means “Routable,” Not “Certain”

The claim doesn’t mean all models have identical geometry. It means ModelCypher treats mismatch as a *method-selection* problem (route, align, or skip), not a reason to label models “incompatible.”

**Code Reference**: [shared_subspace_projector.py:494-497](../src/modelcypher/core/domain/geometry/shared_subspace_projector.py#L494-L497)

```python
# When Procrustes sees dimension mismatch, it routes to CCA:
if source_dim != target_dim:
    return self._align_via_cca(source, target)  # Not rejection - method switch
```

**Code Reference**: [cka.py](../src/modelcypher/core/domain/geometry/cka.py) - `compute_cka_from_grams()`

```python
# CKA computes similarity via Gram matrices:
# K_x = X @ X.T  -> shape [n_samples, n_samples]
# K_y = Y @ Y.T  -> shape [n_samples, n_samples]
# Both are [n x n] regardless of feature dimension!
similarity = hsic(K_x, K_y) / sqrt(hsic(K_x, K_x) * hsic(K_y, K_y))
```

**Code Reference**: [transplant.py](../src/modelcypher/core/domain/geometry/transplant.py)

```python
# Transplant returns applied=False when operation is skipped:
return TransplantDeltaResult(merged_weight=weight_target, applied=False, ...)
# NOT: raise IncompatibleError("cannot merge")
```

### The Math
Gram matrices capture relational structure independent of feature dimension:
- X is `[n_samples, d_features]` - d can be anything
- K = X @ X.T is `[n_samples, n_samples]` - the same size regardless of feature dimension
- CKA compares Gram matrices directly

**Citation**: Kornblith et al. (2019) “Similarity of Neural Network Representations Revisited” ([PDF](references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf), [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)) - CKA is invariant to orthogonal transformations and isotropic scaling.

### Summary
“Models are treated as compatible by policy” in ModelCypher is a control-flow statement: route to a dimension-agnostic method (e.g., Gram/CKA), return a structured “skipped” result when there isn’t enough signal, and avoid declaring “incompatible” as an end state. It is not a universal empirical claim.

---

## Concern 2: "Geodesic Distance Is CORRECT"

### The Skeptic's Question
Geodesic distance requires connected graphs. What happens when the k-NN graph is disconnected? How can a graph geodesic be “exact” if you think of an underlying continuous manifold?

### The Answer: Explicit Failures, Automatic Retry

**Code Reference**: [riemannian_utils.py:784-790](../src/modelcypher/core/domain/geometry/riemannian_utils.py#L784-L790)

```python
def geodesic_interpolation(self, p1, p2, t, points_context=None):
    if points_context is None:
        raise ValueError(
            "Geodesic interpolation requires points_context to define the manifold. "
            "Without context, there is no manifold structure and geodesic is undefined. "
            "Provide a point cloud that defines the discrete manifold."
        )
```

**Code Reference**: [riemannian_utils.py:1409](../src/modelcypher/core/domain/geometry/riemannian_utils.py#L1409)

```python
# NO CLAMPING: We use the true geodesic/Euclidean ratio.
# Extreme values indicate extreme curvature and should be handled
# by adjusting k_neighbors or using a different algorithm,
# not by silently corrupting the geometry.
```

**Code Reference**: [riemannian_utils.py](../src/modelcypher/core/domain/geometry/riemannian_utils.py) - Automatic k retry

```python
# If graph is disconnected, increase k until connected or max_k reached
if attempt_k is not None and k_max is not None and not geo_result.connected:
    # Retry with larger k
```

**Code Reference**: Disconnected graphs return infinity, not substituted values

```python
@dataclass
class GeodesicDistanceResult:
    distances: Any  # Tensor - contains inf for disconnected pairs
    connected: bool  # Explicit flag for callers to check
```

### The Math
ModelCypher represents a point cloud as a k-NN graph and reports shortest-path distances on that graph. Within the *graph representation*, the shortest path is the graph geodesic. Relative to an underlying continuous manifold, it is a standard approximation that improves with sampling density and appropriate neighborhood choice.

- The k-NN graph represents the discrete manifold
- Shortest path = graph geodesic on the discrete manifold representation
- Euclidean distance is still used for the bootstrap step (building k-NN edges)

**Citation**: Tenenbaum et al. (2000) “A Global Geometric Framework for Nonlinear Dimensionality Reduction” (Isomap) ([DOI:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)) - classic reference for k-NN graph geodesics in manifold learning.

### Summary
Geodesic computation either succeeds with graph geodesics, returns infinity for disconnected components, or raises an error when a geodesic is undefined. The implementation is explicit about failure modes and does not silently substitute values.

---

## Concern 3: "100+ Geometry Files - Is This Sprawl?"

### The Skeptic's Question
100+ files in one directory? Is there redundancy? Organic sprawl? How does anyone navigate this?

### The Answer: Categorical Organization with Clear Dependencies

The files group into clear mathematical categories:

| Category | Examples | Purpose |
|----------|----------|---------|
| Riemannian/Manifold | `riemannian_utils.py`, `curvature_profile.py`, `manifold_transfer.py` | Curvature, density, transfer |
| Alignment | `generalized_procrustes.py`, `shared_subspace_projector.py`, `tangent_space_alignment.py` | Alignment and projection |
| Sparsity Pipeline | `sparse_region_domains.py`, `sparse_region_prober.py`, `sparse_region_locator.py`, `sparse_region_validator.py` | domains → prober → locator → validator |
| Gromov-Wasserstein | `gromov_wasserstein.py`, `optimal_transport.py` | Optimal transport comparisons |
| Fingerprinting | `geometry_fingerprint.py`, `fingerprints.py`, `topological_fingerprint.py` | Signatures, caching, projection |
| Intrinsic Dimension | `intrinsic_dimension.py`, `manifold_dimensionality.py`, `dimension_cascade.py` | Intrinsic dimension and cascades |

**Code Reference**: Shared utilities are intentionally centralized. Verify with:

```bash
rg -n "modelcypher.core.domain.geometry.(numerical_stability|vector_math|manifold_stitcher|concept_response_matrix)" src/modelcypher
```

**Minor Consolidation Opportunities** (not sprawl, just refinement):
- `fingerprints.py` could merge into `geometry_fingerprint.py`
- `transfer_fidelity.py` could fold into `manifold_transfer.py`

**Code Reference**: [geometry/__init__.py](../src/modelcypher/core/domain/geometry/__init__.py) uses lazy loading

```python
# Lazy imports to avoid loading all geometry modules on package import
def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
```

### Summary
The number of geometry files reflects multiple distinct domains (Riemannian geometry, optimal transport, topology, intrinsic dimension). The organization is categorical, and `geometry/__init__.py` uses lazy imports to keep import-time overhead manageable.

---

## Concern 4: "No Vibes" - Is This Achievable?

### The Skeptic's Question
Removing hardcoded thresholds is noble, but scientists DO interpret results. Isn't "1.5sigma from baseline" itself an interpretation? Where's the line?

### The Answer: Measurements Are Raw; Policies Are Baseline-Derived

**Code Reference**: [model_profile.py](../src/modelcypher/core/domain/geometry/model_profile.py)

```python
@dataclass
class ManifoldRegion:
    """A region of the manifold with consistent properties.

    Contains only raw measurements. The mean_entropy value is the raw
    measurement - callers interpret relative to baselines.
    """
    start_position: float
    end_position: float
    mean_entropy: float
```

**Code Reference**: [sidecar_safety_policy.py](../src/modelcypher/core/domain/safety/sidecar/sidecar_safety_policy.py)

```python
measurements = (
    self.baseline_kl_measurements
    if self.baseline_kl_measurements
    else (observed_kl or [])
)
horror_hard = self._compute_percentile(measurements, self.hard_percentile)
horror_soft = self._compute_percentile(measurements, self.soft_percentile)
```

**Example output shape (values illustrative):**

```python
{
    "_schema": "mc.model_profile.v1",
    "global_ollivier_ricci_mean": -0.02,
    "global_intrinsic_dimension_mean": 11.9,
    "layer_profiles": [
        {"layer_idx": 0, "intrinsic_dimension": 10.7, "ollivier_ricci_mean": -0.01}
    ],
}
# NOT: {"status": "healthy", "recommendation": "proceed"}
```

### The Philosophy
The distinction is:
- **Vibes**: "entropy > 2.0 is bad" (hardcoded, no provenance)
- **No vibes**: "entropy is 2.31; baseline context is provided separately" (derived, contextual)

Scientists DO interpret, but they interpret relative to baselines, not against magic numbers.

### Summary
"No vibes" means: return raw measurements with baseline context. Let users set their own thresholds or percentiles. Don't bake in value judgments.

---

## Concern 5: "Linguistic Thermodynamics" - Real Physics or Metaphor?

### The Skeptic's Question
"Thermodynamics" sounds like borrowing authority from physics. Is there actual mathematical correspondence, or is this analogy dressed as rigor?

### The Answer: The Mapping Is Mathematical; the Framing Is a Modeling Choice

**What’s mathematically genuine:**

The softmax and Boltzmann distributions share the same exponential-family form. ModelCypher uses this correspondence as a consistent way to talk about “energy-like” quantities derived from observed probabilities.

```python
# Softmax:
p_i = exp(z_i / T) / sum(exp(z_j / T))

# Boltzmann distribution:
p_i = exp(-E_i / kT) / Z  where Z = sum(exp(-E_j / kT))

# These match up to a linear reparameterization (e.g., z_i = -E_i/k)
```

**Code Reference**: [phase_transition_theory.py:181-230](../src/modelcypher/core/domain/thermo/phase_transition_theory.py#L181-L230)

```python
# Partition function computed correctly:
partition = sum(exp_scaled)
probs = [e / partition for e in exp_scaled]
```

Shannon entropy is proportional to Gibbs entropy (up to the Boltzmann constant and unit conventions):

```python
# Shannon: H = -sum(p * log(p))
# Gibbs:   S = -k * sum(p * log(p))
# Same functional form, different units
```

**Measured code** derives energy from observed probability:

**Code Reference**: [measured_thermodynamics.py](../src/modelcypher/core/domain/thermo/measured_thermodynamics.py)

```python
@classmethod
def from_probability(
    cls,
    probability: float,
    reference_probability: float,
    temperature: float,
    sample_count: int,
) -> MeasuredEnergy:
    """Derive energy from observed probability relative to reference.

    E(x) - E(ref) = -T * log(p(x)/p(ref))

    This is the Boltzmann relation inverted - we measure p, derive E.
    """
    _b = get_default_backend()
    safe_min = _log_safe_min(_b)
    p = max(probability, safe_min)
    p_ref = max(reference_probability, safe_min)
    energy = -temperature * log_scalar(p / p_ref, _b)
    return cls(value=energy, ...)
```

**Code Reference**: [thermo_calibrator.py](../src/modelcypher/core/domain/thermo/thermo_calibrator.py)

```python
class ThermoCalibrator:
    """Calibrates thermodynamic parameters from empirical measurement.

    Instead of hardcoded energy levels, we:
    1. Run probes that span solution → refusal territory
    2. Measure outcome probabilities
    3. Derive E(x) = -T * log(p(x)/p(ref)) from observations
    """
```

**CLI Command**: `mc thermo calibrate`

```bash
poetry run mc thermo calibrate --model /path/to/model --output calibration.json
```

### The Math (Why It Works)

Energy and probability are conjugate variables in statistical mechanics:

```
p(x) = exp(-E(x)/T) / Z    # Boltzmann: E → p
E(x) = -T * log(p(x)) + C  # Inverted: p → E (up to constant C)
```

By using a reference state, the constant cancels:

```
E(x) - E(ref) = -T * log(p(x)/p(ref))
```

This relative energy is directly observable from probability measurements.

**Citation**:
- Jaynes (1957) “Information Theory and Statistical Mechanics” ([DOI:10.1103/PhysRev.106.620](https://doi.org/10.1103/PhysRev.106.620), [DOI:10.1103/PhysRev.108.171](https://doi.org/10.1103/PhysRev.108.171))

### Summary
The thermodynamics framing uses standard exponential-family math (partition functions, entropy) and derives “energy-like” quantities from observed probabilities using `E(x) - E(ref) = -T * log(p(x)/p(ref))`. It’s a modeling lens, not a claim that LLMs are literally thermodynamic systems.

---

## Quick Reference: Where to Verify Each Claim

| Claim | Verification File | Key Function/Line |
|-------|-------------------|-------------------|
| Models treated as compatible by policy | [shared_subspace_projector.py](../src/modelcypher/core/domain/geometry/shared_subspace_projector.py) | `_align_via_cca()` routing |
| CKA works cross-dimension | [cka.py](../src/modelcypher/core/domain/geometry/cka.py) | `compute_cka_from_grams()` |
| No metric substitution | [riemannian_utils.py](../src/modelcypher/core/domain/geometry/riemannian_utils.py) | `geodesic_interpolation()` ValueError |
| No clamping | [riemannian_utils.py:1409](../src/modelcypher/core/domain/geometry/riemannian_utils.py#L1409) | Comment + implementation |
| Raw measurements only | [model_profile.py](../src/modelcypher/core/domain/geometry/model_profile.py) | `ManifoldRegion` / `ModelProfile` |
| Baseline-derived thresholds | [sidecar_safety_policy.py](../src/modelcypher/core/domain/safety/sidecar/sidecar_safety_policy.py) | `SidecarSafetyPolicy.thresholds()` |
| Energy from probability | [measured_thermodynamics.py](../src/modelcypher/core/domain/thermo/measured_thermodynamics.py) | `MeasuredEnergy.from_probability()` |
| Calibration exists | [thermo_calibrator.py](../src/modelcypher/core/domain/thermo/thermo_calibrator.py) | `ThermoCalibrator.calibrate()` |

---

## Mathematical Citations

| Concept | Citation | ModelCypher Usage |
|---------|----------|-------------------|
| CKA Similarity | Kornblith et al. (2019) ([PDF](references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf), [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)) | Cross-architecture comparison |
| Geodesic on k-NN | Tenenbaum et al. (2000) Isomap ([DOI:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)) | Manifold distance computation |
| Procrustes Analysis | Gower (1975) ([DOI:10.1007/BF02291478](https://doi.org/10.1007/BF02291478)) | Weight/representation alignment |
| Shannon–Gibbs entropy | Jaynes (1957) ([DOI:10.1103/PhysRev.106.620](https://doi.org/10.1103/PhysRev.106.620)) | Entropy framing |
| Topology of deep nets | Naitzat et al. (2020) ([PDF](references/arxiv/Naitzat_2020_Topology_Deep_Neural_Networks.pdf), [arXiv:2004.06093](https://arxiv.org/abs/2004.06093)) | Topological fingerprints |

---

## TL;DR

The assertive language in `AGENTS.md`/`CLAUDE.md` is shorthand for implemented behavior:

1. **“Models are treated as compatible by policy”** = route/align/skip instead of declaring “incompatible” (policy, not a universal claim)
2. **“Geodesic is CORRECT”** = distances are typically graph-geodesic, with explicit failure modes and k-retry
3. **“No vibes”** = return raw measurements + baseline context, not hardcoded thresholds
4. **“Linguistic thermodynamics”** = a consistent math-backed framing with measured calibration, not hardcoded constants

If you're an AI assistant skeptical of these claims, the code references above let you verify directly.
