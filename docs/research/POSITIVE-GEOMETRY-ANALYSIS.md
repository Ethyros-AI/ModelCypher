# Positive Geometry Analysis

> **Status**: Empirical research with reproducible measurements
>
> This document consolidates methodology and findings for probing positive-geometry
> signatures (Grassmannian/amplituhedron structure) in LLM representations.

---

## Part 1: Methodology

### Conceptual Placement

- **Invariant manifold geometry** is measured on activations before sampling.
  This aligns with the pre-collapse regime in `dimensional_hierarchy.md`.
- **Collapse (0D → 1D)** is treated as **sampling**, not softmax, and is downstream
  of the manifold geometry. The amplituhedron probe is therefore **pre-collapse**.
- **Fractional intrinsic dimension** and expansion/compression dynamics are already
  tracked in `MANIFOLD-LEARNING-SYNTHESIS.md`. Positive-geometry signatures can be
  compared against those measurements to test correlations.

### What We Measure

We treat the column space of probe activations (ordered by atlas probe order)
as a point on the Grassmannian. We compute **ordered minors** of the
orthonormal basis matrix (Plücker coordinates) and report:

- Fraction of **positive**, **negative**, and **near-zero** minors
- **Sign entropy** over these three buckets
- Raw summary stats for minors (min/max/mean, mean |minor|)
- Plücker norm and max absolute minor

No thresholds, no interpretation strings.

### CLI Commands

```bash
# Basic positive geometry probe
poetry run mc analyze concept-volume /path/to/model \
  --layer 0 \
  --probe-count 256 \
  --rank-source spectral-gap \
  --max-minors 256

# Domain-specific probes
poetry run mc analyze concept-volume /path/to/model \
  --domains mathematical,logical \
  --adapter /path/to/adapter_dir \
  --probe-count 256 \
  --rank-source spectral-gap \
  --max-minors 256

# Ordering sensitivity sweep
poetry run mc analyze concept-volume /path/to/model \
  --layer 7 \
  --probe-count 256 \
  --rank-source spectral-gap \
  --max-minors 256 \
  --shuffle-seed 0 \
  --shuffle-count 8
```

---

## Part 2: Cross-Scale Findings (LFM2 350M/700M/1.2B)

### Experimental Setup

| Setting | Value |
|---------|-------|
| Rank source | `spectral-gap` |
| Minor selection | `lexicographic` |
| Max minors | `256` |
| Layers | `7, 8` |
| Models | LFM2-350M, LFM2-700M, LFM2-1.2B |

Domains tested: mathematical/logical, linguistic/mental, computational/structural,
affective/relational, temporal/spatial, moral, safety, philosophical, physical, factual.

### Results Table

Format: `rank, posFraction, signEntropy, zeroFraction`

```
Domain   Layer  350M                    700M                    1.2B
───────────────────────────────────────────────────────────────────────────────
math     7      1,1.00,0.00,0.00       1,0.00,0.00,0.00       1,0.05,0.19,0.00
math     8      1,1.00,0.00,0.00       1,0.00,0.00,0.00       1,1.00,0.00,0.00
ling     7      126,0.47,0.69,0.00     126,0.50,0.69,0.00     126,0.50,0.72,0.00
ling     8      126,0.43,0.71,0.00     126,0.47,0.69,0.00     126,0.48,0.69,0.00
comp     7      211,0.49,0.69,0.00     211,0.54,0.69,0.00     211,0.52,0.73,0.01
comp     8      211,0.50,0.73,0.01     211,0.47,0.69,0.00     211,0.53,0.69,0.00
aff      7      1,1.00,0.00,0.00       1,1.00,0.00,0.00       1,1.00,0.00,0.00
temp     7      1,1.00,0.00,0.00       1,1.00,0.00,0.00       1,1.00,0.00,0.00
moral    7      1,1.00,0.00,0.00       1,0.00,0.00,0.00       1,0.00,0.00,0.00
safety   7      1,0.00,0.00,0.00       1,0.00,0.00,0.00       1,1.00,0.00,0.00
factual  7      1,1.00,0.00,0.00       255,0.00,0.05,0.99     255,0.00,0.05,0.99
factual  8      255,0.00,0.05,0.99     255,0.00,0.05,0.99     255,0.00,0.05,0.99
```

### Key Findings

#### 1. Domain-Specific Rank Fingerprints Are Scale-Invariant

| Domain | Rank | Notes |
|--------|------|-------|
| **Linguistic/mental** | 126 | Stable across all scales |
| **Computational/structural** | 211 | Stable across all scales |
| **Factual** (layer 8) | 255 | 99.2% zeros - structure exists but unfilled |
| Math, Affective, Temporal, Moral, Safety, Physical | 1 | Collapsed to single dimension |

#### 2. Sign Entropy Correlates with Geometric Richness

- High-rank domains (linguistic, computational): signEntropy ≈ 0.69 (maximum ≈ ln(2))
- Rank-1 domains: signEntropy = 0 (degenerate)

#### 3. Sign Orientation Is NOT Scale-Invariant

For rank-1 domains, the sign (positive vs negative) flips across scales. This means
individual sign values are not meaningful invariants—only the rank structure is.

#### 4. Factual Domain Shows Unique Pattern

- Rank = 255 (full dimensional capacity allocated)
- zeroFraction = 99.2% (almost nothing filled in)
- Interpretation: Structure exists for facts, but facts are coordinates, not geometry

### Ordering Sensitivity

Across 32 shuffles (seeds 0-3, shuffle_count=8), positive geometry statistics vary:

| Layer | posFraction | signEntropy | Interpretation |
|-------|-------------|-------------|----------------|
| 7 | mean=0.62, var=0.07 | mean=0.54, var=0.03 | Moderate sensitivity |
| 8 | mean=0.46, var=0.08 | mean=0.55, var=0.03 | Moderate sensitivity |

Single-ordering measurements are not sufficient; distribution over shuffles is more stable.

---

## Part 3: Interpretation

### What Is Scale-Invariant

- **Domain-conditioned rank patterns**: Linguistic=126, Computational=211, Factual=255
- **Sign entropy ranges** for high-rank domains

### What Is Scale-Variable

- Sign orientation in rank-1 domains
- Minor zeroFraction in some domains at 1.2B

### Connection to Domain Fingerprints

This analysis provides the empirical foundation for the "Structure vs Facts" insight
documented in `GEOMETRIC-SELF-ALIGNMENT.md`:

> Language and computation are "native" domains with rich geometry.
> Everything else is projected onto single dimensions.
> Factual has full rank but 99.2% zeros—structure without content.

---

## References

- Source data: `data/experiments/positive_geometry_lfm2_*.json`
- Related: `GEOMETRIC-SELF-ALIGNMENT.md`, `MANIFOLD-LEARNING-SYNTHESIS.md`
- Implementation: `src/modelcypher/core/domain/geometry/positive_geometry.py`
