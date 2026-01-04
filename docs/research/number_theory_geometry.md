# Number Theory Geometry: Prime Distribution Diagnostics

> **Status**: Experimental
> **Implementation**: `src/modelcypher/core/domain/geometry/prime_geometry.py`
> **CLI**: `mc geometry number-theory ...`

---

## Overview

Number Theory Geometry analyzes prime gaps with the same geometric tools used
for model representations. It provides raw spectral, topological, and curvature
measurements plus baseline comparisons. No interpretations or thresholds are
added beyond what the core code reports.

---

## CLI Commands

- `spectral` - time-delay embedding + spectral metrics vs random baseline
- `topology` - persistent homology fingerprint on prime embeddings
- `curvature` - Ollivier-Ricci curvature on prime vs random embeddings
- `sweep` - log-spaced prime-count sweep (KS + participation + ID)
- `full-analysis` - multi-baseline analysis with hypothesis tests + bootstrap
- `scale-study` - scale sweep with effect sizes and scale_invariance flag
- `perturbation` - noise robustness study (stability scores)
- `hypothesis-summary` - summarizes a saved full-analysis JSON

Examples:
```bash
mc geometry number-theory spectral --n-primes 1000
mc geometry number-theory topology --n-primes 500
mc geometry number-theory curvature --n-primes 1000
mc geometry number-theory full-analysis --n-primes 5000
mc geometry number-theory scale-study --max-primes 50000
mc geometry number-theory perturbation --n-primes 1000
```

---

## Embeddings Used

ModelCypher supports multiple prime embeddings in `prime_geometry.py`:

- **time_delay** - sliding windows of prime gaps
- **residue** - residues mod small primorials
- **digit** - digit-pattern encodings
- **position** - index/position features

---

## Baselines

Baseline generators in `prime_geometry.py` include:

- **exponential** - Poisson gap baseline
- **uniform** - uniform gap baseline
- **poisson** - Poisson count baseline
- **cramer** - Cramer probabilistic model
- **shuffled** - shuffled prime gaps (same marginal, no order)

`full-analysis` uses exponential, uniform, and shuffled by default.

---

## Metrics Returned (Raw)

- **Spectral metrics**: participation ratio, spectral entropy, condition number,
  top-k ratio
- **Distribution comparisons**: Wasserstein distance, KS statistic
- **Intrinsic dimension**: TwoNN estimate for prime and baseline embeddings
- **CKA**: gap vs position embedding coherence
- **Hypothesis tests**: effect sizes, p-values, bootstrap CIs (from full-analysis)

---

## Notes

- `scale-study` computes `scale_invariance_passed` from the sign stability of
  effect sizes in the sweep.
- All outputs are raw measurements; interpretation is left to callers.

---

## Related Docs

- [math/prime_spectral_geometry.md](math/prime_spectral_geometry.md)
- [math/geodesic_distance.md](math/geodesic_distance.md)
- [math/persistent_homology.md](math/persistent_homology.md)
