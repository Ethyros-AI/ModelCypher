# Wow! Signal Geometric Analysis `[EMPIRICAL]`

Application of high-dimensional geometry methods to the 1977 Wow! signal.

## Data Source

- **Archive**: Big Ear Radio Observatory, Ohio State University
- **Date**: August 15, 1977, 23:16 UTC
- **Format**: SNR matrix, 82 time samples × 50 frequency channels
- **Peak**: 6EQUJ5 sequence at time index 60, frequency channel 1

## Methods

### Singular Value Decomposition (SVD)

Standard decomposition: `U, S, Vt = svd(signal)`

Metrics computed:
- Singular value ratios: S[i]/S[j]
- Variance explained per mode
- Effective rank (Shannon entropy): `exp(-Σ p·log(p))` where `p = s²/Σs²`
- Participation ratio: `(Σs²)² / Σs⁴`

### Intrinsic Dimension (TwoNN)

Facco et al. 2017 estimator using ratio of 2nd to 1st nearest neighbor distances.

### Persistent Homology

Vietoris-Rips complex computed via ripser. H0 (connected components) and H1 (1-dimensional holes) tracked.

### Control Group

45 Fast Radio Bursts from CHIME catalog, same analysis pipeline applied.

## Results

### SVD Structure

| Metric | Value |
|--------|-------|
| S[0] | 62.53 |
| S[1] | 40.01 |
| S[2] | 12.15 |
| S[0]/S[1] | 1.56 |
| S[1]/S[2] | 3.29 |
| Variance (mode 0) | 56.8% |
| Variance (modes 0+1) | 80.0% |
| Effective rank (Shannon) | 5.16 |
| Participation ratio | 2.64 |

### Intrinsic Dimension

| View | Dimension |
|------|-----------|
| Time steps as samples | 19.65 |
| Freq channels as samples | 20.29 |

### Persistent Homology

| Betti Number | Count | Max Persistence | Total Persistence |
|--------------|-------|-----------------|-------------------|
| H0 | 82 | 10.95 | — |
| H1 | 29 | 0.27 | 4.64 |

### FRB Comparison (n=45)

| Metric | Wow! | FRB μ | FRB σ | z |
|--------|------|-------|-------|---|
| S[1]/S[2] | 3.29 | 1.12 | 0.38 | 5.78 |
| Variance modes 0+1 (%) | 80.05 | 9.66 | 10.19 | 6.91 |
| Effective rank | 5.16 | 74.47 | 105.63 | -0.66 |
| H1 count | 29 | 2.36 | 4.69 | 5.69 |
| H1 max persistence | 0.27 | 0.02 | 0.03 | 9.73 |
| H1 total persistence | 4.64 | 0.05 | 0.11 | 40.36 |

### Percentile Ranks vs FRBs

| Metric | Wow! Percentile |
|--------|-----------------|
| S[1]/S[2] | 97.8% |
| Effective rank | 0.0% (lowest) |
| Intrinsic dimension | 17.8% |

## Code

Analysis scripts available in external archive. Core computation:

```python
from scipy.linalg import svd
from scipy.io import readsav
from ripser import ripser

data = readsav('wow_signal.sav')
signal = data['oseti'][0]['SNR'].astype(float)  # shape: (82, 50)

U, S, Vt = svd(signal, full_matrices=False)
topo = ripser(signal, maxdim=1)
```

## Notes

- All statistics computed with standard scipy/numpy
- FRB data from CHIME public catalog (h5 format)
- Persistent homology via ripser (Vietoris-Rips complex)
- No corrections or transformations applied to raw SNR values

---

*Analysis: January-February 2026*
