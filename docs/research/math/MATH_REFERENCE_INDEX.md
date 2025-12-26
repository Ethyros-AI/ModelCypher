# Mathematical Foundations Reference Library

> *"Standing on the shoulders of giants"*

This directory contains rigorous mathematical documentation for every geometric and high-dimensional concept used in ModelCypher. Each file includes:
- Formal definitions and theorems
- The exact formulas we implement
- Full citations to foundational and 2025 research
- How we apply the concept in model merging

---

## Core Geometric Concepts

| File | Concept | Primary Use in ModelCypher |
|------|---------|---------------------------|
| [frechet_mean.md](frechet_mean.md) | Fréchet Mean (Karcher Mean) | Riemannian center of mass for embedding averaging |
| [geodesic_distance.md](geodesic_distance.md) | Geodesic Distance on k-NN Graphs | True distance on discrete manifolds |
| [manifold_curvature.md](manifold_curvature.md) | Ricci and Sectional Curvature | Curvature-aware geometry |
| [intrinsic_dimension.md](intrinsic_dimension.md) | Intrinsic Dimension Estimation | Manifold complexity measurement |
| [tangent_space.md](tangent_space.md) | Tangent Space & Exp/Log Maps | Local linearization for Riemannian operations |
| [persistent_homology.md](persistent_homology.md) | Persistent Homology (TDA) | Multi-scale topological fingerprints |

---

## Representation Similarity

| File | Concept | Primary Use in ModelCypher |
|------|---------|---------------------------|
| [centered_kernel_alignment.md](centered_kernel_alignment.md) | CKA (Centered Kernel Alignment) | Cross-dimensional representation similarity |
| [hsic.md](hsic.md) | HSIC (Hilbert-Schmidt Independence Criterion) | Foundation underlying CKA |
| [gromov_wasserstein.md](gromov_wasserstein.md) | Gromov-Wasserstein Optimal Transport | Cross-dimensional weight projection |
| [procrustes_analysis.md](procrustes_analysis.md) | Procrustes Analysis | Orthogonal alignment of representations |
| [relative_representations.md](relative_representations.md) | Relative Representations | Dimension-agnostic transfer via anchors |

---

## Model Merging Methods

| File | Concept | Primary Use in ModelCypher |
|------|---------|---------------------------|
| [slerp.md](slerp.md) | SLERP (Spherical Linear Interpolation) | Geodesic weight interpolation |
| [ties_merge.md](ties_merge.md) | TIES-Merging (Trim, Elect, Merge) | Sign conflict resolution |
| [dare_sparsity.md](dare_sparsity.md) | DARE (Drop And REscale) | Sparse delta parameter merging |
| [task_singular_vectors.md](task_singular_vectors.md) | Task Singular Vectors (CVPR 2025) | Skill/structure separation |
| [fisher_information.md](fisher_information.md) | Fisher Information Matrix | Importance-weighted merging |
| [permutation_alignment.md](permutation_alignment.md) | Git Re-Basin / Permutation Alignment | Linear mode connectivity |

---

## Weight Analysis

| File | Concept | Primary Use in ModelCypher |
|------|---------|---------------------------|
| [dora_decomposition.md](dora_decomposition.md) | DoRA (Weight-Decomposed Low-Rank) | Magnitude/direction separation |
| [spectral_analysis.md](spectral_analysis.md) | Spectral Analysis (RMT) | Eigenvalue distributions of weights |

---

## The Geometric Thesis

ModelCypher is built on a fundamental observation:

> **Neural network representations live on curved manifolds. Euclidean geometry is the approximation; geodesic geometry is the reality.**

This has profound implications:
- **Averaging**: Arithmetic mean is wrong on curved manifolds → use Fréchet mean
- **Distance**: Euclidean distance is wrong in high dimensions → use geodesic via k-NN
- **Similarity**: Pointwise comparison fails across dimensions → use CKA on Gram matrices
- **Transfer**: Linear projection loses geometry → use Gromov-Wasserstein transport
- **Interpolation**: Linear interpolation crosses loss barriers → use SLERP

---

## Key Mathematical Relationships

```
                    ┌─────────────────────────────────────┐
                    │         REPRESENTATION SPACE        │
                    │    (High-dimensional, curved)       │
                    └─────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
    ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
    │   Geodesic   │        │   Intrinsic  │        │   Manifold   │
    │   Distance   │        │   Dimension  │        │   Curvature  │
    │  (k-NN graph)│        │  (MLE/TwoNN) │        │(Ricci/Ollivier)│
    └──────────────┘        └──────────────┘        └──────────────┘
            │                        │                        │
            ▼                        ▼                        ▼
    ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
    │   Tangent    │        │  Persistent  │        │   Spectral   │
    │    Space     │        │   Homology   │        │   Analysis   │
    │  (Exp/Log)   │        │    (TDA)     │        │  (RMT/SVD)   │
    └──────────────┘        └──────────────┘        └──────────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
    ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
    │   Fréchet    │        │  CKA (HSIC)  │        │  Gromov-     │
    │    Mean      │        │  Similarity  │        │  Wasserstein │
    │  (averaging) │        │ (comparison) │        │  (transport) │
    └──────────────┘        └──────────────┘        └──────────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                       MODEL MERGING                         │
    │  SLERP • TIES • DARE • TSV • Fisher • Procrustes • Re-Basin │
    └─────────────────────────────────────────────────────────────┘
```

---

## Citation Format

All citations follow this format:
```
Author et al. (Year). "Title." Venue.
DOI/arXiv: [link]
```

For 2024-2025 papers, we include arXiv IDs for immediate access.

---

## Acknowledgments

ModelCypher's geometric approach to model merging would not be possible without the foundational work of:

### Geometry & Topology
- **Maurice Fréchet** (1948) - Metric space generalization of means
- **Hermann Karcher** (1977) - Riemannian center of mass
- **Manfredo do Carmo** (1992) - Riemannian geometry foundations
- **Herbert Edelsbrunner** (2002) - Persistent homology

### Representation Similarity
- **Arthur Gretton et al.** (2005) - HSIC
- **Simon Kornblith et al.** (2019) - CKA for neural networks
- **Facundo Mémoli** (2011) - Gromov-Wasserstein distances
- **Peter Schönemann** (1966) - Orthogonal Procrustes

### Model Merging
- **Ken Shoemake** (1985) - SLERP for quaternions
- **Prateek Yadav et al.** (2023) - TIES-Merging
- **Lingkai Yu et al.** (2024) - DARE sparsification
- **Samuel Ainsworth et al.** (2023) - Git Re-Basin
- **Antonio Gargiulo et al.** (2025) - Task Singular Vectors

### Efficient Fine-Tuning
- **Shih-Yang Liu et al.** (2024) - DoRA decomposition
- **Michael Mahoney et al.** (2021) - Spectral analysis of neural networks
- **Luca Moschella et al.** (2023) - Relative representations

And the many researchers advancing these fields in 2024-2025.

---

## File Count: 20 Reference Documents

| Category | Count | Files |
|----------|-------|-------|
| Core Geometry | 6 | frechet_mean, geodesic_distance, manifold_curvature, intrinsic_dimension, tangent_space, persistent_homology |
| Similarity | 5 | cka, hsic, gromov_wasserstein, procrustes, relative_representations |
| Merging | 6 | slerp, ties_merge, dare_sparsity, task_singular_vectors, fisher_information, permutation_alignment |
| Weight Analysis | 2 | dora_decomposition, spectral_analysis |

---

*Last updated: 2025-12-25*
