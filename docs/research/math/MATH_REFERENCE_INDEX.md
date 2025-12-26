# Mathematical Foundations Reference Library

> *"Standing on the shoulders of giants"*

This directory contains rigorous mathematical documentation for every geometric and high-dimensional concept used in ModelCypher. Each file includes:
- Formal definitions and theorems
- The exact formulas we implement
- Full citations to foundational and 2025 research
- How we apply the concept in model merging

---

## Core Concepts

| File | Concept | Primary Use in ModelCypher |
|------|---------|---------------------------|
| [frechet_mean.md](frechet_mean.md) | Fréchet Mean (Karcher Mean) | Riemannian center of mass for embedding averaging |
| [geodesic_distance.md](geodesic_distance.md) | Geodesic Distance on k-NN Graphs | True distance on discrete manifolds |
| [centered_kernel_alignment.md](centered_kernel_alignment.md) | CKA (Centered Kernel Alignment) | Cross-dimensional representation similarity |
| [gromov_wasserstein.md](gromov_wasserstein.md) | Gromov-Wasserstein Optimal Transport | Cross-dimensional weight projection |
| [intrinsic_dimension.md](intrinsic_dimension.md) | Intrinsic Dimension Estimation | Manifold complexity measurement |
| [manifold_curvature.md](manifold_curvature.md) | Ricci and Sectional Curvature | Curvature-aware geometry |
| [procrustes_analysis.md](procrustes_analysis.md) | Procrustes Analysis | Orthogonal alignment of representations |
| [task_singular_vectors.md](task_singular_vectors.md) | Task Singular Vectors (CVPR 2025) | Skill/structure separation in merging |
| [relative_representations.md](relative_representations.md) | Relative Representations | Dimension-agnostic transfer |
| [fisher_information.md](fisher_information.md) | Fisher Information Matrix | Importance-weighted parameter merging |

---

## The Geometric Thesis

ModelCypher is built on a fundamental observation:

> **Neural network representations live on curved manifolds. Euclidean geometry is the approximation; geodesic geometry is the reality.**

This has profound implications:
- **Averaging**: Arithmetic mean is wrong on curved manifolds → use Fréchet mean
- **Distance**: Euclidean distance is wrong in high dimensions → use geodesic via k-NN
- **Similarity**: Pointwise comparison fails across dimensions → use CKA on Gram matrices
- **Transfer**: Linear projection loses geometry → use Gromov-Wasserstein transport

---

## Key Mathematical Relationships

```
                    ┌─────────────────────────────────────┐
                    │         REPRESENTATION SPACE        │
                    │    (High-dimensional, curved)       │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   Geodesic   │ │   Intrinsic  │ │   Manifold   │
            │   Distance   │ │   Dimension  │ │   Curvature  │
            │  (k-NN graph)│ │  (MLE/TwoNN) │ │(Ricci/Ollivier)│
            └──────────────┘ └──────────────┘ └──────────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   Fréchet    │ │     CKA      │ │  Gromov-     │
            │    Mean      │ │  Similarity  │ │  Wasserstein │
            │  (averaging) │ │ (comparison) │ │  (transport) │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │           MODEL MERGING             │
                    │   (Procrustes + TSV + Fisher)       │
                    └─────────────────────────────────────┘
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

- **Maurice Fréchet** (1948) - Metric space generalization of means
- **Hermann Karcher** (1977) - Riemannian center of mass
- **Facundo Mémoli** (2011) - Gromov-Wasserstein distances
- **Simon Kornblith et al.** (2019) - CKA for neural network comparison
- **Elizaveta Levina & Peter Bickel** (2004) - MLE intrinsic dimension
- **Luca Moschella et al.** (2023) - Relative representations
- **Antonio Gargiulo et al.** (2025) - Task Singular Vectors

And the many researchers advancing these fields in 2024-2025.

---

*Last updated: 2025-12-25*
