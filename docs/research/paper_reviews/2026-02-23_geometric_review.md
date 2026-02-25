# ModelCypher Paper Review — 2026-02-23

**Generated**: 2026-02-25 19:58
**Papers scanned**: 18
**Papers with geometric relevance**: 2

---

## 1. Spanning the Visual Analogy Space with a Weight Basis of LoRAs

**arXiv**: [2602.15727](https://arxiv.org/abs/2602.15727)
**Authors**: Hila Manor, Rinon Gal, Haggai Maron, Tomer Michaeli, Gal Chechik
**HF Upvotes**: 12
**Geometric Relevance Score**: 5.0
**Code**: [https://github.com/NVlabs/LoRWeB](https://github.com/NVlabs/LoRWeB)

### Summary
Visual analogy learning enables image manipulation through demonstration rather than textual description, allowing users to specify complex transformations difficult to articulate in words. Given a triplet {a, a', b}, the goal is to generate b' such that a : a' :: b : b'. Recent methods adapt text-to-image models to this task using a single Low-Rank Adaptation (LoRA) module, but they face a fundamental limitation: attempting to capture the diverse space of visual transformations within a fixed adaptation module constrains generalization capabilities. Inspired by recent work showing that LoRAs in constrained domains span meaningful, interpolatable semantic spaces, we propose LoRWeB, a novel approach that specializes the model for each analogy task at inference time through dynamic composition of learned transformation primitives, informally, choosing a point in a "space of LoRAs". We introduce two key components: (1) a learnable basis of LoRA modules, to span the space of different visual transformations, and (2) a lightweight encoder that dynamically selects and weighs these basis LoRAs based on the input analogy pair. Comprehensive evaluations demonstrate our approach achieves state-of-the-art performance and significantly improves generalization to unseen visual transformations. Our findings suggest that LoRA basis decompositions are a promising direction for flexible visual manipulation. Code and data are in https://research.nvidia.com/labs/par/lorweb

### Keyword Matches
- **Core geometric** (1): \blow.rank
- **Mechanistic/structural** (1): \bcomposition\b.*\b(transform|operator|function|layer)
- **Adjacent** (1): \blora\b

### ModelCypher Integration Notes
<!-- FILL: Claude deep-dive pass populates this section -->
_Pending deep analysis — run with --deep flag or review manually._

---

## 2. 4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere

**arXiv**: [2602.10094](https://arxiv.org/abs/2602.10094)
**Authors**: Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, Chen Change Loy
**HF Upvotes**: 1
**Geometric Relevance Score**: 3.0

### Summary
We present 4RC, a unified feed-forward framework for 4D reconstruction from monocular videos. Unlike existing approaches that typically decouple motion from geometry or produce limited 4D attributes such as sparse trajectories or two-view scene flow, 4RC learns a holistic 4D representation that jointly captures dense scene geometry and motion dynamics. At its core, 4RC introduces a novel encode-once, query-anywhere and anytime paradigm: a transformer backbone encodes the entire video into a compact spatio-temporal latent space, from which a conditional decoder can efficiently query 3D geometry and motion for any query frame at any target timestamp. To facilitate learning, we represent per-view 4D attributes in a minimally factorized form by decomposing them into base geometry and time-dependent relative motion. Extensive experiments demonstrate that 4RC outperforms prior and concurrent methods across a wide range of 4D reconstruction tasks.

### Keyword Matches
- **Mechanistic/structural** (2): \brepresentation\b.*\b(geometry|structure|space|manifold|alignment), \btransformer\b.*\b(geometry|spectral|composition|operator)

### ModelCypher Integration Notes
<!-- FILL: Claude deep-dive pass populates this section -->
_Pending deep analysis — run with --deep flag or review manually._

---

## Quick Reference

| # | Score | Paper | arXiv | Code |
|---|-------|-------|-------|------|
| 1 | 5.0 | Spanning the Visual Analogy Space with a Weight Basis of LoR... | [2602.15727](https://arxiv.org/abs/2602.15727) | [repo](https://github.com/NVlabs/LoRWeB) |
| 2 | 3.0 | 4RC: 4D Reconstruction via Conditional Querying Anytime and ... | [2602.10094](https://arxiv.org/abs/2602.10094) | — |
