# Daily HuggingFace Paper Review — ModelCypher Geometric Filter

## Purpose

Each morning, pull yesterday's HuggingFace Daily Papers, filter for geometric/algebraic/mechanistic relevance to ModelCypher, and produce a concise report with extracted math and code for incorporation into the repository.

## Filtering Criteria (Tight: 3-5 papers/day target)

### INCLUDE papers that address any of:

**Core Geometry**
- SVD / spectral analysis of weight matrices or activations
- Singular value perturbation bounds (Weyl-type results)
- Riemannian optimization on Stiefel/Grassmann manifolds
- Cayley transforms, orthogonal parameterizations, or norm-bounded parameterizations
- Effective rank, numerical rank, stable rank of neural network matrices
- Low-rank adaptation theory (LoRA, NB-LoRA, spectral bounds on adapters)
- Null-space projections for capability preservation

**Manifold & Topology**
- Intrinsic dimension estimation (TwoNN, MLE, correlation dimension) in neural nets
- Persistent homology / Betti numbers of representation spaces
- Manifold learning applied to understanding LLM internals
- Geodesic distances in representation space
- Curvature estimation of loss landscapes or activation manifolds
- CKA, Procrustes alignment, or Gromov-Wasserstein for representation comparison

**Mechanistic / Causal Understanding**
- Mechanistic interpretability that reveals geometric structure (circuits as geometric objects)
- Causal interventions that illuminate how transformers compose representations
- Information geometry of training dynamics
- Layer-wise analysis of how representations transform (not just what they encode)
- Composition of transformations (non-commutativity, operator ordering)

**Training Theory with Geometric Foundation**
- Convergence proofs for Riemannian/constrained optimization
- Spectral analysis of gradient noise, preconditioners, or adaptive methods
- Learning rate theory grounded in spectral structure (not just empirical schedules)
- Weight decay / regularization with spectral interpretation

**Cross-Model Geometry**
- Platonic Representation Hypothesis extensions
- Cross-architecture representation alignment
- Universal structure in learned representations
- Dimensional bottleneck / abstraction phase analysis

### EXCLUDE papers that are:
- Pure benchmark papers (SOTA on X with no geometric insight)
- Prompt engineering or in-context learning (unless geometric analysis of why it works)
- Scaling laws papers (unless they derive the laws from spectral structure)
- Application papers (RAG, agents, tool use) with no geometric content
- Probabilistic interpretations that treat probability as a causal mechanism
- Distillation papers (unless spectral/geometric analysis of what transfers)
- Quantization papers (unless spectral impact analysis)

## Report Format

For each identified paper, produce:

### Paper Entry
```
## [Paper Title]
**arXiv**: https://arxiv.org/abs/XXXX.XXXXX
**Authors**: ...
**HF Upvotes**: N

### Geometric Relevance
[1-3 sentences: What geometric/algebraic insight does this paper offer?
How does it connect to ModelCypher's framework?]

### Key Math
[Extract the core mathematical result — theorems, bounds, formulas.
Use LaTeX notation. Be precise.]

### Extractable Code/Algorithms
[If the paper has code or pseudocode for geometric operations,
extract or summarize it here. Include GitHub links if available.]

### ModelCypher Integration Notes
[Specific suggestions for how this could be incorporated:
- Which ModelCypher module would benefit?
- Does it challenge or extend any existing derivation?
- Does it address any open question from RESEARCH-ROADMAP.md?]

### Evidence Level
[Rate using ModelCypher taxonomy: PROVEN / VALIDATED / EMPIRICAL / CONJECTURAL]
```

## ModelCypher Context for Filtering

ModelCypher trains LoRA adapters using ONLY geometry — every hyperparameter derived from SVD of weight matrices, IEEE 754 machine precision, or measured data. Key concepts:

- **NB-LoRA via Cayley transform**: Spectral bounds by construction
- **MASS step size**: min(Weyl ceiling, SPS, Weyl displacement) — no Lipschitz
- **Weyl perturbation monitoring**: Per-layer adapter saturation tracking
- **Shannon effective rank → LoRA rank**: tail_dims = full_rank - floor(effective_rank)
- **Cayley-Stiefel retraction**: Orthogonality constraint on adapter factors
- **CKA verification**: Post-training capability preservation
- **Geometric stopping**: Loss stability via SE_diff, adapter saturation via Weyl

The project rejects probability as a causal mechanism. A forward pass is a deterministic geometric map. Softmax is a readout projection, not the mechanism.

Open research questions that are especially high-priority:
1. Per-layer vs global learning rate (when does per-layer geometry matter?)
2. Can we derive geometry from architecture parameters?
3. Information-theoretic characterization of layer-wise mutual information
4. Training dynamics → geometry (how do hyperparameters affect spectral structure?)
5. Cross-architecture representation alignment at bottleneck layers
