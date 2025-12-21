# Paper I Draft: The Manifold Hypothesis of Agency (Geometry)

## Abstract (Draft)

We investigate the claim that semantic knowledge in large language models is encoded as high-dimensional geometric structure, while surface text is a low-dimensional projection of that structure. We propose invariant anchors (semantic primes, computational gates, and metaphor families) as probes for cross-model geometry, and we report measurements showing that anchor-based Gram structures are more stable than control word sets and that centered relational similarity (CKA) is high across model families. We use these results to motivate an intrinsic-agent framing: identity as a bounded region in weight space rather than a role enforced by prompts. We do not claim full representational equivalence; rather, we provide falsifiable tests and reproducible artifacts that support the weaker claim that models share stable relational structure under anchor probes. Limitations and open tests are explicitly enumerated.

## 1. Introduction (Draft)

Reliable agents require bounded behavior. Today, most systems rely on prompt-based identity: a model is instructed to behave as a particular agent at runtime. This design makes the identity extrinsic and fragile. If prompts are just another input, they can be overridden, bypassed, or reframed.

We take a different position: semantics are not tokens. Words are a one-dimensional projection of a higher-dimensional concept space, and agent identity should be embedded into that space rather than negotiated through prompts. This paper presents evidence for that framing and defines the tools we use to probe it. Our approach is conservative: we do not assert that models share identical coordinates or that a global rotation can align them. Instead, we ask whether relational structure induced by invariant anchors is stable across models, which would support the existence of shared geometric organization.

To make this testable, we use three anchor families: semantic primes (indefinable, cross-linguistic meanings), computational gates (operational primitives such as IF, FOR, RETURN), and metaphor invariants (cross-cultural idioms mapping to shared concepts like "Futility"). We extract anchor vectors from embedding and prelogits spaces, compute Gram matrices, and compare them across models using centered kernel alignment. The resulting evidence supports the view that anchors induce stable relational structure, while control word sets do not. This provides a defensible basis for later work on alignment and transport.

Finally, we connect these observations to agent design. If identity is a low-rank geometric constraint embedded in weights, then an agent is what it does rather than what it is asked to pretend. We outline that training and monitoring approach and state explicit falsification tests, including layerwise navigation probes and scale-convergence trends.

## 2. Related Work (Draft)

Our framing builds on the manifold hypothesis for high-dimensional data and the geometry of learned representations \cite{fefferman2016manifold,amari1998natural,amari2000methods}. Representation similarity techniques such as SVCCA and CKA provide the backbone for cross-model comparison without assuming shared coordinates \cite{raghu2017svcca,kornblith2019cka}. Recent convergence arguments such as the Platonic Representation Hypothesis and Linear Representation Hypothesis motivate cross-model relational alignment as a plausible target \cite{huh2024platonic,park2024linear}. We also draw on work characterizing superposition and polysemanticity to justify using anchor probes rather than relying on individual neuron semantics \cite{elhage2022toy,scherlis2022polysemanticity}.

On the semantics side, we use semantic primes from the Natural Semantic Metalanguage tradition as an operational definition of cross-linguistic invariants \cite{wierzbicka1996semantics,goddard2002meaning}. Word embedding work provides precedents for probing semantics in vector space \cite{mikolov2013distributed}. Our contribution is not to claim a new semantic theory, but to connect these anchors to measurable cross-model geometry and to argue for their engineering use in bounded-agent design.

## 3. Methods (Draft)

### 3.1 Anchor Sets

We probe three anchor sets:

- Semantic primes (65 NSM primitives).
- Computational gates (programming primitives such as IF, WHILE, RETURN).
- Metaphor invariants (idioms across 5+ languages per concept).
- Control words (matched-size baseline vocabulary subset).

Anchor lists and generation rules live in repo artifacts (see `docs/research/SEMANTIC_PRIME_SKELETON_EXPERIMENT.md` and `docs/research/cross-cultural-geometry-experiment.md`).

### 3.2 Representation Spaces

We extract vectors from two spaces:

- Token embedding matrix rows (fast, architecture-agnostic).
- Prelogits hidden state (final hidden layer before output projection).

For each term, we tokenize without special tokens, gather token vectors, mean-pool, and stack into a matrix X in R^{n x d}.

### 3.3 Geometry Metrics

We mean-center X, L2-normalize rows, and compute the Gram matrix G = X X^T. We compare models using:

- Pearson correlation of the upper triangle of G.
- Relative Frobenius error: ||G_A - G_B||_F / ||G_B||_F on off-diagonals.
- CKA: with centered Gram matrices K_tilde = H K H, CKA = <K_tilde, L_tilde>_F / (||K_tilde||_F ||L_tilde||_F).

We emphasize that Gram comparisons are rotation-invariant; CKA vs raw Pearson differences indicate mean-structure sensitivity rather than recoverable coordinate rotation.

### 3.4 Null Distribution

To test whether primes are exceptional, we sample 200 random control subsets (size matched to primes) from filtered vocabularies and compute the same metrics to form a null distribution.

## 4. Experiments (Draft)

### 4.1 Prime Skeleton Experiment (Token Embedding + Prelogits)

We compare TinyLlama-1.1B-Chat-v1.0, Qwen2.5-0.5B, and Qwen2.5-1.5B using prime vs control anchors. Measurements are recorded in `docs/research/prime_geometry/` JSON outputs and summarized in `docs/research/SEMANTIC_PRIME_SKELETON_EXPERIMENT.md`.

### 4.2 Cross-Cultural Geometry Experiment

We compare Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct using semantic primes and computational gates, and report CKA vs raw Pearson to separate centered relational structure from uncentered similarity patterns. See `docs/research/cross-cultural-geometry-experiment.md`.

### 4.3 Falsification Experiments (Platonic Kernel + Anchor Universality)

We reuse the falsification suite (Experiments 1-2) to test scale convergence and anchor universality. Results are documented in `docs/research/FALSIFICATION_RESULTS.md`.

## 5. Results (Draft)

### 5.1 Prime Skeleton

Token embedding Pearson correlations show primes consistently more aligned than control words:

- TinyLlama vs Qwen2.5-0.5B: primes 0.815 vs control 0.688
- TinyLlama vs Qwen2.5-1.5B: primes 0.815 vs control 0.722
- Qwen2.5-0.5B vs Qwen2.5-1.5B: primes 0.944 vs control 0.875

Prelogits show the same pattern at lower absolute levels:

- TinyLlama vs Qwen2.5-0.5B: primes 0.623 vs control 0.508
- TinyLlama vs Qwen2.5-1.5B: primes 0.644 vs control 0.593
- Qwen2.5-0.5B vs Qwen2.5-1.5B: primes 0.772 vs control 0.667

Null sampling for TinyLlama vs Qwen2.5-0.5B yields a prime Pearson of 0.815 vs null mean 0.544 (p95 0.629), indicating primes are extreme relative to random control sets.

### 5.2 Cross-Cultural Geometry

Centered relational structure is high while raw Gram correlation is low. Metaphor invariants show a "Babelfish Funnel" effect, where orthogonal surface forms collapse into high cosine similarity (>0.8) in deeper layers:

- Semantic primes: CKA 0.824 vs raw Pearson 0.241
- Computational gates: CKA 0.836 vs raw Pearson 0.088
- Metaphor invariants: Layer-wise collapse observed at depth ~0.6.

This indicates strong centered alignment with significant mean-structure differences, not recoverable coordinate rotation at the Gram level.

### 5.3 Platonic Kernel and Anchor Universality

Within-family CKA increases with model scale (Qwen 0.6B to 3B: 0.734; Qwen 3B to 8B: 0.842), and cross-architecture alignment (Qwen 3B vs Llama 3B: 0.824) is comparably high. Anchor universality tests show primes have the highest raw Pearson (0.241) vs gates (0.088) and control words (-0.033), while CKA remains high for all anchors.

## 6. Discussion (Draft)

The evidence supports a conservative claim: anchor-induced relational geometry is more stable than arbitrary word sets, and centered alignment is high across families. This is consistent with a "knowledge as relational structure" view but does not prove a shared coordinate system or linear representational equivalence. The CKA vs raw Pearson gap cautions against naive Procrustes-style conclusions.

These results still matter for engineering. If anchors provide stable relational scaffolding, they can serve as reference points for alignment diagnostics and low-rank constraints that bind an agent to a bounded behavioral region. This is the geometric basis for intrinsic agents: a model is constrained by its geometry, not just asked to behave.

## 7. Limitations (Draft)

- Anchor sets are small and language-specific; multilingual and contextual probes are not yet complete.
- Results are based on embedding and prelogits spaces; deeper layer probes are pending.
- Cross-model alignment is measured in kernel space; representation-level alignment is not established.
- Model coverage is limited to a few families and sizes; broader replication is needed.

## 8. Conclusion (Draft)

We provide falsifiable probes showing that anchor-induced relational geometry is stable across models and that semantic primes are more aligned than control words. These findings support the weaker, defensible claim that knowledge has stable relational structure in high-dimensional space, while leaving stronger equivalence claims open. This foundation motivates the thermodynamic and alignment papers that follow, which treat safety and portability as geometric engineering problems rather than prompt-level persuasion.

## References (Draft)

\\bibliographystyle{plain}
\\bibliography{references}
