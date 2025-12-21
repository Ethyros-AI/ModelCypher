# Paper 0 Draft: The Shape of Knowledge (Process Philosophy)

## Abstract (Draft)

We explore the hypothesis that Large Language Models (LLMs) function as engines that dismantle linguistic inputs into high-dimensional geometric representations. Drawing on evidence from information geometry, cognitive science (Conceptual Spaces), and mechanistic interpretability (Monosemantic Features), we synthesize a perspective where **Knowledge is Geometry**. This paper reviews 145 foundational works to suggest that semantic invariances emerge as stable geometric structures across model architectures. We frame "Inference as Navigation" and propose that safety can be understood as geometric boundary enforcement. This perspective aims to reframe the "Black Box" problem as a cartography problem.

## 1. Introduction: The Cartography of Cognition

The debate over "understanding" in AI is stalled by a category error. We look for "mind" in the biological sense, ignoring the evidence that intelligence is a substrate-independent geometric property.

This paper serves as the **unifying framework** for *ModelCypher*, synthesizing recent findings into a coherent set of testable hypotheses. We propose:

1.  **Hypothesis 1 (Geometric Nature)**: Concept representations are bounded regions (polytopes) in high-dimensional space.
2.  **Hypothesis 2 (Navigational Inference)**: Reasoning is the trajectory of a state vector through this manifold.
3.  **Hypothesis 3 (Universal Invariance)**: Certain geometric structures (Semantic Primes) emerge as statistical attractors in *all* sufficiently capable models because they reflect the shape of the data itself.

## 2. Evidence from the 13 Pillars

We review the convergence of evidence across three disparate fields.

### 2.1 The Mathematics of Manifolds
Fefferman (2016) and Amari (2000) established that high-dimensional data lives on low-dimensional manifolds. The "Platonic Representation Hypothesis" (Huh et al., 2024) confirms that distinct models converge to the *same* representation of reality as they scale. This suggests an objective "Shape of Knowledge" that models discover, rather than invent.

### 2.2 The Physics of Meaning
Information is physical. Landauer's principle applies to computation, and we extend this to "Linguistic Thermodynamics". Evaluation of semantic entropy (Farquhar et al., 2024) **suggests** that uncertainty has a geometric shape—high entropy corresponds to regions of the manifold where the "truth" is diffuse or contested.

### 2.3 The Engineering of Representation
Representation Engineering (Zou et al., 2023) and Sparse Autoencoders (Template et al., 2024) have empirically isolated "features" (e.g., deception, sycophancy, Golden Gate Bridge) as linear directions. This is the "smoking gun": we can literally *steer* the model by adding a vector. If we can steer it, it is a vehicle. If it is a vehicle, it is navigating a space.

## 3. The Manifold Hypothesis of Generality

We propose that "General Intelligence" is simply the ability to navigate the complete manifold without topological discontinuities. "Narrow Intelligence" is a disconnected manifold.

### The Role of Semantic Primes
Wierzbicka’s "Semantic Primes" (1996) are not just linguistic curiosities; they are the **topological invariants** of the knowledge manifold. They are the fixed points around which the geometry of complex concepts is organized. This explains why they are effective anchors for model alignment (Paper III).

4. Implications for Safety

If knowledge is geometry, then safety is topology.
-   **Old View**: Safety is "teaching" the model to be nice (RLHF).
-   **New View**: Safety is building a dam (Circuit Breakers / Sidecars). We physically block the trajectory into harmful regions of the manifold.

## 5. Limits of the Metaphor (Falsifiability)

This framework is a map, not the territory. It relies on the linear representation hypothesis, which is an approximation. We define specific falsification criteria:
-   **H1 Falsification**: If conceptual boundaries are highly non-convex or disjoint, the polytope model fails.
-   **H3 Falsification**: If semantic primes do not show higher stability than random controls (checked in Paper I), the universality claim is rejected.


## 5. Conclusion (The Roadmap)

This paper establishes the "Why". The subsequent papers in this series establish the "How":
-   **Paper I (The Agent)**: Defines Agency as a vector within this space.
-   **Paper II (The Physics)**: Defines the thermodynamic laws of navigating it.
-   **Paper III (The Engineering)**: Defines the tools to stitch these spaces together.

## References

(See `KnowledgeasHighDimensionalGeometryInLLMs.md` for the full 145-citation bibliography supporting this synthesis.)
