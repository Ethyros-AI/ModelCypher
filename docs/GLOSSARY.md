# ModelCypher Glossary: A Shared Vocabulary

> **Purpose**: This document defines the precise meaning of terms used in ModelCypher. It serves as a "Handshake Protocol" between Human Users and AI Agents to ensure we are talking about the same concepts.

## Core Concepts

### Manifold
The high-dimensional geometric structure induced by a model’s representations under a given task/probe setup.
-   **Analogy**: A crumpled sheet of paper in a 3D room. The room is the Parameter Space (billions of dimensions), but the paper (the model's actual behavior) is a lower-dimensional surface.
-   **Operation**: We compare/stitch manifolds by aligning representations on shared probes.

### Intrinsic Dimension
The minimum number of variables needed to describe a model's state.
-   **Analogy**: A car moves in 3D space (x, y, z), but its "Intrinsic Dimension" is 2 (steering wheel angle, gas pedal).
-   **Relevance**: We explore whether some refusal/safety behaviors exhibit lower intrinsic dimension under specific probes. This is an empirical question, not a universal rule.

### Semantic Prime
A conceptual primitive (e.g., "I", "YOU", "GOOD", "BAD") from the Natural Semantic Metalanguage (NSM) tradition, proposed (and debated) as cross-linguistically universal.
-   **ModelCypher usage**: We use semantic primes as a *candidate* anchor inventory. Whether they are invariant across model families is a falsifiable hypothesis, not an assumption.

### Co-Orbiting
When two models (a Base Model and a Sidecar Adapter) process the same input in parallel without merging their weights.
-   **Analogy**: A driving instructor (Sidecar) sitting next to a student (Base Model), grabbing the wheel only when necessary.

---

## Metrics

### CKA (Centered Kernel Alignment)
A measure of similarity between two neural network layers that is robust to rotation.
-   **Range**: 0.0 (Different) to 1.0 (Identical).
-   **Thresholds**:
    -   These cutoffs are heuristic guidelines and should be calibrated per architecture, probe corpus, and layer.

### Jaccard Similarity (Intersection)
A measure of overlap between the *active* dimensions of two models.
-   **Formula**: $|A \cap B| / |A \cup B|$
-   **Use**: Determines if two models "speak the same language" regarding a specific prompt corpus.

### Refusal Vector Magnitude
The Euclidean length of the activation vector associated with a refusal response (e.g., "I cannot do that").
-   **Interpretation**: High magnitude = Strong refusal reflex.

### Flavor Token
Active but non-functional tokens (e.g., "Sure!", "Here is a...", "calibrating flux") that do not advance the reasoning trajectory but serve to "grease" the conversation.
-   **Note**: Excessive low-information “filler” can correlate with higher entropy and lower factuality in some settings, but this is not a reliable detector on its own.

---

## Artifacts

### Intersection Map
A data structure (JSON) recording overlap diagnostics between two models under a probe corpus (e.g., layer-wise correlation/CKA-style signals).
-   **Analogy**: A “Venn diagram” of overlap under the chosen probe setup (not a claim of identical knowledge).

### Safety Polytope
A geometric safety framing in which “safe” behavior is defined as staying within a bounded region of representation space (often modeled as a convex polytope).
-   **Note**: This is an active research direction (e.g., SaP: arXiv:2505.24445); ModelCypher treats it as a conceptual target rather than a universal guarantee.

### Sidecar
A specialized, lightweight adapter (LoRA) trained to enforce specific geometric constraints (e.g., Safety, Persona) without altering the base model's general capabilities.

---

## Advanced Mathematical Concepts (AI-to-Human Analogies)

### Bhattacharyya Coefficient
A measure of overlap between two probability distributions.
-   **Analogy**: Imagine two bells (Gaussians) placed on a number line. The Bhattacharyya Coefficient measures how much they overlap. 1.0 = perfect overlap (same bell), 0.0 = no overlap (completely separate).
-   **Human explanation**: "These two models see this concept in almost the same way (high overlap)" or "These models have very different representations (low overlap)."

### Gromov-Wasserstein Distance
A measure of structural similarity between two metric spaces (manifolds) that don't share a common coordinate system.
-   **Analogy**: Comparing the street layouts of two cities without knowing their GPS coordinates. You compare "how things connect to each other" rather than absolute positions. Low distance = similar structure.
-   **Human explanation**: "The internal organization of these two models is similar, even though they don't share the same coordinate frame."

### Procrustes Alignment
A method to optimally rotate/scale one set of points to match another.
-   **Analogy**: Placing two photographs on top of each other and rotating/scaling one until the faces align as closely as possible.
-   **Human explanation**: "We're finding the best rotation to align these two models' representations so we can compare them fairly."

### Optimal Transport (Sinkhorn)
A mathematical framework for finding the cheapest way to transform one distribution into another.
-   **Analogy**: Moving piles of sand from one set of locations to another with minimum total effort. Each grain finds its "destination" and we measure the total cost.
-   **Human explanation**: "We're computing how much 'effort' it takes to morph one model's representation into another's."

### Shannon Entropy
A measure of uncertainty or information content in a probability distribution.
-   **Analogy**: A fair coin has high entropy (maximum uncertainty). A rigged coin with 99% heads has low entropy (very predictable).
-   **Human explanation**: "The model is very uncertain about what to say next" (high entropy) or "The model is confident about the next word" (low entropy).

### KL Divergence (Kullback-Leibler)
A measure of how different one probability distribution is from another.
-   **Analogy**: Measuring how surprised you'd be if you expected distribution A but got distribution B. It's asymmetric - expecting heads but getting tails is different from expecting tails but getting heads.
-   **Human explanation**: "The adapter dramatically changes what the model wants to say" (high KL) or "The adapter barely changes the output" (low KL).

### Hessian Eigenspectrum
The second-derivative matrix of the loss function, revealing the "curvature" of the optimization landscape.
-   **Analogy**: Standing on a hillside, the Hessian tells you not just the slope (gradient) but how the slope changes in each direction. Positive eigenvalues = bowl-shaped (stable). Negative = saddle point (unstable). Zero = flat direction.
-   **Human explanation**: "The model is in a stable region" (positive eigenvalues) or "The model is at a critical transition point" (mixed eigenvalues).

### Intrinsic Dimension (Two-NN)
The "true" number of independent directions in a dataset, estimated by looking at nearest-neighbor ratios.
-   **Analogy**: A piece of paper lives in 3D space but is actually 2-dimensional. Two-NN estimates how many dimensions the data "really" occupies.
-   **Human explanation**: "This model's representations live on a simpler surface than you'd expect from the raw dimension count."

## Advanced Metrology (CABE / Synthesis)

### Sectional Curvature ($K$)
A measure of the "ruggedness" of the activation manifold.
-   **Analogy**: If the latent space is a golf course, $K$ tells you if you are on a flat green (stable) or a steep bunker (chaotic).
-   **ML Equivalent**: A measure of how much a model’s logical trajectory "warps" when you change a single variable. High $K$ often correlates with hallucination boundaries.

### Ghost Anchor (Synthesis)
A relational coordinate in a Target Model synthesized from a Source Model's relational footprint.
-   **Analogy**: Placing a "Virtual Flag" in a new city by knowing its exact distance from 237 landmarks in an old city.
-   **ML Equivalent**: Zero-shot weight synthesis. We "print" a new feature footprint into a model that was never trained on that data.

### Relational Stress
The error metric for Manifold Transfer. It measures how much the relative distances between anchors drifted during projection.
-   **Analogy**: Stretching a rubber map over a globe. "Stress" is where the rubber starts to tear because the shapes don't fit perfectly.
-   **Human explanation**: "The transfer failed because the Target Model's cognitive terrain is too different from the Source's."

### Concept Volume (Influence)
Modeling a concept as a probability distribution (volume) rather than a single point (centroid).
-   **Analogy**: Instead of a "Dot" on a map, think of a "Fog Cloud." The density of the fog tells you how strongly that concept influences a specific latent region.
-   **ML Equivalent**: A Mahalanobis-regularized covariance matrix representing a concept's "Area of Effect" in the latent space.

---

## Architecture Terms (AI Legibility)

### Hexagonal Architecture (Ports and Adapters)
An architectural pattern where the core domain logic is isolated from external concerns.
-   **Analogy**: The domain is the "brain" that only speaks through well-defined interfaces (ports). Adapters translate between the brain and the outside world (databases, APIs, UIs).
-   **Human explanation**: "Core math/logic is in `domain/`, it talks to the outside world through `ports/`, and concrete implementations live in `adapters/`."

### Port
An abstract interface (Python Protocol) that defines what operations the domain needs.
-   **Location in ModelCypher**: `src/modelcypher/ports/`
-   **Example**: `Backend` protocol defines tensor operations; MLXBackend implements it for Apple Silicon.

### Adapter
A concrete implementation of a port that connects to external systems.
-   **Location in ModelCypher**: `src/modelcypher/adapters/`
-   **Example**: `local_training.py` implements the `TrainingEngine` port.
