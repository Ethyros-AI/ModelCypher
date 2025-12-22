# ModelCypher Glossary: A Shared Vocabulary

> **Purpose**: This document defines the precise meaning of terms used in ModelCypher. It serves as a "Handshake Protocol" between Human Users and AI Agents to ensure we are talking about the same concepts.

## Core Concepts

### Manifold
The high-dimensional geometric shape formed by a model's knowledge.
-   **Analogy**: A crumpled sheet of paper in a 3D room. The room is the Parameter Space (billions of dimensions), but the paper (the model's actual behavior) is a lower-dimensional surface.
-   **Operation**: We "stitch" manifolds together by aligning their curvatures.

### Intrinsic Dimension
The minimum number of variables needed to describe a model's state.
-   **Analogy**: A car moves in 3D space (x, y, z), but its "Intrinsic Dimension" is 2 (steering wheel angle, gas pedal).
-   **Relevance**: We explore whether some refusal/safety behaviors exhibit lower intrinsic dimension under specific probes. This is an empirical question, not a universal rule.

### Semantic Prime
A conceptual primitive (e.g., "I", "YOU", "GOOD", "BAD") derived from Natural Semantic Metalanguage (NSM), proposed as universal across human languages.
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
A data structure (JSON) recording the layer-wise correlations between two models. It is the "Venn Diagram" of their knowledge.

### Safety Polytope
A bounded convex region in activation space defined by linear constraints. If the model's trajectory exits this polytope, the **Circuit Breaker** trips.

### Sidecar
A specialized, lightweight adapter (LoRA) trained to enforce specific geometric constraints (e.g., Safety, Persona) without altering the base model's general capabilities.
