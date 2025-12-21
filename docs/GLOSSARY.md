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
-   **Relevance**: Safe models often have *lower* intrinsic dimensions in their refusal mechanisms (they are simpler).

### Semantic Prime
A universal conceptual unit (e.g., "I", "YOU", "GOOD", "BAD") derived from Natural Semantic Metalanguage (NSM).
-   **Analogy**: The "GPS Satellites" of the latent space. We use them to triangulate position because they are invariant across languages and models.

### Co-Orbiting
When two models (a Base Model and a Sidecar Adapter) process the same input in parallel without merging their weights.
-   **Analogy**: A driving instructor (Sidecar) sitting next to a student (Base Model), grabbing the wheel only when necessary.

---

## Metrics

### CKA (Centered Kernel Alignment)
A measure of similarity between two neural network layers that is robust to rotation.
-   **Range**: 0.0 (Different) to 1.0 (Identical).
-   **Thresholds**:
    -   $> 0.9$: Identical structure.
    -   $> 0.7$: Compatible for merging.
    -   $< 0.4$: Disjoint manifolds.

### Jaccard Similarity (Intersection)
A measure of overlap between the *active* dimensions of two models.
-   **Formula**: $|A \cap B| / |A \cup B|$
-   **Use**: Determines if two models "speak the same language" regarding a specific prompt corpus.

### Refusal Vector Magnitude
The Euclidean length of the activation vector associated with a refusal response (e.g., "I cannot do that").
-   **Interpretation**: High magnitude = Strong refusal reflex.

### Flavor Token
Active but non-functional tokens (e.g., "Sure!", "Here is a...", "calibrating flux") that do not advance the reasoning trajectory but serve to "grease" the conversation.
-   **danger**: High-entropy flavor tokens can pull the model off the optimal manifold path (hallucination).

---

## Artifacts

### Intersection Map
A data structure (JSON) recording the layer-wise correlations between two models. It is the "Venn Diagram" of their knowledge.

### Safety Polytope
A bounded convex region in activation space defined by linear constraints. If the model's trajectory exits this polytope, the **Circuit Breaker** trips.

### Sidecar
A specialized, lightweight adapter (LoRA) trained to enforce specific geometric constraints (e.g., Safety, Persona) without altering the base model's general capabilities.
