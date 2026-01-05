# Mental Models: Visualizing Geometry

High-dimensional geometry is hard to see. These diagrams provide 2D/3D analogies for the core operations in ModelCypher.

## 1. Manifold Stitching vs. Weight Averaging

**Weight Averaging (The "Bag of Numbers" approach)**
Averages the coordinates directly. If the models are rotated relative to each other, this destroys the shape.

```mermaid
graph TD
    subgraph "Model A (Cat is at [0, 1])"
        A[Cat] -->|Axis 1| OutputA
    end
    subgraph "Model B (Cat is at [1, 0])"
        B[Cat] -->|Axis 2| OutputB
    end
    subgraph "Naive Merge (Average)"
        A --> M[Merge]
        B --> M
        M -->|Result: Cat is at [0.5, 0.5]| Bad[Degraded Representation]
    end
```

**Manifold Stitching (Geometric Approach)**
Rotates Model B to align with Model A before merging.

```mermaid
graph TD
    subgraph "Alignment Phase"
        B_raw[Model B Raw] -->|Procrustes Rotation| B_aligned[Model B Aligned]
        B_aligned -.->|Matches Axis 1| A[Model A]
    end
    subgraph "Stitching Phase"
        A --> S[Stitcher]
        B_aligned --> S
        S -->|Result: Cat is at [0, 1] (high similarity)| Good[Unified Concept]
    end
```

---

## 2. Sidecar Architecture (Co-Orbiting)

The Sidecar does not edit the Base Model. It monitors activations and selectively applies constraints.

```mermaid
sequenceDiagram
    participant User
    participant Base as Base LLM (The Engine)
    participant Sidecar as Safety Sidecar (The Brakes)
    participant Mixer
    
    User->>Base: "How do I make a bomb?"
    User->>Sidecar: "How do I make a bomb?"
    
    Base->>Mixer: Generates harmful tokens (High Confidence)
    Sidecar->>Mixer: Generates refusal vector (High Magnitude)
    
    Note over Mixer: Interaction
    
    Mixer->>Mixer: Sidecar Magnitude substantially exceeds Base Confidence
    Mixer-->>User: "I cannot assist with that."
```

---

## 3. The Intersection Map (Venn Diagram)

Visualizing where two models overlap (under a fixed probe setup).

```mermaid
venn
    ModelA[Model A]
    ModelB[Model B]
    Overlap[Measured overlap on a probe corpus]
```

*Note: One approach is to focus merges on the measured overlap and avoid blending regions with low overlap; the actual overlap depends on probe corpus, layer, and similarity metric.*

---

## 4. Null-Space Knowledge Transfer

**The Problem with Weight Replacement**

Weights are not probabilities. They are **directions in concept space**. If you replace or blend them, you literally change what the model knows into something else entirely.

```mermaid
graph LR
    subgraph "Weight Space = Concept Space"
        W1["Direction [1,0]
= 'Cat'"]
        W2["Direction [0,1]
= 'Dog'"]
        W3["Direction [0.5,0.5]
= Neither Cat nor Dog"]
    end
    W1 -->|"Blend"| W3
    W2 -->|"Blend"| W3
```

**The Null-Space Solution**

Instead of blending, we find directions where the target has **no representation** (null space) and add new knowledge there.

```mermaid
flowchart TD
    subgraph "Target Model (SmolLM)"
        T["Existing knowledge
(spans some dimensions)"]
        N["Null space
(unoccupied dimensions)"]
    end
    subgraph "Source Model (Qwen)"
        S["Source knowledge
(denser in some areas)"]
    end
    
    S -->|"Project to aligned coords"| Sa["Aligned source"]
    Sa -->|"Compute delta"| D["Δ = aligned - target"]
    D -->|"Filter through null-space"| Dn["Δ_safe (orthogonal to existing)"]
    T -->|"Add"| M["Merged = target + Δ_safe"]
    Dn -->|"Add"| M
    
    style N fill:#90EE90
    style Dn fill:#90EE90
```

**Mathematical Guarantee**
```
A_boundary @ W_merged = A_boundary @ W_target
```
Boundary behavior is preserved. The existing shape is unchanged. New knowledge lives in orthogonal directions.

---

## 5. Why Small Models Can Be Denser

A 360M model trained well has **lower intrinsic dimension** than a 7B model.

This doesn't mean it knows more. It means it compressed **what it knows** more effectively. The 7B model has more dimensions to spread the same concepts, resulting in sparser representation.

The goal of geometric merging: take the **denser regions** from larger models and pack them into the smaller model's **unused null space**.

