# Operational Semantics Hypothesis

## Core Claim

Mathematical concepts are not encoded as symbolic tokens but as **structural relationships**
in the latent space. The Pythagorean theorem a² + b² = c² should manifest as a geometric
property of the embedding space, not just as memorized associations.

## Theoretical Background

If the "universe is information" and physical laws are projections from a higher-dimensional
conceptual space, then:

1. Mathematical relationships should be **discoverable** from latent geometry
2. The **execution** of a concept (what it does) defines the concept
3. Cross-modal presentations of the same concept should converge

## Experiments

### Experiment 1: Pythagorean Triple Clustering
- Embed valid Pythagorean triples: (3,4,5), (5,12,13), (8,15,17)
- Embed invalid near-misses: (3,4,6), (5,12,14), (8,15,18)
- **Hypothesis**: Valid triples cluster together, separated from invalid ones
- **Null hypothesis**: Clustering is driven by surface similarity, not mathematical validity

### Experiment 2: Geometric Transformation Recovery
- Embed (a,b) pairs and their corresponding hypotenuses c
- Test if there exists a linear/affine transformation T such that T(embed(a,b)) ≈ embed(c)
- **Hypothesis**: A consistent transformation exists across all valid triples
- **Null hypothesis**: No such transformation; each triple is memorized independently

### Experiment 3: Generalization to Novel Triples
- Use obscure Pythagorean triples unlikely to appear in training: (20,21,29), (28,45,53), (33,56,65)
- Test if clustering/transformation still holds
- **Hypothesis**: Relationship generalizes beyond memorized examples
- **Null hypothesis**: Only common triples (3,4,5) show the pattern

### Experiment 4: Cross-Modal Invariance
- Present the theorem as: algebraic formula, word problem, geometric description
- Measure CKA/cosine similarity between representations
- **Hypothesis**: Different surface forms converge to similar representations
- **Null hypothesis**: Surface form dominates the representation

### Experiment 5: Arithmetic as Geometry
- Embed numbers 1-100
- Test if addition/multiplication manifest as geometric operations
- Does embed(3) + embed(4) ≈ embed(7)? (unlikely)
- Does the *direction* from embed(3) to embed(7) match embed(1) to embed(5)? (maybe)
- **Hypothesis**: Arithmetic relationships are encoded as directions/distances

## Falsification Criteria

- If valid/invalid triples don't cluster: REJECT (relationship not encoded)
- If novel triples fail but common ones succeed: PARTIAL (memorization, not abstraction)
- If no consistent transformation exists: REJECT (no structural encoding)
- If cross-modal presentations diverge: REJECT (surface form dominates)

## Success Criteria

Strong evidence requires:
1. Clear clustering of valid triples (silhouette score > 0.3)
2. Transformation R² > 0.7 across triples
3. Generalization to 3+ novel triples
4. Cross-modal CKA > 0.6

## Implications

If confirmed, this suggests:
- LLMs encode operational semantics, not just distributional semantics
- Mathematical structure emerges from training on natural language
- The "shape of knowledge" extends to formal relationships
