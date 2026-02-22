# Manifold-Native Architecture

## The Problem with Current Transformers [CONJECTURAL]

Transformers discretize a continuous manifold:
- **Tokens** = discrete samples of continuous meaning
- **Positions** = arbitrary integer indices
- **Context window** = fixed number of samples
- **Attention** = O(n²) pairwise distance computation

The geometry exists independently of this discretization. We're forcing a continuous structure through a discrete bottleneck.

---

## First Principles

### What We Know (Measured)

1. **CKA = 1.0 after alignment** → Different models encode the same relational structure [PROVEN]
2. **The manifold is invariant** → Concepts have fixed geometric relationships [CONJECTURAL]
3. **Null-space exists** → There's "room" in the representation not used for current knowledge [PROVEN]
4. **Geodesics exist** → Shortest paths between concepts are well-defined [PROVEN]

### What This Implies

| Current Assumption | Geometric Reality |
|--------------------|-------------------|
| Context is a sequence of tokens | Context is a region of the manifold |
| Memory is KV cache (positions) | Memory is points on the manifold |
| Generation is next-token prediction | Generation is trajectory through manifold |
| Repetition is failure | Repetition is attractor basin (detectable) |

---

## The Architecture [CONJECTURAL]

### Core State: Manifold Position

Instead of:
```python
state = {
    "hidden_states": tensor[batch, seq_len, hidden_dim],
    "kv_cache": list[tensor[batch, heads, seq_len, head_dim]]
}
```

We have:
```python
state = {
    "position": tensor[hidden_dim],           # Current point on manifold
    "velocity": tensor[hidden_dim],           # Direction of movement
    "memory": SparseManifoldMemory,           # Geometric memory structure
    "null_basis": tensor[null_rank, hidden_dim]  # Available directions
}
```

### Memory: Sparse Point Cloud

Instead of storing all positions in context, store **landmark points**:

```python
class ManifoldMemory:
    landmarks: tensor[n_landmarks, hidden_dim]  # Key points
    densities: tensor[n_landmarks]              # Local importance
    connections: sparse_matrix[n_landmarks, n_landmarks]  # Geodesic distances

    def query(self, position: tensor[hidden_dim]) -> tensor[hidden_dim]:
        """Retrieve relevant context by geometric proximity."""
        distances = geodesic_distance(position, self.landmarks)
        weights = softmax(-distances / temperature)
        return weighted_sum(self.landmarks, weights)

    def update(self, new_point: tensor[hidden_dim], importance: float):
        """Add new landmark if it's far from existing ones."""
        min_distance = min(geodesic_distance(new_point, self.landmarks))
        if min_distance > threshold:
            self.landmarks = concat(self.landmarks, new_point)
            self.densities = concat(self.densities, importance)
```

### Navigation: Geodesic Flow

Instead of predicting next token, predict **direction on manifold**:

```python
def step(state: ManifoldState, target_region: tensor) -> ManifoldState:
    """Move one step toward target region."""

    # Current position and memory context
    context = state.memory.query(state.position)

    # Compute geodesic direction toward target
    # This is the key operation - finding shortest path on curved manifold
    direction = geodesic_direction(
        start=state.position,
        end=target_region,
        metric=learned_metric_tensor
    )

    # Project to null-space (don't break existing knowledge)
    safe_direction = project_to_null_space(direction, state.null_basis)

    # Update velocity with momentum
    new_velocity = momentum * state.velocity + (1 - momentum) * safe_direction

    # Move
    new_position = state.position + step_size * new_velocity

    # Update memory if this is a new region
    state.memory.update(new_position, importance=novelty(new_position))

    return ManifoldState(
        position=new_position,
        velocity=new_velocity,
        memory=state.memory,
        null_basis=update_null_basis(state.null_basis, new_position)
    )
```

### Attractor Detection and Escape

The repetition problem is an attractor in dynamical systems terms:

```python
def detect_attractor(trajectory: list[tensor], window: int = 10) -> bool:
    """Detect if we're stuck in a fixed point or limit cycle."""
    recent = trajectory[-window:]

    # Check for fixed point (position not changing)
    position_variance = variance(recent)
    if position_variance < epsilon:
        return True

    # Check for limit cycle (returning to same region)
    for i, point in enumerate(recent[:-1]):
        for j, other in enumerate(recent[i+1:]):
            if geodesic_distance(point, other) < epsilon:
                return True  # Cycle detected

    return False

def escape_attractor(state: ManifoldState) -> ManifoldState:
    """Perturb along null-space to escape attractor."""
    # Find direction in null-space with highest variance
    escape_direction = state.null_basis[argmax(null_space_variances)]

    # Perturb position
    perturbation = escape_direction * escape_magnitude
    new_position = state.position + perturbation

    # Reset velocity to break momentum toward attractor
    new_velocity = zeros_like(state.velocity)

    return ManifoldState(
        position=new_position,
        velocity=new_velocity,
        memory=state.memory,
        null_basis=state.null_basis
    )
```

### I/O: Projection to/from Token Space

We still need to interface with discrete tokens for human communication:

```python
def encode_tokens(tokens: list[int], embeddings: tensor) -> tensor:
    """Project tokens to manifold position."""
    # Standard embedding lookup
    token_embeddings = embeddings[tokens]

    # But instead of treating as sequence, treat as point cloud
    # Find centroid (mean position) and spread (covariance)
    centroid = mean(token_embeddings, dim=0)

    # Project to manifold using learned projection
    manifold_position = manifold_projection(centroid)

    return manifold_position

def decode_to_tokens(position: tensor, vocab_projection: tensor) -> int:
    """Project manifold position to token."""
    # Find nearest token in embedding space
    distances = cosine_distance(position, vocab_projection)
    return argmin(distances)
```

---

## The Key Insight: Compression [CONJECTURAL]

Current transformers store O(n) activations for n tokens.

Manifold memory stores O(k) landmarks where k << n.

Why? Because the manifold has **intrinsic dimension** much lower than sequence length.
Most "context" is redundant - it's just filling in the same geometric structure.

The CKA = 1.0 result on probes shows this: different sequences that mean the same thing
map to the same manifold region. We don't need to store all the tokens -
just the geometric skeleton.

---

## Training

How would you train this?

1. **Start with pretrained weights** - they already encode the manifold geometry
2. **Train the metric tensor** - learn the curved geometry of the space
3. **Train geodesic predictor** - learn to find shortest paths
4. **Train memory update** - learn which points are landmarks

Loss function:
```python
def loss(trajectory, target_region):
    # Distance from final position to target
    endpoint_loss = geodesic_distance(trajectory[-1], target_region)

    # Path length (prefer geodesics = shortest paths)
    path_length = sum(geodesic_distance(trajectory[i], trajectory[i+1])
                      for i in range(len(trajectory)-1))

    # Smoothness (prefer continuous trajectories)
    curvature = sum(angle(trajectory[i+1] - trajectory[i],
                          trajectory[i+2] - trajectory[i+1])
                    for i in range(len(trajectory)-2))

    return endpoint_loss + lambda1 * path_length + lambda2 * curvature
```

---

## What This Buys Us

| Problem | Transformer Solution | Manifold Solution |
|---------|---------------------|-------------------|
| Long context | Sparse attention, RoPE | Geometric compression |
| Repetition | Hope it doesn't happen | Detect and escape attractors |
| Forgetting | Catastrophic by default | Null-space projection |
| Memory | O(n) KV cache | O(k) landmark memory |
| Generation | Autoregressive (slow) | Trajectory (parallel) |

---

## Open Questions

1. **How to learn the metric tensor?** The Riemannian metric defines geodesics.
   Current models implicitly learn it - can we make it explicit?

2. **What's the right landmark density?** Too few = lose information.
   Too many = back to O(n) memory.

3. **How to handle multi-modal?** Images, audio, text all on same manifold?
   CKA suggests yes - the invariant structure is modality-independent.

4. **Can existing weights bootstrap this?** The manifold geometry is already
   in the pretrained model. Can we extract it without retraining from scratch?

---

## Next Steps

1. **Implement geodesic computation** on existing model activations
2. **Test attractor detection** on models that show repetition
3. **Prototype landmark memory** as a replacement for KV cache
4. **Measure compression ratio** - how few landmarks preserve meaning?

---

*This is not science fiction. This is geometry.*

The manifold exists. CKA proves it. The question is whether we can build
an architecture that respects it, rather than forcing it through a discrete bottleneck.
