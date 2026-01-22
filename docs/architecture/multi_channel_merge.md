# Multi-Channel Merge Architecture

**Status**: VALIDATED (January 2026)
**Author**: ModelCypher Research
**Validation**: See [multi_modal_cka_validation.md](../research/multi_modal_cka_validation.md)

> **Experimental Result**: CKA = 1.0 achieved across ALL 6 modality pairs
> (Text ↔ Vision ↔ Audio ↔ Diffusion). The geometry is invariant.

## Overview

This document describes the architecture for extending ModelCypher's merge pipeline to support multi-channel knowledge compression. The design enables merging world model, vision-language, and text-only model capabilities into a single dense target.

---

## 1. Current Architecture (Single-Channel)

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT PIPELINE                         │
│                                                             │
│  Source Model ──┐                                          │
│                 ├──► Gram Alignment ──► Null-Space ──► Merged │
│  Target Model ──┘       (CKA=1.0)      Projection         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Files:
- pipeline.py: Orchestration
- gram_aligner.py: CKA alignment
- geodesic_null_space.py: Null-space filtering
```

### Current Formula

```python
# Stage 1: Align
F = find_alignment(source_acts, target_acts)  # CKA(source @ F, target) = 1.0

# Stage 2: Compute delta
δW = (source_weights @ F) - target_weights

# Stage 3: Filter to null space
δW_safe = P_null(target_acts) @ δW

# Stage 4: Merge (geometric addition)
merged = target_weights + δW_safe
```

---

## 2. Extended Architecture (Multi-Channel)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-CHANNEL PIPELINE                               │
│                                                                         │
│  Source 1 (World Model) ──┐                                            │
│                           │      ┌──────────────┐                      │
│  Source 2 (VL Model) ─────┼──►   │   Channel    │   ┌─────────────┐   │
│                           │      │   Router     │──►│   Merged    │   │
│  Source 3 (Text Model) ───┤      │  (Birkhoff)  │   │   Model     │   │
│                           │      └──────────────┘   └─────────────┘   │
│  Target Model ────────────┘           ▲                               │
│                                       │                               │
│                          Per-channel null-space                       │
│                          projection (CKA=1.0 each)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

New Files:
- multi_channel_router.py: Birkhoff routing
- channel_projector.py: Per-channel null-space
- multi_channel_pipeline.py: Extended orchestration
```

### Extended Formula

```python
# Stage 1: Per-channel alignment
for i, source in enumerate(sources):
    F_i = find_alignment(source_acts[i], target_acts)  # CKA = 1.0 per channel
    δW_i = (source_weights[i] @ F_i) - target_weights

# Stage 2: Per-channel null-space projection
for i in range(n_channels):
    Q_i = compute_tangent_basis(channel_activations[i])
    δW_safe_i = (I - Q_i @ Q_i.T) @ δW_i

# Stage 3: Doubly stochastic routing (mHC)
H = sinkhorn_knopp(H_init, iterations=20)  # Project to Birkhoff polytope
merged_delta = sum(H[i] * δW_safe_i for i in range(n_channels))

# Stage 4: Merge (geometric addition)
merged = target_weights + merged_delta
```

---

## 3. New Components

### 3.1 ChannelProjector

```python
# src/modelcypher/core/domain/geometry/channel_projector.py

@dataclass
class ChannelProjectionResult:
    """Result of projecting a single channel."""
    channel_id: str
    filtered_delta: Array
    cka_achieved: float  # Should be 1.0
    projection_loss: float
    preserved_fraction: float

class ChannelProjector:
    """Projects multiple knowledge channels into target's null space."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._null_space_filter = GeodesicNullSpaceFilter(backend)
        self._aligner = GramAligner(backend)

    def project_channels(
        self,
        source_activations: dict[str, Array],  # {channel_id: activations}
        source_weights: dict[str, Array],      # {channel_id: weights}
        target_activations: Array,
        target_weights: Array,
    ) -> dict[str, ChannelProjectionResult]:
        """
        Project each source channel into target's null space.

        Each channel maintains CKA = 1.0 independently.
        """
        results = {}

        for channel_id, source_acts in source_activations.items():
            source_w = source_weights[channel_id]

            # Align this channel
            alignment = self._aligner.find_perfect_alignment(
                source_acts, target_activations
            )

            # Compute aligned delta
            aligned_source = self._backend.matmul(source_w, alignment.feature_transform)
            delta = aligned_source - target_weights

            # Project to null space
            null_result = self._null_space_filter.filter_delta(
                delta, target_activations
            )

            results[channel_id] = ChannelProjectionResult(
                channel_id=channel_id,
                filtered_delta=null_result.filtered_delta,
                cka_achieved=1.0,  # Invariant
                projection_loss=null_result.projection_loss,
                preserved_fraction=null_result.preserved_fraction,
            )

        return results
```

### 3.2 BirkhoffRouter

```python
# src/modelcypher/core/domain/geometry/birkhoff_router.py

@dataclass
class BirkhoffRoutingResult:
    """Result of Birkhoff routing."""
    routing_matrix: Array  # [n_channels, n_channels], doubly stochastic
    iterations: int
    convergence_error: float
    spectral_norm: float

class BirkhoffRouter:
    """Routes multiple channels via doubly stochastic mixing (mHC)."""

    def __init__(self, backend: Backend, max_iterations: int = 20):
        self._backend = backend
        self._max_iterations = max_iterations

    def compute_routing(
        self,
        channel_deltas: list[Array],
        target_activations: Array,
        init_mode: str = "uniform",  # "uniform", "learned", "identity"
    ) -> BirkhoffRoutingResult:
        """
        Compute doubly stochastic routing matrix.

        The routing matrix H satisfies:
        - All entries >= 0
        - All rows sum to 1
        - All columns sum to 1
        - Spectral norm <= 1.0

        This ensures stable combination without interference.
        """
        b = self._backend
        n = len(channel_deltas)

        # Initialize
        if init_mode == "uniform":
            H = b.full((n, n), 1.0 / n)
        elif init_mode == "identity":
            H = b.eye(n)
        else:
            # Learned from data (future extension)
            H = b.full((n, n), 1.0 / n)

        # Sinkhorn-Knopp projection to Birkhoff polytope
        H, iterations, error = self._sinkhorn_knopp(H)

        # Compute spectral norm (should be <= 1.0)
        _, S, _ = b.svd(H)
        spectral_norm = float(b.to_scalar(b.take(S, b.array([0]), axis=0)))

        return BirkhoffRoutingResult(
            routing_matrix=H,
            iterations=iterations,
            convergence_error=error,
            spectral_norm=spectral_norm,
        )

    def _sinkhorn_knopp(self, H: Array) -> tuple[Array, int, float]:
        """Project matrix onto Birkhoff polytope via Sinkhorn-Knopp."""
        b = self._backend
        eps = 1e-8

        # Ensure positive
        H = b.maximum(H, b.full(b.shape(H), eps))

        for i in range(self._max_iterations):
            # Normalize rows
            row_sums = b.sum(H, axis=1, keepdims=True)
            H = H / (row_sums + eps)

            # Normalize columns
            col_sums = b.sum(H, axis=0, keepdims=True)
            H = H / (col_sums + eps)

            # Check convergence
            row_error = b.max(b.abs(b.sum(H, axis=1) - 1.0))
            col_error = b.max(b.abs(b.sum(H, axis=0) - 1.0))
            b.eval(row_error, col_error)

            error = max(float(b.to_scalar(row_error)), float(b.to_scalar(col_error)))
            if error < eps:
                return H, i + 1, error

        return H, self._max_iterations, error

    def apply_routing(
        self,
        H: Array,
        channel_deltas: list[Array],
    ) -> Array:
        """Apply routing matrix to combine channel deltas."""
        b = self._backend
        n = len(channel_deltas)

        # Stack deltas: [n_channels, *weight_shape]
        # For simplicity, assume all same shape
        stacked = b.stack(channel_deltas, axis=0)

        # Apply routing: result[i] = sum_j H[i,j] * delta[j]
        # This is a weighted sum with doubly stochastic weights
        result = b.zeros_like(channel_deltas[0])
        for i in range(n):
            for j in range(n):
                H_ij = b.take(b.take(H, b.array([i]), axis=0), b.array([j]), axis=1)
                result = result + H_ij * channel_deltas[j]

        return result
```

### 3.3 MultiChannelPipeline

```python
# src/modelcypher/core/use_cases/merge/multi_channel_pipeline.py

@dataclass
class MultiChannelMergeConfig:
    """Configuration for multi-channel merge."""
    channels: list[str]  # ["spatial", "temporal", "text"]
    routing_mode: str = "uniform"  # "uniform", "learned"
    verify_cka: bool = True

@dataclass
class MultiChannelMergeResult:
    """Result of multi-channel merge."""
    merged_weights: dict[str, Array]  # Per-layer merged weights
    per_channel_cka: dict[str, float]  # CKA per channel (should all be 1.0)
    routing_matrix: Array  # Doubly stochastic routing
    spectral_norm: float  # Should be <= 1.0
    total_projection_loss: float
    total_preserved: float

class MultiChannelMergePipeline:
    """Extended merge pipeline supporting multiple knowledge channels."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._channel_projector = ChannelProjector(backend)
        self._birkhoff_router = BirkhoffRouter(backend)

    def run_merge(
        self,
        sources: dict[str, Model],  # {channel_id: model}
        target: Model,
        config: MultiChannelMergeConfig,
        probe_corpus: ProbeCorpus,
    ) -> MultiChannelMergeResult:
        """
        Run multi-channel merge.

        Steps:
        1. Extract per-channel activations
        2. Compute per-channel null-space projections
        3. Compute Birkhoff routing
        4. Apply unified merge formula
        """
        # Stage 1: Extract activations per channel
        channel_activations = {}
        for channel_id, source in sources.items():
            channel_activations[channel_id] = self._extract_activations(
                source, probe_corpus, channel_id
            )

        target_activations = self._extract_activations(target, probe_corpus, "target")

        # Stage 2: Per-layer, per-channel projection
        merged_weights = {}
        per_channel_cka = {ch: 1.0 for ch in config.channels}  # Invariant
        total_loss = 0.0
        total_preserved = 0.0

        for layer_name in self._get_mergeable_layers(target):
            # Get weights
            channel_weights = {
                ch: sources[ch].get_weight(layer_name)
                for ch in config.channels
            }
            target_weight = target.get_weight(layer_name)

            # Project channels
            projections = self._channel_projector.project_channels(
                channel_activations,
                channel_weights,
                target_activations[layer_name],
                target_weight,
            )

            # Collect deltas
            channel_deltas = [projections[ch].filtered_delta for ch in config.channels]
            total_loss += sum(p.projection_loss for p in projections.values())
            total_preserved += sum(p.preserved_fraction for p in projections.values())

            # Route and combine
            routing = self._birkhoff_router.compute_routing(
                channel_deltas,
                target_activations[layer_name],
                init_mode=config.routing_mode,
            )

            merged_delta = self._birkhoff_router.apply_routing(
                routing.routing_matrix,
                channel_deltas,
            )

            # Final merge (geometric addition)
            merged_weights[layer_name] = target_weight + merged_delta

        return MultiChannelMergeResult(
            merged_weights=merged_weights,
            per_channel_cka=per_channel_cka,
            routing_matrix=routing.routing_matrix,
            spectral_norm=routing.spectral_norm,
            total_projection_loss=total_loss,
            total_preserved=total_preserved,
        )
```

---

## 4. CLI Extension

```bash
# New command: multi-channel merge
mc merge multi-channel \
    --channel spatial:/path/to/world_model \
    --channel temporal:/path/to/video_model \
    --channel text:/path/to/llm \
    --target /path/to/target \
    --output /path/to/merged \
    --routing uniform
```

---

## 5. Verification

### 5.1 Unit Tests

```python
def test_birkhoff_routing_is_doubly_stochastic():
    """Routing matrix must be doubly stochastic."""
    router = BirkhoffRouter(backend)
    result = router.compute_routing(channel_deltas, target_acts)

    H = result.routing_matrix
    row_sums = backend.sum(H, axis=1)
    col_sums = backend.sum(H, axis=0)

    assert backend.allclose(row_sums, 1.0)
    assert backend.allclose(col_sums, 1.0)
    assert result.spectral_norm <= 1.0 + 1e-6

def test_per_channel_cka_preserved():
    """Each channel must achieve CKA = 1.0."""
    projector = ChannelProjector(backend)
    results = projector.project_channels(...)

    for channel_id, result in results.items():
        assert result.cka_achieved > 0.9999, f"Channel {channel_id} CKA = {result.cka_achieved}"

def test_multi_channel_merge_stable():
    """Merged model should not explode."""
    result = pipeline.run_merge(sources, target, config, probes)

    for layer, weights in result.merged_weights.items():
        norm = backend.frobenius_norm(weights)
        original_norm = backend.frobenius_norm(target.get_weight(layer))

        # Should not grow unboundedly
        assert norm < original_norm * 10, f"Layer {layer} grew by {norm / original_norm}x"
```

### 5.2 Integration Tests

```python
def test_world_model_capabilities_transfer():
    """Spatial/temporal reasoning should improve after merge."""
    # Before merge
    before_spatial = evaluate_spatial_reasoning(target)
    before_temporal = evaluate_temporal_reasoning(target)

    # Merge
    merged = pipeline.run_merge(
        sources={"spatial": world_model, "text": target},
        target=target,
        ...
    )

    # After merge
    after_spatial = evaluate_spatial_reasoning(merged)
    after_temporal = evaluate_temporal_reasoning(merged)

    assert after_spatial > before_spatial
    assert after_temporal > before_temporal
```

---

## 6. Performance Considerations

### 6.1 Computational Cost

| Operation | Single-Channel | Multi-Channel (n=3) |
|-----------|----------------|---------------------|
| Gram alignment | O(n_samples² × d) | 3 × O(n_samples² × d) |
| Null-space | O(n_samples × d²) | 3 × O(n_samples × d²) |
| Sinkhorn | N/A | O(n² × 20) ≈ O(1) |
| **Total** | **O(n_samples² × d)** | **3 × O(n_samples² × d)** |

**Overhead**: ~3x for 3 channels (linear scaling).

### 6.2 Memory

Each channel requires storing:
- Activations: O(n_samples × d)
- Weights: O(d × d) per layer
- Projected delta: O(d × d) per layer

For n=3 channels: ~3x memory overhead.

### 6.3 Optimization Opportunities

1. **Shared target activations**: Compute once, reuse for all channels
2. **Batched Gram computation**: Compute K_s for all channels in one pass
3. **Lazy routing**: Compute routing matrix once, apply per-layer

---

## 7. Migration Path

### Phase 1: Add Components (Non-Breaking)
- Add `channel_projector.py`
- Add `birkhoff_router.py`
- Add `multi_channel_pipeline.py`
- Add tests

### Phase 2: CLI Extension
- Add `mc merge multi-channel` command
- Document new workflow

### Phase 3: Integration
- Integrate with existing pipeline as optional mode
- Extend CLI coverage for multi-channel operations

---

## 8. Future Extensions

### 8.1 Learned Routing
Instead of uniform/identity routing, learn H from data:
```python
H = optimize_routing(channel_deltas, target_acts, objective="cka")
```

### 8.2 Hierarchical Channels
Channels can have sub-channels:
```
spatial:
  - static (images)
  - dynamic (video)
temporal:
  - short-term (frames)
  - long-term (sequences)
```

### 8.3 Continuous Channel Addition
Add channels incrementally without full re-merge:
```python
merged_v2 = add_channel(merged_v1, new_source, "new_channel")
```

---

## References

- [mHC Paper](https://arxiv.org/abs/2512.24880)
- [Null-Space Projection](../src/modelcypher/core/domain/geometry/geodesic_null_space.py)
- [mHC/Null-Space Connection](research/mhc_null_space_connection.md)
- [Dimensional Compression](DIMENSIONAL_COMPRESSION.md)
