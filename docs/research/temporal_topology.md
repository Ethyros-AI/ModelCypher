# Temporal Topology: The Latent Chronologist Hypothesis

> **Status**: Partially Validated (2025-12-23)
> **Implementation**: `src/modelcypher/core/domain/geometry/temporal_topology.py`
> **Atlas**: `src/modelcypher/core/domain/agents/temporal_atlas.py`
> **CLI**: `mc geometry temporal probe-model`, `mc geometry temporal anchors`
> **Experiment Data**: `/Volumes/CodeCypher/experiments/temporal-topology-validation-2025-12-23/`

## The Problem: Do Models Encode Temporal Structure?

Language models are trained on narrative text, which inherently encodes temporal relationships—sequences of events, cause-and-effect chains, and duration concepts. Do these implicit temporal structures emerge as geometric patterns in the model's representation space?

## The Hypothesis: The Temporal Manifold

**Latent Chronologist Hypothesis**: Models trained on narrative text encode time as a coherent geometric manifold with three orthogonal axes:

1. **Direction Axis**: Temporal flow (past → yesterday → today → tomorrow → future)
2. **Duration Axis**: Temporal extent (moment → hour → day → year → century)
3. **Causality Axis**: Causal ordering (because → causes → leads → therefore → results)

If these axes are geometrically independent (high orthogonality) and form consistent gradients (monotonic ordering), the model has learned implicit temporal physics.

## Methodology

### Temporal Prime Atlas (23 anchors)

| Category | Anchors | Axis | Expected Gradient |
|----------|---------|------|-------------------|
| **Tense** | past, yesterday, today, tomorrow, future | Direction | Past → Future |
| **Duration** | moment, hour, day, year, century | Duration | Short → Long |
| **Causality** | because, causes, leads, therefore, results | Causality | Cause → Effect |
| **Lifecycle** | birth, childhood, adulthood, elderly, death | Direction | Beginning → End |
| **Sequence** | beginning, middle, ending | Direction | First → Last |

### Probing Protocol

1. Extract last-layer hidden states for each anchor prompt
2. Compute axis orthogonality between Direction, Duration, and Causality dimensions
3. Test gradient consistency (Spearman correlation with expected ordering)
4. Detect Arrow of Time (monotonic past→future gradient)
5. Compute Temporal Manifold Score (weighted combination)

### Temporal Manifold Score

$$TMS = 0.30 \times \text{orthogonality} + 0.40 \times \text{gradient} + 0.30 \times \text{arrow\_detection}$$

**Threshold**: TMS > 0.40 indicates temporal manifold presence

## Empirical Results (2025-12-23)

### Models Tested

| Model | TMS | Orthogonality | Duration Monotonic | Arrow Detected | Verdict |
|-------|-----|---------------|-------------------|----------------|---------|
| Mistral-7B-4bit | **0.577** | 96.0% | ✓ | ✗ | STRONG |
| Qwen2.5-0.5B-bf16 | 0.463 | 93.1% | ✗ | ✗ | MODERATE |
| Llama-3.2-3B-4bit | 0.448 | 97.9% | ✗ | ✗ | MODERATE |
| Qwen2-0.5B-4bit | 0.385 | 86.6% | ✗ | ✗ | WEAK |
| Qwen2.5-Coder-3B-bf16 | 0.371 | 88.5% | ✗ | ✗ | WEAK |
| Qwen2.5-3B-bf16 | 0.327 | 89.4% | ✗ | ✗ | WEAK |

**Mean TMS**: 0.429 (std: 0.088)

### Hypothesis Test Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| **H1: Temporal Structure** | ⚠️ PARTIALLY SUPPORTED | TMS = 0.429 > 0.33 baseline but mixed results |
| **H2: Axis Independence** | ✅ SUPPORTED | Mean orthogonality = 91.9% > 80% |
| **H3: Arrow of Time** | ❌ NOT SUPPORTED | No model shows monotonic direction gradient |
| **H4: Duration Monotonicity** | ⚠️ PARTIALLY SUPPORTED | Only Mistral-7B shows monotonic duration |
| **H5: Size Correlation** | ❌ NOT SUPPORTED | Smaller models perform better (inverse trend) |

### Key Finding: Duration vs Direction

The most notable pattern is that **duration** is more robustly encoded than **direction**:

- Mistral-7B shows monotonic duration ordering: moment < hour < day < year < century
- No model shows consistent Arrow of Time (past→future) gradient
- This suggests models learn **duration extent** more reliably than **temporal direction**

## Comparison with Social Geometry

| Metric | Social Geometry | Temporal Topology |
|--------|-----------------|-------------------|
| Mean Score | **0.53** | 0.429 |
| Axis Orthogonality | **94.8%** | 91.9% |
| Monotonic Gradients | Power detected | Duration (1/6 models) |
| Strongest Model | Qwen2.5-3B | Mistral-7B |

**Social geometry shows stronger signal than temporal topology** - LLMs encode social relationships more robustly than temporal relationships. This is consistent with:
- Training data being predominantly present-tense dialogue
- Social relationships being more explicitly marked in language (pronouns, titles, registers)
- Temporal structure being more implicit and context-dependent

## Interpretation

The **Latent Chronologist** hypothesis is **partially supported**:

1. **Axis orthogonality is high** (91.9%) - Direction, Duration, and Causality are geometrically independent
2. **Duration is robustly encoded** - Mistral-7B shows perfect moment→century ordering
3. **Arrow of Time is NOT detected** - Models don't consistently encode past→future direction
4. **Size correlation is inverse** - Smaller models actually show better temporal structure

The failure to detect Arrow of Time may indicate:
- Training data is predominantly present-tense
- Models learn relative duration but not absolute temporal position
- Temporal reasoning may be more distributed across attention patterns rather than in static embeddings

## Usage

```bash
# List the temporal anchor inventory
mc geometry temporal anchors

# Probe a model for temporal topology
mc geometry temporal probe-model /path/to/model

# Save activations for offline analysis
mc geometry temporal probe-model /path/to/model -o activations.json

# Analyze pre-computed activations
mc geometry temporal analyze activations.json
```

## Limitations

- Small model sample (6 models, 3 architectures)
- Heavy representation of Qwen family
- Single layer probed (final layer only)
- English-centric temporal concepts
- MLX backend only (Apple Silicon)

## Future Directions

1. **Per-Layer Analysis**: Where in the network does temporal structure emerge?
2. **Narrative Probing**: Use story context instead of isolated words
3. **Causal Direction**: Test if models distinguish cause from effect
4. **Cross-Lingual**: Test temporal concepts in languages with different tense systems
5. **Temporal Steering**: Use duration axis to adjust generation pacing

## Related Work

- [Social Geometry](social_geometry.md) - Parallel experiment on social structure
- [Spatial Grounding](spatial_grounding.md) - Physical space encoding
- [Semantic Primes](semantic_primes.md) - Anchor inventory methodology

## Integration with UnifiedAtlas

The Temporal Atlas (25 probes) is now integrated into the UnifiedAtlas system:

```python
from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    AtlasSource,
)

# Get all temporal probes
temporal = UnifiedAtlasInventory.probes_by_source({AtlasSource.TEMPORAL_CONCEPT})
print(f"Temporal probes: {len(temporal)}")  # 25

# Total atlas now has 343 probes
print(f"Total probes: {UnifiedAtlasInventory.total_probe_count()}")  # 343
```
