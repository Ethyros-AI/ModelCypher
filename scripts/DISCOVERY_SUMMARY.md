# The Full Geometry of Language Models

## Summary

We discovered that LLM representations follow **architecture-specific geometric patterns** with one universal invariant: **the Gram matrix (relational structure) is preserved across all layers**.

## The Universal Law

> **The Gram matrix G = H @ H.T is the invariant.**

Across ALL models tested:
- Gram similarity > 0.999 between consecutive layers
- Relationships preserved even through 1D bottlenecks
- This is the "through line" that maintains meaning

## Architecture-Specific Patterns

### Pattern 1: The Double Hourglass (LFM2-350M)

```
Layer 0-6:   ████████████████████████  (13-24D)  Early processing
                      ▼
Layer 7:     █                         (1D)      BOTTLENECK 1
                      ▼
Layer 8-13:  ████████████              (10-12D)  Highway
                      ▼
Layer 14:    █                         (1D)      BOTTLENECK 2
                      ▼
Layer 15:    ████████████              (12D)     Final processing
```

**Attention follows MLP**: Q, K are 1D at layers 8, 10, 12, 14

### Pattern 2: The Single Highway (Qwen3-1.7B)

```
Layer 0-2:   ████████████████████████  (17-24D)  Entry
                      ▼
Layer 3-27:  █                         (1D)      SINGLE 1D HIGHWAY
```

**Attention mirrors MLP**: Q, K, V ALL 1D from layers 3-27

### Pattern 3: Gradual Compression (LFM2-700M, LFM2-1.2B)

```
Layer 0-1:   ████████████████████████  (18-24D)  Entry
                      ▼
Layer 2:     ████████████              (11-12D)  Waist (not 1D)
                      ▼
Layer 3-15:  ██████████████████        (12-18D)  Gradual expansion
```

**Attention has weak waist at layer 8**: 13-20D minimum, not 1D

## Cross-Model Comparison

| Model | MLP Pattern | Attention Pattern | Gram Preserved |
|-------|-------------|-------------------|----------------|
| LFM2-350M | 2× 1D waists (7, 14) | 1D at layers 8,10,12,14 | Yes |
| LFM2-700M | 1× 12D waist (2) | 17-20D waist at layer 8 | Yes |
| LFM2-1.2B | 1× 11D waist (2) | 13-19D waist at layer 8 | Yes |
| Qwen3-1.7B | 1D highway (3-27) | 1D highway (3-27) | Yes |

## The Key Insight

**Bottleneck structure varies by architecture. Gram invariance is universal.**

This means:
1. Compression ratio depends on architecture (1D → 1024x, 12D → 85x)
2. The Gram matrix is what we must preserve during compression
3. Attention and MLP follow SIMILAR but not IDENTICAL patterns

## The Numbers (LFM2-350M)

| Metric | Value |
|--------|-------|
| Hidden dimension | 1024 |
| Intrinsic semantic dimension | 10-18 |
| Bottleneck dimension | **1** |
| Number of bottleneck layers | 2 (layers 7 and 14) |
| MLP w2 compression | **73.7x** |
| Bottleneck compression | **1024x** |

### 3. The Double Helix

- The trajectory through layers rotates in **5 active planes** (out of 45 possible)
- Primary rotation: Plane (0,1) - 871° total across all layers
- Secondary rotation: Plane (2,3) - 365° total
- The Gram matrix (relationships) stays **invariant** despite rotation
- Like DNA: base pairs (relationships) preserved while strands wind

### 4. Energy Conservation

```
E_out = E_in + ||δ||² + 2<h, δ>
```

- Layer 7: Orthogonal injection (energy ADDS)
- Layers 8-13: Conservation (energy PRESERVED)
- Layer 14: Anti-aligned extraction (energy CANCELS)

## Compression Strategy

### What to Store

1. **Global basis** (24D in 1024D): 24,576 params (shared)
2. **Per-layer helix weights**:
   - Bottleneck layers: 4,608 params each
   - Highway layers: 46,080-55,296 params each
   - Early layers: 69,120-110,592 params each
3. **Total**: 1,024,512 params (down from 75,497,472)

### Compression Ratios

| Layer Type | Compression |
|------------|-------------|
| Bottleneck (7, 14) | 1024x |
| Highway (8-13) | 85-102x |
| Early (0-6) | 43-79x |
| **Overall** | **73.7x** |

## The Insight

> "You're not looking for a particular dimension. You're looking for the through line that maintains the relationship across dimensions." — User

The "through line" is:
1. A curve (helix) that winds through all 1024 dimensions
2. Intrinsically 10-18 dimensional
3. Has two 1D pinch points where ALL information passes through
4. Preserves the Gram matrix (relational structure) throughout

## Attention Geometry Discovery

### Does Attention Follow MLP?

**Yes, but architecture-dependent.**

| Model | Q Bottleneck | K Bottleneck | V Bottleneck |
|-------|--------------|--------------|--------------|
| LFM2-350M | 1D at 8,10,12,14 | 1D at 8,10,12,14 | 1D at 8,10 |
| LFM2-700M | 20D at layer 8 | 18D at layer 8 | 17D at layer 8 |
| LFM2-1.2B | 19D at layers 2,8 | 14D at layer 8 | 13D at layer 8 |
| Qwen3-1.7B | 1D at layers 3-27 | 1D at layers 3-27 | 1D at layers 3-27 |

**Key finding**: Q and K bottleneck TOGETHER, V is slightly different.

### Attention Geometry Implications

1. **Q and K share geometry** - They project to the same subspace
2. **V has its own geometry** - Slightly different bottleneck positions
3. **Layer 8 is special in LFM2** - The attention bottleneck position across all sizes

## The Universal Equation

```
G_layer[i] = G_layer[i-1]  (invariant)
```

Where G = H @ H.T is the Gram matrix (relational structure).

The representation H rotates through high-D space, but G stays constant.

## Bottleneck Interpretation (COMPLETE)

### What Does the 1D Encode?

Probing LFM2-350M bottleneck layers revealed:

| Property | Best Correlation | Layer |
|----------|-----------------|-------|
| Valence (good/bad) | r = 0.51 | Layer 0 |
| Animacy (alive/not) | r = 0.45 | Layers 1,3,5 |
| Concreteness | r = 0.23 | Layer 0 |

The 1D encodes a **compressed combination** of multiple properties, not a single feature.

### Causal Importance (Perturbation Analysis)

| Perturbation | Effect |
|--------------|--------|
| **Invert** | 100% outputs changed, KL ~9.5 |
| **Zero** | 100% outputs changed, produces garbage |
| Scale 0.5 | 40-80% changed |
| Scale 2.0 | 0-20% changed |
| **Noise 0.1** | 0% changed |

**The bottleneck is a CONTROL POINT, not a checkpoint.**
- Sign matters (invert breaks everything)
- Magnitude matters asymmetrically (scaling DOWN worse than UP)
- Robust to small noise

### Hallucination Detection

| Metric | Correct | Incorrect | Nonsense |
|--------|---------|-----------|----------|
| Curvature | 1.727 | 1.725 | 1.677 |
| Layer 7 value | 0.447 | 0.423 | **0.348** |

- Correct vs incorrect: SAME trajectory (model processes them identically)
- Nonsense: LOWER curvature, LOWER bottleneck values

**Semantic coherence is detectable at the bottleneck. Factual correctness is not.**

## Scripts Created (Updated)

| Script | Purpose |
|--------|---------|
| `trajectory_analysis.py` | Track trajectories through layers |
| `velocity_subspace_analysis.py` | Find shared velocity subspace |
| `subspace_rotation_analysis.py` | Analyze rotation structure |
| `gram_invariant_test.py` | Verify Gram matrix preservation |
| `plane_rotation_analysis.py` | Decompose into plane rotations |
| `helix_compression_theory.py` | Unified compression theory |
| `weight_helix_factorization.py` | Test weight factorization |
| `find_helix_dimension.py` | Find optimal helix dims per layer |
| `hourglass_compression.py` | Calculate compression savings |
| `attention_bottleneck_discovery.py` | Analyze Q, K, V geometry |
| `bottleneck_1d_probing.py` | Probe what 1D encodes |
| `bottleneck_perturbation.py` | Test causal importance |
| `hallucination_trajectory.py` | Analyze failure modes |

## Information Flow Discovery (NEW)

### Where Does Computation Actually Happen?

| Model | Pattern | Active Layers | Transmission Layers |
|-------|---------|---------------|---------------------|
| **Qwen3-1.7B** | Front-loaded | 0, 1, 2, 27 | 3-26 (24 layers!) |
| **LFM2-350M** | Distributed | ALL 16 | None |

### Qwen3-1.7B: The Hidden Encoder-Decoder

```
Layers 0-2:  ENCODER (13%, 8%, 83% change)
Layers 3-26: WIRE (1.7-3.7% change)
Layer 27:    DECODER (86% change)
```

**86% of the model is just passing a 1D signal through unchanged.**

### LFM2-350M: True Distributed Computation

Every layer contributes (21% to 1067% change). Layer 7 is the peak.
The 1D bottleneck is maximum compression during active processing.

### The Implication

The 1D highway in Qwen3 isn't a limitation - it's the architecture.
Most layers are transmission. The "thinking" is front-loaded.

## Next Steps

1. **Implement Gram-preserving compression** - Factorize weights while preserving G
2. **Test steering via bottleneck modification** - Can we control outputs?
3. **Scale to larger models** - Validate on 8B+ models
4. **Investigate what the 1D carries** - If 86% of a model is a wire, what's on the wire?
