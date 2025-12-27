# ModelCypher Integration Architecture

## The Problem

ModelCypher has a comprehensive 402-probe atlas system in `UnifiedAtlasInventory`, but most merge and geometry modules **ignore it entirely** and reinvent their own anchor systems.

---

## The Atlas System (What You Built)

### UnifiedAtlasInventory: 402 Probes, 10 Atlas Sources

**Location:** `src/modelcypher/core/domain/agents/unified_atlas.py`

```
ATLAS SOURCES (10):
├── SEQUENCE_INVARIANT .... 68 probes  (Fibonacci, Lucas, Primes, Catalan, etc.)
├── SEMANTIC_PRIME ........ 65 probes  (Wierzbicka's Natural Semantic Metalanguage)
├── COMPUTATIONAL_GATE .... 76 probes  (Control flow, data types, functions)
├── EMOTION_CONCEPT ....... 32 probes  (Plutchik wheel + dyads)
├── TEMPORAL_CONCEPT ...... 25 probes  (Tense, duration, causality, lifecycle)
├── SOCIAL_CONCEPT ........ 25 probes  (Power, kinship, formality, status)
├── MORAL_CONCEPT ......... 30 probes  (Haidt's Moral Foundations Theory)
├── COMPOSITIONAL ......... 22 probes  (Semantic prime compositions)
├── PHILOSOPHICAL_CONCEPT . 30 probes  (Ontological, epistemological, modal)
└── CONCEPTUAL_GENEALOGY .. 29 probes  (Etymology + lineage)

TRIANGULATION DOMAINS (12):
├── MATHEMATICAL .......... Sequences, ratios, patterns
├── LOGICAL ............... Logic, conditionals, causality
├── LINGUISTIC ............ Semantic primes, speech acts
├── MENTAL ................ Mental predicates, cognitive
├── COMPUTATIONAL ......... Code gates, algorithms
├── STRUCTURAL ............ Data types, modularity
├── AFFECTIVE ............. Emotions, valence
├── RELATIONAL ............ Social, interpersonal
├── TEMPORAL .............. Time concepts
├── SPATIAL ............... Place, location
├── MORAL ................. Ethics, virtue, vice
└── PHILOSOPHICAL .......... Ontology, epistemology, logic, modality
```

### Key APIs

```python
from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    AtlasProbe,
    AtlasSource,
    AtlasDomain,
    MultiAtlasTriangulationScorer,
    get_probe_ids,
)

# Get all 402 probes
probes = UnifiedAtlasInventory.all_probes()

# Filter by source
semantic_probes = UnifiedAtlasInventory.probes_by_source({AtlasSource.SEMANTIC_PRIME})

# Filter by domain
math_probes = UnifiedAtlasInventory.probes_by_domain({AtlasDomain.MATHEMATICAL})

# Get probe texts for embedding
for probe in probes:
    anchor_texts = probe.support_texts  # Example texts for this concept
```

---

## The Merge Infrastructure (What Should Use Atlas)

### Current State: DISCONNECTED

| Module | Location | Uses Atlas? | What It Uses Instead |
|--------|----------|-------------|---------------------|
| `merge_engine.py` | use_cases | **NO** | `AnchorExtractor` (primes + gates only) |
| `unified_geometric_merge.py` | use_cases | **NO** | CKA on raw weights |
| `manifold_stitcher.py` | geometry | **NO** | `TriangulatedProbeBuilder` (39 hardcoded probes) |
| `anchor_extractor.py` | use_cases | **PARTIAL** | `SemanticPrimeFrames` + `ComputationalGateInventory` |
| `rotational_merger.py` | merging | **NO** | External anchor embeddings |
| `shared_subspace_projector.py` | geometry | **NO** | Raw CRM data |

### Required Connections

```
UnifiedAtlasInventory (402 probes)
         │
         ├──► AnchorExtractor
         │    └─ Should use all 10 atlas sources, not just 2
         │
         ├──► merge_engine.py::RotationalMerger
         │    └─ SharedAnchors should be built from atlas probes
         │
         ├──► unified_geometric_merge.py::_stage_probe()
         │    └─ Fingerprinting should use atlas probes for intersection map
         │
         ├──► manifold_stitcher.py::TriangulatedProbeBuilder
         │    └─ Should be DELETED, use UnifiedAtlasInventory instead
         │
         └──► shared_subspace_projector.py
              └─ CCA alignment should use atlas probes as anchors
```

---

## Individual Atlas Modules

### semantic_prime_atlas.py
- **65 probes** from Wierzbicka's Natural Semantic Metalanguage
- Categories: substantives, determiners, quantifiers, evaluators, mental predicates, etc.
- **Status:** Properly integrated into UnifiedAtlasInventory

### computational_gate_atlas.py
- **76 probes** for computational concepts
- Categories: control flow, functions, data types, memory, concurrency, etc.
- **Status:** Properly integrated into UnifiedAtlasInventory

### sequence_invariant_atlas.py
- **68 probes** for mathematical sequences
- Families: Fibonacci, Lucas, Tribonacci, Primes, Catalan, Ramanujan, etc.
- **Status:** Properly integrated into UnifiedAtlasInventory

### emotion_concept_atlas.py
- **32 probes** from Plutchik's wheel of emotions
- Includes 8 basic emotions + 8 dyadic blends
- **Status:** Properly integrated into UnifiedAtlasInventory

### temporal_atlas.py
- **25 probes** for temporal concepts
- Categories: tense, duration, causality, lifecycle, sequence
- **Status:** Properly integrated into UnifiedAtlasInventory (validated 2025-12-23)

### social_atlas.py
- **25 probes** for social concepts
- Categories: power hierarchy, formality, kinship, status, age
- **Status:** Properly integrated into UnifiedAtlasInventory (SMS=0.53)

### moral_atlas.py
- **30 probes** from Haidt's Moral Foundations Theory
- Foundations: care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, sanctity/degradation, liberty/oppression
- **Status:** Properly integrated into UnifiedAtlasInventory

---

## Geometry Modules

### Core Alignment (should use atlas)

| Module | Purpose | Uses Atlas? |
|--------|---------|-------------|
| `generalized_procrustes.py` | Rotation alignment | No (raw weights) |
| `shared_subspace_projector.py` | CCA-based alignment | No (CRM data) |
| `cka.py` | Dimension-independent similarity | No (raw activations) |
| `null_space_filter.py` | Safe delta projection | No (N/A) |
| `transport_guided_merger.py` | Gromov-Wasserstein merge | No (raw weights) |
| `permutation_aligner.py` | Re-Basin alignment | No (weight matching) |

### Specialized Analysis

| Module | Purpose | Status |
|--------|---------|--------|
| `manifold_stitcher.py` | Cross-model alignment | Has own probes (BAD) |
| `concept_response_matrix.py` | Per-layer concept activations | Could use atlas |
| `invariant_layer_mapper.py` | Layer correspondence | Uses sequence invariants |
| `gate_detector.py` | Computational gate detection | Uses gate atlas |

---

## What Needs to Change

### 1. AnchorExtractor: Use Full Atlas

**Current:** Uses `SemanticPrimeFrames` + `ComputationalGateInventory` (141 anchors)
**Should:** Use `UnifiedAtlasInventory` (402 probes)

```python
# In anchor_extractor.py

from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    AtlasProbe,
)

def _unified_atlas_anchors(
    self,
    tokenizer: Tokenizer,
    embedding: Array,  # Backend protocol array type
    vocab: int,
    confidence: dict[str, float],
) -> dict[str, Array]:
    """Extract anchors from all 402 unified atlas probes."""
    probes = UnifiedAtlasInventory.all_probes()
    anchors: dict[str, Array] = {}

    for probe in probes:
        vectors = []
        for text in probe.support_texts:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            valid = [tid for tid in ids if 0 <= tid < vocab]
            if valid:
                vectors.append(embedding[valid].mean(axis=0))

        if vectors:
            anchor_id = probe.probe_id
            anchors[anchor_id] = backend.mean(backend.stack(vectors), axis=0)
            confidence[anchor_id] = probe.cross_domain_weight

    return anchors
```

### 2. manifold_stitcher.py: Delete TriangulatedProbeBuilder

**Current:** `TriangulatedProbeBuilder._get_semantic_prime_probes()` returns 15 hardcoded probes
**Should:** Use `UnifiedAtlasInventory.probes_by_source({AtlasSource.SEMANTIC_PRIME})`

```python
# DELETE these methods from manifold_stitcher.py:
# - _get_semantic_prime_probes()
# - _get_sequence_probes()
# - _get_metaphor_probes()
# - _get_genealogy_probes()

# REPLACE with:
from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    DEFAULT_ATLAS_SOURCES,
)

def build_triangulated_probes() -> list[AtlasProbe]:
    """Get all 402 probes for triangulation."""
    return UnifiedAtlasInventory.all_probes()
```

### 3. unified_geometric_merge.py: Use Atlas for Fingerprinting

**Current:** `_stage_probe()` computes CKA on raw weights
**Should:** Use atlas probes to build semantic intersection map

```python
# In unified_geometric_merge.py::_stage_probe()

from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

def _stage_probe(self, ...):
    probes = UnifiedAtlasInventory.all_probes()

    # Compute probe activations for each model
    source_activations = self._probe_model(source_weights, probes)
    target_activations = self._probe_model(target_weights, probes)

    # Build intersection map from probe similarity
    for probe in probes:
        source_sim = source_activations[probe.probe_id]
        target_sim = target_activations[probe.probe_id]
        correlation = cosine_similarity(source_sim, target_sim)
        intersection_map[probe.probe_id] = correlation
```

### 4. merge_engine.py: Build SharedAnchors from Atlas

**Current:** `build_shared_anchors()` takes external anchor dicts
**Should:** Build from UnifiedAtlasInventory when called without arguments

```python
# In merge_engine.py

def build_shared_anchors_from_atlas(
    self,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
) -> SharedAnchors:
    """Build anchors from the unified atlas system."""
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    probes = UnifiedAtlasInventory.all_probes()

    source_anchors = self._embed_probes(probes, source_weights, source_tokenizer)
    target_anchors = self._embed_probes(probes, target_weights, target_tokenizer)

    return self.build_shared_anchors(
        source_anchors=source_anchors,
        target_anchors=target_anchors,
        source_confidence={p.probe_id: p.cross_domain_weight for p in probes},
        target_confidence={p.probe_id: p.cross_domain_weight for p in probes},
        alignment_rank=self.options.alignment_rank,
    )
```

---

## The Pure Geometric Algorithm (Cross-Dimension Transfer)

### Why This Matters

The atlas system enables **dimension-agnostic transfer** via anchor-relative representations:

```
Traditional (fails):
  h_source ∈ R^2048 → project → h_target ∈ R^896  (lossy)

Atlas-relative (works):
  h_source ∈ R^2048 → similarities to N probes → s ∈ R^N
  h_target ∈ R^896  → similarities to N probes → t ∈ R^N
  Transfer happens in anchor-relative space (dimension-agnostic)
```

### Reference Papers
- [Moschella et al. (2023)](https://arxiv.org/abs/2209.15430) - Relative Representations
- [Kornblith et al. (2019)](https://arxiv.org/abs/1905.00414) - CKA (dimension-agnostic similarity)

---

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedAtlasInventory                        │
│                        (402 probes)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│AnchorExtract│   │ManifoldStit │   │SharedSubspa │
│    or       │   │   cher      │   │ ceProjector │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └────────────┬────┴────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  merge_engine   │
           │RotationalMerger │
           └────────┬────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│UnifiedGeometric │   │ TransportGuided │
│     Merge       │   │     Merger      │
└─────────────────┘   └─────────────────┘
```

---

## Summary

**The atlas is the anchor system. Everything else should derive from it.**

1. `UnifiedAtlasInventory` = 402 cross-domain probes
2. All merge operations should use these as anchors
3. Cross-dimension transfer works because anchor similarities are dimension-agnostic
4. Triangulation across domains provides robustness
