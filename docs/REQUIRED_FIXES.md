# Required Fixes: Connect Orphaned Modules to UnifiedAtlasInventory

These are the specific code changes needed to unite the existing infrastructure.

---

## Fix 1: anchor_extractor.py

**File:** `src/modelcypher/core/use_cases/anchor_extractor.py`

**Problem:** Uses only 2 of 7 atlas sources (primes + gates = 141 anchors instead of 321)

**Current imports (line 11-15):**
```python
from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory
from modelcypher.core.domain.agents.semantic_prime_frames import SemanticPrimeFrames
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
```

**Should add:**
```python
from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    AtlasProbe,
)
```

**Current extraction methods (lines 171-312):**
- `_enriched_prime_anchors()` - uses SemanticPrimeFrames
- `_basic_prime_anchors()` - uses SemanticPrimeInventory
- `_computational_gate_anchors()` - uses ComputationalGateInventory

**Should add new method:**
```python
def _unified_atlas_anchors(
    self,
    tokenizer: Tokenizer,
    embedding: np.ndarray,
    vocab: int,
    confidence: dict[str, float],
) -> dict[str, np.ndarray]:
    """Extract anchors from all 321 unified atlas probes."""
    probes = UnifiedAtlasInventory.all_probes()
    anchors: dict[str, np.ndarray] = {}

    for probe in probes:
        vectors = []
        for text in probe.support_texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            valid = [tid for tid in ids if 0 <= tid < vocab]
            if valid:
                vectors.append(embedding[valid].mean(axis=0))

        if vectors:
            anchors[probe.probe_id] = np.mean(np.stack(vectors), axis=0)
            confidence[probe.probe_id] = probe.cross_domain_weight

    return anchors
```

**Modify `extract()` method (lines 40-87):**
Add config option `use_unified_atlas: bool = False` and call `_unified_atlas_anchors()` when True.

---

## Fix 2: manifold_stitcher.py

**File:** `src/modelcypher/core/domain/geometry/manifold_stitcher.py`

**Problem:** Lines 706-878 define `TriangulatedProbeBuilder` with 33 hardcoded probes that duplicate (poorly) what the atlas already has.

**DELETE these methods (lines 751-843):**
```python
@staticmethod
def _get_semantic_prime_probes() -> List[Dict[str, str]]:  # 15 probes
    ...

@staticmethod
def _get_sequence_probes() -> List[Dict[str, str]]:  # 6 probes
    ...

@staticmethod
def _get_metaphor_probes() -> List[Dict[str, str]]:  # 7 probes
    ...

@staticmethod
def _get_genealogy_probes() -> List[Dict[str, str]]:  # 5 probes
    ...
```

**REPLACE with:**
```python
from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    AtlasProbe,
    AtlasSource,
)

class TriangulatedProbeBuilder:
    """Builds probe sets from UnifiedAtlasInventory."""

    @staticmethod
    def build_triangulated_probes(
        config: Optional[TriangulatedProbingConfig] = None,
    ) -> list[AtlasProbe]:
        """Get probes from the unified atlas (321 total)."""
        return UnifiedAtlasInventory.all_probes()

    @staticmethod
    def build_probes_for_sources(
        sources: set[AtlasSource],
    ) -> list[AtlasProbe]:
        """Get probes from specific atlas sources."""
        return UnifiedAtlasInventory.probes_by_source(sources)
```

---

## Fix 3: unified_geometric_merge.py

**File:** `src/modelcypher/core/use_cases/unified_geometric_merge.py`

**Problem:** `_stage_probe()` (lines 577-705) computes CKA on raw weights without using semantic anchors.

**Current (line 597-604):**
```python
from modelcypher.core.domain.geometry.cka import (
    compute_layer_cka,
    ensemble_similarity,
)

intersection_map = {}
layer_confidences = {}
cka_scores = {}
```

**Should also import and use:**
```python
from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    MultiAtlasTriangulationScorer,
)
```

**Modify `_stage_probe()` to also compute atlas-based correlation:**
The intersection map should include probe-level correlations, not just weight-level CKA.

Add after line 680:
```python
# Atlas-based triangulation score (supplements weight CKA)
probes = UnifiedAtlasInventory.all_probes()
# For each layer, compute which probes activate similarly in both models
# This gives semantic-level intersection confidence
```

---

## Fix 4: merge_engine.py

**File:** `src/modelcypher/core/use_cases/merge_engine.py`

**Problem:** `build_shared_anchors()` (lines 664-705) takes external anchor dicts but has no method to generate them from the atlas.

**Should add convenience method:**
```python
def build_shared_anchors_from_atlas(
    self,
    source_path: str,
    target_path: str,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    source_quantization: QuantizationConfig | None = None,
    target_quantization: QuantizationConfig | None = None,
) -> SharedAnchors:
    """Build shared anchors using all 321 probes from UnifiedAtlasInventory."""
    extractor = AnchorExtractor()

    # Extract with unified atlas
    source_anchors, source_conf = extractor.extract(
        source_path,
        source_weights,
        config=AnchorExtractionConfig(use_unified_atlas=True),
        quantization=source_quantization,
        backend=self.backend,
    )

    target_anchors, target_conf = extractor.extract(
        target_path,
        target_weights,
        config=AnchorExtractionConfig(use_unified_atlas=True),
        quantization=target_quantization,
        backend=self.backend,
    )

    return self.build_shared_anchors(
        source_anchors=source_anchors,
        target_anchors=target_anchors,
        source_confidence=source_conf,
        target_confidence=target_conf,
        alignment_rank=self.options.alignment_rank,
    )
```

---

## Fix 5: shared_subspace_projector.py

**File:** `src/modelcypher/core/domain/geometry/shared_subspace_projector.py`

**Problem:** CCA alignment uses raw CRM data without atlas-based anchoring.

**Current state:** Uses `ConceptResponseMatrix` which stores per-layer concept activations, but the concept IDs come from wherever the CRM was built - not necessarily the atlas.

**Should add:**
```python
from modelcypher.core.domain.agents.unified_atlas import get_probe_ids

def validate_crm_uses_atlas(crm: ConceptResponseMatrix) -> bool:
    """Check if CRM was built using unified atlas probes."""
    atlas_ids = set(get_probe_ids())
    crm_ids = set(crm.concept_ids)
    return atlas_ids.issubset(crm_ids) or crm_ids.issubset(atlas_ids)
```

---

## Fix 6: Add relative_representation.py (NEW)

**File:** `src/modelcypher/core/domain/geometry/relative_representation.py`

**Purpose:** Implement the core algorithm from Moschella et al. (2023) "Relative Representations".

This is the mathematical foundation for dimension-agnostic transfer:

```python
"""
Relative Representations for Dimension-Agnostic Transfer.

Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Latent Space Communication"
https://arxiv.org/abs/2209.15430

Key insight: Pairwise similarities to a fixed anchor set are quasi-isometric across models,
regardless of their hidden dimension.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory


@dataclass(frozen=True)
class RelativeRepresentation:
    """Anchor-relative representation (dimension-agnostic)."""
    similarities: np.ndarray  # [n_samples, 321]
    anchor_ids: list[str]
    hidden_dim: int  # Original dimension (for reference)


def compute_relative_representation(
    hidden_states: np.ndarray,  # [n, d_model]
    anchor_embeddings: np.ndarray,  # [321, d_model]
) -> np.ndarray:  # [n, 321]
    """
    Compute anchor-relative representation.

    This maps any hidden state h ∈ R^d to s ∈ R^321 via:
        s_i = cos(h, anchor_i)

    The result is dimension-agnostic: models with d=2048 and d=896
    both produce s ∈ R^321.
    """
    # Normalize anchors
    anchor_norms = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
    anchors_normalized = anchor_embeddings / np.maximum(anchor_norms, 1e-8)

    # Normalize hidden states
    hidden_norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
    hidden_normalized = hidden_states / np.maximum(hidden_norms, 1e-8)

    # Compute cosine similarities: [n, d] @ [d, 321] = [n, 321]
    similarities = hidden_normalized @ anchors_normalized.T

    return similarities


def align_relative_representations(
    source_rel: np.ndarray,  # [n, 321]
    target_rel: np.ndarray,  # [n, 321]
) -> tuple[np.ndarray, float]:
    """
    Find optimal rotation in anchor space using Procrustes.

    Returns (rotation_matrix [321, 321], alignment_error)
    """
    # Procrustes: find R such that ||R @ source - target||_F is minimized
    M = source_rel.T @ target_rel  # [321, 321]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # Ensure proper rotation (det = +1)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute alignment error
    aligned = source_rel @ R.T
    error = np.linalg.norm(aligned - target_rel) / np.linalg.norm(target_rel)

    return R, float(error)
```

---

## Dependency Order for Fixes

1. **First:** Fix `anchor_extractor.py` (adds `_unified_atlas_anchors`)
2. **Second:** Fix `manifold_stitcher.py` (removes redundant probes)
3. **Third:** Add `relative_representation.py` (core algorithm)
4. **Fourth:** Fix `merge_engine.py` (adds `build_shared_anchors_from_atlas`)
5. **Fifth:** Fix `unified_geometric_merge.py` (uses atlas in `_stage_probe`)
6. **Sixth:** Fix `shared_subspace_projector.py` (validates CRM uses atlas)

---

## The Core Equation

After these fixes, cross-dimension transfer works:

```
Source model (d=2048):
  h_s ∈ R^2048 → cos_sim(h_s, anchors) → s_s ∈ R^321

Target model (d=896):
  h_t ∈ R^896 → cos_sim(h_t, anchors) → s_t ∈ R^321

Alignment happens in R^321 (dimension-agnostic):
  R = procrustes(s_s, s_t)  # [321, 321] rotation

Transfer:
  s_aligned = s_s @ R.T
  h_transferred = pseudo_inverse(target_anchors) @ s_aligned
```

The 321 anchors from `UnifiedAtlasInventory` are the bridge.
