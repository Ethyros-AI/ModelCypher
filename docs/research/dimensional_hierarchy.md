# Dimensional Hierarchy for Alignment

## Core claim

Representations are nested compressions, not independent feature spaces:

- **Binary (1D):** Bytes/bits are the base coordinate system.
- **Vocabulary + syntax (2D):** Token lattices compress the binary stream.
- **Physical structure (3D):** Spatial/causal relations are the next projection.
- **Conceptual manifold (4D+):** Abstractions are higher-order compressions.

Each layer is a compression of the layer below it. Alignment must start at the
lowest compression level and propagate upward.

## CKA as a barometer (not a scorecard)

CKA does not measure "merge quality." It signals whether two representations
are **exactly kernel-aligned** (CKA = 1.0) for a given anchor set. We keep searching
for the transformation until CKA reaches 1.0, then merge.

## Implementation touchpoints

Binary + vocabulary exact kernel alignment (1D -> 2D):
- `src/modelcypher/core/use_cases/merge_stages/stage_0_vocabulary.py`
  - Byte anchors (0-255) are aligned with `GramAligner`.
  - UnifiedAtlas anchors are kernel-aligned for vocabulary alignment.

Activation exact kernel alignment (3D+):
- `src/modelcypher/core/use_cases/merge_stages/stage_1_probe.py`
  - Probes run on each model's tokenizer; activations are compared on shared texts.
- `src/modelcypher/core/domain/geometry/gram_aligner.py`
  - Finds the exact feature transform that achieves CKA = 1.0.

## Practical implication

If the 1D/2D alignment is missing, higher-dimensional alignment is a rotation
in the wrong coordinate system. The merge must wait until the base geometry
is exactly kernel-aligned.
