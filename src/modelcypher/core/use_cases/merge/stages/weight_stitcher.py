# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Unified Weight Stitching via Activation Space Alignment.

All weight transforms follow a single geometric principle:

    W_target = F_out @ W_source @ F_in.T

Where:
    - F_out maps source output activation space → target output activation space
    - F_in maps source input activation space → target input activation space
    - The .T on F_in accounts for right-multiplication with column dimension

Weight dimensions correspond to activation spaces. A 2D weight matrix has:
    - Rows (dim 0) = output activation space
    - Cols (dim 1) = input activation space

MLP weight [intermediate, hidden] has:
    - output_space = "intermediate"
    - input_space = "hidden"

Attention weight [attn_dim, hidden] has:
    - output_space = "attention" (or "kv" for K/V in GQA)
    - input_space = "hidden"

The stitch registry maps each space to its transform. Weight stitching is then
dimension detection + registry lookup + single matmul sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

__all__ = [
    "ActivationSpace",
    "StitchRegistry",
    "stitch_weight",
    "detect_weight_spaces",
]


class ActivationSpace(Enum):
    """Known activation spaces in transformer models."""

    HIDDEN = auto()  # Main hidden dimension
    INTERMEDIATE = auto()  # MLP intermediate dimension
    ATTENTION = auto()  # Q attention dimension (num_heads * head_dim)
    KV = auto()  # K/V dimension (may differ from Q in GQA)
    V = auto()  # V-specific dimension (when K and V differ)
    EMBEDDING = auto()  # Vocabulary embedding dimension
    IDENTITY = auto()  # No transform needed (same dimensions)


@dataclass(frozen=True)
class SpaceStitch:
    """Transform pair for one activation space.

    Attributes:
        output_transform: F.T [tgt_dim, src_dim] for left-multiply (output/rows)
        input_transform: pinv(F).T [src_dim, tgt_dim] for right-multiply (input/cols)
        src_dim: Source dimension for this space
        tgt_dim: Target dimension for this space
    """

    output_transform: "Array"  # [tgt_dim, src_dim]
    input_transform: "Array"  # [src_dim, tgt_dim]
    src_dim: int
    tgt_dim: int


class StitchRegistry:
    """Registry of activation space transforms for one layer.

    The registry maps each activation space to its stitch (transform pair).
    Weight stitching is then:
        1. Detect which spaces the weight dimensions belong to
        2. Look up stitches from registry
        3. Apply: W_target = out_stitch @ W @ in_stitch
    """

    def __init__(self, backend: "Backend"):
        self._backend = backend
        self._stitches: dict[ActivationSpace, SpaceStitch] = {}
        self._dim_to_space: dict[tuple[int, str], ActivationSpace] = {}  # (dim, role) -> space

    def register(
        self,
        space: ActivationSpace,
        output_transform: "Array",
        input_transform: "Array",
    ) -> None:
        """Register a stitch for an activation space.

        Args:
            space: The activation space this stitch applies to
            output_transform: F.T [tgt_dim, src_dim] for output dimension
            input_transform: pinv(F).T [src_dim, tgt_dim] for input dimension
        """
        b = self._backend
        src_dim = int(b.shape(output_transform)[1])
        tgt_dim = int(b.shape(output_transform)[0])

        self._stitches[space] = SpaceStitch(
            output_transform=output_transform,
            input_transform=input_transform,
            src_dim=src_dim,
            tgt_dim=tgt_dim,
        )

        # Register dimension mappings for detection
        self._dim_to_space[(src_dim, "source")] = space
        self._dim_to_space[(tgt_dim, "target")] = space

    def get(self, space: ActivationSpace) -> SpaceStitch | None:
        """Get the stitch for an activation space."""
        return self._stitches.get(space)

    def detect_space(self, dim: int, role: str = "source") -> ActivationSpace | None:
        """Detect which space a dimension belongs to.

        Args:
            dim: The dimension to identify
            role: "source" or "target" to disambiguate overlapping dims

        Returns:
            The activation space, or None if unknown
        """
        return self._dim_to_space.get((dim, role))

    def get_dims(self, space: ActivationSpace) -> tuple[int, int] | None:
        """Get (src_dim, tgt_dim) for a space, or None if not registered."""
        stitch = self._stitches.get(space)
        if stitch:
            return (stitch.src_dim, stitch.tgt_dim)
        return None


def detect_weight_spaces(
    source_shape: tuple[int, int],
    registry: StitchRegistry,
) -> tuple[ActivationSpace | None, ActivationSpace | None]:
    """Detect which activation spaces a weight matrix belongs to.

    Args:
        source_shape: Shape of source weight [output_dim, input_dim]
        registry: Stitch registry with known dimensions

    Returns:
        (output_space, input_space) or (None, None) if undetectable
    """
    dim0, dim1 = source_shape

    # Try to match dimensions to known spaces
    output_space = registry.detect_space(dim0, "source")
    input_space = registry.detect_space(dim1, "source")

    return output_space, input_space


def stitch_weight(
    source_weight: "Array",
    registry: StitchRegistry,
    backend: "Backend",
    output_space: ActivationSpace | None = None,
    input_space: ActivationSpace | None = None,
) -> "Array | None":
    """Apply activation space stitches to a weight matrix.

    This is the unified weight transform:
        W_target = F_out @ W_source @ F_in.T

    Args:
        source_weight: Source weight matrix [out_dim, in_dim]
        registry: Stitch registry for this layer
        backend: Compute backend
        output_space: Override output space detection (optional)
        input_space: Override input space detection (optional)

    Returns:
        Stitched weight matrix, or None if no stitch needed/available
    """
    b = backend
    shape = b.shape(source_weight)

    if len(shape) != 2:
        return None

    dim0, dim1 = int(shape[0]), int(shape[1])

    # Detect spaces if not provided
    if output_space is None or input_space is None:
        detected_out, detected_in = detect_weight_spaces((dim0, dim1), registry)
        output_space = output_space or detected_out
        input_space = input_space or detected_in

    # Get stitches
    out_stitch = registry.get(output_space) if output_space else None
    in_stitch = registry.get(input_space) if input_space else None

    if out_stitch is None and in_stitch is None:
        return None  # No stitching needed

    result = source_weight

    # Apply output stitch (left multiply)
    if out_stitch is not None:
        # Validate dimension
        if dim0 != out_stitch.src_dim:
            logger.warning(
                "Output dim mismatch: weight[%d, _] vs stitch src_dim=%d",
                dim0,
                out_stitch.src_dim,
            )
            return None
        result = b.matmul(out_stitch.output_transform, result)

    # Apply input stitch (right multiply)
    if in_stitch is not None:
        # Validate dimension
        expected_dim = dim1 if out_stitch is None else in_stitch.src_dim
        if dim1 != in_stitch.src_dim:
            logger.warning(
                "Input dim mismatch: weight[_, %d] vs stitch src_dim=%d",
                dim1,
                in_stitch.src_dim,
            )
            return None
        result = b.matmul(result, in_stitch.input_transform)

    b.eval(result)
    return result


def build_registry_from_stitches(
    hidden_stitch: tuple["Array", "Array"] | None,
    intermediate_stitch: tuple["Array", "Array"] | None,
    attention_stitch: tuple["Array", "Array"] | None,
    k_stitch: tuple["Array", "Array"] | None,
    v_stitch: tuple["Array", "Array"] | None,
    backend: "Backend",
) -> StitchRegistry:
    """Build a stitch registry from the legacy per-space stitch tuples.

    Each stitch tuple is (output_transform, input_transform).

    Args:
        hidden_stitch: (F.T, pinv(F).T) for hidden space
        intermediate_stitch: (I.T, pinv(I).T) for intermediate space
        attention_stitch: (A.T, pinv(A).T) for Q attention space
        k_stitch: K attention stitch (may differ from Q in GQA)
        v_stitch: V attention stitch (may differ from K)
        backend: Compute backend

    Returns:
        Populated StitchRegistry
    """
    registry = StitchRegistry(backend)

    if hidden_stitch is not None:
        registry.register(ActivationSpace.HIDDEN, hidden_stitch[0], hidden_stitch[1])

    if intermediate_stitch is not None:
        registry.register(
            ActivationSpace.INTERMEDIATE, intermediate_stitch[0], intermediate_stitch[1]
        )

    if attention_stitch is not None:
        registry.register(
            ActivationSpace.ATTENTION, attention_stitch[0], attention_stitch[1]
        )

    if k_stitch is not None:
        registry.register(ActivationSpace.KV, k_stitch[0], k_stitch[1])

    if v_stitch is not None:
        registry.register(ActivationSpace.V, v_stitch[0], v_stitch[1])

    return registry
