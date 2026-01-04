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

"""Permutation Aligner (Git Re-Basin).

Solves the permutation symmetry problem for neural network merging.

Neural networks have N! permutation symmetries per layer. If Model A has Neuron[1]="CAT"
and Model B has Neuron[5]="CAT", naive weight averaging fails because it mixes unrelated
features. This aligner finds the optimal permutation P that "un-spins" the neurons.

SCOPE
-----
- MLP blocks (up_proj, gate_proj, down_proj): FULLY RE-BASINED
- Attention blocks: NOT re-basined (multi-head structure makes generic
  permutation unsafe without head-aware alignment - see is_attention_weight())
- Embeddings/norms: Passed through unchanged

DIMENSION HANDLING
------------------
For transformer MLP triplets (up_proj, gate_proj, down_proj):
- up_proj/gate_proj: Permute OUTPUT dimension (intermediate_dim, align_output=True)
- down_proj: Permute INPUT dimension (same permutation, align_input=True)

This maintains functional equivalence through the MLP:
    output = down_proj(gate_proj(x) * up_proj(x))

The SAME permutation P is applied consistently:
    up_proj:   P @ W (permute rows)
    gate_proj: P @ W (permute rows, same P)
    down_proj: W @ P^T (permute columns, same P)

ASSUMPTIONS
-----------
- Neuron identity lives on the INTERMEDIATE dimension
- Hidden dimension identity is preserved across layers
- LoRA adapters follow standard PEFT conventions (B @ A decomposition)

ALGORITHM
---------
1. Use semantic prime anchors to probe each model's neuron responses
2. Compute geodesic cosine similarity between source and target neuron activations
3. Hungarian algorithm for optimal bipartite matching (O(N³))
4. Sign correction: handle ±1 symmetry per neuron
5. Return: P (permutation), S (signs) such that W_aligned = S @ P @ W @ P^T @ S^T

References:
    - Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2022).
      "Git Re-Basin: Merging Models modulo Permutation Symmetries."
      arXiv:2209.04836. https://arxiv.org/abs/2209.04836
    - Kuhn, H. W. (1955). "The Hungarian Method for the Assignment Problem."
      Naval Research Logistics Quarterly 2(1-2):83-97.
      https://doi.org/10.1002/nav.3800020109
    - Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023).
      "TIES-Merging: Resolving Interference When Merging Models."
      arXiv:2306.01708. https://arxiv.org/abs/2306.01708
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.hungarian import hungarian_assignment
from modelcypher.core.domain.geometry.vector_math import geodesic_cosine_between_sets

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.geometry.permutation_aligner")

# Memory-derived threshold for sparse vs dense permutation representation.
# Dense matrix = N × N × 4 bytes (float32). At N=4096, matrix is 64MB.
# Beyond this, use sparse representation (assignment indices) to avoid OOM.
_DENSE_MATRIX_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024  # 64 MB
_SPARSE_THRESHOLD_N = int((_DENSE_MATRIX_MEMORY_BUDGET_BYTES / 4) ** 0.5)  # = 4096


class PermutationAlignerError(Exception):
    """Error during permutation alignment."""

    pass


class PermutationAlignerErrorKind(str, Enum):
    """Kind of permutation alignment error."""

    INVALID_SHAPE = "invalid_shape"
    DIMENSION_MISMATCH = "dimension_mismatch"
    ALIGNMENT_FAILED = "alignment_failed"


@dataclass(frozen=True)
class AlignmentResult:
    """Result of permutation alignment between two weight matrices."""

    permutation: "Array"  # [N, N]
    signs: "Array"  # [N, N] diagonal or [N] vector
    match_quality: float
    match_confidences: list[float]
    sign_flip_count: int
    is_sparse_permutation: bool = False
    assignment_indices: list[int] | None = None


@dataclass(frozen=True)
class AnchorActivationContext:
    """Anchor activation snapshots for layer-aware permutation alignment."""

    anchor_ids: list[str]
    source_by_layer: dict[int, list[list[float]]]
    target_by_layer: dict[int, list[list[float]]]

    def activations(self, layer: int) -> tuple[list[list[float]], list[list[float]]] | None:
        """Get source and target activations for a specific layer."""
        source = self.source_by_layer.get(layer)
        target = self.target_by_layer.get(layer)
        if source is None or target is None:
            return None
        if len(source) != len(target):
            return None
        return source, target


class PermutationAligner:
    """
    Solves the permutation symmetry problem for neural network merging.

    Ported 1:1 from the reference Swift implementation.
    """

    @staticmethod
    def align(
        source_weight: "Array",
        target_weight: "Array",
        anchors: "Array | None" = None,
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        """
        Computes the optimal permutation and sign alignment between two weight matrices.
        """
        b = backend or get_default_backend()

        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError(
                f"Weights must be 2D matrices. Got source={source_weight.ndim}D, target={target_weight.ndim}D"
            )

        source_out, source_in = source_weight.shape
        target_out, target_in = target_weight.shape

        if source_out != target_out or source_in != target_in:
            raise ValueError(
                f"Weight dimensions must match. Source: [{source_out}, {source_in}], Target: [{target_out}, {target_in}]"
            )

        N = source_out
        source_signatures = None
        target_signatures = None

        if anchors is not None:
            # Anchor-grounded: project weights through anchors
            anchor_dim = anchors.shape[1]
            if source_in == anchor_dim:
                # Direct: sourceWeight @ anchors.T gives [N, numAnchors]
                source_signatures = b.matmul(
                    b.astype(source_weight, "float32"), b.transpose(anchors)
                )
                target_signatures = b.matmul(
                    b.astype(target_weight, "float32"), b.transpose(anchors)
                )
                logger.debug(f"Anchor-grounded (input match): [{N}, {anchors.shape[0]}]")
            else:
                raise PermutationAlignerError(
                    f"Anchor dim {anchor_dim} does not match weight dims "
                    f"[{source_out}, {source_in}]"
                )
        else:
            # Direct: use weight rows as signatures
            source_signatures = b.astype(source_weight, "float32")
            target_signatures = b.astype(target_weight, "float32")
            logger.debug(f"Using direct weight signatures: [{N}, {source_in}]")

        if source_signatures.shape[0] != N or target_signatures.shape[0] != N:
            raise PermutationAlignerError("Anchor signatures shape mismatch")

        # Compute full geodesic similarity matrix: [N, N]
        similarity = geodesic_cosine_between_sets(source_signatures, target_signatures, b)
        b.eval(similarity)

        # Convert similarity to cost matrix on backend to avoid O(N^2) Python loops.
        # We want to MAXIMIZE similarity, but Hungarian MINIMIZES cost:
        # cost = max_abs_sim - abs(sim)
        abs_similarity = b.abs(similarity)
        max_abs_sim_arr = b.max(abs_similarity)
        b.eval(max_abs_sim_arr)
        max_abs_sim = float(b.to_scalar(max_abs_sim_arr))
        cost_matrix_arr = max_abs_sim - abs_similarity
        b.eval(cost_matrix_arr)
        assignment_arr = hungarian_assignment(cost_matrix_arr, b)
        assignment = [int(x) for x in b.tolist(assignment_arr)]

        # Compute signed similarity for assigned pairs without pulling full matrix to CPU.
        row_idx = b.arange(N)
        flat_idx = row_idx * N + assignment_arr
        flat_sim = b.reshape(similarity, (-1,))
        sim_selected = b.take(flat_sim, flat_idx, axis=0)
        sim_abs = b.abs(sim_selected)
        neg_mask = sim_selected < 0
        signs_arr = b.where(
            neg_mask,
            b.full(sim_selected.shape, -1.0),
            b.full(sim_selected.shape, 1.0),
        )
        sign_flip_count_arr = b.sum(b.astype(neg_mask, "float32"))
        b.eval(sim_selected, sim_abs, signs_arr, sign_flip_count_arr)
        sim_abs_list = b.tolist(sim_abs)
        signs_list = b.tolist(signs_arr)
        sign_flip_count = int(b.to_scalar(sign_flip_count_arr))

        # Compute signs and confidences from the optimal assignment
        match_confidences = [float(v) for v in sim_abs_list]
        signs = [float(v) for v in signs_list]

        # Build target-ordered sign/confidence arrays
        signs_target = [1.0] * N
        confidences_target = [0.0] * N

        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        # Build permutation via backend column take: P = I[:, assignment]
        identity = b.eye(N)
        permutation = b.take(identity, assignment_arr, axis=1)
        permutation = b.astype(permutation, "float32")
        sign_matrix = b.astype(b.diag(b.array(signs_target)), "float32")
        b.eval(permutation, sign_matrix)

        mean_quality_arr = b.mean(sim_abs)
        b.eval(mean_quality_arr)
        mean_quality = float(b.to_scalar(mean_quality_arr))

        logger.info(f"Aligned {N} neurons: quality={mean_quality:.3f}, signFlips={sign_flip_count}")

        return AlignmentResult(
            permutation=permutation,
            signs=sign_matrix,
            match_quality=mean_quality,
            match_confidences=confidences_target,
            sign_flip_count=sign_flip_count,
        )

    @staticmethod
    def apply(
        weight: "Array",
        alignment: AlignmentResult,
        align_output: bool = True,
        align_input: bool = False,
        backend: "Backend | None" = None,
    ) -> "Array":
        """Applies permutation and sign alignment to a weight matrix."""
        b = backend or get_default_backend()
        w = b.astype(weight, "float32")

        if alignment.is_sparse_permutation and alignment.assignment_indices is not None:
            # Sparse logic
            indices = alignment.assignment_indices
            count = len(indices)

            # Inverse permutation logic
            inverse = [0] * count
            for i, tgt in enumerate(indices):
                if 0 <= tgt < count:
                    inverse[tgt] = i

            sign_values = [float(x) for x in b.tolist(alignment.signs)]

            index_tensor = b.array(inverse)

            if align_output:
                w = b.take(w, index_tensor, axis=0)
                sign_row = b.astype(b.reshape(b.array(sign_values), (count, 1)), "float32")
                w = w * sign_row

            if align_input:
                w = b.take(w, index_tensor, axis=1)
                sign_col = b.astype(b.reshape(b.array(sign_values), (1, count)), "float32")
                w = w * sign_col

            b.eval(w)
            return w

        # Dense logic
        if align_output:
            # W' = S @ P @ W
            permuted = b.matmul(alignment.permutation, w)
            if alignment.signs.ndim == 1:
                sign_row = b.reshape(alignment.signs, (-1, 1))
                w = permuted * sign_row
            else:
                w = b.matmul(alignment.signs, permuted)

        if align_input:
            # W' = W @ P^T @ S^T (S is diagonal => S^T = S)
            permuted = b.matmul(w, b.transpose(alignment.permutation))
            if alignment.signs.ndim == 1:
                sign_col = b.reshape(alignment.signs, (1, -1))
                w = permuted * sign_col
            else:
                w = b.matmul(permuted, alignment.signs)

        b.eval(w)
        return w

    @staticmethod
    def align_via_anchor_activations(
        source_weight: "Array",
        target_weight: "Array",
        source_anchors: "Array",
        target_anchors: "Array",
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        """Aligns neurons using per-layer anchor activations."""
        b = backend or get_default_backend()

        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError("Weights must be 2D")

        input_dim = source_weight.shape[1]
        source_anchor_dim = source_anchors.shape[1]
        target_anchor_dim = target_anchors.shape[1]

        if source_anchor_dim != input_dim or target_anchor_dim != input_dim:
            raise PermutationAlignerError(
                "Anchor activation dim mismatch."
            )

        source_signatures = b.matmul(
            b.astype(source_weight, "float32"), b.transpose(b.astype(source_anchors, "float32"))
        )
        target_signatures = b.matmul(
            b.astype(target_weight, "float32"), b.transpose(b.astype(target_anchors, "float32"))
        )
        b.eval(source_signatures, target_signatures)

        return PermutationAligner._align_from_signatures(
            source_signatures, target_signatures, backend=b
        )

    @staticmethod
    def _align_from_signatures(
        source_signatures: "Array",
        target_signatures: "Array",
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        """Aligns neurons using exact Hungarian assignment over signature similarity."""
        b = backend or get_default_backend()

        if source_signatures.ndim != 2 or target_signatures.ndim != 2:
            raise ValueError("Signatures must be 2D matrices")

        N = source_signatures.shape[0]
        if target_signatures.shape[0] != N:
            raise PermutationAlignerError("Signature count mismatch")

        source_fp32 = b.astype(source_signatures, "float32")
        target_fp32 = b.astype(target_signatures, "float32")

        similarity = geodesic_cosine_between_sets(source_fp32, target_fp32, b)
        b.eval(similarity)

        abs_similarity = b.abs(similarity)
        max_abs_sim_arr = b.max(abs_similarity)
        b.eval(max_abs_sim_arr)
        max_abs_sim = float(b.to_scalar(max_abs_sim_arr))
        cost_matrix_arr = max_abs_sim - abs_similarity
        b.eval(cost_matrix_arr)
        assignment_arr = hungarian_assignment(cost_matrix_arr, b)
        assignment = [int(x) for x in b.tolist(assignment_arr)]

        row_idx = b.arange(N)
        flat_idx = row_idx * N + assignment_arr
        flat_sim = b.reshape(similarity, (-1,))
        sim_selected = b.take(flat_sim, flat_idx, axis=0)
        sim_abs = b.abs(sim_selected)
        b.eval(sim_selected, sim_abs)
        sim_selected_list = b.tolist(sim_selected)
        sim_abs_list = b.tolist(sim_abs)

        signs = [1.0] * N
        match_confidences = [0.0] * N
        sign_flip_count = 0

        for src_idx in range(N):
            sim = float(sim_selected_list[src_idx])
            match_confidences[src_idx] = float(sim_abs_list[src_idx])
            if sim < 0:
                signs[src_idx] = -1.0
                sign_flip_count += 1

        signs_target = [1.0] * N
        confidences_target = [0.0] * N
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        avg_quality_arr = b.mean(sim_abs)
        b.eval(avg_quality_arr)
        avg_quality = float(b.to_scalar(avg_quality_arr))

        if N > _SPARSE_THRESHOLD_N:
            return AlignmentResult(
                permutation=b.astype(b.array(assignment), "float32"),
                signs=b.astype(b.array(signs_target), "float32"),
                match_quality=avg_quality,
                match_confidences=confidences_target,
                sign_flip_count=sign_flip_count,
                is_sparse_permutation=True,
                assignment_indices=assignment,
            )

        identity = b.eye(N)
        permutation = b.take(identity, assignment_arr, axis=1)
        permutation = b.astype(permutation, "float32")
        sign_matrix = b.astype(b.diag(b.array(signs_target)), "float32")

        return AlignmentResult(
            permutation=permutation,
            signs=sign_matrix,
            match_quality=avg_quality,
            match_confidences=confidences_target,
            sign_flip_count=sign_flip_count,
        )

    @staticmethod
    def rebasin_mlp_with_activations(
        source_weights: "dict[str, Array]",
        target_weights: "dict[str, Array]",
        source_anchors: "Array",
        target_anchors: "Array",
        anchor_activations: AnchorActivationContext | None = None,
        backend: "Backend | None" = None,
    ) -> "tuple[dict[str, Array], float, int]":
        """Performs MLP-only re-basin alignment with separate source/target anchors.

        Each model needs its own anchor embeddings because different models encode
        concepts at different locations, even for same-architecture models.

        Args:
            source_weights: Source model weights by key.
            target_weights: Target model weights by key.
            source_anchors: Source model anchor embeddings [numAnchors, anchorDim].
            target_anchors: Target model anchor embeddings [numAnchors, anchorDim].
            anchor_activations: Optional per-layer anchor activation context.
            backend: Optional backend for array operations.

        Returns:
            Tuple of (aligned_weights, average_quality, mlp_blocks_aligned).
        """
        b = backend or get_default_backend()

        aligned_weights: "dict[str, Array]" = {}
        total_quality = 0.0
        mlp_blocks_aligned = 0

        up_proj_keys = [
            k for k in source_weights.keys() if "up_proj" in k and k.endswith(".weight")
        ]
        up_proj_keys.sort()

        logger.info(f"Found {len(up_proj_keys)} MLP blocks for anchor-projected re-basin")

        for up_key in up_proj_keys:
            gate_key = up_key.replace("up_proj", "gate_proj")
            down_key = up_key.replace("up_proj", "down_proj")

            source_up = source_weights.get(up_key)
            target_up = target_weights.get(up_key)
            source_gate = source_weights.get(gate_key)
            target_gate = target_weights.get(gate_key)
            source_down = source_weights.get(down_key)
            target_down = target_weights.get(down_key)

            if not all(
                [
                    source_up is not None,
                    target_up is not None,
                    source_gate is not None,
                    target_gate is not None,
                    source_down is not None,
                    target_down is not None,
                ]
            ):
                logger.warning(f"Incomplete MLP block for {up_key}, skipping")
                continue

            # Compute alignment using per-layer activations if available, else global anchors
            layer_idx = PermutationAligner._extract_layer_index(up_key)
            alignment: AlignmentResult

            # Per-layer anchor activations take priority if available
            use_per_layer = False
            if anchor_activations is not None and layer_idx is not None:
                activations = anchor_activations.activations(layer_idx)
                if activations is not None and len(activations[0]) > 0 and len(activations[1]) > 0:
                    logger.debug(f"Using anchor activations for layer {layer_idx}")
                    src_act = PermutationAligner._array_from_matrix(
                        activations[0], backend=b
                    )
                    tgt_act = PermutationAligner._array_from_matrix(
                        activations[1], backend=b
                    )
                    alignment = PermutationAligner.align_via_anchor_activations(
                        source_up, target_up, src_act, tgt_act, backend=b
                    )
                    use_per_layer = True

            if not use_per_layer:
                # Use separate source/target anchors for proper cross-model alignment
                alignment = PermutationAligner.align_via_anchor_activations(
                    source_up, target_up, source_anchors, target_anchors, backend=b
                )

            # Apply permutation (sparse or dense)
            if alignment.is_sparse_permutation and alignment.assignment_indices is not None:
                signed_up, signed_gate, aligned_down = (
                    PermutationAligner._apply_sparse_mlp_permutation(
                        b.astype(source_up, "float32"),
                        b.astype(source_gate, "float32"),
                        b.astype(source_down, "float32"),
                        alignment.assignment_indices,
                        alignment.signs,
                        backend=b,
                    )
                )
            else:
                # Dense application
                aligned_up = b.matmul(alignment.permutation, b.astype(source_up, "float32"))
                signed_up = b.matmul(alignment.signs, aligned_up)

                aligned_gate = b.matmul(alignment.permutation, b.astype(source_gate, "float32"))
                signed_gate = b.matmul(alignment.signs, aligned_gate)

                permuted_down = b.matmul(
                    b.astype(source_down, "float32"), b.transpose(alignment.permutation)
                )
                aligned_down = b.matmul(permuted_down, alignment.signs)

            # Get dtype string from source array
            source_dtype = str(source_up.dtype) if hasattr(source_up, "dtype") else "float32"
            aligned_weights[up_key] = b.astype(signed_up, source_dtype)
            aligned_weights[gate_key] = b.astype(signed_gate, source_dtype)
            aligned_weights[down_key] = b.astype(aligned_down, source_dtype)

            total_quality += alignment.match_quality
            mlp_blocks_aligned += 1

            logger.debug(
                f"MLP block {up_key}: quality={alignment.match_quality:.3f}, "
                f"signFlips={alignment.sign_flip_count}"
            )

        # Copy all other weights unchanged
        for key, value in source_weights.items():
            if key not in aligned_weights:
                aligned_weights[key] = value

        avg_quality = total_quality / max(1, mlp_blocks_aligned)
        logger.info(
            f"MLP re-basin complete: {mlp_blocks_aligned} blocks aligned, avg quality: {avg_quality:.3f}"
        )
        return aligned_weights, avg_quality, mlp_blocks_aligned

    @staticmethod
    def _apply_sparse_mlp_permutation(
        source_up: "Array",
        source_gate: "Array",
        source_down: "Array",
        indices: list[int],
        signs: "Array",
        backend: "Backend | None" = None,
    ) -> "tuple[Array, Array, Array]":
        """Apply sparse permutation to MLP weights without building full [N, N] matrix.

        For large intermediate dimensions (e.g., 14336), this avoids 800MB+ memory allocation.
        Instead, we use index-based reordering which is O(N) memory.

        Args:
            source_up: up_proj weight [intermediate, hidden].
            source_gate: gate_proj weight [intermediate, hidden].
            source_down: down_proj weight [hidden, intermediate].
            indices: Assignment indices where indices[i] = target index for source i.
            signs: Sign diagonal matrix or vector (target order).
            backend: Optional backend for array operations.

        Returns:
            Tuple of aligned (up, gate, down) weights.
        """
        b = backend or get_default_backend()

        intermediate = source_up.shape[0]

        # Extract sign values (target order)
        sign_values = PermutationAligner._extract_sign_values(signs, intermediate, backend=b)

        # Build inverse permutation: invP[target] = source
        inv_indices = PermutationAligner._inverse_permutation(indices, intermediate)

        # Create index tensor for gather operation
        index_tensor = b.astype(b.array(inv_indices), "int32")

        # Gather rows: result[j, :] = source[invIndices[j], :]
        permuted_up = b.take(source_up, index_tensor, axis=0)
        permuted_gate = b.take(source_gate, index_tensor, axis=0)

        # Apply signs: multiply each row by its sign
        sign_col = b.astype(b.reshape(b.array(sign_values), (intermediate, 1)), "float32")
        signed_up = permuted_up * sign_col
        signed_gate = permuted_gate * sign_col

        # For down_proj: permute columns
        permuted_down = b.take(source_down, index_tensor, axis=1)

        # Apply signs: multiply each column by its sign
        sign_row = b.astype(b.reshape(b.array(sign_values), (1, intermediate)), "float32")
        signed_down = permuted_down * sign_row

        b.eval(signed_up, signed_gate, signed_down)
        return signed_up, signed_gate, signed_down

    @staticmethod
    def _extract_layer_index(key: str) -> int | None:
        """Extract layer index from weight key."""
        patterns = [".layers.", ".h.", ".blocks.", ".block."]
        for pattern in patterns:
            if pattern in key:
                idx = PermutationAligner._parse_index_after(pattern, key)
                if idx is not None:
                    return idx
        return None

    @staticmethod
    def _parse_index_after(needle: str, haystack: str) -> int | None:
        """Parse integer index after a substring."""
        idx = haystack.find(needle)
        if idx < 0:
            return None
        suffix = haystack[idx + len(needle) :]
        digits = ""
        for ch in suffix:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return None
        return int(digits)

    @staticmethod
    def _array_from_matrix(
        matrix: list[list[float]],
        backend: "Backend | None" = None,
    ) -> "Array":
        """Convert 2D list to Array."""
        b = backend or get_default_backend()
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        flat = [x for row in matrix for x in row]
        return b.reshape(b.array(flat), (rows, cols))

    @staticmethod
    def _inverse_permutation(indices: list[int], count: int) -> list[int]:
        """Compute inverse permutation."""
        inverse = list(range(count))
        for src, tgt in enumerate(indices):
            if 0 <= tgt < count:
                inverse[tgt] = src
        return inverse

    @staticmethod
    def _extract_sign_values(
        signs: "Array",
        expected_count: int,
        backend: "Backend | None" = None,
    ) -> list[float]:
        """Extract sign values from matrix or vector."""
        b = backend or get_default_backend()

        if signs.ndim == 1:
            values = [float(x) for x in b.tolist(signs)]
        else:
            diag = b.diag(signs)
            b.eval(diag)
            values = [float(x) for x in b.tolist(diag)]

        if len(values) != expected_count:
            raise PermutationAlignerError(
                f"Sign vector size mismatch (expected {expected_count}, got {len(values)})"
            )
        return values

    @staticmethod
    def is_mlp_weight(key: str) -> bool:
        """Check if a weight key is part of the MLP (safe to permute)."""
        return any(
            pattern in key
            for pattern in [
                "up_proj",
                "gate_proj",
                "down_proj",
                "w1",
                "w2",
                "w3",
            ]
        )

    @staticmethod
    def is_attention_weight(key: str) -> bool:
        """Check if a weight key is attention (NOT safe to permute with generic aligner)."""
        return any(
            pattern in key
            for pattern in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "wq",
                "wk",
                "wv",
                "wo",
            ]
        )
