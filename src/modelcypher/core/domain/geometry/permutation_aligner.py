"""Permutation Aligner (Git Re-Basin).

Solves the permutation symmetry problem for neural network merging.

Neural networks have N! permutation symmetries per layer. If Model A has Neuron[1]="CAT"
and Model B has Neuron[5]="CAT", naive weight averaging fails because it mixes unrelated
features. This aligner finds the optimal permutation P that "un-spins" the neurons.

Based on: Ainsworth et al. (2022) "Git Re-Basin" and Yadav et al. (2023) "TIES-Merging"

Algorithm:
    1. Use semantic prime anchors to probe each model's neuron responses
    2. Compute cosine similarity between source and target neuron activations
    3. Greedy assignment: O(N²) instead of O(N³) Hungarian
    4. Sign correction: handle ±1 symmetry per neuron
    5. Return: P (permutation), S (signs) such that W_aligned = S @ P @ W @ P^T @ S^T
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.geometry.permutation_aligner")


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
    assignment_indices: Optional[list[int]] = None


@dataclass(frozen=True)
class Config:
    """Configuration for permutation alignment."""
    min_match_threshold: float = 0.1
    use_anchor_grounding: bool = True
    top_k: int = 5


@dataclass(frozen=True)
class AnchorActivationContext:
    """Anchor activation snapshots for layer-aware permutation alignment."""

    anchor_ids: list[str]
    source_by_layer: dict[int, list[list[float]]]
    target_by_layer: dict[int, list[list[float]]]

    def activations(self, layer: int) -> Optional[Tuple[list[list[float]], list[list[float]]]]:
        """Get source and target activations for a specific layer."""
        source = self.source_by_layer.get(layer)
        target = self.target_by_layer.get(layer)
        if source is None or target is None:
            return None
        if len(source) != len(target):
            return None
        return source, target


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for confidence-weighted fusion (TIES-Merging).

    Implements TIES-Merging principles:
    1. Only merge neurons that are geometrically aligned (high confidence).
    2. For unaligned neurons, preserve the dominant signal (or base).
    """

    interference_threshold: float = 0.5
    """Threshold for constructive interference (averaging).
    Matches with confidence > this will be averaged."""

    source_alpha: float = 0.5
    """Weight for the source model (0.0-1.0)."""

    normalize: bool = False
    """Whether to normalize weights before averaging."""

    @staticmethod
    def default() -> FusionConfig:
        """Default fusion configuration."""
        return FusionConfig()


class PermutationAligner:
    """
    Solves the permutation symmetry problem for neural network merging.
    
    Ported 1:1 from the reference Swift implementation.
    """

    @staticmethod
    def align(
        source_weight: "Array",
        target_weight: "Array",
        anchors: "Optional[Array]" = None,
        config: Config = Config(),
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        """
        Computes the optimal permutation and sign alignment between two weight matrices.
        """
        b = backend or get_default_backend()

        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError(f"Weights must be 2D matrices. Got source={source_weight.ndim}D, target={target_weight.ndim}D")

        source_out, source_in = source_weight.shape
        target_out, target_in = target_weight.shape

        if source_out != target_out or source_in != target_in:
            raise ValueError(f"Weight dimensions must match. Source: [{source_out}, {source_in}], Target: [{target_out}, {target_in}]")

        N = source_out
        source_signatures = None
        target_signatures = None

        if config.use_anchor_grounding and anchors is not None:
            # Anchor-grounded: project weights through anchors
            anchor_dim = anchors.shape[1]
            if source_in == anchor_dim:
                # Direct: sourceWeight @ anchors.T gives [N, numAnchors]
                source_signatures = b.matmul(b.astype(source_weight, "float32"), b.transpose(anchors))
                target_signatures = b.matmul(b.astype(target_weight, "float32"), b.transpose(anchors))
                logger.debug(f"Anchor-grounded (input match): [{N}, {anchors.shape[0]}]")
            elif source_out == anchor_dim:
                logger.warning(f"Anchor dim {anchor_dim} matches output; using direct weight signatures for alignment")
                source_signatures = b.astype(source_weight, "float32")
                target_signatures = b.astype(target_weight, "float32")
            else:
                logger.warning(f"Anchor dim {anchor_dim} doesn't match weight dims [{source_out}, {source_in}], using direct")
                source_signatures = b.astype(source_weight, "float32")
                target_signatures = b.astype(target_weight, "float32")
        else:
            # Direct: use weight rows as signatures
            source_signatures = b.astype(source_weight, "float32")
            target_signatures = b.astype(target_weight, "float32")
            logger.debug(f"Using direct weight signatures: [{N}, {source_in}]")

        if source_signatures.shape[0] != N or target_signatures.shape[0] != N:
            logger.warning(f"Anchor signatures shape mismatch; using direct weight signatures")
            source_signatures = b.astype(source_weight, "float32")
            target_signatures = b.astype(target_weight, "float32")

        # Normalize signatures
        source_norms = b.sqrt(b.sum(source_signatures * source_signatures, axis=1, keepdims=True)) + 1e-8
        target_norms = b.sqrt(b.sum(target_signatures * target_signatures, axis=1, keepdims=True)) + 1e-8
        source_normalized = source_signatures / source_norms
        target_normalized = target_signatures / target_norms

        # Compute full similarity matrix: [N, N]
        similarity = b.matmul(source_normalized, b.transpose(target_normalized))
        b.eval(similarity)

        # Pull to CPU (numpy) for greedy assignment
        sim_data = b.to_numpy(similarity).tolist()

        # Greedy assignment: O(N^2)
        assignment = [-1] * N
        signs = [1.0] * N
        match_confidences = [0.0] * N
        used_targets = set()
        sign_flip_count = 0

        # Sort source neurons by max similarity
        source_order = []
        for i in range(N):
            row = sim_data[i]
            max_sim = max(abs(x) for x in row)
            source_order.append((i, max_sim))

        source_order.sort(key=lambda x: x[1], reverse=True)

        for src_idx, _ in source_order:
            best_target = -1
            best_sim = 0.0
            best_abs = -float('inf')

            row = sim_data[src_idx]
            for tgt_idx in range(N):
                if tgt_idx in used_targets:
                    continue
                sim = row[tgt_idx]
                abs_sim = abs(sim)
                if abs_sim > best_abs:
                    best_target = tgt_idx
                    best_sim = sim
                    best_abs = abs_sim

            if best_target >= 0 and best_abs >= config.min_match_threshold:
                assignment[src_idx] = best_target
                used_targets.add(best_target)
                match_confidences[src_idx] = float(best_abs)

                if best_sim < 0:
                    signs[src_idx] = -1.0
                    sign_flip_count += 1

        # Handle unassigned
        remaining_targets = set(range(N)) - used_targets
        sorted_remaining = sorted(list(remaining_targets))

        for src_idx in range(N):
            if assignment[src_idx] < 0:
                if src_idx in remaining_targets:
                    assignment[src_idx] = src_idx
                    remaining_targets.remove(src_idx)
                elif sorted_remaining:
                    tgt = sorted_remaining.pop(0)
                    assignment[src_idx] = tgt
                    match_confidences[src_idx] = 0.0
                    remaining_targets.discard(tgt)

        # Build target-ordered sign/confidence arrays
        signs_target = [1.0] * N
        confidences_target = [0.0] * N

        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        # Build permutation
        perm_data = [0.0] * (N * N)
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                perm_data[tgt * N + src] = 1.0

        permutation = b.astype(b.reshape(b.array(perm_data), (N, N)), "float32")
        sign_matrix = b.astype(b.diag(b.array(signs_target)), "float32")
        b.eval(permutation, sign_matrix)

        mean_quality = sum(confidences_target) / max(N, 1)

        logger.info(f"Aligned {N} neurons: quality={mean_quality:.3f}, signFlips={sign_flip_count}")

        return AlignmentResult(
            permutation=permutation,
            signs=sign_matrix,
            match_quality=mean_quality,
            match_confidences=confidences_target,
            sign_flip_count=sign_flip_count
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

            # Extract signs
            sign_values = b.to_numpy(alignment.signs).tolist()
            if hasattr(sign_values, 'tolist'):
                sign_values = sign_values if isinstance(sign_values, list) else b.to_numpy(alignment.signs).tolist()

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
    def align_via_anchor_projection(
        source_weight: "Array",
        target_weight: "Array",
        anchors: "Array",
        config: Config = Config(),
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        """Aligns neurons using low-dimensional anchor projections."""
        b = backend or get_default_backend()

        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError(f"Weights must be 2D. Got source={source_weight.ndim}D, target={target_weight.ndim}D")

        N = source_weight.shape[0]
        if N != target_weight.shape[0]:
            raise ValueError(f"Output dimensions must match. Source: {N}, Target: {target_weight.shape[0]}")

        input_dim = source_weight.shape[1]
        anchor_dim = anchors.shape[1]

        source_signatures = None
        target_signatures = None

        if input_dim == anchor_dim:
            source_signatures = b.matmul(b.astype(source_weight, "float32"), b.transpose(anchors))
            target_signatures = b.matmul(b.astype(target_weight, "float32"), b.transpose(anchors))
        else:
            logger.warning(f"Weight dim {input_dim} != anchor dim {anchor_dim}, using weight row norms")
            source_fp32 = b.astype(source_weight, "float32")
            target_fp32 = b.astype(target_weight, "float32")
            source_norms = b.sqrt(b.sum(source_fp32 * source_fp32, axis=1, keepdims=True))
            target_norms = b.sqrt(b.sum(target_fp32 * target_fp32, axis=1, keepdims=True))
            source_signatures = source_norms
            target_signatures = target_norms

        b.eval(source_signatures, target_signatures)
        return PermutationAligner._align_from_signatures(source_signatures, target_signatures, config, backend=b)

    @staticmethod
    def align_via_anchor_activations(
        source_weight: "Array",
        target_weight: "Array",
        source_anchors: "Array",
        target_anchors: "Array",
        config: Config = Config(),
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        """Aligns neurons using per-layer anchor activations."""
        b = backend or get_default_backend()

        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError("Weights must be 2D")

        N = source_weight.shape[0]
        input_dim = source_weight.shape[1]
        source_anchor_dim = source_anchors.shape[1]
        target_anchor_dim = target_anchors.shape[1]

        if source_anchor_dim != input_dim or target_anchor_dim != input_dim:
            logger.warning(f"Anchor activation dim mismatch. Returning projection alignment.")
            return PermutationAligner.align_via_anchor_projection(
                source_weight, target_weight, source_anchors, config, backend=b
            )

        source_signatures = b.matmul(b.astype(source_weight, "float32"), b.transpose(b.astype(source_anchors, "float32")))
        target_signatures = b.matmul(b.astype(target_weight, "float32"), b.transpose(b.astype(target_anchors, "float32")))
        b.eval(source_signatures, target_signatures)

        return PermutationAligner._align_from_signatures(source_signatures, target_signatures, config, backend=b)

    @staticmethod
    def _align_from_signatures(
        source_signatures: "Array",
        target_signatures: "Array",
        config: Config,
        backend: "Backend | None" = None,
    ) -> AlignmentResult:
        b = backend or get_default_backend()

        if source_signatures.ndim != 2 or target_signatures.ndim != 2:
            raise ValueError("Signatures must be 2D matrices")

        N = source_signatures.shape[0]

        source_fp32 = b.astype(source_signatures, "float32")
        target_fp32 = b.astype(target_signatures, "float32")

        # Normalize
        source_norms = b.sqrt(b.sum(source_fp32 * source_fp32, axis=1, keepdims=True)) + 1e-8
        target_norms = b.sqrt(b.sum(target_fp32 * target_fp32, axis=1, keepdims=True)) + 1e-8

        source_normalized = source_fp32 / source_norms
        target_normalized = target_fp32 / target_norms

        # Batched Similarity
        batch_size = 512
        if N >= 4096: batch_size = 128
        if N >= 8000: batch_size = 32
        if N > 12000: batch_size = 16

        logger.debug(f"Using batch size {batch_size} for N={N}")

        assignment = [-1] * N
        signs = [1.0] * N
        match_confidences = [0.0] * N
        used_targets = set()
        sign_flip_count = 0

        source_order = []

        # Batched computation on GPU
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            source_slice = source_normalized[batch_start:batch_end]
            sim_slice = b.matmul(source_slice, b.transpose(target_normalized))  # [batch, N]

            abs_sim = b.abs(sim_slice)
            best_targets_gpu = b.argmax(abs_sim, axis=1)  # [batch]
            best_sims_gpu = b.max(abs_sim, axis=1)  # [batch]

            # For signed sim, we need to gather
            # Use indexing: sim_slice[batch_indices, best_targets_gpu]
            batch_indices = b.arange(batch_end - batch_start)
            best_signed_gpu = sim_slice[batch_indices, best_targets_gpu]

            b.eval(best_targets_gpu, best_sims_gpu, best_signed_gpu)

            best_targets = b.to_numpy(best_targets_gpu).tolist()
            best_sims = b.to_numpy(best_sims_gpu).tolist()
            best_signed = b.to_numpy(best_signed_gpu).tolist()

            for i in range(len(best_targets)):
                source_order.append((
                    batch_start + i,
                    best_sims[i],
                    int(best_targets[i]),
                    best_signed[i]
                ))

        # Sort
        source_order.sort(key=lambda x: x[1], reverse=True)

        needs_recompute = []

        for src_idx, max_sim, best_target, signed_sim in source_order:
            if max_sim < config.min_match_threshold:
                continue

            if best_target not in used_targets:
                assignment[src_idx] = best_target
                match_confidences[src_idx] = float(max_sim)
                used_targets.add(best_target)
                if signed_sim < 0:
                    signs[src_idx] = -1.0
                    sign_flip_count += 1
            else:
                needs_recompute.append(src_idx)

        # Recompute conflicts
        if needs_recompute:
            logger.debug(f"Recomputing {len(needs_recompute)} sources")
            sim_full_arr = b.matmul(source_normalized, b.transpose(target_normalized))
            b.eval(sim_full_arr)
            sim_full = b.to_numpy(sim_full_arr).tolist()

            for src_idx in needs_recompute:
                row = sim_full[src_idx]
                best_target = -1
                best_sim = 0.0
                best_abs = -float('inf')

                for tgt_idx in range(N):
                    if tgt_idx in used_targets:
                        continue
                    sim = row[tgt_idx]
                    abs_sim_val = abs(sim)
                    if abs_sim_val > best_abs:
                        best_target = tgt_idx
                        best_sim = sim
                        best_abs = abs_sim_val

                if best_target >= 0 and best_abs >= config.min_match_threshold:
                    assignment[src_idx] = best_target
                    match_confidences[src_idx] = float(best_abs)
                    used_targets.add(best_target)
                    if best_sim < 0:
                        signs[src_idx] = -1.0
                        sign_flip_count += 1

        # Fill remaining
        remaining_targets = set(range(N)) - used_targets
        sorted_remaining = sorted(list(remaining_targets))

        for src_idx in range(N):
            if assignment[src_idx] < 0:
                if src_idx in remaining_targets:
                    assignment[src_idx] = src_idx
                    remaining_targets.remove(src_idx)
                elif sorted_remaining:
                    tgt = sorted_remaining.pop(0)
                    assignment[src_idx] = tgt
                    match_confidences[src_idx] = 0.0
                    remaining_targets.discard(tgt)

        # Target arrays
        signs_target = [1.0] * N
        confidences_target = [0.0] * N
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        avg_quality = sum(confidences_target) / max(1, N)

        if N > 4096:
            # Sparse return
            return AlignmentResult(
                permutation=b.astype(b.array(assignment), "float32"),  # abuse of notation, but keeps ID
                signs=b.astype(b.array(signs_target), "float32"),
                match_quality=avg_quality,
                match_confidences=confidences_target,
                sign_flip_count=sign_flip_count,
                is_sparse_permutation=True,
                assignment_indices=assignment
            )
        else:
            # Dense return
            perm_data = [0.0] * (N * N)
            for src, tgt in enumerate(assignment):
                if tgt >= 0:
                    perm_data[tgt * N + src] = 1.0
                else:
                    # Identity fallback for safety
                    perm_data[src * N + src] = 1.0

            permutation = b.astype(b.reshape(b.array(perm_data), (N, N)), "float32")
            sign_matrix = b.astype(b.diag(b.array(signs_target)), "float32")

            return AlignmentResult(
                permutation=permutation,
                signs=sign_matrix,
                match_quality=avg_quality,
                match_confidences=confidences_target,
                sign_flip_count=sign_flip_count
            )

    @staticmethod
    def rebasin_mlp_only(
        source_weights: "dict[str, Array]",
        target_weights: "dict[str, Array]",
        anchors: "Array",
        config: Config = Config(),
        backend: "Backend | None" = None,
    ) -> "Tuple[dict[str, Array], float, int]":
        """Performs MLP-only re-basin alignment."""
        b = backend or get_default_backend()

        aligned_weights: "dict[str, Array]" = {}
        total_quality = 0.0
        mlp_blocks_aligned = 0

        up_proj_keys = [k for k in source_weights.keys() if "up_proj" in k and k.endswith(".weight")]
        up_proj_keys.sort()

        for up_key in up_proj_keys:
            gate_key = up_key.replace("up_proj", "gate_proj")
            down_key = up_key.replace("up_proj", "down_proj")

            source_up = source_weights.get(up_key)
            target_up = target_weights.get(up_key)
            source_gate = source_weights.get(gate_key)
            target_gate = target_weights.get(gate_key)
            source_down = source_weights.get(down_key)
            target_down = target_weights.get(down_key)

            if not all([source_up is not None, target_up is not None, source_gate is not None,
                        target_gate is not None, source_down is not None, target_down is not None]):
                continue

            # Align based on up_proj
            # up_proj: [intermediate, hidden] => align rows (dim 0)
            alignment = PermutationAligner.align_via_anchor_projection(
                source_up, target_up, anchors, config, backend=b
            )

            # Apply to source
            # up_proj: align output
            aligned_up = PermutationAligner.apply(source_up, alignment, align_output=True, align_input=False, backend=b)

            # gate_proj: align output (same permutation)
            aligned_gate = PermutationAligner.apply(source_gate, alignment, align_output=True, align_input=False, backend=b)

            # down_proj: align input (dim 1)
            aligned_down = PermutationAligner.apply(source_down, alignment, align_output=False, align_input=True, backend=b)

            aligned_weights[up_key] = aligned_up
            aligned_weights[gate_key] = aligned_gate
            aligned_weights[down_key] = aligned_down

            total_quality += alignment.match_quality
            mlp_blocks_aligned += 1

        # Copy all other weights unchanged (attention, norms, embeddings)
        for key, value in source_weights.items():
            if key not in aligned_weights:
                aligned_weights[key] = value

        avg_quality = total_quality / max(1, mlp_blocks_aligned)
        logger.info(
            f"MLP re-basin complete: {mlp_blocks_aligned} blocks aligned, avg quality: {avg_quality:.3f}"
        )
        return aligned_weights, avg_quality, mlp_blocks_aligned

    @staticmethod
    def rebasin_mlp_with_activations(
        source_weights: "dict[str, Array]",
        target_weights: "dict[str, Array]",
        anchors: "Array",
        anchor_activations: Optional[AnchorActivationContext] = None,
        config: Config = Config(),
        backend: "Backend | None" = None,
    ) -> "Tuple[dict[str, Array], float, int]":
        """Performs MLP-only re-basin alignment with optional per-layer anchor activations.

        Args:
            source_weights: Source model weights by key.
            target_weights: Target model weights by key.
            anchors: Semantic prime anchor embeddings [numAnchors, anchorDim].
            anchor_activations: Optional per-layer anchor activation context.
            config: Alignment configuration.
            backend: Optional backend for array operations.

        Returns:
            Tuple of (aligned_weights, average_quality, mlp_blocks_aligned).
        """
        b = backend or get_default_backend()

        aligned_weights: "dict[str, Array]" = {}
        total_quality = 0.0
        mlp_blocks_aligned = 0

        up_proj_keys = [k for k in source_weights.keys() if "up_proj" in k and k.endswith(".weight")]
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

            if not all([
                source_up is not None,
                target_up is not None,
                source_gate is not None,
                target_gate is not None,
                source_down is not None,
                target_down is not None,
            ]):
                logger.warning(f"Incomplete MLP block for {up_key}, skipping")
                continue

            # Check for per-layer anchor activations
            layer_idx = PermutationAligner._extract_layer_index(up_key)
            alignment: AlignmentResult

            if (
                anchor_activations is not None
                and layer_idx is not None
            ):
                activations = anchor_activations.activations(layer_idx)
                if activations is not None and len(activations[0]) > 0 and len(activations[1]) > 0:
                    logger.debug(f"Using anchor activations for layer {layer_idx}")
                    source_anchors = PermutationAligner._array_from_matrix(activations[0], backend=b)
                    target_anchors = PermutationAligner._array_from_matrix(activations[1], backend=b)
                    alignment = PermutationAligner.align_via_anchor_activations(
                        source_up, target_up, source_anchors, target_anchors, config, backend=b
                    )
                else:
                    alignment = PermutationAligner.align_via_anchor_projection(
                        source_up, target_up, anchors, config, backend=b
                    )
            else:
                alignment = PermutationAligner.align_via_anchor_projection(
                    source_up, target_up, anchors, config, backend=b
                )

            # Apply permutation (sparse or dense)
            if alignment.is_sparse_permutation and alignment.assignment_indices is not None:
                signed_up, signed_gate, aligned_down = PermutationAligner._apply_sparse_mlp_permutation(
                    b.astype(source_up, "float32"),
                    b.astype(source_gate, "float32"),
                    b.astype(source_down, "float32"),
                    alignment.assignment_indices,
                    alignment.signs,
                    backend=b,
                )
            else:
                # Dense application
                aligned_up = b.matmul(alignment.permutation, b.astype(source_up, "float32"))
                signed_up = b.matmul(alignment.signs, aligned_up)

                aligned_gate = b.matmul(alignment.permutation, b.astype(source_gate, "float32"))
                signed_gate = b.matmul(alignment.signs, aligned_gate)

                permuted_down = b.matmul(b.astype(source_down, "float32"), b.transpose(alignment.permutation))
                aligned_down = b.matmul(permuted_down, alignment.signs)

            # Get dtype string from source array
            source_dtype = str(source_up.dtype) if hasattr(source_up, 'dtype') else "float32"
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
    def fuse(
        source_weight: "Array",
        aligned_target_weight: "Array",
        alignment: AlignmentResult,
        config: FusionConfig = FusionConfig(),
        backend: "Backend | None" = None,
    ) -> "Array":
        """Fuses source and aligned target weights using confidence-weighted averaging.

        Implements TIES-Merging principles:
        1. Only merge neurons that are geometrically aligned (high confidence).
        2. For unaligned neurons, preserve the dominant signal (or base).

        Args:
            source_weight: Base model weight [Out, In].
            aligned_target_weight: Aligned target weight [Out, In].
            alignment: Alignment result with match confidences.
            config: Fusion configuration.
            backend: Optional backend for array operations.

        Returns:
            Fused weight matrix.
        """
        b = backend or get_default_backend()

        confidence = b.astype(b.array(alignment.match_confidences), "float32")

        # Broadcast confidence to shape [Out, 1] for row-wise masking
        mask = b.reshape(confidence, (-1, 1))

        # Standard weighted average
        alpha = config.source_alpha
        avg = (source_weight * alpha) + (aligned_target_weight * (1 - alpha))

        # Confidence-gated blend:
        # If confidence is high, use average.
        # If confidence is low, stick to source (base model stability).
        # W_final = confidence * avg + (1 - confidence) * source
        fused = (avg * mask) + (source_weight * (1 - mask))

        b.eval(fused)
        return fused

    @staticmethod
    def _apply_sparse_mlp_permutation(
        source_up: "Array",
        source_gate: "Array",
        source_down: "Array",
        indices: list[int],
        signs: "Array",
        backend: "Backend | None" = None,
    ) -> "Tuple[Array, Array, Array]":
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
    def _extract_layer_index(key: str) -> Optional[int]:
        """Extract layer index from weight key."""
        patterns = [".layers.", ".h.", ".blocks.", ".block."]
        for pattern in patterns:
            if pattern in key:
                idx = PermutationAligner._parse_index_after(pattern, key)
                if idx is not None:
                    return idx
        return None

    @staticmethod
    def _parse_index_after(needle: str, haystack: str) -> Optional[int]:
        """Parse integer index after a substring."""
        idx = haystack.find(needle)
        if idx < 0:
            return None
        suffix = haystack[idx + len(needle):]
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
            values = b.to_numpy(signs).tolist()
        else:
            values = b.to_numpy(b.diag(signs)).tolist()

        if len(values) == expected_count:
            return values

        logger.warning(
            f"Sign vector size mismatch (expected {expected_count}, got {len(values)}); "
            "falling back to +1"
        )
        return [1.0] * expected_count

    @staticmethod
    def is_mlp_weight(key: str) -> bool:
        """Check if a weight key is part of the MLP (safe to permute)."""
        return any(pattern in key for pattern in [
            "up_proj", "gate_proj", "down_proj",
            "w1", "w2", "w3",
        ])

    @staticmethod
    def is_attention_weight(key: str) -> bool:
        """Check if a weight key is attention (NOT safe to permute with generic aligner)."""
        return any(pattern in key for pattern in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "wq", "wk", "wv", "wo",
        ])
