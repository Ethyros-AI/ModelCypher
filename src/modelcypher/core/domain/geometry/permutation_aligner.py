from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("modelcypher.geometry.permutation_aligner")


@dataclass(frozen=True)
class AlignmentResult:
    """Result of permutation alignment between two weight matrices."""
    permutation: mx.array  # [N, N]
    signs: mx.array  # [N, N] diagonal or [N] vector
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
        source_weight: mx.array,
        target_weight: mx.array,
        anchors: Optional[mx.array] = None,
        config: Config = Config(),
    ) -> AlignmentResult:
        """
        Computes the optimal permutation and sign alignment between two weight matrices.
        """
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
                source_signatures = source_weight.astype(mx.float32) @ anchors.T
                target_signatures = target_weight.astype(mx.float32) @ anchors.T
                logger.debug(f"Anchor-grounded (input match): [{N}, {anchors.shape[0]}]")
            elif source_out == anchor_dim:
                logger.warning(f"Anchor dim {anchor_dim} matches output; using direct weight signatures for alignment")
                source_signatures = source_weight.astype(mx.float32)
                target_signatures = target_weight.astype(mx.float32)
            else:
                logger.warning(f"Anchor dim {anchor_dim} doesn't match weight dims [{source_out}, {source_in}], using direct")
                source_signatures = source_weight.astype(mx.float32)
                target_signatures = target_weight.astype(mx.float32)
        else:
             # Direct: use weight rows as signatures
            source_signatures = source_weight.astype(mx.float32)
            target_signatures = target_weight.astype(mx.float32)
            logger.debug(f"Using direct weight signatures: [{N}, {source_in}]")
            
        if source_signatures.shape[0] != N or target_signatures.shape[0] != N:
             logger.warning(f"Anchor signatures shape mismatch; using direct weight signatures")
             source_signatures = source_weight.astype(mx.float32)
             target_signatures = target_weight.astype(mx.float32)

        # Normalize signatures
        source_norms = mx.sqrt(mx.sum(source_signatures * source_signatures, axis=1, keepdims=True)) + 1e-8
        target_norms = mx.sqrt(mx.sum(target_signatures * target_signatures, axis=1, keepdims=True)) + 1e-8
        source_normalized = source_signatures / source_norms
        target_normalized = target_signatures / target_norms
        
        # Compute full similarity matrix: [N, N]
        similarity = source_normalized @ target_normalized.T
        mx.eval(similarity)
        
        # Pull to CPU (numpy) for greedy assignment
        sim_data = similarity.tolist() # List of lists
        
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
        sorted_remaining = sorted(list(remaining_targets)) # For determinism
        
        for src_idx in range(N):
            if assignment[src_idx] < 0:
                # Try to assign to self if available
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
        
        permutation = mx.array(perm_data).reshape((N, N)).astype(mx.float32)
        sign_matrix = mx.diag(mx.array(signs_target)).astype(mx.float32)
        mx.eval(permutation, sign_matrix)
        
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
        weight: mx.array,
        alignment: AlignmentResult,
        align_output: bool = True,
        align_input: bool = False,
    ) -> mx.array:
        """Applies permutation and sign alignment to a weight matrix."""
        w = weight.astype(mx.float32)
        
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
            # signs is sparse vector [N]
            sign_values = alignment.signs.tolist() 
            if hasattr(sign_values, 'tolist'): # if mx array
                sign_values = sign_values if isinstance(sign_values, list) else alignment.signs.tolist()

            index_tensor = mx.array(inverse, dtype=mx.int32)
            
            if align_output:
                w = mx.take(w, index_tensor, axis=0)
                sign_row = mx.array(sign_values).reshape((count, 1)).astype(mx.float32)
                w = w * sign_row
            
            if align_input:
                w = mx.take(w, index_tensor, axis=1)
                sign_col = mx.array(sign_values).reshape((1, count)).astype(mx.float32)
                w = w * sign_col
                
            mx.eval(w)
            return w

        # Dense logic
        if align_output:
            # W' = S @ P @ W
            permuted = alignment.permutation @ w
            if alignment.signs.ndim == 1:
                sign_row = alignment.signs.reshape((-1, 1))
                w = permuted * sign_row
            else:
                w = alignment.signs @ permuted
        
        if align_input:
           # W' = W @ P^T @ S^T (S is diagonal => S^T = S)
           permuted = w @ alignment.permutation.T
           if alignment.signs.ndim == 1:
               sign_col = alignment.signs.reshape((1, -1))
               w = permuted * sign_col
           else:
               w = permuted @ alignment.signs

        mx.eval(w)
        return w

    @staticmethod
    def align_via_anchor_projection(
        source_weight: mx.array,
        target_weight: mx.array,
        anchors: mx.array,
        config: Config = Config(),
    ) -> AlignmentResult:
        """Aligns neurons using low-dimensional anchor projections."""
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
            source_signatures = source_weight.astype(mx.float32) @ anchors.T
            target_signatures = target_weight.astype(mx.float32) @ anchors.T
        else:
            logger.warning(f"Weight dim {input_dim} != anchor dim {anchor_dim}, using weight row norms")
            source_norms = mx.sqrt(mx.sum(source_weight.astype(mx.float32) ** 2, axis=1, keepdims=True))
            target_norms = mx.sqrt(mx.sum(target_weight.astype(mx.float32) ** 2, axis=1, keepdims=True))
            source_signatures = source_norms
            target_signatures = target_norms
            
        mx.eval(source_signatures, target_signatures)
        return PermutationAligner._align_from_signatures(source_signatures, target_signatures, config)

    @staticmethod
    def align_via_anchor_activations(
        source_weight: mx.array,
        target_weight: mx.array,
        source_anchors: mx.array,
        target_anchors: mx.array,
        config: Config = Config(),
    ) -> AlignmentResult:
        """Aligns neurons using per-layer anchor activations."""
        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError("Weights must be 2D")

        N = source_weight.shape[0]
        input_dim = source_weight.shape[1]
        source_anchor_dim = source_anchors.shape[1]
        target_anchor_dim = target_anchors.shape[1]
        
        if source_anchor_dim != input_dim or target_anchor_dim != input_dim:
             logger.warning(f"Anchor activation dim mismatch. Returning projection alignment.")
             return PermutationAligner.align_via_anchor_projection(
                 source_weight, target_weight, source_anchors, config
             )
        
        source_signatures = source_weight.astype(mx.float32) @ source_anchors.astype(mx.float32).T
        target_signatures = target_weight.astype(mx.float32) @ target_anchors.astype(mx.float32).T
        mx.eval(source_signatures, target_signatures)
        
        return PermutationAligner._align_from_signatures(source_signatures, target_signatures, config)

    @staticmethod
    def _align_from_signatures(
        source_signatures: mx.array,
        target_signatures: mx.array,
        config: Config,
    ) -> AlignmentResult:
        if source_signatures.ndim != 2 or target_signatures.ndim != 2:
            raise ValueError("Signatures must be 2D matrices")
            
        N = source_signatures.shape[0]
        
        source_fp32 = source_signatures.astype(mx.float32)
        target_fp32 = target_signatures.astype(mx.float32)
        
        # Normalize
        source_norms = mx.sqrt(mx.sum(source_fp32 ** 2, axis=1, keepdims=True)) + 1e-8
        target_norms = mx.sqrt(mx.sum(target_fp32 ** 2, axis=1, keepdims=True)) + 1e-8
        
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
            sim_slice = source_slice @ target_normalized.T # [batch, N]
            
            abs_sim = mx.abs(sim_slice)
            best_targets_gpu = mx.argmax(abs_sim, axis=1) # [batch]
            best_sims_gpu = mx.max(abs_sim, axis=1) # [batch]
            
            # For signed sim, we need to gather
            # MLX doesn't have take_along_axis equivalent easily for 2D, we can use indexing
            # OR simple gather
            # Since [batch, N], specific indices
            batch_indices = mx.arange(batch_end - batch_start)
            best_signed_gpu = sim_slice[batch_indices, best_targets_gpu]
            
            mx.eval(best_targets_gpu, best_sims_gpu, best_signed_gpu)
            
            best_targets = best_targets_gpu.tolist()
            best_sims = best_sims_gpu.tolist()
            best_signed = best_signed_gpu.tolist()
            
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
            sim_full = (source_normalized @ target_normalized.T).tolist()
            
            for src_idx in needs_recompute:
                row = sim_full[src_idx]
                best_target = -1
                best_sim = 0.0
                best_abs = -float('inf')
                
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
                permutation=mx.array(assignment).astype(mx.float32), # abuse of notation, but keeps ID
                signs=mx.array(signs_target).astype(mx.float32),
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
            
            permutation = mx.array(perm_data).reshape((N, N)).astype(mx.float32)
            sign_matrix = mx.diag(mx.array(signs_target)).astype(mx.float32)
            
            return AlignmentResult(
                permutation=permutation,
                signs=sign_matrix,
                match_quality=avg_quality,
                match_confidences=confidences_target,
                sign_flip_count=sign_flip_count
            )

    @staticmethod
    def rebasin_mlp_only(
        source_weights: dict[str, mx.array],
        target_weights: dict[str, mx.array],
        anchors: mx.array,
        config: Config = Config(),
    ) -> Tuple[dict[str, mx.array], float, int]:
        """Performs MLP-only re-basin alignment."""
        aligned_weights = {}
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
                source_up, target_up, anchors, config
            )
            
            # Apply to source
            # up_proj: align output
            aligned_up = PermutationAligner.apply(source_up, alignment, align_output=True, align_input=False)
            
            # gate_proj: align output (same permutation)
            aligned_gate = PermutationAligner.apply(source_gate, alignment, align_output=True, align_input=False)
            
            # down_proj: align input (dim 1)
            aligned_down = PermutationAligner.apply(source_down, alignment, align_output=False, align_input=True)
            
            aligned_weights[up_key] = aligned_up
            aligned_weights[gate_key] = aligned_gate
            aligned_weights[down_key] = aligned_down
            
            total_quality += alignment.match_quality
            mlp_blocks_aligned += 1
            
        avg_quality = total_quality / max(1, mlp_blocks_aligned)
        return aligned_weights, avg_quality, mlp_blocks_aligned
