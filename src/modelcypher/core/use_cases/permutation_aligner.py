from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from modelcypher.ports.backend import Backend, Array


@dataclass(frozen=True)
class AlignmentResult:
    permutation: Array
    signs: Array
    match_quality: float
    match_confidences: list[float]
    sign_flip_count: int
    is_sparse_permutation: bool = False
    assignment_indices: Optional[list[int]] = None


@dataclass(frozen=True)
class Config:
    min_match_threshold: float = 0.1
    use_anchor_grounding: bool = True
    top_k: int = 5


@dataclass(frozen=True)
class FusionConfig:
    interference_threshold: float = 0.5
    source_alpha: float = 0.5
    normalize: bool = False


class PermutationAligner:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def align(
        self,
        source_weight: Array,
        target_weight: Array,
        anchors: Array | None = None,
        config: Config = Config(),
    ) -> AlignmentResult:
        source = self.backend.astype(source_weight, np.float32)
        target = self.backend.astype(target_weight, np.float32)
        self.backend.eval(source, target)

        source_shape = source.shape
        target_shape = target.shape
        if len(source_shape) != 2 or len(target_shape) != 2:
            raise ValueError("Weights must be 2D matrices")
        if source_shape != target_shape:
            raise ValueError("Weight dimensions must match")

        n = int(source_shape[0])
        in_dim = int(source_shape[1])

        if config.use_anchor_grounding and anchors is not None:
            anchor = self.backend.astype(anchors, np.float32)
            self.backend.eval(anchor)
            anchor_dim = int(anchor.shape[1])
            if in_dim == anchor_dim:
                source_signatures = self.backend.matmul(source, self.backend.transpose(anchor))
                target_signatures = self.backend.matmul(target, self.backend.transpose(anchor))
            elif n == anchor_dim:
                source_signatures = source
                target_signatures = target
            else:
                source_signatures = source
                target_signatures = target
        else:
            source_signatures = source
            target_signatures = target

        self.backend.eval(source_signatures, target_signatures)
        source_np = self.backend.to_numpy(source_signatures)
        target_np = self.backend.to_numpy(target_signatures)

        source_norms = np.linalg.norm(source_np, axis=1, keepdims=True) + 1e-8
        target_norms = np.linalg.norm(target_np, axis=1, keepdims=True) + 1e-8
        source_normed = source_np / source_norms
        target_normed = target_np / target_norms

        similarity = source_normed @ target_normed.T

        assignment = [-1] * n
        signs = [1.0] * n
        match_confidences = [0.0] * n
        used_targets: set[int] = set()
        sign_flip_count = 0

        source_order = [
            (i, float(np.max(np.abs(similarity[i])) if n else 0.0)) for i in range(n)
        ]
        source_order.sort(key=lambda item: item[1], reverse=True)

        for src_idx, _ in source_order:
            best_target = -1
            best_sim = 0.0
            best_abs = -float("inf")
            for tgt_idx in range(n):
                if tgt_idx in used_targets:
                    continue
                sim = float(similarity[src_idx, tgt_idx])
                abs_sim = abs(sim)
                if abs_sim > best_abs:
                    best_abs = abs_sim
                    best_target = tgt_idx
                    best_sim = sim
            if best_target >= 0 and best_abs >= config.min_match_threshold:
                assignment[src_idx] = best_target
                used_targets.add(best_target)
                match_confidences[src_idx] = best_abs
                if best_sim < 0:
                    signs[src_idx] = -1.0
                    sign_flip_count += 1

        remaining_targets = set(range(n)).difference(used_targets)
        for src_idx in range(n):
            if assignment[src_idx] >= 0:
                continue
            if src_idx in remaining_targets:
                assignment[src_idx] = src_idx
                remaining_targets.remove(src_idx)
            elif remaining_targets:
                assignment[src_idx] = remaining_targets.pop()
                match_confidences[src_idx] = 0.0

        signs_target = [1.0] * n
        confidences_target = [0.0] * n
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        perm_matrix = np.zeros((n, n), dtype=np.float32)
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                perm_matrix[tgt, src] = 1.0

        sign_matrix = np.diag(np.array(signs_target, dtype=np.float32))
        permutation = self.backend.array(perm_matrix, dtype=np.float32)
        signs_array = self.backend.array(sign_matrix, dtype=np.float32)
        self.backend.eval(permutation, signs_array)

        mean_quality = float(np.mean(confidences_target)) if n else 0.0
        return AlignmentResult(
            permutation=permutation,
            signs=signs_array,
            match_quality=mean_quality,
            match_confidences=confidences_target,
            sign_flip_count=sign_flip_count,
        )

    def align_via_anchor_projection(
        self,
        source_weight: Array,
        target_weight: Array,
        anchors: Array,
        config: Config = Config(),
    ) -> AlignmentResult:
        source = self.backend.astype(source_weight, np.float32)
        target = self.backend.astype(target_weight, np.float32)
        anchor = self.backend.astype(anchors, np.float32)
        self.backend.eval(source, target, anchor)

        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("Weights must be 2D.")
        if source.shape != target.shape:
            raise ValueError("Weight dimensions must match.")

        n = int(source.shape[0])
        input_dim = int(source.shape[1])
        anchor_dim = int(anchor.shape[1])

        if input_dim == anchor_dim:
            source_signatures = self.backend.matmul(source, self.backend.transpose(anchor))
            target_signatures = self.backend.matmul(target, self.backend.transpose(anchor))
        else:
            source_np = self.backend.to_numpy(source)
            target_np = self.backend.to_numpy(target)
            source_signatures = self.backend.array(
                np.linalg.norm(source_np, axis=1, keepdims=True).astype(np.float32),
                dtype=np.float32,
            )
            target_signatures = self.backend.array(
                np.linalg.norm(target_np, axis=1, keepdims=True).astype(np.float32),
                dtype=np.float32,
            )

        self.backend.eval(source_signatures, target_signatures)
        source_np = self.backend.to_numpy(source_signatures)
        target_np = self.backend.to_numpy(target_signatures)

        source_norms = np.linalg.norm(source_np, axis=1, keepdims=True) + 1e-8
        target_norms = np.linalg.norm(target_np, axis=1, keepdims=True) + 1e-8
        source_normed = source_np / source_norms
        target_normed = target_np / target_norms

        if n < 4096:
            similarity = source_normed @ target_normed.T
            return self._assign_from_similarity(similarity, config)

        batch_size = 512 if n < 4096 else 128 if n < 8000 else 32 if n < 12000 else 16
        source_order: list[tuple[int, float, int, float]] = []
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = source_normed[batch_start:batch_end]
            sim_slice = batch @ target_normed.T
            abs_sim = np.abs(sim_slice)
            best_targets = abs_sim.argmax(axis=1)
            best_sims = abs_sim.max(axis=1)
            signed = sim_slice[np.arange(len(batch)), best_targets]
            for i in range(len(batch)):
                source_order.append(
                    (batch_start + i, float(best_sims[i]), int(best_targets[i]), float(signed[i]))
                )

        source_order.sort(key=lambda item: item[1], reverse=True)

        assignment = [-1] * n
        signs = [1.0] * n
        match_confidences = [0.0] * n
        used_targets: set[int] = set()
        sign_flip_count = 0
        needs_recompute: list[int] = []

        for src_idx, max_sim, best_target, signed_sim in source_order:
            if max_sim < config.min_match_threshold:
                continue
            if best_target not in used_targets:
                assignment[src_idx] = best_target
                match_confidences[src_idx] = max_sim
                used_targets.add(best_target)
                if signed_sim < 0:
                    signs[src_idx] = -1.0
                    sign_flip_count += 1
            else:
                needs_recompute.append(src_idx)

        if needs_recompute:
            recompute_batch = 256
            for batch_start in range(0, len(needs_recompute), recompute_batch):
                batch_indices = needs_recompute[batch_start : batch_start + recompute_batch]
                for src_idx in batch_indices:
                    sim_row = source_normed[src_idx : src_idx + 1] @ target_normed.T
                    best_target = -1
                    best_sim = 0.0
                    best_abs = -float("inf")
                    for j in range(n):
                        if j in used_targets:
                            continue
                        sim = float(sim_row[0, j])
                        abs_sim = abs(sim)
                        if abs_sim > best_abs:
                            best_abs = abs_sim
                            best_sim = sim
                            best_target = j
                    if best_target >= 0 and best_abs >= config.min_match_threshold:
                        assignment[src_idx] = best_target
                        match_confidences[src_idx] = best_abs
                        used_targets.add(best_target)
                        if best_sim < 0:
                            signs[src_idx] = -1.0
                            sign_flip_count += 1

        remaining_targets = set(range(n)).difference(used_targets)
        for src_idx in range(n):
            if assignment[src_idx] >= 0:
                continue
            if src_idx in remaining_targets:
                assignment[src_idx] = src_idx
                remaining_targets.remove(src_idx)
            elif remaining_targets:
                assignment[src_idx] = remaining_targets.pop()
                match_confidences[src_idx] = 0.0

        signs_target = [1.0] * n
        confidences_target = [0.0] * n
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        avg_quality = float(np.mean(confidences_target)) if n else 0.0

        if n > 4096:
            permutation = self.backend.array(np.array(assignment, dtype=np.float32), dtype=np.float32)
            signs_array = self.backend.array(np.array(signs_target, dtype=np.float32), dtype=np.float32)
            return AlignmentResult(
                permutation=permutation,
                signs=signs_array,
                match_quality=avg_quality,
                match_confidences=confidences_target,
                sign_flip_count=sign_flip_count,
                is_sparse_permutation=True,
                assignment_indices=assignment,
            )

        perm_matrix = np.zeros((n, n), dtype=np.float32)
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                perm_matrix[tgt, src] = 1.0
            else:
                perm_matrix[src, src] = 1.0
        sign_matrix = np.diag(np.array(signs_target, dtype=np.float32))
        permutation = self.backend.array(perm_matrix, dtype=np.float32)
        signs_array = self.backend.array(sign_matrix, dtype=np.float32)
        return AlignmentResult(
            permutation=permutation,
            signs=signs_array,
            match_quality=avg_quality,
            match_confidences=confidences_target,
            sign_flip_count=sign_flip_count,
        )

    def apply(
        self,
        weight: Array,
        alignment: AlignmentResult,
        align_output: bool = True,
        align_input: bool = False,
    ) -> Array:
        w = self.backend.astype(weight, np.float32)

        if alignment.is_sparse_permutation and alignment.assignment_indices:
            indices = alignment.assignment_indices
            inverse = self._inverse_permutation(indices)
            sign_values = self._extract_sign_values(alignment.signs, len(indices))
            index_tensor = self.backend.array(np.array(inverse, dtype=np.int32))

            if align_output:
                w = self._take(w, index_tensor, axis=0)
                sign_row = self.backend.array(np.array(sign_values, dtype=np.float32)).reshape((len(indices), 1))
                w = w * sign_row

            if align_input:
                w = self._take(w, index_tensor, axis=1)
                sign_col = self.backend.array(np.array(sign_values, dtype=np.float32)).reshape((1, len(indices)))
                w = w * sign_col

            self.backend.eval(w)
            return w

        if align_output:
            permuted = self.backend.matmul(alignment.permutation, w)
            w = self.backend.matmul(alignment.signs, permuted)

        if align_input:
            permuted = self.backend.matmul(w, self.backend.transpose(alignment.permutation))
            w = self.backend.matmul(permuted, alignment.signs)

        self.backend.eval(w)
        return w

    def rebasin_mlp_only(
        self,
        source_weights: dict[str, Array],
        target_weights: dict[str, Array],
        anchors: Array,
        config: Config = Config(),
    ) -> tuple[dict[str, Array], float, int]:
        aligned_weights: dict[str, Array] = {}
        total_quality = 0.0
        mlp_blocks_aligned = 0

        up_proj_keys = sorted([key for key in source_weights if "up_proj" in key and key.endswith(".weight")])

        for up_key in up_proj_keys:
            gate_key = up_key.replace("up_proj", "gate_proj")
            down_key = up_key.replace("up_proj", "down_proj")
            source_up = source_weights.get(up_key)
            target_up = target_weights.get(up_key)
            source_gate = source_weights.get(gate_key)
            target_gate = target_weights.get(gate_key)
            source_down = source_weights.get(down_key)
            target_down = target_weights.get(down_key)
            if not all([source_up is not None, target_up is not None, source_gate is not None, target_gate is not None, source_down is not None, target_down is not None]):
                continue
            if source_up.ndim != 2 or source_gate.ndim != 2 or source_down.ndim != 2:
                continue
            if source_up.shape != target_up.shape or source_gate.shape != target_gate.shape or source_down.shape != target_down.shape:
                continue

            alignment = self.align_via_anchor_projection(source_up, target_up, anchors, config=config)

            if alignment.is_sparse_permutation and alignment.assignment_indices:
                signed_up, signed_gate, aligned_down = self._apply_sparse_mlp_permutation(
                    source_up=source_up,
                    source_gate=source_gate,
                    source_down=source_down,
                    indices=alignment.assignment_indices,
                    signs=alignment.signs,
                )
            else:
                aligned_up_dense = self.backend.matmul(alignment.permutation, self.backend.astype(source_up, np.float32))
                signed_up = self.backend.matmul(alignment.signs, aligned_up_dense)
                aligned_gate_dense = self.backend.matmul(alignment.permutation, self.backend.astype(source_gate, np.float32))
                signed_gate = self.backend.matmul(alignment.signs, aligned_gate_dense)
                aligned_down_dense = self.backend.matmul(self.backend.astype(source_down, np.float32), self.backend.transpose(alignment.permutation))
                aligned_down = self.backend.matmul(aligned_down_dense, alignment.signs)

            aligned_weights[up_key] = signed_up
            aligned_weights[gate_key] = signed_gate
            aligned_weights[down_key] = aligned_down

            total_quality += alignment.match_quality
            mlp_blocks_aligned += 1

        for key, value in source_weights.items():
            if key not in aligned_weights:
                aligned_weights[key] = value

        avg_quality = total_quality / mlp_blocks_aligned if mlp_blocks_aligned else 0.0
        return aligned_weights, avg_quality, mlp_blocks_aligned

    def fuse(
        self,
        source_weight: Array,
        aligned_target_weight: Array,
        alignment: AlignmentResult,
        config: FusionConfig = FusionConfig(),
    ) -> Array:
        confidence = np.array(alignment.match_confidences, dtype=np.float32)
        mask = confidence.reshape((confidence.shape[0], 1))
        source = self.backend.astype(source_weight, np.float32)
        target = self.backend.astype(aligned_target_weight, np.float32)
        alpha = config.source_alpha
        avg = (source * alpha) + (target * (1 - alpha))
        fused = (avg * mask) + (source * (1 - mask))
        self.backend.eval(fused)
        return fused

    @staticmethod
    def _inverse_permutation(indices: list[int]) -> list[int]:
        inverse = [0] * len(indices)
        for src, tgt in enumerate(indices):
            if tgt < 0 or tgt >= len(indices):
                continue
            inverse[tgt] = src
        return inverse

    def _extract_sign_values(self, signs: Array, expected_count: int) -> list[float]:
        values = self.backend.to_numpy(signs)
        if values.ndim == 1:
            return [float(v) for v in values]
        diag = np.diag(values)
        if diag.shape[0] != expected_count:
            diag = np.pad(diag, (0, max(0, expected_count - diag.shape[0])), constant_values=1.0)
        return [float(v) for v in diag]

    def _take(self, array: Array, indices: Array, axis: int) -> Array:
        data = self.backend.to_numpy(array)
        idx = self.backend.to_numpy(indices)
        return self.backend.array(np.take(data, idx, axis=axis), dtype=np.float32)

    def _apply_sparse_mlp_permutation(
        self,
        source_up: Array,
        source_gate: Array,
        source_down: Array,
        indices: list[int],
        signs: Array,
    ) -> tuple[Array, Array, Array]:
        intermediate = len(indices)
        sign_values = self._extract_sign_values(signs, intermediate)
        inv_indices = self._inverse_permutation(indices)
        idx = np.array(inv_indices, dtype=np.int32)
        permuted_up = np.take(self.backend.to_numpy(source_up), idx, axis=0)
        permuted_gate = np.take(self.backend.to_numpy(source_gate), idx, axis=0)
        sign_col = np.array(sign_values, dtype=np.float32).reshape((intermediate, 1))
        signed_up = permuted_up * sign_col
        signed_gate = permuted_gate * sign_col
        permuted_down = np.take(self.backend.to_numpy(source_down), idx, axis=1)
        sign_row = np.array(sign_values, dtype=np.float32).reshape((1, intermediate))
        signed_down = permuted_down * sign_row
        return (
            self.backend.array(signed_up, dtype=np.float32),
            self.backend.array(signed_gate, dtype=np.float32),
            self.backend.array(signed_down, dtype=np.float32),
        )

    @staticmethod
    def is_mlp_weight(key: str) -> bool:
        return any(token in key for token in ("up_proj", "gate_proj", "down_proj", "w1", "w2", "w3"))

    @staticmethod
    def is_attention_weight(key: str) -> bool:
        return any(token in key for token in ("q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo"))

    def _assign_from_similarity(self, similarity: np.ndarray, config: Config) -> AlignmentResult:
        n = similarity.shape[0]
        assignment = [-1] * n
        signs = [1.0] * n
        match_confidences = [0.0] * n
        used_targets: set[int] = set()
        sign_flip_count = 0

        source_order = [
            (i, float(np.max(np.abs(similarity[i])) if n else 0.0)) for i in range(n)
        ]
        source_order.sort(key=lambda item: item[1], reverse=True)

        for src_idx, _ in source_order:
            best_target = -1
            best_sim = 0.0
            best_abs = -float("inf")
            for tgt_idx in range(n):
                if tgt_idx in used_targets:
                    continue
                sim = float(similarity[src_idx, tgt_idx])
                abs_sim = abs(sim)
                if abs_sim > best_abs:
                    best_target = tgt_idx
                    best_sim = sim
                    best_abs = abs_sim
            if best_target >= 0 and best_abs >= config.min_match_threshold:
                assignment[src_idx] = best_target
                used_targets.add(best_target)
                match_confidences[src_idx] = best_abs
                if best_sim < 0:
                    signs[src_idx] = -1.0
                    sign_flip_count += 1

        remaining_targets = set(range(n)).difference(used_targets)
        for src_idx in range(n):
            if assignment[src_idx] >= 0:
                continue
            if src_idx in remaining_targets:
                assignment[src_idx] = src_idx
                remaining_targets.remove(src_idx)
            elif remaining_targets:
                assignment[src_idx] = remaining_targets.pop()
                match_confidences[src_idx] = 0.0

        signs_target = [1.0] * n
        confidences_target = [0.0] * n
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                signs_target[tgt] = signs[src]
                confidences_target[tgt] = match_confidences[src]

        perm_matrix = np.zeros((n, n), dtype=np.float32)
        for src, tgt in enumerate(assignment):
            if tgt >= 0:
                perm_matrix[tgt, src] = 1.0
        sign_matrix = np.diag(np.array(signs_target, dtype=np.float32))
        permutation = self.backend.array(perm_matrix, dtype=np.float32)
        signs_array = self.backend.array(sign_matrix, dtype=np.float32)
        mean_quality = float(np.mean(confidences_target)) if n else 0.0
        return AlignmentResult(
            permutation=permutation,
            signs=signs_array,
            match_quality=mean_quality,
            match_confidences=confidences_target,
            sign_flip_count=sign_flip_count,
        )
