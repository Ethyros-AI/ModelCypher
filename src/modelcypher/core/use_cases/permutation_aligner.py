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
