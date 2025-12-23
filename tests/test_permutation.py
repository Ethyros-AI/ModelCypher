from __future__ import annotations

import numpy as np

from modelcypher.core.use_cases.permutation_aligner import AlignmentResult, FusionConfig, PermutationAligner
from tests.conftest import NumpyBackend


def test_permutation_alignment_identity():
    backend = NumpyBackend()
    aligner = PermutationAligner(backend)
    weights = np.eye(3, dtype=np.float32)
    result = aligner.align(weights, weights)
    perm = backend.to_numpy(result.permutation)
    assert np.allclose(perm, np.eye(3))


def test_permutation_alignment_swapped_rows():
    backend = NumpyBackend()
    aligner = PermutationAligner(backend)
    source = np.eye(3, dtype=np.float32)
    target = source[[1, 0, 2], :]
    result = aligner.align(source, target)
    aligned = aligner.apply(source, result, align_output=True, align_input=False)
    assert np.allclose(backend.to_numpy(aligned), target)


def test_anchor_projected_alignment():
    backend = NumpyBackend()
    aligner = PermutationAligner(backend)
    source = np.eye(4, dtype=np.float32)
    target = source[[2, 1, 0, 3], :]
    anchors = np.eye(4, dtype=np.float32)
    result = aligner.align_via_anchor_projection(source, target, anchors)
    aligned = aligner.apply(source, result, align_output=True, align_input=False)
    assert np.allclose(backend.to_numpy(aligned), target)


def test_permutation_alignment_sign_flipping():
    backend = NumpyBackend()
    aligner = PermutationAligner(backend)
    source = np.eye(2, dtype=np.float32)
    # Target is source with a row flipped
    target = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    
    result = aligner.align(source, target)
    aligned = aligner.apply(source, result)
    
    assert np.allclose(backend.to_numpy(aligned), target)
    signs_arr = backend.to_numpy(result.signs)
    assert signs_arr.flat[0] == -1.0


def test_permutation_fusion():
    backend = NumpyBackend()
    aligner = PermutationAligner(backend)
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)
    
    # Perfect match alignment
    alignment = AlignmentResult(
        permutation=np.eye(2),
        signs=np.ones(2),
        match_quality=1.0,
        match_confidences=[1.0, 1.0],
        sign_flip_count=0
    )
    
    # 50/50 fusion
    fused = aligner.fuse(source, target, alignment, FusionConfig(source_alpha=0.5))
    expected = (source + target) / 2.0
    assert np.allclose(backend.to_numpy(fused), expected)


def test_permutation_fusion_low_confidence():
    backend = NumpyBackend()
    aligner = PermutationAligner(backend)
    source = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    
    # Low confidence (0.0) should favor source
    alignment = AlignmentResult(
        permutation=np.eye(2),
        signs=np.ones(2),
        match_quality=0.0,
        match_confidences=[0.0, 0.0],
        sign_flip_count=0
    )
    
    fused = aligner.fuse(source, target, alignment)
    assert np.allclose(backend.to_numpy(fused), source)
