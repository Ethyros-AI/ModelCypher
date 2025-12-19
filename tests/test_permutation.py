from __future__ import annotations

import numpy as np

from modelcypher.core.use_cases.permutation_aligner import PermutationAligner
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
