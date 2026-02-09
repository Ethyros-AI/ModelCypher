# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.prompt_manifold import (
    apply_prompt_basis,
    derive_prompt_manifold_basis,
)


def test_derive_prompt_manifold_basis_handles_scalar_input(any_backend) -> None:
    b = any_backend
    scalar = b.array(3.0)

    result = derive_prompt_manifold_basis(scalar, backend=b)

    assert result.sample_count == 0
    assert result.feature_dim == 0
    assert result.basis_rank == 0
    assert result.scale == 0.0
    assert tuple(result.basis.shape) == (0, 0)


def test_derive_prompt_manifold_basis_with_explicit_rank(any_backend) -> None:
    b = any_backend
    embeddings = b.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )

    result = derive_prompt_manifold_basis(
        embeddings,
        basis_rank=2,
        backend=b,
    )

    assert result.sample_count == 6
    assert result.feature_dim == 4
    assert result.basis_rank == 2
    assert tuple(result.mean.shape) == (4,)
    assert tuple(result.basis.shape) == (2, 4)
    assert result.scale >= 0.0


def test_derive_prompt_manifold_basis_zero_rank_returns_empty_basis(any_backend) -> None:
    b = any_backend
    embeddings = b.array([[1.0, 2.0], [3.0, 4.0]])
    result = derive_prompt_manifold_basis(
        embeddings,
        basis_rank=0,
        backend=b,
    )

    assert result.basis_rank == 0
    assert tuple(result.basis.shape) == (0, 2)
    assert result.scale == 0.0


def test_apply_prompt_basis_no_basis_is_noop(any_backend) -> None:
    b = any_backend
    base = b.array([1.0, 2.0, 3.0])
    basis = b.zeros((0, 3))
    coeffs = b.array([])

    result = apply_prompt_basis(base, basis, coeffs, backend=b)
    assert b.tolist(result) == [1.0, 2.0, 3.0]


def test_apply_prompt_basis_vector_and_matrix(any_backend) -> None:
    b = any_backend
    basis = b.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coeffs = b.array([2.0, -1.0])

    vector = b.array([1.0, 2.0, 3.0])
    vector_out = apply_prompt_basis(vector, basis, coeffs, backend=b)
    assert b.tolist(vector_out) == pytest.approx([3.0, 1.0, 3.0])

    matrix = b.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    matrix_out = apply_prompt_basis(matrix, basis, coeffs, backend=b)
    matrix_list = b.tolist(matrix_out)
    assert matrix_list[0] == pytest.approx([3.0, 1.0, 3.0])
    assert matrix_list[1] == pytest.approx([12.0, 19.0, 30.0])
