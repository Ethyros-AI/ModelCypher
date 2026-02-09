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

from modelcypher.core.domain.geometry.activation_buffer import ActivationBuffer


def test_activation_buffer_running_stats_and_rollover(any_backend) -> None:
    b = any_backend
    buffer = ActivationBuffer(hidden_dim=2, backend=b)

    buffer.add(b.array([1.0, 0.0]))
    buffer.add(b.array([0.0, 1.0]))
    buffer.add(b.array([1.0, 1.0]))

    stats = buffer.get_stats()
    variance = b.tolist(stats.variance)
    assert stats.n_samples == 3
    assert variance[0] == pytest.approx(1.0 / 3.0, rel=1e-4, abs=1e-4)
    assert variance[1] == pytest.approx(1.0 / 3.0, rel=1e-4, abs=1e-4)
    assert stats.total_variance == pytest.approx(2.0 / 3.0, rel=1e-4, abs=1e-4)
    assert buffer.is_full is True

    # Buffer capacity is hidden_dim + 1; adding another sample should roll over.
    buffer.add(b.array([2.0, 2.0]))
    assert buffer.current_size == buffer.buffer_size == 3
    assert buffer.get_stats().n_samples == 3


def test_activation_buffer_svd_and_direction_accessors(any_backend) -> None:
    b = any_backend
    buffer = ActivationBuffer(hidden_dim=2, backend=b)
    buffer.add(b.array([1.0, 0.0]))
    buffer.add(b.array([0.0, 1.0]))
    buffer.add(b.array([1.0, 1.0]))

    assert buffer.should_update_svd() is True
    buffer.update_svd()
    assert buffer.should_update_svd() is False

    stats = buffer.get_stats()
    assert stats.svd_update_count == 1
    assert stats.svd_rank >= 0

    singular = buffer.get_singular_values()
    assert singular is not None
    assert int(singular.shape[0]) == 2

    principal_all = buffer.get_principal_directions()
    principal_top1 = buffer.get_principal_directions(k=1)
    assert principal_all is not None
    assert principal_top1 is not None
    assert tuple(principal_all.shape) == (2, 2)
    assert tuple(principal_top1.shape) == (1, 2)


def test_activation_buffer_null_directions_and_reset(any_backend) -> None:
    b = any_backend
    buffer = ActivationBuffer(hidden_dim=2, backend=b)
    constant = b.array([1.0, 1.0])
    buffer.add(constant)
    buffer.add(constant)
    buffer.add(constant)
    buffer.update_svd()

    null_dirs = buffer.get_null_directions()
    assert null_dirs is not None
    assert tuple(null_dirs.shape)[1] == 2

    cov_1 = b.tolist(buffer.get_covariance())
    cov_2 = b.tolist(buffer.get_covariance())
    assert cov_1 == cov_2

    buffer.reset()
    assert buffer.current_size == 0
    assert buffer.get_singular_values() is None
    assert buffer.get_principal_directions() is None
    assert buffer.get_null_directions() is None

