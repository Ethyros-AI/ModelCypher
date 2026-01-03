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

"""Tests for EntropyWindow sliding window tracker."""

import uuid

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_window import EntropyWindow
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestEntropyWindow:
    """Tests for EntropyWindow."""

    def test_initialization(self):
        """Should initialize with derived window size."""
        window = EntropyWindow(sample_count=400)

        assert window.window_id is not None

    def test_custom_window_id(self):
        """Should accept custom window ID."""
        custom_id = str(uuid.uuid4())
        window = EntropyWindow(sample_count=100, window_id=custom_id)

        assert window.window_id == custom_id

    def test_add_single_sample(self):
        """Should add a single sample."""
        window = EntropyWindow(sample_count=100)
        status = window.add(entropy=2.0, variance=0.1, token_index=0)

        assert status.sample_count == 1
        eps = _eps(status.current_entropy, status.moving_average)
        assert abs(status.current_entropy - 2.0) <= eps
        assert abs(status.moving_average - 2.0) <= eps

    def test_add_multiple_samples(self):
        """Should compute moving average correctly."""
        window = EntropyWindow(sample_count=100)
        window.add(entropy=1.0, variance=0.1, token_index=0)
        window.add(entropy=2.0, variance=0.1, token_index=1)
        status = window.add(entropy=3.0, variance=0.1, token_index=2)

        assert status.sample_count == 3
        eps = _eps(status.moving_average, status.current_entropy)
        assert abs(status.moving_average - 2.0) <= eps  # (1+2+3)/3
        assert abs(status.current_entropy - 3.0) <= eps

    def test_window_size_limit(self):
        """Should maintain window size limit."""
        window = EntropyWindow(sample_count=25)

        for i in range(10):
            window.add(entropy=float(i), variance=0.1, token_index=i)

        status = window.status()
        assert status.sample_count == 5
        assert abs(status.min_entropy - 5.0) <= _eps(status.min_entropy)

    def test_reset(self):
        """Reset should clear all state."""
        window = EntropyWindow(sample_count=25)
        window.add(entropy=4.0, variance=0.1, token_index=0)
        window.add(entropy=4.0, variance=0.1, token_index=1)

        window.reset()

        status = window.status()
        assert status.sample_count == 0

    def test_add_batch(self):
        """Should add multiple samples via batch."""
        window = EntropyWindow(sample_count=25)
        batch = [
            (1.0, 0.1, 0),
            (2.0, 0.2, 1),
            (3.0, 0.3, 2),
        ]
        status = window.add_batch(batch)

        assert status.sample_count == 3
        assert abs(status.moving_average - 2.0) <= _eps(status.moving_average)

    def test_moving_average_reflects_raw_entropy(self):
        """Moving average should reflect raw entropy values."""
        window = EntropyWindow(sample_count=25)

        # Low entropy value
        window.reset()
        window.add(entropy=1.0, variance=0.1, token_index=0)
        assert abs(window.status().moving_average - 1.0) <= _eps(
            window.status().moving_average
        )

        # Moderate entropy value
        window.reset()
        window.add(entropy=2.0, variance=0.1, token_index=0)
        assert abs(window.status().moving_average - 2.0) <= _eps(
            window.status().moving_average
        )

        # High entropy value
        window.reset()
        window.add(entropy=4.0, variance=0.1, token_index=0)
        assert abs(window.status().moving_average - 4.0) <= _eps(
            window.status().moving_average
        )

    def test_to_entropy_summary(self):
        """Should produce summary dict."""
        window = EntropyWindow(sample_count=25)
        window.add(entropy=2.0, variance=0.1, token_index=0)

        summary = window.to_entropy_summary()

        assert "window_id" in summary
        assert "logit_entropy" in summary
        assert summary["sample_count"] == 1


class TestEntropyWindowAsync:
    """Tests for async operations."""

    @pytest.mark.asyncio
    async def test_add_async(self):
        """Should add sample asynchronously."""
        window = EntropyWindow(sample_count=25)

        status = await window.add_async(entropy=2.0, variance=0.1, token_index=0)

        assert status.sample_count == 1
