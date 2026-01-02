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

from modelcypher.core.domain.entropy.entropy_window import (
    EntropyWindow,
    EntropyWindowConfig,
)


class TestEntropyWindowConfig:
    """Tests for EntropyWindowConfig."""

    def test_requires_explicit_values(self):
        """Should require explicit configuration."""
        with pytest.raises(TypeError):
            EntropyWindowConfig()  # type: ignore[call-arg]

    def test_custom_values(self):
        """Should accept custom values."""
        config = EntropyWindowConfig(window_size=10)

        assert config.window_size == 10


class TestEntropyWindow:
    """Tests for EntropyWindow."""

    def test_initialization(self):
        """Should initialize with config."""
        config = EntropyWindowConfig(window_size=20)
        window = EntropyWindow(config=config)

        assert window.config is not None
        assert window.window_id is not None

    def test_custom_window_id(self):
        """Should accept custom window ID."""
        custom_id = str(uuid.uuid4())
        config = EntropyWindowConfig(window_size=10)
        window = EntropyWindow(config=config, window_id=custom_id)

        assert window.window_id == custom_id

    def test_add_single_sample(self):
        """Should add a single sample."""
        config = EntropyWindowConfig(window_size=10)
        window = EntropyWindow(config=config)
        status = window.add(entropy=2.0, variance=0.1, token_index=0)

        assert status.sample_count == 1
        assert status.current_entropy == 2.0
        assert status.moving_average == 2.0

    def test_add_multiple_samples(self):
        """Should compute moving average correctly."""
        config = EntropyWindowConfig(window_size=10)
        window = EntropyWindow(config=config)
        window.add(entropy=1.0, variance=0.1, token_index=0)
        window.add(entropy=2.0, variance=0.1, token_index=1)
        status = window.add(entropy=3.0, variance=0.1, token_index=2)

        assert status.sample_count == 3
        assert status.moving_average == 2.0  # (1+2+3)/3
        assert status.current_entropy == 3.0

    def test_window_size_limit(self):
        """Should maintain window size limit."""
        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config=config)

        for i in range(10):
            window.add(entropy=float(i), variance=0.1, token_index=i)

        status = window.status()
        assert status.sample_count == 5
        assert status.min_entropy == 5.0  # First 5 should be evicted

    def test_reset(self):
        """Reset should clear all state."""
        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config=config)
        window.add(entropy=4.0, variance=0.1, token_index=0)
        window.add(entropy=4.0, variance=0.1, token_index=1)

        window.reset()

        status = window.status()
        assert status.sample_count == 0

    def test_add_batch(self):
        """Should add multiple samples via batch."""
        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config=config)
        batch = [
            (1.0, 0.1, 0),
            (2.0, 0.2, 1),
            (3.0, 0.3, 2),
        ]
        status = window.add_batch(batch)

        assert status.sample_count == 3
        assert status.moving_average == 2.0

    def test_moving_average_reflects_raw_entropy(self):
        """Moving average should reflect raw entropy values."""
        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config=config)

        # Low entropy value
        window.reset()
        window.add(entropy=1.0, variance=0.1, token_index=0)
        assert window.status().moving_average == 1.0

        # Moderate entropy value
        window.reset()
        window.add(entropy=2.0, variance=0.1, token_index=0)
        assert window.status().moving_average == 2.0

        # High entropy value
        window.reset()
        window.add(entropy=4.0, variance=0.1, token_index=0)
        assert window.status().moving_average == 4.0

    def test_to_entropy_summary(self):
        """Should produce summary dict."""
        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config=config)
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
        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config=config)

        status = await window.add_async(entropy=2.0, variance=0.1, token_index=0)

        assert status.sample_count == 1
