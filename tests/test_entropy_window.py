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

"""
Tests for EntropyWindow sliding window tracker.
"""
import pytest
import asyncio
import uuid
from datetime import datetime

from modelcypher.core.domain.entropy.entropy_window import (
    EntropyWindow,
    EntropyWindowConfig,
    EntropyWindowStatus,
    EntropyLevel,
)


class TestEntropyWindowConfig:
    """Tests for EntropyWindowConfig."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        config = EntropyWindowConfig()
        
        assert config.window_size == 20
        assert config.minimum_samples == 5
        assert config.high_entropy_threshold == 3.0
        assert config.circuit_breaker_threshold == 4.0
        assert config.sustained_high_count == 3
    
    def test_custom_values(self):
        """Should accept custom values."""
        config = EntropyWindowConfig(
            window_size=10,
            circuit_breaker_threshold=5.0,
        )
        
        assert config.window_size == 10
        assert config.circuit_breaker_threshold == 5.0


class TestEntropyLevel:
    """Tests for EntropyLevel enum."""
    
    def test_values(self):
        """Should have expected values."""
        assert EntropyLevel.LOW.value == "low"
        assert EntropyLevel.MODERATE.value == "moderate"
        assert EntropyLevel.HIGH.value == "high"


class TestEntropyWindow:
    """Tests for EntropyWindow."""
    
    def test_initialization(self):
        """Should initialize with default config."""
        window = EntropyWindow()
        
        assert window.config is not None
        assert window.window_id is not None
    
    def test_custom_window_id(self):
        """Should accept custom window ID."""
        custom_id = str(uuid.uuid4())
        window = EntropyWindow(window_id=custom_id)
        
        assert window.window_id == custom_id
    
    def test_add_single_sample(self):
        """Should add a single sample."""
        window = EntropyWindow()
        status = window.add(entropy=2.0, variance=0.1, token_index=0)
        
        assert status.sample_count == 1
        assert status.current_entropy == 2.0
        assert status.moving_average == 2.0
    
    def test_add_multiple_samples(self):
        """Should compute moving average correctly."""
        window = EntropyWindow()
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
    
    def test_circuit_breaker_high_average(self):
        """Should trip circuit breaker on high moving average."""
        config = EntropyWindowConfig(circuit_breaker_threshold=3.5)
        window = EntropyWindow(config=config)
        
        # Add high entropy samples
        for i in range(5):
            window.add(entropy=4.0, variance=0.1, token_index=i)
        
        status = window.status()
        assert status.should_trip_circuit_breaker
    
    def test_circuit_breaker_sustained_high(self):
        """Should trip on sustained high count."""
        config = EntropyWindowConfig(
            high_entropy_threshold=3.0,
            sustained_high_count=3,
        )
        window = EntropyWindow(config=config)
        
        # Add 3 consecutive high entropy samples
        for i in range(3):
            window.add(entropy=3.5, variance=0.1, token_index=i)
        
        status = window.status()
        assert status.consecutive_high_count == 3
        assert status.should_trip_circuit_breaker
    
    def test_consecutive_high_count_reset(self):
        """Low entropy should reset consecutive count."""
        config = EntropyWindowConfig(high_entropy_threshold=3.0)
        window = EntropyWindow(config=config)
        
        window.add(entropy=4.0, variance=0.1, token_index=0)
        window.add(entropy=4.0, variance=0.1, token_index=1)
        status = window.add(entropy=1.0, variance=0.1, token_index=2)  # Low
        
        assert status.consecutive_high_count == 0
    
    def test_reset(self):
        """Reset should clear all state."""
        window = EntropyWindow()
        window.add(entropy=4.0, variance=0.1, token_index=0)
        window.add(entropy=4.0, variance=0.1, token_index=1)
        
        window.reset()
        
        status = window.status()
        assert status.sample_count == 0
    
    def test_reset_circuit_breaker(self):
        """Should reset only circuit breaker state."""
        config = EntropyWindowConfig(sustained_high_count=2)
        window = EntropyWindow(config=config)
        
        window.add(entropy=5.0, variance=0.1, token_index=0)
        window.add(entropy=5.0, variance=0.1, token_index=1)
        
        assert window.status().should_trip_circuit_breaker
        
        window.reset_circuit_breaker()
        
        status = window.status()
        assert not status.should_trip_circuit_breaker
        assert status.sample_count == 2  # Samples preserved
    
    def test_add_batch(self):
        """Should add multiple samples via batch."""
        window = EntropyWindow()
        batch = [
            (1.0, 0.1, 0),
            (2.0, 0.2, 1),
            (3.0, 0.3, 2),
        ]
        status = window.add_batch(batch)
        
        assert status.sample_count == 3
        assert status.moving_average == 2.0
    
    def test_entropy_level_classification(self):
        """Should classify entropy levels correctly."""
        window = EntropyWindow()
        
        # Low: < 1.5
        window.reset()
        window.add(entropy=1.0, variance=0.1, token_index=0)
        assert window.status().level == EntropyLevel.LOW
        
        # Moderate: 1.5-3.0
        window.reset()
        window.add(entropy=2.0, variance=0.1, token_index=0)
        assert window.status().level == EntropyLevel.MODERATE
        
        # High: > 3.0
        window.reset()
        window.add(entropy=4.0, variance=0.1, token_index=0)
        assert window.status().level == EntropyLevel.HIGH
    
    def test_to_entropy_summary(self):
        """Should produce summary dict."""
        window = EntropyWindow()
        window.add(entropy=2.0, variance=0.1, token_index=0)
        
        summary = window.to_entropy_summary()
        
        assert "window_id" in summary
        assert "logit_entropy" in summary
        assert summary["sample_count"] == 1
    
    def test_circuit_breaker_alert(self):
        """Should generate alert when tripped."""
        config = EntropyWindowConfig(sustained_high_count=1, high_entropy_threshold=2.0)
        window = EntropyWindow(config=config)
        
        window.add(entropy=3.0, variance=0.1, token_index=0)
        
        alert = window.circuit_breaker_alert()
        
        assert alert is not None
        assert alert["type"] == "circuit_breaker_tripped"
    
    def test_circuit_breaker_alert_none(self):
        """Should return None when not tripped."""
        window = EntropyWindow()
        window.add(entropy=1.0, variance=0.1, token_index=0)
        
        alert = window.circuit_breaker_alert()
        
        assert alert is None


class TestEntropyWindowAsync:
    """Tests for async operations."""
    
    @pytest.mark.asyncio
    async def test_add_async(self):
        """Should add sample asynchronously."""
        window = EntropyWindow()
        
        status = await window.add_async(entropy=2.0, variance=0.1, token_index=0)
        
        assert status.sample_count == 1
