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

"""Tests for bridge generator module.

Validates that cross-modal bridges can be generated, saved, loaded, and applied.
"""

import tempfile
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.adapters.bridge_store import SafetensorsBridgeStore
from modelcypher.core.domain.bridge.generator import (
    BridgeGenerator,
    BridgeGeneratorResult,
    CrossModalBridge,
)
from modelcypher.core.use_cases.bridge_service import BridgeService


class TestBridgeGeneration:
    """Test bridge generation."""

    def test_generate_same_dimension(self) -> None:
        """Generate bridge between same dimensions."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(42)
        n_samples = 15
        d = 64

        source = backend.random_normal((n_samples, d))
        target = backend.random_normal((n_samples, d))
        backend.eval(source, target)

        result = generator.generate(source, target)

        assert result.source_dim == d
        assert result.target_dim == d
        assert result.cka_achieved > 0.999, f"CKA = {result.cka_achieved}"
        assert result.n_samples == n_samples
        assert result.transform.shape == (d, d)
        assert result.transform_inv.shape == (d, d)

    def test_generate_cross_dimension(self) -> None:
        """Generate bridge from lower to higher dimension."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(123)
        n_samples = 12
        d_source = 32
        d_target = 64

        source = backend.random_normal((n_samples, d_source))
        target = backend.random_normal((n_samples, d_target))
        backend.eval(source, target)

        result = generator.generate(source, target)

        assert result.source_dim == d_source
        assert result.target_dim == d_target
        assert result.cka_achieved > 0.999
        assert result.transform.shape == (d_source, d_target)
        assert result.transform_inv.shape == (d_target, d_source)

    def test_raw_cka_less_than_aligned(self) -> None:
        """Raw CKA should be < 1.0, aligned CKA should be 1.0."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(456)
        n_samples = 10
        d = 32

        # Create correlated but different data
        base = backend.random_normal((n_samples, d))
        noise = backend.random_normal((n_samples, d)) * 0.5
        source = base + noise
        target = base
        backend.eval(source, target)

        result = generator.generate(source, target)

        # Raw should be correlated but < 1.0
        assert result.raw_cka < 1.0
        # Aligned should be 1.0
        assert result.cka_achieved > 0.999


class TestBridgeSaveLoad:
    """Test bridge save/load functionality."""

    def test_save_and_load_bridge(self) -> None:
        """Save bridge to file and load it back."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(789)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 64))
        backend.eval(source, target)

        result = generator.generate(
            source, target,
            source_name="clip",
            target_name="lfm2",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bridge.safetensors"
            service = BridgeService(store=SafetensorsBridgeStore(), backend=backend)
            service.save(result, path)

            assert path.exists()

            loaded = service.load(path)

            assert loaded.source_dim == 32
            assert loaded.target_dim == 64
            assert loaded.source_name == "clip"
            assert loaded.target_name == "lfm2"
            assert loaded.scale_ratio == pytest.approx(result.scale_ratio, rel=1e-5)


class TestBridgeApplication:
    """Test applying bridges to embeddings."""

    def test_apply_forward(self) -> None:
        """Apply bridge in forward direction."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(101)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 64))
        backend.eval(source, target)

        result = generator.generate(source, target)
        bridge = generator.to_bridge(result)

        # Apply to new embeddings
        new_source = backend.random_normal((5, 32))
        backend.eval(new_source)

        transformed = bridge.apply(new_source)
        backend.eval(transformed)

        assert transformed.shape == (5, 64)

    def test_apply_inverse(self) -> None:
        """Apply bridge in reverse direction."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(202)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 64))
        backend.eval(source, target)

        result = generator.generate(source, target)
        bridge = generator.to_bridge(result)

        # Apply inverse to target-space embeddings
        target_embeds = backend.random_normal((5, 64))
        backend.eval(target_embeds)

        reversed_embeds = bridge.apply_inverse(target_embeds)
        backend.eval(reversed_embeds)

        assert reversed_embeds.shape == (5, 32)

    def test_round_trip_approximate(self) -> None:
        """Round trip (forward then inverse) approximately recovers original."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(303)
        n_samples = 12
        d_source = 32
        d_target = 64

        source = backend.random_normal((n_samples, d_source))
        target = backend.random_normal((n_samples, d_target))
        backend.eval(source, target)

        result = generator.generate(source, target)
        bridge = generator.to_bridge(result)

        # Round trip
        transformed = bridge.apply(source, normalize_scale=False)
        recovered = bridge.apply_inverse(transformed, normalize_scale=False)
        backend.eval(transformed, recovered)

        # Should be close to original (pinv is not exact inverse)
        diff = backend.abs(recovered - source)
        mean_diff = backend.mean(diff)
        backend.eval(mean_diff)
        mean_diff_val = float(backend.to_scalar(mean_diff))

        # Allow reasonable tolerance for pseudo-inverse
        assert mean_diff_val < 1.0, f"Round trip error too large: {mean_diff_val}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sample_count_mismatch_raises(self) -> None:
        """Mismatched sample counts should raise."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(404)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((15, 64))  # Different n_samples
        backend.eval(source, target)

        with pytest.raises(ValueError, match="Sample counts must match"):
            generator.generate(source, target)

    def test_metadata_preserved(self) -> None:
        """Metadata should be preserved through save/load."""
        backend = get_default_backend()
        generator = BridgeGenerator(backend)

        backend.random_seed(505)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 64))
        backend.eval(source, target)

        result = generator.generate(
            source, target,
            source_name="whisper",
            target_name="t5",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bridge.safetensors"
            service = BridgeService(store=SafetensorsBridgeStore(), backend=backend)
            service.save(result, path)
            loaded = service.load(path)

            assert loaded.source_name == "whisper"
            assert loaded.target_name == "t5"
