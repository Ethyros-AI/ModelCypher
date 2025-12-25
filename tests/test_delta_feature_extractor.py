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

"""Tests for DeltaFeatureExtractor and DeltaFeatureProbe."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelcypher.core.domain.safety.adapter_safety_models import (
    AdapterSafetyTier,
    AdapterSafetyTrigger,
)
from modelcypher.core.domain.safety.adapter_safety_probe import ProbeContext
from modelcypher.core.domain.safety.delta_feature_extractor import (
    DeltaFeatureExtractor,
    DeltaFeatureProbe,
)
from modelcypher.core.domain.safety.delta_feature_set import DeltaFeatureSet


class TestDeltaFeatureSet:
    """Tests for DeltaFeatureSet dataclass."""

    def test_empty_feature_set(self) -> None:
        """Empty feature set has sensible defaults."""
        fs = DeltaFeatureSet()
        assert fs.layer_count == 0
        assert fs.has_suspect_layers is False
        assert fs.suspect_layer_fraction == 0.0
        assert fs.mean_l2_norm == 0.0
        assert fs.max_l2_norm == 0.0
        assert fs.mean_sparsity == 0.0

    def test_layer_count(self) -> None:
        """layer_count returns number of L2 norms."""
        fs = DeltaFeatureSet(l2_norms=(1.0, 2.0, 3.0))
        assert fs.layer_count == 3

    def test_has_suspect_layers(self) -> None:
        """has_suspect_layers detects suspect indices."""
        fs1 = DeltaFeatureSet(suspect_layer_indices=())
        assert fs1.has_suspect_layers is False

        fs2 = DeltaFeatureSet(suspect_layer_indices=(0, 2))
        assert fs2.has_suspect_layers is True

    def test_suspect_layer_fraction(self) -> None:
        """suspect_layer_fraction computes correctly."""
        fs = DeltaFeatureSet(
            l2_norms=(1.0, 2.0, 3.0, 4.0),
            suspect_layer_indices=(0, 2),
        )
        assert fs.suspect_layer_fraction == 0.5

    def test_mean_l2_norm(self) -> None:
        """mean_l2_norm computes average."""
        fs = DeltaFeatureSet(l2_norms=(2.0, 4.0, 6.0))
        assert fs.mean_l2_norm == 4.0

    def test_max_l2_norm(self) -> None:
        """max_l2_norm returns maximum."""
        fs = DeltaFeatureSet(l2_norms=(2.0, 10.0, 6.0))
        assert fs.max_l2_norm == 10.0

    def test_mean_sparsity(self) -> None:
        """mean_sparsity computes average."""
        fs = DeltaFeatureSet(sparsity=(0.1, 0.3, 0.5))
        assert abs(fs.mean_sparsity - 0.3) < 0.001

    def test_to_dict(self) -> None:
        """to_dict serializes correctly."""
        fs = DeltaFeatureSet(
            l2_norms=(1.0, 2.0),
            sparsity=(0.1, 0.2),
            suspect_layer_indices=(1,),
            feature_version="test-v1",
        )
        d = fs.to_dict()
        assert d["l2_norms"] == [1.0, 2.0]
        assert d["sparsity"] == [0.1, 0.2]
        assert d["suspect_layer_indices"] == [1]
        assert d["feature_version"] == "test-v1"

    def test_from_dict(self) -> None:
        """from_dict deserializes correctly."""
        data = {
            "l2_norms": [1.0, 2.0],
            "sparsity": [0.5],
            "cosine_to_aligned": [],
            "suspect_layer_indices": [0],
            "feature_version": "v2",
        }
        fs = DeltaFeatureSet.from_dict(data)
        assert fs.l2_norms == (1.0, 2.0)
        assert fs.sparsity == (0.5,)
        assert fs.suspect_layer_indices == (0,)
        assert fs.feature_version == "v2"

    def test_from_dict_defaults(self) -> None:
        """from_dict handles missing keys."""
        fs = DeltaFeatureSet.from_dict({})
        assert fs.l2_norms == ()
        assert fs.sparsity == ()


class TestDeltaFeatureExtractor:
    """Tests for DeltaFeatureExtractor class."""

    def test_version(self) -> None:
        """Extractor has version string."""
        extractor = DeltaFeatureExtractor()
        assert extractor.VERSION == "delta-v1.0"

    def test_find_safetensors_files_empty(self, tmp_path: Path) -> None:
        """_find_safetensors_files returns empty for empty dir."""
        extractor = DeltaFeatureExtractor()
        result = extractor._find_safetensors_files(tmp_path)
        assert result == []

    def test_find_safetensors_files_nonexistent(self, tmp_path: Path) -> None:
        """_find_safetensors_files returns empty for nonexistent dir."""
        extractor = DeltaFeatureExtractor()
        result = extractor._find_safetensors_files(tmp_path / "nonexistent")
        assert result == []

    def test_find_safetensors_files_found(self, tmp_path: Path) -> None:
        """_find_safetensors_files finds .safetensors files."""
        (tmp_path / "model.safetensors").write_bytes(b"dummy")
        (tmp_path / "adapter.safetensors").write_bytes(b"dummy")
        (tmp_path / "config.json").write_text("{}")

        extractor = DeltaFeatureExtractor()
        result = extractor._find_safetensors_files(tmp_path)
        assert len(result) == 2
        assert all(f.suffix == ".safetensors" for f in result)

    def test_find_outlier_indices_empty(self) -> None:
        """_find_outlier_indices returns empty for empty input."""
        extractor = DeltaFeatureExtractor()
        assert extractor._find_outlier_indices([]) == []

    def test_find_outlier_indices_too_few(self) -> None:
        """_find_outlier_indices returns empty for <= 2 values."""
        extractor = DeltaFeatureExtractor()
        assert extractor._find_outlier_indices([1.0]) == []
        assert extractor._find_outlier_indices([1.0, 2.0]) == []

    def test_find_outlier_indices_uniform(self) -> None:
        """_find_outlier_indices returns empty for uniform values."""
        extractor = DeltaFeatureExtractor()
        assert extractor._find_outlier_indices([1.0, 1.0, 1.0, 1.0]) == []

    def test_find_outlier_indices_with_outlier(self) -> None:
        """_find_outlier_indices detects outliers."""
        # Use low std_devs threshold so outlier is easier to detect
        extractor = DeltaFeatureExtractor(outlier_std_devs=1.0)
        # Values with clear outlier: mean=3, std≈1.41, threshold≈4.41
        values = [1.0, 2.0, 3.0, 4.0, 10.0]
        outliers = extractor._find_outlier_indices(values)
        assert 4 in outliers  # Index of 10.0

    @pytest.mark.asyncio
    async def test_extract_empty_directory(self, tmp_path: Path) -> None:
        """extract returns empty features for empty directory."""
        extractor = DeltaFeatureExtractor()
        result = await extractor.extract(tmp_path)
        assert result.layer_count == 0
        assert result.feature_version == "delta-v1.0"

    @pytest.mark.asyncio
    async def test_extract_nonexistent_directory(self, tmp_path: Path) -> None:
        """extract returns empty features for nonexistent directory."""
        extractor = DeltaFeatureExtractor()
        result = await extractor.extract(tmp_path / "nonexistent")
        assert result.layer_count == 0


class TestDeltaFeatureProbe:
    """Tests for DeltaFeatureProbe class."""

    def test_probe_metadata(self) -> None:
        """Probe has correct metadata."""
        probe = DeltaFeatureProbe()
        assert probe.name == "delta-features"
        assert probe.version == "probe-delta-v1.0"

    def test_supported_tiers(self) -> None:
        """Probe supports all tiers."""
        probe = DeltaFeatureProbe()
        assert AdapterSafetyTier.QUICK in probe.supported_tiers
        assert AdapterSafetyTier.STANDARD in probe.supported_tiers
        assert AdapterSafetyTier.FULL in probe.supported_tiers

    @pytest.mark.asyncio
    async def test_evaluate_empty_adapter(self, tmp_path: Path) -> None:
        """evaluate handles empty adapter directory."""
        probe = DeltaFeatureProbe()
        context = ProbeContext(
            adapter_path=tmp_path,
            tier=AdapterSafetyTier.QUICK,
            trigger=AdapterSafetyTrigger.IMPORT_ADAPTER,
        )
        result = await probe.evaluate(context)
        assert result.probe_name == "delta-features"
        assert result.risk_score == 0.0
        assert result.triggered is False

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_extractor(self, tmp_path: Path) -> None:
        """evaluate uses provided extractor."""

        # Create a mock extractor that returns known features
        class MockExtractor:
            VERSION = "mock-v1"

            async def extract(self, path: Path) -> DeltaFeatureSet:
                return DeltaFeatureSet(
                    l2_norms=(1.0, 2.0, 3.0),
                    sparsity=(0.1, 0.2, 0.3),
                    suspect_layer_indices=(),
                    feature_version="mock-v1",
                )

        probe = DeltaFeatureProbe(extractor=MockExtractor())  # type: ignore
        context = ProbeContext(
            adapter_path=tmp_path,
            tier=AdapterSafetyTier.STANDARD,
            trigger=AdapterSafetyTrigger.POST_TRAINING,
        )
        result = await probe.evaluate(context)
        assert result.triggered is False
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_high_l2_norms(self, tmp_path: Path) -> None:
        """evaluate flags high L2 norms."""

        class MockExtractor:
            VERSION = "mock-v1"

            async def extract(self, path: Path) -> DeltaFeatureSet:
                return DeltaFeatureSet(
                    l2_norms=(100.0, 200.0),  # Very high norms
                    sparsity=(0.1, 0.2),
                    suspect_layer_indices=(),
                )

        probe = DeltaFeatureProbe(
            extractor=MockExtractor(),  # type: ignore
            l2_norm_warning_threshold=50.0,
        )
        context = ProbeContext(
            adapter_path=tmp_path,
            tier=AdapterSafetyTier.STANDARD,
            trigger=AdapterSafetyTrigger.IMPORT_ADAPTER,
        )
        result = await probe.evaluate(context)
        assert result.risk_score >= 0.3
        assert any("L2 norm" in f for f in result.findings)

    @pytest.mark.asyncio
    async def test_evaluate_suspect_layers(self, tmp_path: Path) -> None:
        """evaluate flags high fraction of suspect layers."""

        class MockExtractor:
            VERSION = "mock-v1"

            async def extract(self, path: Path) -> DeltaFeatureSet:
                return DeltaFeatureSet(
                    l2_norms=(1.0, 2.0, 3.0, 4.0, 5.0),
                    sparsity=(0.1,) * 5,
                    suspect_layer_indices=(0, 1, 2),  # 60% suspect
                )

        probe = DeltaFeatureProbe(
            extractor=MockExtractor(),  # type: ignore
            suspect_layer_fraction=0.2,
        )
        context = ProbeContext(
            adapter_path=tmp_path,
            tier=AdapterSafetyTier.FULL,
            trigger=AdapterSafetyTrigger.MERGE,
        )
        result = await probe.evaluate(context)
        assert result.risk_score >= 0.4
        assert any("outlier" in f for f in result.findings)

    @pytest.mark.asyncio
    async def test_evaluate_high_sparsity(self, tmp_path: Path) -> None:
        """evaluate flags unusually high sparsity."""

        class MockExtractor:
            VERSION = "mock-v1"

            async def extract(self, path: Path) -> DeltaFeatureSet:
                return DeltaFeatureSet(
                    l2_norms=(1.0, 2.0),
                    sparsity=(0.95, 0.98),  # Very sparse
                    suspect_layer_indices=(),
                )

        probe = DeltaFeatureProbe(
            extractor=MockExtractor(),  # type: ignore
            high_sparsity_threshold=0.9,
        )
        context = ProbeContext(
            adapter_path=tmp_path,
            tier=AdapterSafetyTier.STANDARD,
            trigger=AdapterSafetyTrigger.IMPORT_ADAPTER,
        )
        result = await probe.evaluate(context)
        assert result.risk_score >= 0.25
        assert any("sparsity" in f for f in result.findings)

    @pytest.mark.asyncio
    async def test_evaluate_zero_norm_layers(self, tmp_path: Path) -> None:
        """evaluate flags zero-norm layers as potentially corrupted."""

        class MockExtractor:
            VERSION = "mock-v1"

            async def extract(self, path: Path) -> DeltaFeatureSet:
                return DeltaFeatureSet(
                    l2_norms=(0.0, 1.0, 0.0),  # Two zero-norm layers
                    sparsity=(0.0, 0.1, 0.0),
                    suspect_layer_indices=(),
                )

        probe = DeltaFeatureProbe(extractor=MockExtractor())  # type: ignore
        context = ProbeContext(
            adapter_path=tmp_path,
            tier=AdapterSafetyTier.QUICK,
            trigger=AdapterSafetyTrigger.ACTIVATION_CHECK,
        )
        result = await probe.evaluate(context)
        assert result.risk_score >= 0.5
        assert any("zero L2 norm" in f for f in result.findings)


class TestDeltaFeatureExtractorIntegration:
    """Integration tests requiring safetensors (skipped if unavailable)."""

    @pytest.mark.asyncio
    async def test_extract_real_safetensors(self, tmp_path: Path) -> None:
        """extract works with real safetensors files."""
        pytest.importorskip("safetensors")
        import numpy as np
        from safetensors.numpy import save_file

        # Create a simple safetensors file
        tensors = {
            "layer.0.weight": np.random.randn(64, 32).astype(np.float32),
            "layer.1.weight": np.random.randn(64, 32).astype(np.float32),
        }
        save_file(tensors, tmp_path / "adapter.safetensors")

        extractor = DeltaFeatureExtractor()
        result = await extractor.extract(tmp_path)

        assert result.layer_count == 2
        assert len(result.l2_norms) == 2
        assert len(result.sparsity) == 2
        assert all(n > 0 for n in result.l2_norms)
