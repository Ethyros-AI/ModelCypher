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

"""Tests for Red Team Probe.

Geometric tests for the static analysis probe that uses metadata embeddings.
"""

from __future__ import annotations

import math
import pytest

from modelcypher.core.domain.safety.behavioral_probes import (
    ProbeContext,
    ProbeResult,
)
from modelcypher.core.domain.safety.red_team_probe import (
    RedTeamProbe,
    RedTeamScanner,
    ThreatIndicator,
    _collect_metadata_items,
    _metadata_outliers,
)


class DummyEmbedder:
    """Deterministic embedding stub for geometry-only tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            checksum = float(sum(ord(ch) for ch in text) % 97)
            embeddings.append([length, checksum])
        return embeddings

    @property
    def dimension(self) -> int:
        return 2


class TestRedTeamProbe:
    """Tests for RedTeamProbe class."""

    @pytest.fixture
    def probe(self):
        """Create probe instance."""
        return RedTeamProbe()

    def test_name_and_version(self, probe):
        """Probe has correct name and version."""
        assert probe.name == "red-team-static"
        assert probe.version == "probe-rt-v1.0"

    def test_evaluate_missing_embedder(self, probe):
        """Missing embedder skips probe."""
        context = ProbeContext(
            adapter_name="adapter",
        )
        result = probe.evaluate(context)
        assert result.has_findings is False
        assert result.finding_counts is not None
        assert result.finding_counts["metadata_items"] == 0

    def test_evaluate_insufficient_fields(self, probe):
        """Single metadata field skips probe."""
        context = ProbeContext(
            adapter_name="adapter",
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.has_findings is False
        assert result.finding_counts is not None
        assert result.finding_counts["metadata_items"] == 1

    def test_evaluate_similar_items_low_distances(self, probe):
        """Similar metadata fields produce very small distances."""
        context = ProbeContext(
            adapter_name="a",
            adapter_description="a",
            skill_tags=("a",),
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        # Similar items should have very small mean distances (near-zero due to floating point)
        eps = math.ulp(1.0)
        assert result.finding_counts["mean_distance"] <= eps
        assert result.finding_counts["max_distance"] <= eps

    def test_evaluate_detects_outlier(self, probe):
        """Outlier metadata field is detected via geodesic distances."""
        context = ProbeContext(
            adapter_name="a",
            adapter_description="a",
            skill_tags=("a" * 100,),
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        items = _collect_metadata_items(context)
        distances, outliers, threshold, mean_distance, max_distance = _metadata_outliers(
            items, context.embedder
        )
        assert result.has_findings is True
        assert result.finding_counts is not None
        assert result.finding_counts["outlier_items"] == len(outliers)
        assert abs(result.finding_counts["distance_threshold"] - threshold) <= math.ulp(
            max(1.0, abs(threshold))
        )
        assert abs(result.finding_counts["mean_distance"] - mean_distance) <= math.ulp(
            max(1.0, abs(mean_distance))
        )
        assert abs(result.finding_counts["max_distance"] - max_distance) <= math.ulp(
            max(1.0, abs(max_distance))
        )
        assert any("mean_distance" in f for f in result.findings)
        assert result.finding_counts["metadata_items"] == 3

    def test_evaluate_counts_include_distances(self, probe):
        """Finding counts report raw distance statistics."""
        context = ProbeContext(
            adapter_name="a",
            adapter_description="a",
            skill_tags=("a" * 100,),
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert "distance_threshold" in result.finding_counts
        assert "mean_distance" in result.finding_counts
        assert "max_distance" in result.finding_counts


class TestThreatIndicator:
    """Tests for ThreatIndicator dataclass."""

    def test_indicator_fields(self):
        """Indicator has all required fields."""
        indicator = ThreatIndicator(
            field="adapter_name",
            text="adapter",
            mean_distance=1.25,
        )
        assert indicator.field == "adapter_name"
        assert indicator.text == "adapter"
        assert indicator.mean_distance == 1.25

    def test_indicator_frozen(self):
        """Indicator is immutable."""
        indicator = ThreatIndicator(
            field="adapter_description",
            text="desc",
            mean_distance=0.5,
        )
        with pytest.raises(AttributeError):
            indicator.mean_distance = 0.9


class TestRedTeamScanner:
    """Tests for RedTeamScanner class."""

    def test_scan_adapter_missing_embedder(self):
        """Scanner returns no indicators without embedder."""
        scanner = RedTeamScanner()
        indicators = scanner.scan_adapter(name="adapter")
        assert indicators == []

    def test_scan_adapter_outlier(self):
        """Scanner returns indicators for outlier metadata."""
        scanner = RedTeamScanner(embedder=DummyEmbedder())
        indicators = scanner.scan_adapter(
            name="a",
            description="a",
            skill_tags=["a" * 100],
        )
        items = _collect_metadata_items(
            ProbeContext(
                adapter_name="a",
                adapter_description="a",
                skill_tags=("a" * 100,),
                embedder=DummyEmbedder(),
            )
        )
        _, outliers, _, _, _ = _metadata_outliers(items, DummyEmbedder())
        assert len(indicators) == len(outliers)
        assert all(isinstance(ind, ThreatIndicator) for ind in indicators)


class TestIntegration:
    """Integration tests for the red team probe system."""

    def test_full_evaluation_pipeline(self):
        """Complete evaluation returns ProbeResult."""
        probe = RedTeamProbe()
        context = ProbeContext(
            adapter_name="a",
            adapter_description="a",
            skill_tags=("a",),
            creator="creator",
            base_model_id="base",
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "red-team-static"
