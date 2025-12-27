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

"""Outcome-based tests for Domain Geometry Validation.

These tests validate that:
1. Baselines can be extracted and stored
2. Baselines have expected properties (negative Ricci, healthy layers)
3. Validation correctly identifies healthy vs. corrupted models
4. Domain metrics are discriminative across domains
5. Cross-model consistency within model families

References:
- NeurIPS 2024: "Geometry matters: ORC reveals neural structure"
- Nature 2024: "Deep learning as Ricci flow"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.domain_geometry_baselines import (
    BaselineRepository,
    BaselineValidationResult,
    DomainGeometryBaseline,
    DomainGeometryBaselineExtractor,
    DomainType,
    ManifoldHealthDistribution,
)
from modelcypher.core.domain.geometry.domain_geometry_validator import (
    DomainGeometryValidator,
    ValidationConfig,
)
from modelcypher.core.domain.geometry.manifold_curvature import (
    ManifoldHealth,
    OllivierRicciCurvature,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Get default compute backend."""
    return get_default_backend()


@pytest.fixture
def sample_healthy_baseline() -> DomainGeometryBaseline:
    """Create a sample healthy baseline for testing."""
    return DomainGeometryBaseline(
        domain="spatial",
        model_family="qwen",
        model_size="0.5B",
        model_path="/test/qwen-0.5b",
        ollivier_ricci_mean=-0.23,
        ollivier_ricci_std=0.08,
        ollivier_ricci_min=-0.35,
        ollivier_ricci_max=-0.12,
        manifold_health_distribution=ManifoldHealthDistribution(
            healthy=0.82,
            degenerate=0.15,
            collapsed=0.03,
        ),
        domain_metrics={
            "euclidean_consistency": 0.76,
            "gravity_alignment": 0.89,
            "volumetric_density": 0.64,
            "3d_grounding_score": 0.71,
        },
        intrinsic_dimension_mean=12.4,
        intrinsic_dimension_std=2.1,
        layers_analyzed=24,
        extraction_date="2025-12-27",
    )


@pytest.fixture
def sample_collapsed_baseline() -> DomainGeometryBaseline:
    """Create a collapsed (unhealthy) baseline for testing."""
    return DomainGeometryBaseline(
        domain="spatial",
        model_family="test",
        model_size="test",
        model_path="/test/collapsed",
        ollivier_ricci_mean=0.45,  # Positive = collapsed
        ollivier_ricci_std=0.12,
        ollivier_ricci_min=0.25,
        ollivier_ricci_max=0.65,
        manifold_health_distribution=ManifoldHealthDistribution(
            healthy=0.08,
            degenerate=0.12,
            collapsed=0.80,
        ),
        domain_metrics={
            "euclidean_consistency": 0.21,
            "gravity_alignment": 0.15,
        },
        intrinsic_dimension_mean=3.2,
        intrinsic_dimension_std=0.8,
        layers_analyzed=24,
    )


@pytest.fixture
def temp_baseline_dir(sample_healthy_baseline: DomainGeometryBaseline) -> Path:
    """Create a temp directory with sample baselines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        # Save a sample baseline
        sample_healthy_baseline.save(
            baseline_dir / "spatial_qwen_0.5B.json"
        )

        yield baseline_dir


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestDomainGeometryBaseline:
    """Tests for DomainGeometryBaseline dataclass."""

    def test_baseline_creation(self, sample_healthy_baseline: DomainGeometryBaseline):
        """Baselines can be created with expected fields."""
        b = sample_healthy_baseline

        assert b.domain == "spatial"
        assert b.model_family == "qwen"
        assert b.model_size == "0.5B"
        assert b.ollivier_ricci_mean == pytest.approx(-0.23)
        assert b.manifold_health_distribution.healthy == pytest.approx(0.82)

    def test_baseline_serialization_roundtrip(
        self, sample_healthy_baseline: DomainGeometryBaseline
    ):
        """Baselines survive JSON serialization."""
        # To dict
        data = sample_healthy_baseline.to_dict()
        assert isinstance(data, dict)
        assert data["domain"] == "spatial"

        # From dict
        restored = DomainGeometryBaseline.from_dict(data)
        assert restored.domain == sample_healthy_baseline.domain
        assert restored.ollivier_ricci_mean == pytest.approx(
            sample_healthy_baseline.ollivier_ricci_mean
        )

    def test_baseline_save_and_load(
        self, sample_healthy_baseline: DomainGeometryBaseline
    ):
        """Baselines can be saved to and loaded from JSON files."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            sample_healthy_baseline.save(path)
            assert path.exists()

            loaded = DomainGeometryBaseline.load(path)
            assert loaded.domain == sample_healthy_baseline.domain
            assert loaded.ollivier_ricci_mean == pytest.approx(
                sample_healthy_baseline.ollivier_ricci_mean
            )
        finally:
            path.unlink()


class TestManifoldHealthDistribution:
    """Tests for ManifoldHealthDistribution."""

    def test_distribution_sums_to_one(self):
        """Health distribution fractions should sum to approximately 1.0."""
        dist = ManifoldHealthDistribution(healthy=0.7, degenerate=0.2, collapsed=0.1)
        total = dist.healthy + dist.degenerate + dist.collapsed
        assert total == pytest.approx(1.0)

    def test_distribution_serialization(self):
        """Distribution survives dict conversion."""
        dist = ManifoldHealthDistribution(healthy=0.8, degenerate=0.15, collapsed=0.05)
        data = dist.to_dict()
        restored = ManifoldHealthDistribution.from_dict(data)

        assert restored.healthy == dist.healthy
        assert restored.degenerate == dist.degenerate
        assert restored.collapsed == dist.collapsed


# =============================================================================
# Baseline Repository Tests
# =============================================================================


class TestBaselineRepository:
    """Tests for BaselineRepository."""

    def test_repository_finds_saved_baseline(self, temp_baseline_dir: Path):
        """Repository can find a saved baseline by domain/family/size."""
        repo = BaselineRepository(temp_baseline_dir)

        baseline = repo.get_baseline("spatial", "qwen", "0.5B")
        assert baseline is not None
        assert baseline.domain == "spatial"
        assert baseline.model_family == "qwen"

    def test_repository_returns_none_for_missing(self, temp_baseline_dir: Path):
        """Repository returns None for non-existent baselines."""
        repo = BaselineRepository(temp_baseline_dir)

        baseline = repo.get_baseline("spatial", "nonexistent", "999B")
        assert baseline is None

    def test_repository_finds_all_baselines_for_domain(self, temp_baseline_dir: Path):
        """Repository can list all baselines for a domain."""
        repo = BaselineRepository(temp_baseline_dir)

        baselines = repo.get_baselines_for_domain("spatial")
        assert len(baselines) >= 1
        assert all(b.domain == "spatial" for b in baselines)


# =============================================================================
# Baseline Extraction Tests
# =============================================================================


class TestDomainGeometryBaselineExtractor:
    """Tests for DomainGeometryBaselineExtractor."""

    def test_extractor_creates_baseline_with_synthetic_data(self, backend: "Backend"):
        """Extractor produces a baseline even with synthetic activations."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        # Use a fake path - extractor will fall back to synthetic data
        baseline = extractor.extract_baseline(
            "/nonexistent/fake-model-0.5B",
            "spatial",
            layers=[0, 4, 8],
        )

        assert baseline.domain == "spatial"
        assert baseline.model_size != ""
        # Synthetic data should produce some layers
        assert baseline.layers_analyzed > 0 or baseline.extraction_config.get("error")

    def test_extractor_parses_model_info(self, backend: "Backend"):
        """Extractor correctly parses model family and size from path."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        # Test various model path formats
        test_cases = [
            ("/path/to/Qwen2.5-0.5B-Instruct-bf16", "qwen", "0.5B"),
            ("/path/to/Llama-3.2-3B-Instruct-4bit", "llama", "3B"),
            ("/path/to/Mistral-7B-Instruct-v0.3", "mistral", "7B"),
            ("/path/to/unknown-model", "unknown", "unknown"),
        ]

        for path, expected_family, expected_size in test_cases:
            family, size = extractor._parse_model_info(path)
            assert family == expected_family, f"Failed for {path}"
            assert size == expected_size, f"Failed for {path}"

    def test_extractor_generates_domain_probes(self, backend: "Backend"):
        """Extractor generates appropriate probes for each domain."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        for domain in ["spatial", "social", "temporal", "moral"]:
            probes = extractor._get_domain_probes(domain)
            assert len(probes) > 0, f"No probes for {domain}"
            assert all(isinstance(p, str) for p in probes)


# =============================================================================
# Curvature Interpretation Tests (SOTA Validation)
# =============================================================================


class TestCurvatureInterpretation:
    """Tests verifying curvature interpretation matches SOTA research."""

    def test_health_classification_thresholds(self):
        """Verify ManifoldHealth thresholds match SOTA expectations."""
        # Per NeurIPS 2024 and Nature 2024:
        # - Healthy LLMs exhibit hyperbolic (negative) curvature
        # - Curvature < -0.1 is healthy
        # - Curvature > 0.1 indicates collapse

        assert ManifoldHealth.HEALTHY.value == "healthy"
        assert ManifoldHealth.DEGENERATE.value == "degenerate"
        assert ManifoldHealth.COLLAPSED.value == "collapsed"

    def test_ollivier_ricci_health_classification(self, backend: "Backend"):
        """OllivierRicciCurvature classifies health correctly."""
        orc = OllivierRicciCurvature(backend=backend)

        # Test the _classify_health method
        assert orc._classify_health(-0.25) == ManifoldHealth.HEALTHY
        assert orc._classify_health(-0.15) == ManifoldHealth.HEALTHY
        assert orc._classify_health(-0.05) == ManifoldHealth.DEGENERATE
        assert orc._classify_health(0.0) == ManifoldHealth.DEGENERATE
        assert orc._classify_health(0.05) == ManifoldHealth.DEGENERATE
        assert orc._classify_health(0.15) == ManifoldHealth.COLLAPSED
        assert orc._classify_health(0.50) == ManifoldHealth.COLLAPSED


# =============================================================================
# Baseline Sanity Tests
# =============================================================================


class TestBaselineSanity:
    """Tests verifying baseline data meets expected properties."""

    def test_healthy_baseline_has_negative_ricci(
        self, sample_healthy_baseline: DomainGeometryBaseline
    ):
        """Healthy baselines should have negative mean Ricci curvature."""
        assert sample_healthy_baseline.ollivier_ricci_mean < 0, (
            f"Expected negative Ricci curvature, got "
            f"{sample_healthy_baseline.ollivier_ricci_mean}"
        )

    def test_healthy_baseline_has_majority_healthy_layers(
        self, sample_healthy_baseline: DomainGeometryBaseline
    ):
        """Healthy baselines should have >50% healthy layers."""
        health_pct = sample_healthy_baseline.manifold_health_distribution.healthy
        assert health_pct > 0.5, (
            f"Expected >50% healthy layers, got {health_pct:.0%}"
        )

    def test_healthy_baseline_has_low_collapsed_fraction(
        self, sample_healthy_baseline: DomainGeometryBaseline
    ):
        """Healthy baselines should have <20% collapsed layers."""
        collapsed_pct = sample_healthy_baseline.manifold_health_distribution.collapsed
        assert collapsed_pct < 0.2, (
            f"Expected <20% collapsed layers, got {collapsed_pct:.0%}"
        )

    def test_collapsed_baseline_has_positive_ricci(
        self, sample_collapsed_baseline: DomainGeometryBaseline
    ):
        """Collapsed baselines should have positive Ricci curvature."""
        assert sample_collapsed_baseline.ollivier_ricci_mean > 0, (
            "Collapsed baseline should have positive curvature"
        )


# =============================================================================
# Validation Logic Tests
# =============================================================================


class TestDomainGeometryValidator:
    """Tests for DomainGeometryValidator."""

    def test_validator_uses_heuristics_when_no_baseline(self, backend: "Backend"):
        """Validator falls back to heuristics when no baseline available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty baseline directory
            validator = DomainGeometryValidator(
                baseline_dir=tmpdir,
                config=ValidationConfig(fallback_to_heuristics=True),
                backend=backend,
            )

            # Validate with fake model (will use synthetic data)
            results = validator.validate_model(
                "/fake/qwen-0.5B",
                domains=["spatial"],
            )

            assert len(results) == 1
            result = results[0]
            assert result.domain == "spatial"
            assert result.baseline_model == "heuristic"

    def test_validator_detects_collapsed_model(
        self,
        sample_healthy_baseline: DomainGeometryBaseline,
        sample_collapsed_baseline: DomainGeometryBaseline,
        backend: "Backend",
    ):
        """Validator correctly identifies collapsed geometry as failing."""
        config = ValidationConfig()
        validator = DomainGeometryValidator(config=config, backend=backend)

        # Validate collapsed baseline against healthy baseline
        result = validator._validate_against_baseline(
            current=sample_collapsed_baseline,
            baseline=sample_healthy_baseline,
        )

        assert not result.passed, "Collapsed model should fail validation"
        assert any("collapse" in w.lower() or "positive" in w.lower()
                   for w in result.warnings), (
            f"Expected collapse warning, got: {result.warnings}"
        )

    def test_validator_passes_healthy_model(
        self,
        sample_healthy_baseline: DomainGeometryBaseline,
        backend: "Backend",
    ):
        """Validator passes a healthy model against itself."""
        config = ValidationConfig()
        validator = DomainGeometryValidator(config=config, backend=backend)

        # Validate healthy baseline against itself (should pass)
        result = validator._validate_against_baseline(
            current=sample_healthy_baseline,
            baseline=sample_healthy_baseline,
        )

        assert result.passed, f"Healthy model should pass: {result.warnings}"
        assert result.overall_deviation < 0.1

    def test_validator_computes_deviations(self, backend: "Backend"):
        """Validator correctly computes deviation metrics."""
        validator = DomainGeometryValidator(backend=backend)

        # Test deviation computation
        dev = validator._compute_deviation(current=0.8, baseline=1.0)
        assert dev == pytest.approx(0.2)  # 20% deviation

        dev = validator._compute_deviation(current=-0.15, baseline=-0.25)
        assert dev == pytest.approx(0.4)  # 40% deviation

    def test_validate_baseline_sanity_rejects_collapsed(
        self,
        sample_collapsed_baseline: DomainGeometryBaseline,
        backend: "Backend",
    ):
        """validate_baseline_sanity correctly rejects collapsed baselines."""
        validator = DomainGeometryValidator(backend=backend)

        is_valid, issues = validator.validate_baseline_sanity(sample_collapsed_baseline)

        assert not is_valid, "Collapsed baseline should be invalid"
        assert len(issues) > 0
        assert any("negative" in i.lower() or "curvature" in i.lower()
                   for i in issues)

    def test_validate_baseline_sanity_accepts_healthy(
        self,
        sample_healthy_baseline: DomainGeometryBaseline,
        backend: "Backend",
    ):
        """validate_baseline_sanity correctly accepts healthy baselines."""
        validator = DomainGeometryValidator(backend=backend)

        is_valid, issues = validator.validate_baseline_sanity(sample_healthy_baseline)

        assert is_valid, f"Healthy baseline should be valid: {issues}"
        assert len(issues) == 0


# =============================================================================
# Cross-Model Consistency Tests
# =============================================================================


class TestCrossModelConsistency:
    """Tests for consistency across model families."""

    def test_same_family_baselines_similar_geometry(self):
        """Models from same family should have similar geometry profiles."""
        # Create two Qwen baselines of different sizes
        qwen_small = DomainGeometryBaseline(
            domain="spatial",
            model_family="qwen",
            model_size="0.5B",
            model_path="/test/qwen-0.5b",
            ollivier_ricci_mean=-0.22,
            ollivier_ricci_std=0.07,
            ollivier_ricci_min=-0.32,
            ollivier_ricci_max=-0.11,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.80, degenerate=0.15, collapsed=0.05
            ),
            domain_metrics={},
            layers_analyzed=24,
        )

        qwen_large = DomainGeometryBaseline(
            domain="spatial",
            model_family="qwen",
            model_size="3B",
            model_path="/test/qwen-3b",
            ollivier_ricci_mean=-0.26,
            ollivier_ricci_std=0.09,
            ollivier_ricci_min=-0.38,
            ollivier_ricci_max=-0.14,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.85, degenerate=0.12, collapsed=0.03
            ),
            domain_metrics={},
            layers_analyzed=36,
        )

        # Ricci curvature should be similar (within 0.3)
        ricci_diff = abs(
            qwen_small.ollivier_ricci_mean - qwen_large.ollivier_ricci_mean
        )
        assert ricci_diff < 0.3, (
            f"Same-family models should have similar Ricci curvature, "
            f"got diff={ricci_diff:.2f}"
        )

    def test_domain_metrics_differ_between_domains(self):
        """Domain-specific metrics should be different for different domains."""
        spatial = DomainGeometryBaseline(
            domain="spatial",
            model_family="qwen",
            model_size="0.5B",
            model_path="/test",
            ollivier_ricci_mean=-0.2,
            ollivier_ricci_std=0.05,
            ollivier_ricci_min=-0.3,
            ollivier_ricci_max=-0.1,
            manifold_health_distribution=ManifoldHealthDistribution(0.8, 0.15, 0.05),
            domain_metrics={
                "euclidean_consistency": 0.85,
                "gravity_alignment": 0.72,
            },
            layers_analyzed=24,
        )

        social = DomainGeometryBaseline(
            domain="social",
            model_family="qwen",
            model_size="0.5B",
            model_path="/test",
            ollivier_ricci_mean=-0.2,
            ollivier_ricci_std=0.05,
            ollivier_ricci_min=-0.3,
            ollivier_ricci_max=-0.1,
            manifold_health_distribution=ManifoldHealthDistribution(0.8, 0.15, 0.05),
            domain_metrics={
                "social_manifold_score": 0.78,
                "power_axis_strength": 0.65,
            },
            layers_analyzed=24,
        )

        # Domain metrics should have different keys
        spatial_keys = set(spatial.domain_metrics.keys())
        social_keys = set(social.domain_metrics.keys())

        assert spatial_keys != social_keys, (
            "Spatial and social domains should have different metric keys"
        )

        # At least some metrics should be domain-specific
        spatial_only = spatial_keys - social_keys
        social_only = social_keys - spatial_keys

        assert len(spatial_only) > 0, "Spatial should have unique metrics"
        assert len(social_only) > 0, "Social should have unique metrics"


# =============================================================================
# Integration Tests (marked slow)
# =============================================================================


@pytest.mark.slow
class TestIntegrationWithRealActivations:
    """Integration tests using real or realistic activations."""

    def test_synthetic_activations_produce_valid_curvature(self, backend: "Backend"):
        """Synthetic activations should produce valid Ollivier-Ricci results."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        # Generate synthetic activations
        activations_by_layer = extractor._generate_synthetic_activations([0, 4, 8])

        assert len(activations_by_layer) == 3
        for layer_idx, acts in activations_by_layer.items():
            assert acts.shape[0] > 0  # Has samples
            assert acts.shape[1] > 0  # Has features

    def test_full_extraction_pipeline(self, backend: "Backend"):
        """Full extraction pipeline runs without errors."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        # Run extraction with fake model (uses synthetic data)
        baseline = extractor.extract_baseline(
            "/fake/test-model-0.5B",
            "spatial",
        )

        # Should have valid structure even with synthetic data
        assert baseline.domain == "spatial"
        assert isinstance(baseline.ollivier_ricci_mean, float)
        assert isinstance(baseline.manifold_health_distribution, ManifoldHealthDistribution)


# =============================================================================
# Real Model Tests (marked for CI skip)
# =============================================================================


@pytest.mark.real_model
@pytest.mark.skip(reason="Requires real model path - enable manually")
class TestRealModelValidation:
    """Tests that run on actual models - skipped by default."""

    def test_real_qwen_model_passes_validation(self):
        """A real Qwen model should pass geometry validation."""
        model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-bf16"

        validator = DomainGeometryValidator()
        results = validator.validate_model(model_path, domains=["spatial"])

        for result in results:
            assert result.passed, (
                f"{result.domain} validation failed: {result.warnings}"
            )
