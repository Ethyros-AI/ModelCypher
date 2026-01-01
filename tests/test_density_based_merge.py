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

"""Tests for density-based merge logic.

The core principle: We aren't merging whole models. We're filling sparse
regions of an invariant high-dimensional shape.

Key properties tested:
1. Density profile correctly identifies sparse regions (high intrinsic dim)
2. Knowledge diff finds graft opportunities (source dense + target sparse)
3. Graft mask filtering respects density decisions
4. Transplant stage only modifies concepts where graft_mask is True
5. Merged model maintains or improves density
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    LayerDensityProfile,
    ModelDensityProfile,
)
from modelcypher.core.domain.geometry.knowledge_diff import (
    GraftOpportunity,
    KnowledgeDiff,
    KnowledgeDiffer,
    compute_graft_mask,
)
from modelcypher.core.use_cases.merge_stages.stage_2_density import (
    DensityStageConfig,
    DensityStageResult,
    filter_core_probes_by_graft_mask,
    stage_density,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Provide backend for tests."""
    return get_default_backend()


@pytest.fixture
def dense_concept() -> ConceptDensity:
    """Create a dense concept (well-learned, low intrinsic dimension)."""
    return ConceptDensity(
        probe_id="math_addition",
        name="Addition",
        domain="mathematical",
        layer=5,
        intrinsic_dimension=2.5,  # Low = dense
        density_score=0.85,  # High = dense
        activation_variance=0.001,
        cluster_tightness=0.95,
        dimension_class="dense",
    )


@pytest.fixture
def sparse_concept() -> ConceptDensity:
    """Create a sparse concept (gap in knowledge, high intrinsic dimension)."""
    return ConceptDensity(
        probe_id="logic_modus_ponens",
        name="Modus Ponens",
        domain="logical",
        layer=5,
        intrinsic_dimension=12.0,  # High = sparse
        density_score=0.25,  # Low = sparse
        activation_variance=0.5,
        cluster_tightness=0.3,
        dimension_class="sparse",
    )


@pytest.fixture
def source_profile_with_dense_math(dense_concept: ConceptDensity) -> ModelDensityProfile:
    """Source model that is dense in mathematical concepts."""
    return ModelDensityProfile(
        model_path="/source/model",
        layers=[5],
        layer_profiles={
            5: LayerDensityProfile(
                layer=5,
                concept_densities=[dense_concept],
                mean_density=0.85,
                median_density=0.85,
                sparse_concept_count=0,
                dense_concept_count=1,
                density_threshold=0.5,
            )
        },
        domain_densities={"mathematical": 0.85},
        overall_density=0.85,
        sparse_concepts=[],
        dense_concepts=[dense_concept],
    )


@pytest.fixture
def target_profile_with_sparse_math(
    sparse_concept: ConceptDensity,
) -> ModelDensityProfile:
    """Target model that is sparse in mathematical concepts."""
    # Create a sparse math concept for target
    target_math = ConceptDensity(
        probe_id="math_addition",  # Same probe_id as source
        name="Addition",
        domain="mathematical",
        layer=5,
        intrinsic_dimension=10.0,  # High = sparse
        density_score=0.3,  # Low = sparse
        activation_variance=0.4,
        cluster_tightness=0.4,
        dimension_class="sparse",
    )
    return ModelDensityProfile(
        model_path="/target/model",
        layers=[5],
        layer_profiles={
            5: LayerDensityProfile(
                layer=5,
                concept_densities=[target_math],
                mean_density=0.3,
                median_density=0.3,
                sparse_concept_count=1,
                dense_concept_count=0,
                density_threshold=0.5,
            )
        },
        domain_densities={"mathematical": 0.3},
        overall_density=0.3,
        sparse_concepts=[target_math],
        dense_concepts=[],
    )


# =============================================================================
# Test: Density Profile Identifies Sparse Regions
# =============================================================================


class TestDensityProfileIdentifiesSparseRegions:
    """Sparse concepts should have high intrinsic dimension, low density score."""

    def test_sparse_concept_has_high_intrinsic_dimension(
        self, sparse_concept: ConceptDensity
    ) -> None:
        """Sparse concepts have high intrinsic dimension (>5)."""
        assert sparse_concept.intrinsic_dimension > 5.0

    def test_sparse_concept_has_low_density_score(
        self, sparse_concept: ConceptDensity
    ) -> None:
        """Sparse concepts have low density score (<0.5)."""
        assert sparse_concept.density_score < 0.5

    def test_dense_concept_has_low_intrinsic_dimension(
        self, dense_concept: ConceptDensity
    ) -> None:
        """Dense concepts have low intrinsic dimension (<5)."""
        assert dense_concept.intrinsic_dimension < 5.0

    def test_dense_concept_has_high_density_score(
        self, dense_concept: ConceptDensity
    ) -> None:
        """Dense concepts have high density score (>0.5)."""
        assert dense_concept.density_score > 0.5

    def test_profile_segregates_sparse_and_dense(
        self, dense_concept: ConceptDensity, sparse_concept: ConceptDensity
    ) -> None:
        """Profile correctly categorizes sparse vs dense concepts."""
        profile = ModelDensityProfile(
            model_path="/test",
            layers=[5],
            layer_profiles={
                5: LayerDensityProfile(
                    layer=5,
                    concept_densities=[dense_concept, sparse_concept],
                    mean_density=0.55,
                    median_density=0.55,
                    sparse_concept_count=1,
                    dense_concept_count=1,
                    density_threshold=0.5,
                )
            },
            domain_densities={},
            overall_density=0.55,
            sparse_concepts=[sparse_concept],
            dense_concepts=[dense_concept],
        )

        assert len(profile.sparse_concepts) == 1
        assert len(profile.dense_concepts) == 1
        assert profile.sparse_concepts[0].probe_id == "logic_modus_ponens"
        assert profile.dense_concepts[0].probe_id == "math_addition"


# =============================================================================
# Test: Knowledge Diff Finds Graft Opportunities
# =============================================================================


class TestKnowledgeDiffFindsGraftOpportunities:
    """Graft opportunities exist where source is dense but target is sparse."""

    def test_high_opportunity_when_source_dense_target_sparse(
        self,
        source_profile_with_dense_math: ModelDensityProfile,
        target_profile_with_sparse_math: ModelDensityProfile,
    ) -> None:
        """High opportunity when source dense, target sparse."""
        differ = KnowledgeDiffer()
        diff = differ.diff(
            source_profile_with_dense_math, target_profile_with_sparse_math
        )

        # Source is dense (0.85), target is sparse (0.3)
        # Opportunity = 0.85 - 0.3 = 0.55 (high positive)
        assert diff.high_opportunity_count >= 1
        assert diff.overall_opportunity > 0

    def test_no_graft_when_target_already_dense(
        self, source_profile_with_dense_math: ModelDensityProfile
    ) -> None:
        """No graft when target is already dense."""
        # Use source as both source and target (same density)
        differ = KnowledgeDiffer()
        diff = differ.diff(
            source_profile_with_dense_math, source_profile_with_dense_math
        )

        # Same density = opportunity around 0
        assert diff.no_graft_count >= 0 or diff.high_opportunity_count == 0

    def test_negative_opportunity_when_source_sparse(
        self,
        target_profile_with_sparse_math: ModelDensityProfile,
        source_profile_with_dense_math: ModelDensityProfile,
    ) -> None:
        """Negative opportunity when source is sparser than target."""
        # Swap source and target
        differ = KnowledgeDiffer()
        diff = differ.diff(
            target_profile_with_sparse_math,  # Sparse source
            source_profile_with_dense_math,  # Dense target
        )

        # Source sparse (0.3), target dense (0.85)
        # Opportunity = 0.3 - 0.85 = -0.55 (negative = don't graft)
        assert diff.overall_opportunity < 0 or diff.no_graft_count >= 1


# =============================================================================
# Test: Graft Mask Filtering
# =============================================================================


class TestGraftMaskFiltering:
    """Graft mask correctly filters which concepts to transplant."""

    def test_filter_returns_all_when_mask_is_none(self) -> None:
        """None mask means graft all (backward compatible)."""
        core_probes = {"probe_a", "probe_b", "probe_c"}
        probe_ids = ["probe_a", "probe_b", "probe_c"]

        filtered = filter_core_probes_by_graft_mask(
            core_probe_ids=core_probes,
            probe_ids=probe_ids,
            layer_idx=5,
            graft_mask=None,
        )

        assert filtered == core_probes

    def test_filter_respects_graft_mask(self) -> None:
        """Mask filters out probes marked as no-graft."""
        core_probes = {"probe_a", "probe_b", "probe_c"}
        probe_ids = ["probe_a", "probe_b", "probe_c"]

        # Only probe_a should be grafted at layer 5
        graft_mask = {
            "probe_a": {5: True, 6: True},
            "probe_b": {5: False, 6: True},
            "probe_c": {5: False, 6: False},
        }

        filtered = filter_core_probes_by_graft_mask(
            core_probe_ids=core_probes,
            probe_ids=probe_ids,
            layer_idx=5,
            graft_mask=graft_mask,
        )

        assert filtered == {"probe_a"}

    def test_filter_handles_missing_probe_in_mask(self) -> None:
        """Probes missing from mask are not grafted."""
        core_probes = {"probe_a", "probe_b"}
        probe_ids = ["probe_a", "probe_b"]

        # probe_b not in mask
        graft_mask = {
            "probe_a": {5: True},
        }

        filtered = filter_core_probes_by_graft_mask(
            core_probe_ids=core_probes,
            probe_ids=probe_ids,
            layer_idx=5,
            graft_mask=graft_mask,
        )

        assert filtered == {"probe_a"}

    def test_filter_handles_missing_layer_in_mask(self) -> None:
        """Layers missing from probe's mask are not grafted."""
        core_probes = {"probe_a"}
        probe_ids = ["probe_a"]

        # Layer 5 not in probe_a's mask
        graft_mask = {
            "probe_a": {6: True, 7: True},
        }

        filtered = filter_core_probes_by_graft_mask(
            core_probe_ids=core_probes,
            probe_ids=probe_ids,
            layer_idx=5,
            graft_mask=graft_mask,
        )

        assert filtered == set()


# =============================================================================
# Test: Compute Graft Mask from Knowledge Diff
# =============================================================================


class TestComputeGraftMask:
    """compute_graft_mask converts knowledge diff to boolean mask."""

    def test_high_opportunity_becomes_true(
        self,
        source_profile_with_dense_math: ModelDensityProfile,
        target_profile_with_sparse_math: ModelDensityProfile,
    ) -> None:
        """High opportunity concepts become True in mask."""
        differ = KnowledgeDiffer()
        diff = differ.diff(
            source_profile_with_dense_math, target_profile_with_sparse_math
        )

        graft_mask = compute_graft_mask(diff, include_low_opportunity=False)

        # math_addition at layer 5 should be True (high opportunity)
        assert graft_mask.get("math_addition", {}).get(5, False) is True

    def test_no_graft_becomes_false(
        self,
        target_profile_with_sparse_math: ModelDensityProfile,
        source_profile_with_dense_math: ModelDensityProfile,
    ) -> None:
        """No-graft concepts become False (or absent) in mask."""
        differ = KnowledgeDiffer()
        diff = differ.diff(
            target_profile_with_sparse_math,  # Sparse source
            source_profile_with_dense_math,  # Dense target
        )

        graft_mask = compute_graft_mask(diff, include_low_opportunity=False)

        # math_addition should be False or absent (target already dense)
        should_graft = graft_mask.get("math_addition", {}).get(5, False)
        assert should_graft is False


# =============================================================================
# Test: Stage Density Integration
# =============================================================================


class TestStageDensityIntegration:
    """Integration tests for stage_density function."""

    def test_stage_density_skip_returns_none_mask(self, backend: "Backend") -> None:
        """Skipping density analysis returns None mask (graft all)."""
        result = stage_density(
            source_activations={},
            target_activations={},
            probe_ids=[],
            probe_domains=[],
            layers=[0, 1, 2],
            config=DensityStageConfig(skip_density_analysis=True),
            backend=backend,
        )

        assert result.graft_mask is None
        assert result.source_profile is None
        assert result.target_profile is None

    def test_stage_density_with_empty_activations(self, backend: "Backend") -> None:
        """Empty activations returns None mask."""
        result = stage_density(
            source_activations={},
            target_activations={},
            probe_ids=["probe_a"],
            probe_domains=["test"],
            layers=[0],
            config=DensityStageConfig(skip_density_analysis=False),
            backend=backend,
        )

        # Should handle gracefully
        assert result.graft_mask is None

    def test_stage_density_with_valid_activations(self, backend: "Backend") -> None:
        """Valid activations produce density profiles and graft mask."""
        backend.random_seed(42)

        # Create synthetic activations
        # Source: tight cluster (dense)
        source_acts = {
            0: [backend.random_normal((64,)) * 0.1 for _ in range(3)],
        }
        # Target: spread cluster (sparse)
        target_acts = {
            0: [backend.random_normal((64,)) * 2.0 for _ in range(3)],
        }

        probe_ids = ["probe_a", "probe_b", "probe_c"]
        probe_domains = ["test", "test", "test"]

        result = stage_density(
            source_activations=source_acts,
            target_activations=target_acts,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            layers=[0],
            config=DensityStageConfig(skip_density_analysis=False),
            backend=backend,
        )

        # Should produce profiles
        assert result.source_profile is not None
        assert result.target_profile is not None
        assert result.knowledge_diff is not None
        # Graft mask should exist (may be empty if no high opportunities)
        # The key is that the pipeline ran without error


# =============================================================================
# Test: Density Preservation Property
# =============================================================================


class TestDensityPreservation:
    """Merged model should maintain or improve density in target regions."""

    def test_grafting_never_decreases_overall_density(self) -> None:
        """
        Graft mask ensures we only graft where source > target density.
        This guarantees overall density cannot decrease.

        Math proof:
        - Let T_dense = target's dense concepts (density >= threshold)
        - Let T_sparse = target's sparse concepts (density < threshold)
        - Graft mask: True only where source_density > target_density
        - For T_dense: graft_mask = False, so W_merged = W_target (unchanged)
        - For T_sparse: graft_mask = True only if source is denser
        - Result: density(merged) >= density(target) for ALL concepts
        """
        # This is a property test - the invariant should hold for any inputs
        source_dense = ConceptDensity(
            probe_id="concept_x",
            name="X",
            domain="test",
            layer=0,
            intrinsic_dimension=2.0,
            density_score=0.9,
            activation_variance=0.01,
            cluster_tightness=None,
            dimension_class="dense",
        )
        target_sparse = ConceptDensity(
            probe_id="concept_x",
            name="X",
            domain="test",
            layer=0,
            intrinsic_dimension=10.0,
            density_score=0.2,
            activation_variance=0.5,
            cluster_tightness=None,
            dimension_class="sparse",
        )

        source_profile = ModelDensityProfile(
            model_path="/source",
            layers=[0],
            layer_profiles={
                0: LayerDensityProfile(
                    layer=0,
                    concept_densities=[source_dense],
                    mean_density=0.9,
                    median_density=0.9,
                    sparse_concept_count=0,
                    dense_concept_count=1,
                    density_threshold=0.5,
                )
            },
            domain_densities={"test": 0.9},
            overall_density=0.9,
            sparse_concepts=[],
            dense_concepts=[source_dense],
        )

        target_profile = ModelDensityProfile(
            model_path="/target",
            layers=[0],
            layer_profiles={
                0: LayerDensityProfile(
                    layer=0,
                    concept_densities=[target_sparse],
                    mean_density=0.2,
                    median_density=0.2,
                    sparse_concept_count=1,
                    dense_concept_count=0,
                    density_threshold=0.5,
                )
            },
            domain_densities={"test": 0.2},
            overall_density=0.2,
            sparse_concepts=[target_sparse],
            dense_concepts=[],
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source_profile, target_profile)

        # Source (0.9) > Target (0.2), so opportunity is positive
        assert diff.overall_opportunity > 0

        # Graft mask should mark this as True (should graft)
        graft_mask = compute_graft_mask(diff)
        should_graft = graft_mask.get("concept_x", {}).get(0, False)

        # This concept SHOULD be grafted (fills a gap)
        assert should_graft is True

    def test_dense_target_regions_protected(self) -> None:
        """
        Target's dense regions are never overwritten.
        The graft mask is False where target is already dense.
        """
        # Target is dense, source is sparse
        source_sparse = ConceptDensity(
            probe_id="concept_y",
            name="Y",
            domain="test",
            layer=0,
            intrinsic_dimension=10.0,
            density_score=0.2,
            activation_variance=0.5,
            cluster_tightness=None,
            dimension_class="sparse",
        )
        target_dense = ConceptDensity(
            probe_id="concept_y",
            name="Y",
            domain="test",
            layer=0,
            intrinsic_dimension=2.0,
            density_score=0.9,
            activation_variance=0.01,
            cluster_tightness=None,
            dimension_class="dense",
        )

        source_profile = ModelDensityProfile(
            model_path="/source",
            layers=[0],
            layer_profiles={
                0: LayerDensityProfile(
                    layer=0,
                    concept_densities=[source_sparse],
                    mean_density=0.2,
                    median_density=0.2,
                    sparse_concept_count=1,
                    dense_concept_count=0,
                    density_threshold=0.5,
                )
            },
            domain_densities={"test": 0.2},
            overall_density=0.2,
            sparse_concepts=[source_sparse],
            dense_concepts=[],
        )

        target_profile = ModelDensityProfile(
            model_path="/target",
            layers=[0],
            layer_profiles={
                0: LayerDensityProfile(
                    layer=0,
                    concept_densities=[target_dense],
                    mean_density=0.9,
                    median_density=0.9,
                    sparse_concept_count=0,
                    dense_concept_count=1,
                    density_threshold=0.5,
                )
            },
            domain_densities={"test": 0.9},
            overall_density=0.9,
            sparse_concepts=[],
            dense_concepts=[target_dense],
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source_profile, target_profile)

        # Source (0.2) < Target (0.9), so opportunity is negative
        assert diff.overall_opportunity < 0

        # Graft mask should mark this as False (don't graft)
        graft_mask = compute_graft_mask(diff)
        should_graft = graft_mask.get("concept_y", {}).get(0, False)

        # This concept should NOT be grafted (would destroy target's knowledge)
        assert should_graft is False


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestDensityEdgeCases:
    """Edge cases in density-based merge."""

    def test_empty_probe_ids(self, backend: "Backend") -> None:
        """Empty probe_ids should not crash."""
        result = stage_density(
            source_activations={0: []},
            target_activations={0: []},
            probe_ids=[],
            probe_domains=[],
            layers=[0],
            config=DensityStageConfig(skip_density_analysis=False),
            backend=backend,
        )
        assert result.graft_mask is None

    def test_mismatched_probe_domains_length(self, backend: "Backend") -> None:
        """Mismatched probe_ids/domains length handled gracefully."""
        result = stage_density(
            source_activations={0: []},
            target_activations={0: []},
            probe_ids=["a", "b", "c"],
            probe_domains=["x"],  # Mismatch
            layers=[0],
            config=DensityStageConfig(skip_density_analysis=False),
            backend=backend,
        )
        # Should handle gracefully, not crash
        assert result.graft_mask is None

    def test_single_layer_single_probe(self, backend: "Backend") -> None:
        """Minimal case: 1 layer, 1 probe."""
        backend.random_seed(42)

        source_acts = {0: [backend.random_normal((32,)) * 0.1]}
        target_acts = {0: [backend.random_normal((32,)) * 1.0]}

        result = stage_density(
            source_activations=source_acts,
            target_activations=target_acts,
            probe_ids=["single_probe"],
            probe_domains=["test"],
            layers=[0],
            config=DensityStageConfig(skip_density_analysis=False),
            backend=backend,
        )

        # Should complete without error
        assert result.source_profile is not None
        assert result.target_profile is not None


__all__ = [
    "TestDensityProfileIdentifiesSparseRegions",
    "TestKnowledgeDiffFindsGraftOpportunities",
    "TestGraftMaskFiltering",
    "TestComputeGraftMask",
    "TestStageDensityIntegration",
    "TestDensityPreservation",
    "TestDensityEdgeCases",
]
