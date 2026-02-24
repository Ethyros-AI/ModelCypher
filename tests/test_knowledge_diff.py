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

"""Tests for knowledge diff computation."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    LayerDensityProfile,
    ModelDensityProfile,
)
from modelcypher.core.domain.geometry.knowledge_diff import (
    DomainDiff,
    GraftOpportunity,
    KnowledgeDiff,
    KnowledgeDiffer,
    LayerDiff,
    _aggregate_opportunities,
    compute_graft_mask,
)


def _make_concept(
    probe_id: str,
    layer: int,
    density: float,
    domain: str = "test",
) -> ConceptDensity:
    """Create a ConceptDensity for testing."""
    return ConceptDensity(
        probe_id=probe_id,
        name=f"concept_{probe_id}",
        domain=domain,
        layer=layer,
        intrinsic_dimension=1.0 / density if density > 0 else 100.0,
        density_score=density,
    )


def _make_layer_profile(
    layer: int,
    concepts: list[ConceptDensity],
) -> LayerDensityProfile:
    """Create a LayerDensityProfile for testing."""
    densities = [c.density_score for c in concepts]
    mean = sum(densities) / len(densities) if densities else 0.0
    return LayerDensityProfile(
        layer=layer,
        concept_densities=concepts,
        mean_density=mean,
        median_density=mean,
    )


def _make_model_profile(
    model_path: str,
    layer_profiles: dict[int, LayerDensityProfile],
) -> ModelDensityProfile:
    """Create a ModelDensityProfile for testing."""
    return ModelDensityProfile(
        model_path=model_path,
        layers=list(layer_profiles.keys()),
        layer_profiles=layer_profiles,
        domain_densities={},
        overall_density=0.5,
    )


class TestGraftOpportunity:
    """Tests for GraftOpportunity dataclass."""

    def test_frozen_dataclass(self):
        """GraftOpportunity should be immutable."""
        opp = GraftOpportunity(
            probe_id="probe1",
            name="concept",
            domain="test",
            layer=0,
            source_density=0.8,
            target_density=0.2,
            opportunity_score=0.6,
        )
        with pytest.raises(AttributeError):
            opp.opportunity_score = 0.0

    def test_positive_opportunity(self):
        """Positive score means source has knowledge target lacks."""
        opp = GraftOpportunity(
            probe_id="probe1",
            name="concept",
            domain="test",
            layer=0,
            source_density=0.9,
            target_density=0.1,
            opportunity_score=0.8,
        )
        assert opp.opportunity_score == 0.8

    def test_negative_opportunity(self):
        """Negative score means target already has the knowledge."""
        opp = GraftOpportunity(
            probe_id="probe1",
            name="concept",
            domain="test",
            layer=0,
            source_density=0.2,
            target_density=0.8,
            opportunity_score=-0.6,
        )
        assert opp.opportunity_score == -0.6


class TestLayerDiff:
    """Tests for LayerDiff dataclass."""

    def test_frozen_dataclass(self):
        """LayerDiff should be immutable."""
        diff = LayerDiff(
            layer=0,
            opportunities=[],
            mean_opportunity=0.0,
            positive_opportunity_count=0,
            nonpositive_opportunity_count=0,
        )
        with pytest.raises(AttributeError):
            diff.layer = 1


class TestDomainDiff:
    """Tests for DomainDiff dataclass."""

    def test_frozen_dataclass(self):
        """DomainDiff should be immutable."""
        diff = DomainDiff(
            domain="test",
            mean_source_density=0.5,
            mean_target_density=0.5,
            mean_opportunity=0.0,
            concept_count=1,
            positive_opportunity_count=0,
        )
        with pytest.raises(AttributeError):
            diff.domain = "other"


class TestKnowledgeDiff:
    """Tests for KnowledgeDiff dataclass."""

    def test_frozen_dataclass(self):
        """KnowledgeDiff should be immutable."""
        diff = KnowledgeDiff(
            source_path="/source",
            target_path="/target",
            layer_diffs={},
            domain_diffs={},
            overall_source_density=0.5,
            overall_target_density=0.5,
            overall_opportunity=0.0,
            ranked_opportunities=[],
            total_concepts=0,
            positive_opportunity_count=0,
            nonpositive_opportunity_count=0,
        )
        with pytest.raises(AttributeError):
            diff.source_path = "/other"


class TestAggregateOpportunities:
    """Tests for _aggregate_opportunities helper."""

    def test_empty_list(self):
        """Empty list should return zeros."""
        mean_src, mean_tgt, mean_opp, positive = _aggregate_opportunities([])
        assert mean_src == 0.0
        assert mean_tgt == 0.0
        assert mean_opp == 0.0
        assert positive == 0

    def test_single_positive(self):
        """Single positive opportunity."""
        opps = [
            GraftOpportunity(
                probe_id="p1",
                name="c1",
                domain="d1",
                layer=0,
                source_density=0.8,
                target_density=0.2,
                opportunity_score=0.6,
            )
        ]
        mean_src, mean_tgt, mean_opp, positive = _aggregate_opportunities(opps)
        assert mean_src == 0.8
        assert mean_tgt == 0.2
        assert mean_opp == 0.6
        assert positive == 1

    def test_mixed_opportunities(self):
        """Mix of positive and negative opportunities."""
        opps = [
            GraftOpportunity(
                probe_id="p1",
                name="c1",
                domain="d1",
                layer=0,
                source_density=0.8,
                target_density=0.2,
                opportunity_score=0.6,
            ),
            GraftOpportunity(
                probe_id="p2",
                name="c2",
                domain="d1",
                layer=0,
                source_density=0.2,
                target_density=0.8,
                opportunity_score=-0.6,
            ),
        ]
        mean_src, mean_tgt, mean_opp, positive = _aggregate_opportunities(opps)
        assert mean_src == 0.5  # (0.8 + 0.2) / 2
        assert mean_tgt == 0.5  # (0.2 + 0.8) / 2
        assert mean_opp == 0.0  # (0.6 + -0.6) / 2
        assert positive == 1


class TestKnowledgeDiffer:
    """Tests for KnowledgeDiffer class."""

    def test_empty_profiles(self):
        """Empty profiles should produce empty diff."""
        source = _make_model_profile("/source", {})
        target = _make_model_profile("/target", {})

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert diff.total_concepts == 0
        assert diff.positive_opportunity_count == 0
        assert diff.nonpositive_opportunity_count == 0
        assert len(diff.ranked_opportunities) == 0

    def test_no_common_concepts(self):
        """Non-overlapping concepts should produce empty diff."""
        source_concepts = [_make_concept("p1", 0, 0.8)]
        target_concepts = [_make_concept("p2", 0, 0.2)]

        source = _make_model_profile(
            "/source",
            {0: _make_layer_profile(0, source_concepts)},
        )
        target = _make_model_profile(
            "/target",
            {0: _make_layer_profile(0, target_concepts)},
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert diff.total_concepts == 0

    def test_common_concepts_positive_opportunity(self):
        """Common concept with source denser should be positive opportunity."""
        source_concepts = [_make_concept("p1", 0, 0.9)]
        target_concepts = [_make_concept("p1", 0, 0.1)]

        source = _make_model_profile(
            "/source",
            {0: _make_layer_profile(0, source_concepts)},
        )
        target = _make_model_profile(
            "/target",
            {0: _make_layer_profile(0, target_concepts)},
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert diff.total_concepts == 1
        assert diff.positive_opportunity_count == 1
        assert diff.ranked_opportunities[0].opportunity_score == pytest.approx(0.8)

    def test_common_concepts_negative_opportunity(self):
        """Common concept with target denser should be negative opportunity."""
        source_concepts = [_make_concept("p1", 0, 0.1)]
        target_concepts = [_make_concept("p1", 0, 0.9)]

        source = _make_model_profile(
            "/source",
            {0: _make_layer_profile(0, source_concepts)},
        )
        target = _make_model_profile(
            "/target",
            {0: _make_layer_profile(0, target_concepts)},
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert diff.total_concepts == 1
        assert diff.nonpositive_opportunity_count == 1
        assert diff.ranked_opportunities[0].opportunity_score == pytest.approx(-0.8)

    def test_ranked_opportunities_sorted(self):
        """Opportunities should be ranked by score (highest first)."""
        source_concepts = [
            _make_concept("p1", 0, 0.5),
            _make_concept("p2", 0, 0.9),
            _make_concept("p3", 0, 0.1),
        ]
        target_concepts = [
            _make_concept("p1", 0, 0.3),  # +0.2
            _make_concept("p2", 0, 0.1),  # +0.8
            _make_concept("p3", 0, 0.9),  # -0.8
        ]

        source = _make_model_profile(
            "/source",
            {0: _make_layer_profile(0, source_concepts)},
        )
        target = _make_model_profile(
            "/target",
            {0: _make_layer_profile(0, target_concepts)},
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert len(diff.ranked_opportunities) == 3
        assert diff.ranked_opportunities[0].probe_id == "p2"  # +0.8
        assert diff.ranked_opportunities[1].probe_id == "p1"  # +0.2
        assert diff.ranked_opportunities[2].probe_id == "p3"  # -0.8

    def test_layer_diffs_grouped(self):
        """Opportunities should be grouped by layer."""
        source_concepts_l0 = [_make_concept("p1", 0, 0.8)]
        source_concepts_l1 = [_make_concept("p2", 1, 0.7)]
        target_concepts_l0 = [_make_concept("p1", 0, 0.2)]
        target_concepts_l1 = [_make_concept("p2", 1, 0.3)]

        source = _make_model_profile(
            "/source",
            {
                0: _make_layer_profile(0, source_concepts_l0),
                1: _make_layer_profile(1, source_concepts_l1),
            },
        )
        target = _make_model_profile(
            "/target",
            {
                0: _make_layer_profile(0, target_concepts_l0),
                1: _make_layer_profile(1, target_concepts_l1),
            },
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert len(diff.layer_diffs) == 2
        assert 0 in diff.layer_diffs
        assert 1 in diff.layer_diffs
        assert diff.layer_diffs[0].mean_opportunity == pytest.approx(0.6)
        assert diff.layer_diffs[1].mean_opportunity == pytest.approx(0.4)

    def test_domain_diffs_grouped(self):
        """Opportunities should be grouped by domain."""
        source_concepts = [
            _make_concept("p1", 0, 0.8, domain="math"),
            _make_concept("p2", 0, 0.6, domain="code"),
        ]
        target_concepts = [
            _make_concept("p1", 0, 0.2, domain="math"),
            _make_concept("p2", 0, 0.4, domain="code"),
        ]

        source = _make_model_profile(
            "/source",
            {0: _make_layer_profile(0, source_concepts)},
        )
        target = _make_model_profile(
            "/target",
            {0: _make_layer_profile(0, target_concepts)},
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source, target)

        assert len(diff.domain_diffs) == 2
        assert "math" in diff.domain_diffs
        assert "code" in diff.domain_diffs
        assert diff.domain_diffs["math"].mean_opportunity == pytest.approx(0.6)
        assert diff.domain_diffs["code"].mean_opportunity == pytest.approx(0.2)


class TestComputeGraftMask:
    """Tests for compute_graft_mask function."""

    def test_empty_diff(self):
        """Empty diff should produce empty mask."""
        diff = KnowledgeDiff(
            source_path="/source",
            target_path="/target",
            layer_diffs={},
            domain_diffs={},
            overall_source_density=0.0,
            overall_target_density=0.0,
            overall_opportunity=0.0,
            ranked_opportunities=[],
            total_concepts=0,
            positive_opportunity_count=0,
            nonpositive_opportunity_count=0,
        )
        mask = compute_graft_mask(diff)
        assert mask == {}

    def test_positive_opportunity_true(self):
        """Positive opportunity should produce True in mask."""
        opp = GraftOpportunity(
            probe_id="p1",
            name="c1",
            domain="d1",
            layer=0,
            source_density=0.8,
            target_density=0.2,
            opportunity_score=0.6,
        )
        diff = KnowledgeDiff(
            source_path="/source",
            target_path="/target",
            layer_diffs={},
            domain_diffs={},
            overall_source_density=0.8,
            overall_target_density=0.2,
            overall_opportunity=0.6,
            ranked_opportunities=[opp],
            total_concepts=1,
            positive_opportunity_count=1,
            nonpositive_opportunity_count=0,
        )
        mask = compute_graft_mask(diff)
        assert mask["p1"][0] is True

    def test_negative_opportunity_false(self):
        """Negative opportunity should produce False in mask."""
        opp = GraftOpportunity(
            probe_id="p1",
            name="c1",
            domain="d1",
            layer=0,
            source_density=0.2,
            target_density=0.8,
            opportunity_score=-0.6,
        )
        diff = KnowledgeDiff(
            source_path="/source",
            target_path="/target",
            layer_diffs={},
            domain_diffs={},
            overall_source_density=0.2,
            overall_target_density=0.8,
            overall_opportunity=-0.6,
            ranked_opportunities=[opp],
            total_concepts=1,
            positive_opportunity_count=0,
            nonpositive_opportunity_count=1,
        )
        mask = compute_graft_mask(diff)
        assert mask["p1"][0] is False

    def test_multiple_layers_same_probe(self):
        """Same probe at different layers should have separate mask entries."""
        opps = [
            GraftOpportunity(
                probe_id="p1",
                name="c1",
                domain="d1",
                layer=0,
                source_density=0.8,
                target_density=0.2,
                opportunity_score=0.6,
            ),
            GraftOpportunity(
                probe_id="p1",
                name="c1",
                domain="d1",
                layer=1,
                source_density=0.2,
                target_density=0.8,
                opportunity_score=-0.6,
            ),
        ]
        diff = KnowledgeDiff(
            source_path="/source",
            target_path="/target",
            layer_diffs={},
            domain_diffs={},
            overall_source_density=0.5,
            overall_target_density=0.5,
            overall_opportunity=0.0,
            ranked_opportunities=opps,
            total_concepts=2,
            positive_opportunity_count=1,
            nonpositive_opportunity_count=1,
        )
        mask = compute_graft_mask(diff)
        assert mask["p1"][0] is True
        assert mask["p1"][1] is False
