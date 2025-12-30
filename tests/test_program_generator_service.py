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

"""Tests for ProgramGeneratorService.

Tests the auto-generation of TransplantProgram YAML from density profiles,
including profile loading, opportunity calculation, donor selection, and
program building.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from modelcypher.core.use_cases.program_generator_service import (
    DensityProfileSummary,
    DomainSummary,
    DonorCandidate,
    ProgramGeneratorConfig,
    ProgramGeneratorService,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_profile(
    model_path: str,
    domain_data: dict[str, dict],
    total_layers: int = 24,
) -> dict:
    """Create a mock density profile JSON structure.

    Args:
        model_path: Model path to include in profile.
        domain_data: Dict of domain -> {density, strongest, weakest}.
        total_layers: Total layer count.

    Returns:
        Profile dict matching mc.geometry.research.full_profile.v1 schema.
    """
    domain_summaries = []
    for domain, data in domain_data.items():
        domain_summaries.append({
            "domain": domain,
            "overallMeanDensity": data.get("density", 0.5),
            "conceptCount": data.get("count", 10),
            "strongestLayers": data.get("strongest", [0, 1, 2, 3, 4]),
            "weakestLayers": data.get("weakest", [19, 20, 21, 22, 23]),
        })

    return {
        "_schema": "mc.geometry.research.full_profile.v1",
        "modelPath": model_path,
        "totalLayers": total_layers,
        "domainSummaries": domain_summaries,
    }


@pytest.fixture
def temp_profiles_dir(tmp_path: Path) -> Path:
    """Create temp directory with mock profiles."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    return profiles_dir


@pytest.fixture
def target_profile(temp_profiles_dir: Path) -> Path:
    """Create mock target profile with gaps in math and safety."""
    profile = create_mock_profile(
        model_path="/models/target-model",
        domain_data={
            "mathematical": {"density": 0.3, "strongest": [0, 1, 2], "weakest": [20, 21, 22, 23]},
            "safety_ethics": {"density": 0.4, "strongest": [5, 6, 7], "weakest": [18, 19, 20]},
            "computational": {"density": 0.7, "strongest": [10, 11, 12], "weakest": [1, 2, 3]},
            "linguistic": {"density": 0.6, "strongest": [8, 9, 10], "weakest": [15, 16, 17]},
        },
    )
    path = temp_profiles_dir / "target.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.fixture
def math_donor_profile(temp_profiles_dir: Path) -> Path:
    """Create mock math-specialized donor profile."""
    profile = create_mock_profile(
        model_path="/models/mathstral",
        domain_data={
            "mathematical": {"density": 0.9, "strongest": [20, 21, 22, 23], "weakest": [0, 1, 2]},
            "computational": {"density": 0.6, "strongest": [15, 16], "weakest": [5, 6]},
        },
    )
    path = temp_profiles_dir / "math_donor.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.fixture
def safety_donor_profile(temp_profiles_dir: Path) -> Path:
    """Create mock safety-specialized donor profile."""
    profile = create_mock_profile(
        model_path="/models/granite-guardian",
        domain_data={
            "safety_ethics": {"density": 0.85, "strongest": [18, 19, 20], "weakest": [0, 1, 2]},
            "linguistic": {"density": 0.5, "strongest": [10, 11], "weakest": [20, 21]},
        },
    )
    path = temp_profiles_dir / "safety_donor.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.fixture
def weak_donor_profile(temp_profiles_dir: Path) -> Path:
    """Create donor profile weaker than target (should not be selected)."""
    profile = create_mock_profile(
        model_path="/models/weak-model",
        domain_data={
            "mathematical": {"density": 0.2, "strongest": [0, 1], "weakest": [22, 23]},
            "safety_ethics": {"density": 0.3, "strongest": [5, 6], "weakest": [19, 20]},
        },
    )
    path = temp_profiles_dir / "weak_donor.json"
    path.write_text(json.dumps(profile))
    return path


# =============================================================================
# Profile Loading Tests
# =============================================================================


class TestProfileLoading:
    """Tests for _load_profile method."""

    def test_load_valid_profile(self, target_profile: Path):
        """Load valid profile with correct schema."""
        service = ProgramGeneratorService()
        summary = service._load_profile(target_profile)

        assert summary.model_path == "/models/target-model"
        assert summary.total_layers == 24
        assert "mathematical" in summary.domain_summaries
        assert "safety_ethics" in summary.domain_summaries

    def test_load_profile_domain_details(self, target_profile: Path):
        """Verify domain summary details are parsed correctly."""
        service = ProgramGeneratorService()
        summary = service._load_profile(target_profile)

        math_domain = summary.domain_summaries["mathematical"]
        assert math_domain.overall_mean_density == 0.3
        assert math_domain.strongest_layers == (0, 1, 2)
        assert math_domain.weakest_layers == (20, 21, 22, 23)

    def test_load_profile_file_not_found(self):
        """Raise FileNotFoundError for missing profile."""
        service = ProgramGeneratorService()
        with pytest.raises(FileNotFoundError, match="Profile not found"):
            service._load_profile("/nonexistent/profile.json")

    def test_load_profile_invalid_schema(self, tmp_path: Path):
        """Raise ValueError for invalid schema."""
        invalid_profile = tmp_path / "invalid.json"
        invalid_profile.write_text(json.dumps({"_schema": "wrong.schema.v1"}))

        service = ProgramGeneratorService()
        with pytest.raises(ValueError, match="Invalid profile schema"):
            service._load_profile(invalid_profile)


# =============================================================================
# Opportunity Calculation Tests
# =============================================================================


class TestOpportunityCalculation:
    """Tests for _compute_candidates method."""

    def test_positive_opportunity_creates_candidate(
        self, target_profile: Path, math_donor_profile: Path
    ):
        """Donor with higher density creates positive opportunity."""
        service = ProgramGeneratorService()
        target = service._load_profile(target_profile)
        donor = service._load_profile(math_donor_profile)
        config = ProgramGeneratorConfig()

        candidates = service._compute_candidates(target, [donor], config)

        # Math donor (0.9) vs target (0.3) = 0.6 opportunity
        assert "mathematical" in candidates
        assert len(candidates["mathematical"]) == 1
        assert candidates["mathematical"][0].mean_opportunity == pytest.approx(0.6)

    def test_negative_opportunity_excluded(
        self, target_profile: Path, weak_donor_profile: Path
    ):
        """Donor weaker than target is not included as candidate."""
        service = ProgramGeneratorService()
        target = service._load_profile(target_profile)
        donor = service._load_profile(weak_donor_profile)
        config = ProgramGeneratorConfig()

        candidates = service._compute_candidates(target, [donor], config)

        # Weak donor is weaker in all domains, should have no candidates
        assert len(candidates) == 0

    def test_layer_intersection_selection(
        self, target_profile: Path, math_donor_profile: Path
    ):
        """Layer selection uses intersection of target weak and donor strong."""
        service = ProgramGeneratorService()
        target = service._load_profile(target_profile)
        donor = service._load_profile(math_donor_profile)
        config = ProgramGeneratorConfig(layer_selection_mode="intersection")

        candidates = service._compute_candidates(target, [donor], config)

        # Target weak: [20, 21, 22, 23], Donor strong: [20, 21, 22, 23]
        # Intersection: [20, 21, 22, 23]
        math_candidate = candidates["mathematical"][0]
        assert set(math_candidate.recommended_layers) == {20, 21, 22, 23}

    def test_multiple_donors_sorted_by_opportunity(
        self, target_profile: Path, math_donor_profile: Path, temp_profiles_dir: Path
    ):
        """Multiple candidates for same domain sorted by opportunity."""
        # Create a second math donor with lower opportunity
        second_donor = create_mock_profile(
            model_path="/models/math-lite",
            domain_data={
                "mathematical": {"density": 0.5, "strongest": [20, 21], "weakest": [0, 1]},
            },
        )
        second_path = temp_profiles_dir / "math_lite.json"
        second_path.write_text(json.dumps(second_donor))

        service = ProgramGeneratorService()
        target = service._load_profile(target_profile)
        donors = [
            service._load_profile(math_donor_profile),
            service._load_profile(second_path),
        ]
        config = ProgramGeneratorConfig()

        candidates = service._compute_candidates(target, donors, config)

        # Should have 2 candidates, sorted by opportunity (0.6 > 0.2)
        assert len(candidates["mathematical"]) == 2
        assert candidates["mathematical"][0].mean_opportunity > candidates["mathematical"][1].mean_opportunity


# =============================================================================
# Donor Selection Tests
# =============================================================================


class TestDonorSelection:
    """Tests for _select_donors method."""

    def test_selects_top_donor_per_domain(
        self, target_profile: Path, math_donor_profile: Path, safety_donor_profile: Path
    ):
        """Select best donor for each domain."""
        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[math_donor_profile, safety_donor_profile],
            # Use threshold of 0 to accept all positive opportunities
            config=ProgramGeneratorConfig(min_opportunity_threshold=0.0),
        )

        # Should have selected math donor for math, safety donor for safety
        domains_covered = {sel.domain for sel in result.donor_selections}
        assert "mathematical" in domains_covered
        assert "safety_ethics" in domains_covered

    def test_threshold_from_data(
        self, target_profile: Path, math_donor_profile: Path, safety_donor_profile: Path
    ):
        """Threshold derived from median of opportunities when not configured."""
        service = ProgramGeneratorService()
        target = service._load_profile(target_profile)
        donors = [
            service._load_profile(math_donor_profile),
            service._load_profile(safety_donor_profile),
        ]
        config = ProgramGeneratorConfig()

        candidates = service._compute_candidates(target, donors, config)
        selections = service._select_donors(target, candidates, config)

        # Should select donors above median threshold
        assert len(selections) >= 1

    def test_explicit_threshold(
        self, target_profile: Path, math_donor_profile: Path, safety_donor_profile: Path
    ):
        """Use explicit threshold when configured."""
        service = ProgramGeneratorService()
        # Very high threshold should exclude all
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[math_donor_profile, safety_donor_profile],
            config=ProgramGeneratorConfig(min_opportunity_threshold=10.0),
        )

        # No donors should meet threshold of 10.0
        assert len(result.donor_selections) == 0

    def test_max_donors_per_domain(
        self, target_profile: Path, temp_profiles_dir: Path
    ):
        """Respect max_donors_per_domain limit."""
        # Create 3 math donors
        for i, density in enumerate([0.9, 0.8, 0.7]):
            profile = create_mock_profile(
                model_path=f"/models/math-{i}",
                domain_data={"mathematical": {"density": density}},
            )
            path = temp_profiles_dir / f"math_{i}.json"
            path.write_text(json.dumps(profile))

        donor_paths = list(temp_profiles_dir.glob("math_*.json"))

        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=donor_paths,
            config=ProgramGeneratorConfig(
                max_donors_per_domain=2,
                min_opportunity_threshold=0.0,  # Accept all positive
            ),
        )

        # Should only select 2 donors for math domain
        math_selections = [s for s in result.donor_selections if s.domain == "mathematical"]
        assert len(math_selections) <= 2


# =============================================================================
# Program Building Tests
# =============================================================================


class TestProgramBuilding:
    """Tests for program generation output."""

    def test_generates_valid_program(
        self, target_profile: Path, math_donor_profile: Path
    ):
        """Generate valid TransplantProgram."""
        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[math_donor_profile],
        )

        program = result.program
        assert program.name is not None
        assert len(program.bases) == 1
        assert program.bases[0].source == "/models/target-model"

    def test_aggregates_domains_per_donor(
        self, target_profile: Path, temp_profiles_dir: Path
    ):
        """Single donor covering multiple domains aggregated correctly."""
        # Create donor good at both math and computational
        multi_donor = create_mock_profile(
            model_path="/models/multi-expert",
            domain_data={
                "mathematical": {"density": 0.9, "strongest": [20, 21, 22, 23]},
                "linguistic": {"density": 0.8, "strongest": [15, 16, 17]},
            },
        )
        path = temp_profiles_dir / "multi_donor.json"
        path.write_text(json.dumps(multi_donor))

        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[path],
            config=ProgramGeneratorConfig(min_opportunity_threshold=0.0),
        )

        # Should have one donor spec covering both domains
        assert len(result.program.donors) == 1
        donor_spec = result.program.donors[0]
        assert "mathematical" in donor_spec.domains
        assert "linguistic" in donor_spec.domains

    def test_priority_by_opportunity(
        self, target_profile: Path, math_donor_profile: Path, safety_donor_profile: Path
    ):
        """Higher opportunity donors get higher priority."""
        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[math_donor_profile, safety_donor_profile],
            config=ProgramGeneratorConfig(priority_by_opportunity=True),
        )

        # Donors should have different priorities based on opportunity
        priorities = [d.priority for d in result.program.donors]
        assert len(set(priorities)) == len(priorities)  # All unique

    def test_custom_program_name(
        self, target_profile: Path, math_donor_profile: Path
    ):
        """Use custom program name when provided."""
        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[math_donor_profile],
            program_name="my-custom-uber-model",
        )

        assert result.program.name == "my-custom-uber-model"

    def test_skipped_domains_tracked(
        self, target_profile: Path, weak_donor_profile: Path
    ):
        """Track domains with no viable donor."""
        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[weak_donor_profile],
        )

        # All domains should be skipped since weak donor is weaker everywhere
        assert len(result.skipped_domains) > 0


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end generation tests."""

    def test_generate_from_directory(
        self,
        target_profile: Path,
        math_donor_profile: Path,
        safety_donor_profile: Path,
        temp_profiles_dir: Path,
    ):
        """Generate from directory of donor profiles."""
        service = ProgramGeneratorService()
        result = service.generate_from_directory(
            target_profile=target_profile,
            donor_dir=temp_profiles_dir,
        )

        # Should have found and processed donors
        assert len(result.program.donors) >= 1

    def test_excludes_target_from_donors(
        self, target_profile: Path, math_donor_profile: Path, temp_profiles_dir: Path
    ):
        """Target profile excluded when in same directory as donors."""
        # Copy target to donors directory
        target_copy = temp_profiles_dir / "target_copy.json"
        target_copy.write_text(target_profile.read_text())

        service = ProgramGeneratorService()
        result = service.generate_from_directory(
            target_profile=target_copy,
            donor_dir=temp_profiles_dir,
        )

        # Target should not appear as a donor
        donor_paths = {d.source for d in result.program.donors}
        assert "/models/target-model" not in donor_paths

    def test_to_dict_serialization(
        self, target_profile: Path, math_donor_profile: Path
    ):
        """Result serializes to dict correctly."""
        service = ProgramGeneratorService()
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=[math_donor_profile],
        )

        data = result.to_dict()
        assert data["_schema"] == "mc.result.program_generator.v1"
        assert "program" in data
        assert "donorSelections" in data

    def test_empty_donors_raises(self, target_profile: Path):
        """Raise ValueError when no donors provided."""
        service = ProgramGeneratorService()
        with pytest.raises(ValueError, match="At least one donor"):
            service.generate(
                target_profile=target_profile,
                donor_profiles=[],
            )

    def test_empty_directory_raises(self, target_profile: Path, tmp_path: Path):
        """Raise ValueError when donor directory is empty."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        service = ProgramGeneratorService()
        with pytest.raises(ValueError, match="No donor profiles found"):
            service.generate_from_directory(
                target_profile=target_profile,
                donor_dir=empty_dir,
            )


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestDataStructures:
    """Tests for data structure immutability and behavior."""

    def test_domain_summary_frozen(self):
        """DomainSummary is immutable."""
        ds = DomainSummary(
            domain="test",
            overall_mean_density=0.5,
            concept_count=10,
            strongest_layers=(0, 1, 2),
            weakest_layers=(20, 21, 22),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            ds.domain = "changed"

    def test_donor_candidate_frozen(self):
        """DonorCandidate is immutable."""
        dc = DonorCandidate(
            profile_path="/path",
            model_path="/model",
            domain="test",
            mean_opportunity=0.5,
            recommended_layers=(0, 1, 2),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            dc.domain = "changed"

    def test_density_profile_summary_frozen(self):
        """DensityProfileSummary is immutable."""
        dps = DensityProfileSummary(
            profile_path="/path",
            model_path="/model",
            total_layers=24,
            domain_summaries={},
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            dps.total_layers = 48
