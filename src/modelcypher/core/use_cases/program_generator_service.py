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

"""Program Generator Service.

Auto-generates TransplantProgram YAML from density profiles.
Analyzes target and donor profiles to identify optimal donors per domain,
then generates executable transplant programs.

The generator:
1. Loads target model's density profile
2. Loads all donor profiles
3. Computes per-domain opportunity scores (donor_density - target_density)
4. Selects best donor per domain where opportunity is positive
5. Generates layer assignments (target weak + donor strong intersection)

Example usage:
    service = ProgramGeneratorService()
    result = service.generate(
        target_profile="/profiles/target.json",
        donor_profiles=["/profiles/math.json", "/profiles/code.json"],
    )
    result.program.to_yaml("/programs/uber-model.yaml")

CLI usage:
    mc program generate ./target.json --donor-dir ./experts/ -o uber.yaml
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelcypher.core.use_cases.multi_donor_merge import (
    BaseModelSpec,
    DonorSpec,
    EvaluationConfig,
    TransplantProgram,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class DomainSummary:
    """Per-domain summary from density profile."""

    domain: str
    overall_mean_density: float
    concept_count: int
    strongest_layers: tuple[int, ...]  # Top 5 layers by density
    weakest_layers: tuple[int, ...]  # Bottom 5 layers by density


@dataclass(frozen=True)
class DensityProfileSummary:
    """Parsed density profile for comparison."""

    profile_path: str
    model_path: str
    total_layers: int
    domain_summaries: dict[str, DomainSummary]


@dataclass(frozen=True)
class DonorCandidate:
    """Candidate donor for a domain."""

    profile_path: str
    model_path: str
    domain: str
    mean_opportunity: float  # donor_density - target_density (higher = better)
    recommended_layers: tuple[int, ...]


@dataclass
class ProgramGeneratorConfig:
    """Configuration for program generation."""

    # Minimum opportunity score to include a donor (derived from data if None)
    min_opportunity_threshold: float | None = None

    # Maximum donors per domain
    max_donors_per_domain: int = 1

    # Layer selection
    max_layers_per_donor: int = 10
    layer_selection_mode: str = "intersection"  # "intersection" or "all"

    # Priority assignment (higher = processed first)
    priority_by_opportunity: bool = True

    # Defaults for generated program
    default_boundary_k: int | None = None
    default_geodesic_k: int | None = None

    # Evaluation config
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)


@dataclass
class ProgramGeneratorResult:
    """Result of program generation."""

    program: TransplantProgram
    donor_selections: list[DonorCandidate]
    skipped_domains: list[str]  # Domains with no viable donor
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "_schema": "mc.result.program_generator.v1",
            "program": self.program.to_dict(),
            "donorSelections": [
                {
                    "profilePath": d.profile_path,
                    "modelPath": d.model_path,
                    "domain": d.domain,
                    "meanOpportunity": d.mean_opportunity,
                    "recommendedLayers": list(d.recommended_layers),
                }
                for d in self.donor_selections
            ],
            "skippedDomains": self.skipped_domains,
            "warnings": self.warnings,
        }


# =============================================================================
# Service Implementation
# =============================================================================


class ProgramGeneratorService:
    """Service for auto-generating TransplantProgram from density profiles.

    Analyzes target and donor density profiles to automatically select
    optimal donors for each knowledge domain. Outputs a complete
    TransplantProgram ready for execution with `mc program run`.

    The algorithm:
    1. For each domain in target, compute opportunity = donor_density - target_density
    2. Select donors with positive opportunity above threshold
    3. Assign layers where target is weak AND donor is strong
    4. Set priority based on opportunity magnitude

    Example:
        service = ProgramGeneratorService()
        result = service.generate_from_directory(
            target_profile="/profiles/target.json",
            donor_dir="/profiles/experts/",
        )
        result.program.to_yaml("/programs/uber-model.yaml")
    """

    def generate(
        self,
        target_profile: str | Path,
        donor_profiles: list[str | Path],
        config: ProgramGeneratorConfig | None = None,
        program_name: str | None = None,
        output_dir: str | None = None,
    ) -> ProgramGeneratorResult:
        """Generate TransplantProgram from density profiles.

        Args:
            target_profile: Path to target model's density profile JSON.
            donor_profiles: Paths to donor model density profiles.
            config: Generation configuration.
            program_name: Name for generated program (auto-generated if None).
            output_dir: Output directory for merged models.

        Returns:
            ProgramGeneratorResult with generated program and metadata.

        Raises:
            ValueError: If no donor profiles provided or profiles invalid.
            FileNotFoundError: If profile files don't exist.
        """
        resolved_config = config or ProgramGeneratorConfig()
        warnings: list[str] = []

        # Load profiles
        target = self._load_profile(target_profile)
        donors = [self._load_profile(p) for p in donor_profiles]

        if not donors:
            msg = "At least one donor profile is required"
            raise ValueError(msg)

        logger.info(
            "Generating program: target=%s, donors=%d, domains=%d",
            target.model_path,
            len(donors),
            len(target.domain_summaries),
        )

        # Compute opportunity for each donor per domain
        candidates_by_domain = self._compute_candidates(target, donors, resolved_config)

        # Select best donor(s) per domain
        selections = self._select_donors(target, candidates_by_domain, resolved_config)

        # Identify skipped domains
        all_target_domains = set(target.domain_summaries.keys())
        selected_domains = {s.domain for s in selections}
        skipped = sorted(all_target_domains - selected_domains)

        if skipped:
            warnings.append(f"No viable donor found for domains: {', '.join(skipped)}")
            logger.warning("Skipped domains (no viable donor): %s", skipped)

        # Build TransplantProgram
        program = self._build_program(
            target,
            selections,
            resolved_config,
            program_name=program_name,
            output_dir=output_dir,
        )

        logger.info(
            "Generated program: %s with %d donors covering %d domains",
            program.name,
            len(program.donors),
            len(selected_domains),
        )

        return ProgramGeneratorResult(
            program=program,
            donor_selections=selections,
            skipped_domains=skipped,
            warnings=warnings,
        )

    def generate_from_directory(
        self,
        target_profile: str | Path,
        donor_dir: str | Path,
        config: ProgramGeneratorConfig | None = None,
        **kwargs: Any,
    ) -> ProgramGeneratorResult:
        """Generate program from target and all profiles in a directory.

        Args:
            target_profile: Path to target model's density profile.
            donor_dir: Directory containing donor profile JSONs.
            config: Generation configuration.
            **kwargs: Additional args passed to generate().

        Returns:
            ProgramGeneratorResult with generated program.

        Raises:
            ValueError: If no donor profiles found in directory.
            FileNotFoundError: If directory doesn't exist.
        """
        donor_dir = Path(donor_dir)
        if not donor_dir.exists():
            msg = f"Donor directory not found: {donor_dir}"
            raise FileNotFoundError(msg)

        donor_profiles = list(donor_dir.glob("*.json"))

        # Exclude target if it's in the same directory
        target_path = Path(target_profile).resolve()
        donor_profiles = [p for p in donor_profiles if p.resolve() != target_path]

        if not donor_profiles:
            msg = f"No donor profiles found in {donor_dir}"
            raise ValueError(msg)

        logger.info("Found %d donor profiles in %s", len(donor_profiles), donor_dir)

        return self.generate(
            target_profile=target_profile,
            donor_profiles=donor_profiles,
            config=config,
            **kwargs,
        )

    def _load_profile(self, path: str | Path) -> DensityProfileSummary:
        """Load and parse density profile JSON.

        Args:
            path: Path to profile JSON file.

        Returns:
            DensityProfileSummary with parsed data.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If schema is invalid.
        """
        path = Path(path)
        if not path.exists():
            msg = f"Profile not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            data = json.load(f)

        schema = data.get("_schema", "")
        if schema != "mc.geometry.research.full_profile.v1":
            msg = (
                f"Invalid profile schema in {path}. "
                f"Expected 'mc.geometry.research.full_profile.v1', got '{schema}'"
            )
            raise ValueError(msg)

        domain_summaries: dict[str, DomainSummary] = {}
        for ds in data.get("domainSummaries", []):
            domain = ds["domain"]
            domain_summaries[domain] = DomainSummary(
                domain=domain,
                overall_mean_density=ds.get("overallMeanDensity", 0.0),
                concept_count=ds.get("conceptCount", 0),
                strongest_layers=tuple(ds.get("strongestLayers", [])),
                weakest_layers=tuple(ds.get("weakestLayers", [])),
            )

        return DensityProfileSummary(
            profile_path=str(path),
            model_path=data.get("modelPath", str(path)),
            total_layers=data.get("totalLayers", 0),
            domain_summaries=domain_summaries,
        )

    def _compute_candidates(
        self,
        target: DensityProfileSummary,
        donors: list[DensityProfileSummary],
        config: ProgramGeneratorConfig,
    ) -> dict[str, list[DonorCandidate]]:
        """Compute donor candidates per domain.

        For each domain in target, evaluate each donor's opportunity score:
        opportunity = donor_density - target_density

        Higher opportunity means donor is stronger where target is weak.

        Args:
            target: Target model's density profile.
            donors: List of donor density profiles.
            config: Generation configuration.

        Returns:
            Dict mapping domain -> list of DonorCandidate (sorted by opportunity desc).
        """
        candidates: dict[str, list[DonorCandidate]] = {}

        for domain, target_summary in target.domain_summaries.items():
            domain_candidates: list[DonorCandidate] = []

            for donor in donors:
                if domain not in donor.domain_summaries:
                    continue

                donor_summary = donor.domain_summaries[domain]

                # Opportunity = donor strength - target strength
                opportunity = (
                    donor_summary.overall_mean_density
                    - target_summary.overall_mean_density
                )

                # Skip if opportunity is non-positive (donor not better)
                if opportunity <= 0:
                    continue

                # Compute recommended layers:
                # Target's weakest layers that overlap with donor's strongest
                target_weak = set(target_summary.weakest_layers)
                donor_strong = set(donor_summary.strongest_layers)

                if config.layer_selection_mode == "intersection":
                    # Only layers where target is weak AND donor is strong
                    recommended = target_weak & donor_strong
                    if not recommended:
                        # Fallback: use target's weakest layers
                        recommended = target_weak
                else:
                    # Use all target weak layers
                    recommended = target_weak

                # Limit layer count
                recommended_list = sorted(recommended)[: config.max_layers_per_donor]

                domain_candidates.append(
                    DonorCandidate(
                        profile_path=donor.profile_path,
                        model_path=donor.model_path,
                        domain=domain,
                        mean_opportunity=opportunity,
                        recommended_layers=tuple(recommended_list),
                    )
                )

            if domain_candidates:
                # Sort by opportunity (highest first)
                domain_candidates.sort(key=lambda c: c.mean_opportunity, reverse=True)
                candidates[domain] = domain_candidates

        return candidates

    def _select_donors(
        self,
        target: DensityProfileSummary,
        candidates_by_domain: dict[str, list[DonorCandidate]],
        config: ProgramGeneratorConfig,
    ) -> list[DonorCandidate]:
        """Select best donor(s) per domain.

        Args:
            target: Target model's density profile.
            candidates_by_domain: Candidates grouped by domain.
            config: Generation configuration.

        Returns:
            List of selected DonorCandidate objects.
        """
        # Compute threshold from data if not configured
        threshold = config.min_opportunity_threshold
        if threshold is None:
            all_opportunities = [
                c.mean_opportunity
                for candidates in candidates_by_domain.values()
                for c in candidates
            ]
            if all_opportunities:
                # Use median as default threshold (data-driven, no arbitrary constants)
                sorted_opps = sorted(all_opportunities)
                n = len(sorted_opps)
                if n % 2 == 1:
                    threshold = sorted_opps[n // 2]
                else:
                    threshold = (sorted_opps[n // 2 - 1] + sorted_opps[n // 2]) / 2
            else:
                threshold = 0.0

        logger.debug("Using opportunity threshold: %.4f", threshold)

        selections: list[DonorCandidate] = []
        for domain, candidates in candidates_by_domain.items():
            # Filter by threshold
            viable = [c for c in candidates if c.mean_opportunity >= threshold]

            # Take top N
            top_n = viable[: config.max_donors_per_domain]
            selections.extend(top_n)

            if top_n:
                logger.debug(
                    "Domain %s: selected %s (opportunity=%.4f)",
                    domain,
                    Path(top_n[0].model_path).name,
                    top_n[0].mean_opportunity,
                )

        return selections

    def _build_program(
        self,
        target: DensityProfileSummary,
        selections: list[DonorCandidate],
        config: ProgramGeneratorConfig,
        program_name: str | None = None,
        output_dir: str | None = None,
    ) -> TransplantProgram:
        """Build TransplantProgram from selections.

        Args:
            target: Target model's density profile.
            selections: Selected donor candidates.
            config: Generation configuration.
            program_name: Optional program name.
            output_dir: Optional output directory.

        Returns:
            TransplantProgram ready for execution.
        """
        # Group selections by donor model
        donors_map: dict[str, list[DonorCandidate]] = {}
        for sel in selections:
            if sel.model_path not in donors_map:
                donors_map[sel.model_path] = []
            donors_map[sel.model_path].append(sel)

        # Build DonorSpecs with aggregated domains and layers
        donor_specs_with_opp: list[tuple[float, DonorSpec]] = []

        for model_path, model_selections in donors_map.items():
            # Aggregate domains and layers from all selections for this donor
            domains = tuple(sorted({s.domain for s in model_selections}))
            all_layers: set[int] = set()
            for s in model_selections:
                all_layers.update(s.recommended_layers)
            layers = tuple(sorted(all_layers)) if all_layers else None

            # Compute mean opportunity across all domains for this donor
            mean_opp = sum(s.mean_opportunity for s in model_selections) / len(
                model_selections
            )

            # Generate donor ID from model path
            donor_id = Path(model_path).name

            spec = DonorSpec(
                id=donor_id,
                source=model_path,
                domains=domains,
                layers=layers,
                priority=0,  # Will be assigned below based on opportunity
                boundary_k=config.default_boundary_k,
                geodesic_k=config.default_geodesic_k,
            )
            donor_specs_with_opp.append((mean_opp, spec))

        # Sort by opportunity and assign priorities
        donor_specs_with_opp.sort(key=lambda x: x[0], reverse=True)

        final_donors: list[DonorSpec] = []
        for i, (_, spec) in enumerate(donor_specs_with_opp):
            # Highest opportunity = highest priority
            priority = len(donor_specs_with_opp) - i if config.priority_by_opportunity else 0

            final_donors.append(
                DonorSpec(
                    id=spec.id,
                    source=spec.source,
                    domains=spec.domains,
                    layers=spec.layers,
                    priority=priority,
                    boundary_k=spec.boundary_k,
                    geodesic_k=spec.geodesic_k,
                )
            )

        # Build BaseModelSpec
        target_id = Path(target.model_path).name
        base_spec = BaseModelSpec(
            id=target_id,
            source=target.model_path,
            alias=target_id,
        )

        # Generate name if not provided
        if program_name is None:
            domain_count = len({s.domain for s in selections})
            program_name = f"auto-{target_id}-{domain_count}domains"

        # Generate description
        description = (
            f"Auto-generated from density profiles.\n"
            f"Target: {target.model_path}\n"
            f"Donors: {len(final_donors)}\n"
            f"Domains covered: {len({s.domain for s in selections})}"
        )

        return TransplantProgram(
            name=program_name,
            description=description,
            bases=(base_spec,),
            donors=tuple(final_donors),
            evaluation=config.evaluation_config,
            output_dir=output_dir or "~/.modelcypher/merged",
        )


__all__ = [
    "DensityProfileSummary",
    "DomainSummary",
    "DonorCandidate",
    "ProgramGeneratorConfig",
    "ProgramGeneratorResult",
    "ProgramGeneratorService",
]
