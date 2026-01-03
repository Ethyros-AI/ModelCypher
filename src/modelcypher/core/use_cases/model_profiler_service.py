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

"""Service for generating unified ModelProfiles.

Orchestrates the generation of comprehensive model profiles by coordinating
various analysis services (curvature, topology, semantics, density) into
a single unified ModelProfile.

Example:
    service = ModelProfilerService()
    profile = service.generate_profile(
        model_path="/path/to/model",
        sections=["geometry", "topology"],
    )
    profile.save("model_profile.json")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.model_profile import (
    ModelProfile,
    ProfileSection,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.curvature_profile import CurvatureProfile

logger = logging.getLogger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for profile generation.

    Note: Curvature parameters (k_neighbors, num_probes) are derived from
    data at runtime by the underlying geometry algorithms. They are not
    configurable here.
    """

    # Which sections to compute
    sections: list[str] = field(
        default_factory=lambda: [ProfileSection.IDENTITY.value, ProfileSection.GEOMETRY.value]
    )

    # Layer selection (None = all layers)
    layers: list[int] | None = None

    # Output settings
    output_path: str | None = None


class ModelProfilerService:
    """Orchestrates unified ModelProfile generation.

    This service coordinates the generation of comprehensive model profiles
    by calling various analysis components and combining their outputs into
    a unified ModelProfile schema.

    The service supports sectioned computation:
    - identity: Fast metadata extraction from config.json
    - geometry: Curvature and intrinsic dimension (medium cost)
    - topology: Persistent homology (expensive)
    - semantic: Semantic prime signatures (requires inference)
    - density: Knowledge density analysis (expensive)
    - entropy: Shannon/Renyi entropy per layer (medium cost)
    """

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config or ProfilerConfig()

    def generate_from_curvature(
        self,
        curvature_profile: "CurvatureProfile",
    ) -> ModelProfile:
        """Generate a ModelProfile from an existing CurvatureProfile.

        This is the fastest path when curvature has already been computed.

        Args:
            curvature_profile: Pre-computed curvature profile

        Returns:
            ModelProfile with geometry section populated
        """
        return ModelProfile.from_curvature_profile(curvature_profile)

    def load_and_merge(
        self,
        profile_paths: list[str],
    ) -> ModelProfile:
        """Load multiple profile JSONs and merge them.

        Useful for combining partial profiles computed at different times.

        Args:
            profile_paths: Paths to profile JSON files

        Returns:
            Merged ModelProfile with all sections combined
        """
        if not profile_paths:
            msg = "No profile paths provided"
            raise ValueError(msg)

        merged = ModelProfile.load(profile_paths[0])
        for path in profile_paths[1:]:
            other = ModelProfile.load(path)
            merged = merged.merge_with(other)

        return merged

    def import_curvature_file(
        self,
        curvature_path: str,
        output_path: str | None = None,
    ) -> ModelProfile:
        """Import a CurvatureProfile JSON into unified ModelProfile format.

        Args:
            curvature_path: Path to CurvatureProfile JSON
            output_path: Optional path to save the unified profile

        Returns:
            ModelProfile with geometry section from curvature data
        """
        from modelcypher.core.domain.geometry.curvature_profile import CurvatureProfile

        curvature = CurvatureProfile.load(curvature_path)
        profile = self.generate_from_curvature(curvature)

        if output_path:
            profile.save(output_path)

        return profile

    def update_identity(
        self,
        profile: ModelProfile,
        model_path: str,
    ) -> ModelProfile:
        """Update a profile with identity information from a model.

        Reads config.json from the model directory to extract:
        - Architecture type
        - Hidden dimension
        - Number of layers
        - Attention heads
        - Vocabulary size

        Args:
            profile: Existing profile to update
            model_path: Path to model directory

        Returns:
            Updated ModelProfile with identity section
        """
        from modelcypher.core.use_cases.model_probe_service import get_model_probe

        probe = get_model_probe()
        result = probe.probe(model_path)

        # Infer model family from architecture
        model_family = _infer_model_family(result.architecture) or profile.model_family

        # Create updated profile with identity info
        updated = ModelProfile(
            model_path=model_path,
            profile_version=profile.profile_version,
            model_family=model_family,
            architecture=result.architecture or profile.architecture,
            parameter_count=result.parameter_count or profile.parameter_count,
            hidden_dim=result.hidden_size or profile.hidden_dim,
            num_layers=len(result.layers) or profile.num_layers,
            num_attention_heads=result.num_attention_heads or profile.num_attention_heads,
            vocab_size=result.vocab_size or profile.vocab_size,
            # Preserve existing data
            layer_profiles=profile.layer_profiles,
            global_sectional_mean=profile.global_sectional_mean,
            global_sectional_std=profile.global_sectional_std,
            global_ollivier_ricci_mean=profile.global_ollivier_ricci_mean,
            global_ollivier_ricci_std=profile.global_ollivier_ricci_std,
            topology_summary=profile.topology_summary,
            semantic_signature=profile.semantic_signature,
            density_summary=profile.density_summary,
            computed_sections=profile.computed_sections,
            probe_corpus_hash=profile.probe_corpus_hash,
            backend_used=profile.backend_used,
            extraction_config=profile.extraction_config,
            computed_at=profile.computed_at,
        )

        # Add identity to computed sections
        updated.add_section(ProfileSection.IDENTITY)

        return updated

    def compare_profiles(
        self,
        source: ModelProfile | str,
        target: ModelProfile | str,
    ) -> dict[str, Any]:
        """Compare two profiles and return alignment analysis.

        Args:
            source: Source profile or path to profile JSON
            target: Target profile or path to profile JSON

        Returns:
            Comparison result as dictionary
        """
        from modelcypher.core.domain.geometry.profile_comparison import compare_profiles

        # Load profiles if paths provided
        if isinstance(source, str):
            source = ModelProfile.load(source)
        if isinstance(target, str):
            target = ModelProfile.load(target)

        comparison = compare_profiles(source, target)
        return comparison.to_dict()

    def collect_profiles(
        self,
        experiment_dir: str,
        pattern: str = "*.json",
    ) -> list[ModelProfile]:
        """Collect all profiles matching a pattern from a directory.

        Useful for batch processing experiment results.

        Args:
            experiment_dir: Directory containing profile JSONs
            pattern: Glob pattern for matching files

        Returns:
            List of loaded ModelProfiles
        """
        profiles = []
        for path in Path(experiment_dir).glob(pattern):
            try:
                profile = ModelProfile.load(str(path))
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        return profiles


def _infer_model_family(architecture: str) -> str:
    """Infer model family from architecture string."""
    arch_lower = architecture.lower()

    if "llama" in arch_lower:
        return "llama"
    elif "qwen" in arch_lower:
        return "qwen"
    elif "mistral" in arch_lower:
        return "mistral"
    elif "smol" in arch_lower:
        return "smollm"
    elif "phi" in arch_lower:
        return "phi"
    elif "gemma" in arch_lower:
        return "gemma"
    else:
        return "unknown"


__all__ = [
    "ModelProfilerService",
    "ProfilerConfig",
]
