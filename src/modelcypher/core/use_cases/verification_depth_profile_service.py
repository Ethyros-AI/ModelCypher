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

"""Verification-depth manifold observability profiling.

Analyze-only service that measures trajectory-rank and intrinsic-dimension
progression across explicit verification-depth probe levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modelcypher.core.domain.geometry.gram_spectrum import compute_gram_spectrum
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.atlas.unified_atlas import AtlasProbe
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend


VerificationDepthMode = Literal["cumulative", "exact"]


@dataclass(frozen=True)
class VerificationDepthLayerProfile:
    """Raw per-layer observability metrics at a verification-depth level."""

    layer_idx: int
    activation_rank: int
    trajectory_rank: int
    intrinsic_dimension: float
    condition_number: float
    hidden_dim: int
    probe_sample_count: int
    trajectory_sample_count: int
    d_plus_1_minimum: int
    d_plus_1_gap: int
    coverage_ratio_probe: float
    coverage_ratio_trajectory: float
    null_rank: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "layer": self.layer_idx,
            "activationRank": self.activation_rank,
            "trajectoryRank": self.trajectory_rank,
            "intrinsicDimension": self.intrinsic_dimension,
            "conditionNumber": self.condition_number,
            "hiddenDim": self.hidden_dim,
            "probeSampleCount": self.probe_sample_count,
            "trajectorySampleCount": self.trajectory_sample_count,
            "dPlus1Minimum": self.d_plus_1_minimum,
            "dPlus1Gap": self.d_plus_1_gap,
            "coverageRatioProbe": self.coverage_ratio_probe,
            "coverageRatioTrajectory": self.coverage_ratio_trajectory,
            "nullRank": self.null_rank,
        }


@dataclass(frozen=True)
class VerificationDepthLevelProfile:
    """Per-level profile with per-layer observability measurements."""

    level: int
    mode: VerificationDepthMode
    probe_count: int
    layer_profiles: tuple[VerificationDepthLayerProfile, ...]
    canonical_trajectory_rank: int | None
    canonical_intrinsic_dimension: float | None

    def to_dict(self) -> dict[str, int | float | str | None | list[dict[str, int | float]]]:
        return {
            "level": self.level,
            "mode": self.mode,
            "probeCount": self.probe_count,
            "canonicalTrajectoryRank": self.canonical_trajectory_rank,
            "canonicalIntrinsicDimension": self.canonical_intrinsic_dimension,
            "layerResults": [profile.to_dict() for profile in self.layer_profiles],
        }


@dataclass(frozen=True)
class VerificationDepthProfileResult:
    """Full verification-depth profile result."""

    mode: VerificationDepthMode
    levels: tuple[int, ...]
    total_probes: int
    probes_with_verification_depth: int
    max_probes_per_level: int | None
    batch_size: int
    level_profiles: tuple[VerificationDepthLevelProfile, ...]
    rank_plateau_level: int | None
    id_plateau_level: int | None
    plateau_disagreement: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "levels": list(self.levels),
            "totalProbes": self.total_probes,
            "probesWithVerificationDepth": self.probes_with_verification_depth,
            "maxProbesPerLevel": self.max_probes_per_level,
            "batchSize": self.batch_size,
            "rankPlateauLevel": self.rank_plateau_level,
            "idPlateauLevel": self.id_plateau_level,
            "plateauDisagreement": self.plateau_disagreement,
            "levelResults": [profile.to_dict() for profile in self.level_profiles],
        }


class VerificationDepthProfileService:
    """Analyze manifold observability as verification-depth increases."""

    def __init__(
        self,
        backend: "Backend",
        activation_provider: "ActivationProvider",
    ) -> None:
        self._backend = backend
        self._activation_provider = activation_provider

    @property
    def backend(self) -> "Backend":
        """Get the backend used by this service."""
        return self._backend

    @property
    def activation_provider(self) -> "ActivationProvider":
        """Get the activation provider used by this service."""
        return self._activation_provider

    def profile(
        self,
        *,
        model: object,
        tokenizer: object,
        probes: list["AtlasProbe"],
        levels: list[int] | None = None,
        mode: VerificationDepthMode = "cumulative",
        max_probes_per_level: int | None = None,
        batch_size: int = 20,
    ) -> VerificationDepthProfileResult:
        """Run verification-depth profiling for a model."""
        if mode not in ("cumulative", "exact"):
            raise ValueError(f"Unsupported mode: {mode}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_probes_per_level is not None and max_probes_per_level <= 0:
            raise ValueError("max_probes_per_level must be > 0 when provided")

        probes_with_depth = [probe for probe in probes if probe.verification_depth is not None]
        if not probes_with_depth:
            raise ValueError("No probes with verification-depth metadata available")

        resolved_levels = self._resolve_levels(levels, probes_with_depth)
        level_profiles: list[VerificationDepthLevelProfile] = []
        epsilon_reference: "Array | None" = None

        for level in resolved_levels:
            selected_probes = self._select_probes_for_level(
                probes_with_depth=probes_with_depth,
                level=level,
                mode=mode,
            )
            if max_probes_per_level is not None:
                selected_probes = selected_probes[:max_probes_per_level]

            probe_texts = self._extract_probe_texts(selected_probes)
            if not probe_texts:
                level_profiles.append(
                    VerificationDepthLevelProfile(
                        level=level,
                        mode=mode,
                        probe_count=0,
                        layer_profiles=(),
                        canonical_trajectory_rank=None,
                        canonical_intrinsic_dimension=None,
                    )
                )
                continue

            positions, velocities = self._collect_layer_trajectories(
                model=model,
                tokenizer=tokenizer,
                texts=probe_texts,
                batch_size=batch_size,
            )

            if epsilon_reference is None and positions:
                first_layer = sorted(positions.keys())[0]
                epsilon_reference = positions[first_layer]

            layer_profiles = self._build_layer_profiles(positions=positions, velocities=velocities)
            trajectory_ranks = [profile.trajectory_rank for profile in layer_profiles]
            intrinsic_dimensions = [
                profile.intrinsic_dimension
                for profile in layer_profiles
                if math.isfinite(profile.intrinsic_dimension)
            ]
            canonical_trajectory_rank = max(trajectory_ranks) if trajectory_ranks else None
            canonical_intrinsic_dimension = max(intrinsic_dimensions) if intrinsic_dimensions else None

            level_profiles.append(
                VerificationDepthLevelProfile(
                    level=level,
                    mode=mode,
                    probe_count=len(probe_texts),
                    layer_profiles=tuple(layer_profiles),
                    canonical_trajectory_rank=canonical_trajectory_rank,
                    canonical_intrinsic_dimension=canonical_intrinsic_dimension,
                )
            )

        rank_plateau_level, id_plateau_level, plateau_disagreement = self._compute_plateaus(
            level_profiles=level_profiles,
            epsilon_reference=epsilon_reference,
        )

        return VerificationDepthProfileResult(
            mode=mode,
            levels=tuple(resolved_levels),
            total_probes=len(probes),
            probes_with_verification_depth=len(probes_with_depth),
            max_probes_per_level=max_probes_per_level,
            batch_size=batch_size,
            level_profiles=tuple(level_profiles),
            rank_plateau_level=rank_plateau_level,
            id_plateau_level=id_plateau_level,
            plateau_disagreement=plateau_disagreement,
        )

    def _resolve_levels(
        self,
        levels: list[int] | None,
        probes_with_depth: list["AtlasProbe"],
    ) -> list[int]:
        if levels:
            return sorted(set(levels))

        discovered = sorted({int(probe.verification_depth) for probe in probes_with_depth if probe.verification_depth is not None})
        if not discovered:
            raise ValueError("No verification-depth levels found in probes")
        return discovered

    def _select_probes_for_level(
        self,
        *,
        probes_with_depth: list["AtlasProbe"],
        level: int,
        mode: VerificationDepthMode,
    ) -> list["AtlasProbe"]:
        if mode == "cumulative":
            return [
                probe
                for probe in probes_with_depth
                if probe.verification_depth is not None and probe.verification_depth <= level
            ]
        return [
            probe
            for probe in probes_with_depth
            if probe.verification_depth is not None and probe.verification_depth == level
        ]

    def _extract_probe_texts(self, probes: list["AtlasProbe"]) -> list[str]:
        texts: list[str] = []
        for probe in probes:
            selected = self._probe_text(probe)
            if selected is not None:
                texts.append(selected)
        return texts

    def _probe_text(self, probe: "AtlasProbe") -> str | None:
        for text in probe.support_texts:
            if text and text.strip():
                return text.strip()
        if probe.description and probe.description.strip():
            return probe.description.strip()
        if probe.name and probe.name.strip():
            return probe.name.strip()
        return None

    def _collect_layer_trajectories(
        self,
        *,
        model: object,
        tokenizer: object,
        texts: list[str],
        batch_size: int,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
        positions_parts: dict[int, list["Array"]] = {}
        velocities_parts: dict[int, list["Array"]] = {}

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            trajectories = self._activation_provider.collect_trajectory_batch(
                model, tokenizer, batch_texts
            )

            for layer_idx, layer_positions in trajectories.positions.items():
                if int(self._backend.shape(layer_positions)[0]) <= 0:
                    continue
                positions_parts.setdefault(layer_idx, []).append(layer_positions)

            for layer_idx, layer_velocities in trajectories.velocities.items():
                if int(self._backend.shape(layer_velocities)[0]) <= 0:
                    continue
                velocities_parts.setdefault(layer_idx, []).append(layer_velocities)

        positions: dict[int, Array] = {}
        for layer_idx in sorted(positions_parts.keys()):
            parts = positions_parts[layer_idx]
            positions[layer_idx] = parts[0] if len(parts) == 1 else self._backend.concatenate(parts, axis=0)

        velocities: dict[int, Array] = {}
        for layer_idx in sorted(velocities_parts.keys()):
            parts = velocities_parts[layer_idx]
            velocities[layer_idx] = (
                parts[0] if len(parts) == 1 else self._backend.concatenate(parts, axis=0)
            )

        eval_targets = [*positions.values(), *velocities.values()]
        if eval_targets:
            self._backend.eval(*eval_targets)
        return positions, velocities

    def _build_layer_profiles(
        self,
        *,
        positions: dict[int, "Array"],
        velocities: dict[int, "Array"],
    ) -> list[VerificationDepthLayerProfile]:
        profiles: list[VerificationDepthLayerProfile] = []

        for layer_idx in sorted(positions.keys()):
            position_matrix = positions[layer_idx]
            velocity_matrix = velocities.get(layer_idx)

            if velocity_matrix is not None and int(self._backend.shape(velocity_matrix)[0]) > 0:
                trajectory_matrix = self._backend.concatenate(
                    [position_matrix, velocity_matrix], axis=0
                )
            else:
                trajectory_matrix = position_matrix

            self._backend.eval(position_matrix, trajectory_matrix)

            activation_spectrum = compute_gram_spectrum(position_matrix, backend=self._backend)
            trajectory_spectrum = compute_gram_spectrum(trajectory_matrix, backend=self._backend)

            hidden_dim = int(activation_spectrum.d_features)
            probe_sample_count = int(activation_spectrum.n_samples)
            trajectory_sample_count = int(trajectory_spectrum.n_samples)
            d_plus_1_minimum = hidden_dim + 1
            d_plus_1_gap = d_plus_1_minimum - probe_sample_count

            profiles.append(
                VerificationDepthLayerProfile(
                    layer_idx=layer_idx,
                    activation_rank=int(activation_spectrum.numeric_rank),
                    trajectory_rank=int(trajectory_spectrum.numeric_rank),
                    intrinsic_dimension=float(trajectory_spectrum.intrinsic_dimension),
                    condition_number=float(trajectory_spectrum.condition_number),
                    hidden_dim=hidden_dim,
                    probe_sample_count=probe_sample_count,
                    trajectory_sample_count=trajectory_sample_count,
                    d_plus_1_minimum=d_plus_1_minimum,
                    d_plus_1_gap=d_plus_1_gap,
                    coverage_ratio_probe=probe_sample_count / hidden_dim if hidden_dim > 0 else float("nan"),
                    coverage_ratio_trajectory=trajectory_sample_count / hidden_dim if hidden_dim > 0 else float("nan"),
                    null_rank=hidden_dim - int(trajectory_spectrum.numeric_rank),
                )
            )

        return profiles

    def _compute_plateaus(
        self,
        *,
        level_profiles: list[VerificationDepthLevelProfile],
        epsilon_reference: "Array | None",
    ) -> tuple[int | None, int | None, int | None]:
        ranked = [
            (profile.level, profile.canonical_trajectory_rank)
            for profile in level_profiles
            if profile.canonical_trajectory_rank is not None
        ]
        ids = [
            (profile.level, profile.canonical_intrinsic_dimension)
            for profile in level_profiles
            if profile.canonical_intrinsic_dimension is not None
            and math.isfinite(profile.canonical_intrinsic_dimension)
        ]

        rank_plateau_level: int | None = None
        if ranked:
            final_rank = ranked[-1][1]
            for level, rank in ranked:
                if rank == final_rank:
                    rank_plateau_level = level
                    break

        id_plateau_level: int | None = None
        if ids:
            final_id = float(ids[-1][1])
            reference = epsilon_reference
            if reference is None:
                reference = self._backend.array([1.0], dtype="float32")
            eps = machine_epsilon(self._backend, reference)
            tolerance = sqrt_scalar(eps, self._backend) * max(1.0, abs(final_id))
            for level, intrinsic_dimension in ids:
                if abs(float(intrinsic_dimension) - final_id) <= tolerance:
                    id_plateau_level = level
                    break

        plateau_disagreement: int | None = None
        if rank_plateau_level is not None and id_plateau_level is not None:
            plateau_disagreement = id_plateau_level - rank_plateau_level

        return rank_plateau_level, id_plateau_level, plateau_disagreement

