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

from __future__ import annotations

from modelcypher.core.domain.atlas.unified_atlas import AtlasProbe, AtlasSource
from modelcypher.core.domain.domains import AtlasDomain
from modelcypher.core.use_cases.verification_depth_profile_service import (
    VerificationDepthLayerProfile,
    VerificationDepthLevelProfile,
    VerificationDepthProfileService,
)
from modelcypher.ports.activation_provider import TrajectoryActivations


class _MockTrajectoryProvider:
    """Backend-native trajectory provider for verification-depth service tests."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.calls: list[list[str]] = []

    def collect_trajectory_batch(self, model, tokenizer, texts: list[str]) -> TrajectoryActivations:
        self.calls.append(list(texts))

        n = len(texts)
        hidden_dim = 4

        positions_rows = []
        velocity_rows = []
        for idx in range(n):
            x = float(idx + 1)
            positions_rows.append([1.0, x, x * x, x * x * x])
            velocity_rows.append([x, x + 1.0, x + 2.0, x + 3.0])

        positions = {0: self.backend.array(positions_rows)}
        velocities = {0: self.backend.array(velocity_rows)}
        embedding = self.backend.zeros((n, hidden_dim))

        self.backend.eval(positions[0], velocities[0], embedding)
        return TrajectoryActivations(
            positions=positions,
            velocities=velocities,
            intermediate_positions={},
            embedding_positions=embedding,
            q_positions={},
            k_positions={},
            v_positions={},
            gate_positions={},
            text_lengths=[1] * n,
            total_tokens=n,
            n_texts=n,
        )


def _build_probe(index: int, depth: int) -> AtlasProbe:
    return AtlasProbe(
        id=f"probe_{depth}_{index}",
        source=AtlasSource.DOMAIN_SPECIFIC,
        domain=AtlasDomain.LOGICAL,
        name=f"Probe {depth}-{index}",
        description=f"Depth {depth} probe {index}",
        cross_domain_weight=1.0,
        category_name="logical",
        support_texts=(f"Probe depth {depth} index {index}",),
        verification_depth=depth,
    )


def _build_probes() -> list[AtlasProbe]:
    probes: list[AtlasProbe] = []
    for depth in (0, 1, 2):
        for idx in range(3):
            probes.append(_build_probe(idx, depth))
    return probes


def test_profile_groups_levels_for_cumulative_and_exact(any_backend) -> None:
    provider = _MockTrajectoryProvider(any_backend)
    service = VerificationDepthProfileService(backend=any_backend, activation_provider=provider)
    probes = _build_probes()

    cumulative = service.profile(
        model=object(),
        tokenizer=object(),
        probes=probes,
        levels=[0, 1, 2],
        mode="cumulative",
        batch_size=32,
    )
    exact = service.profile(
        model=object(),
        tokenizer=object(),
        probes=probes,
        levels=[0, 1, 2],
        mode="exact",
        batch_size=32,
    )

    assert [level.probe_count for level in cumulative.level_profiles] == [3, 6, 9]
    assert [level.probe_count for level in exact.level_profiles] == [3, 3, 3]


def test_profile_reports_d_plus_1_gap_and_coverage(any_backend) -> None:
    provider = _MockTrajectoryProvider(any_backend)
    service = VerificationDepthProfileService(backend=any_backend, activation_provider=provider)
    probes = _build_probes()

    result = service.profile(
        model=object(),
        tokenizer=object(),
        probes=probes,
        levels=[0],
        mode="exact",
        batch_size=32,
    )

    layer = result.level_profiles[0].layer_profiles[0]
    assert layer.d_plus_1_minimum == layer.hidden_dim + 1
    assert layer.d_plus_1_gap == (layer.hidden_dim + 1) - layer.probe_sample_count
    assert layer.coverage_ratio_probe == layer.probe_sample_count / layer.hidden_dim
    assert layer.coverage_ratio_trajectory == layer.trajectory_sample_count / layer.hidden_dim


def test_plateau_detection_uses_rank_and_id(any_backend) -> None:
    provider = _MockTrajectoryProvider(any_backend)
    service = VerificationDepthProfileService(backend=any_backend, activation_provider=provider)

    level_profiles = [
        VerificationDepthLevelProfile(
            level=0,
            mode="cumulative",
            probe_count=3,
            layer_profiles=(),
            canonical_trajectory_rank=2,
            canonical_intrinsic_dimension=1.0,
        ),
        VerificationDepthLevelProfile(
            level=1,
            mode="cumulative",
            probe_count=6,
            layer_profiles=(),
            canonical_trajectory_rank=4,
            canonical_intrinsic_dimension=2.0,
        ),
        VerificationDepthLevelProfile(
            level=2,
            mode="cumulative",
            probe_count=9,
            layer_profiles=(),
            canonical_trajectory_rank=4,
            canonical_intrinsic_dimension=2.0,
        ),
    ]

    reference = any_backend.array([1.0], dtype="float32")
    rank_plateau_level, id_plateau_level, plateau_disagreement = service._compute_plateaus(
        level_profiles=level_profiles,
        epsilon_reference=reference,
    )

    assert rank_plateau_level == 1
    assert id_plateau_level == 1
    assert plateau_disagreement == 0


def test_profile_returns_backend_native_metrics_without_numpy(any_backend) -> None:
    provider = _MockTrajectoryProvider(any_backend)
    service = VerificationDepthProfileService(backend=any_backend, activation_provider=provider)

    result = service.profile(
        model=object(),
        tokenizer=object(),
        probes=_build_probes(),
        levels=[0, 1, 2],
        mode="cumulative",
        batch_size=32,
    )

    assert result.level_profiles
    first_layer: VerificationDepthLayerProfile = result.level_profiles[0].layer_profiles[0]
    assert isinstance(first_layer.activation_rank, int)
    assert isinstance(first_layer.trajectory_rank, int)
    assert isinstance(first_layer.intrinsic_dimension, float)
