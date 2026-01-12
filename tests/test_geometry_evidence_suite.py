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

"""Tests for geometry evidence utilities and suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alignment_validation import (
    alignment_generalization_by_domain,
    alignment_generalization_report,
)
from modelcypher.core.domain.geometry.analytic_manifolds import (
    analytic_geodesic_distances,
    sample_circle_points,
    sample_sphere_points,
)
from modelcypher.core.domain.geometry.constrained_transplant import (
    causal_intervention_report,
)
from modelcypher.core.domain.geometry.evidence_suite import run_synthetic_evidence
from modelcypher.core.domain.geometry.manifold_accuracy import (
    curvature_accuracy_report,
    geodesic_accuracy_report,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


@pytest.fixture
def backend():
    return get_default_backend()


@dataclass(frozen=True)
class DummyProbe:
    probe_id: str
    name: str
    description: str
    support_texts: list[str]
    source: str
    domain: str
    category_name: str
    cross_domain_weight: float


def _linear_transform_pair(backend, n_samples: int, dim: int, seed: int = 0):
    backend.random_seed(seed)
    source = backend.random_normal((n_samples, dim))
    transform = backend.random_normal((dim, dim))
    target = backend.matmul(source, transform)
    backend.eval(source, transform, target)
    return source, target


def test_alignment_generalization_linear_map(backend):
    n_samples = 64
    dim = 16
    source, target = _linear_transform_pair(backend, n_samples, dim, seed=123)

    indices = list(range(n_samples))
    train_idx = indices[::2]
    holdout_idx = indices[1::2]

    report = alignment_generalization_report(
        source=source,
        target=target,
        train_indices=train_idx,
        holdout_indices=holdout_idx,
        backend=backend,
    )

    tol = division_epsilon(backend, source) * float(dim)
    assert abs(report.train_cka - 1.0) <= tol
    assert abs(report.holdout_cka - 1.0) <= tol


@given(
    dim=st.integers(min_value=2, max_value=12),
    scale=st.integers(min_value=4, max_value=6),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=25)
def test_alignment_generalization_property(dim, scale, seed):
    backend = get_default_backend()
    n_samples = dim * scale
    source, target = _linear_transform_pair(backend, n_samples, dim, seed=seed)

    indices = list(range(n_samples))
    train_idx = indices[::2]
    holdout_idx = indices[1::2]

    report = alignment_generalization_report(
        source=source,
        target=target,
        train_indices=train_idx,
        holdout_indices=holdout_idx,
        backend=backend,
    )

    tol = division_epsilon(backend, source) * float(dim)
    assert abs(report.train_cka - 1.0) <= tol
    assert abs(report.holdout_cka - 1.0) <= tol


def test_alignment_by_domain(backend):
    n_per_domain = 32
    domains = ["domain-a", "domain-b"]
    probes: list[DummyProbe] = []
    for domain in domains:
        for idx in range(n_per_domain):
            probes.append(
                DummyProbe(
                    probe_id=f"{domain}-{idx}",
                    name=domain,
                    description=domain,
                    support_texts=[domain],
                    source="test",
                    domain=domain,
                    category_name="test",
                    cross_domain_weight=1.0,
                )
            )

    n_samples = len(probes)
    dim = 8
    source, target = _linear_transform_pair(backend, n_samples, dim, seed=321)

    report = alignment_generalization_by_domain(
        source=source,
        target=target,
        probes=probes,
        backend=backend,
    )

    assert set(report.domain_reports.keys()) == set(domains)
    assert report.skipped_domains == []
    tol = division_epsilon(backend, source) * float(dim)
    for domain in domains:
        domain_report = report.domain_reports[domain]
        assert abs(domain_report.train_cka - 1.0) <= tol
        assert abs(domain_report.holdout_cka - 1.0) <= tol


def test_geodesic_convergence_circle(backend):
    small = sample_circle_points(32, radius=1.0, backend=backend)
    large = sample_circle_points(64, radius=1.0, backend=backend)

    analytic_small = analytic_geodesic_distances(small, radius=1.0, backend=backend)
    analytic_large = analytic_geodesic_distances(large, radius=1.0, backend=backend)

    geo_small = geodesic_accuracy_report(small, analytic_small, backend=backend)
    geo_large = geodesic_accuracy_report(large, analytic_large, backend=backend)

    eps = division_epsilon(backend, small)
    assert geo_large.mean_abs_error <= geo_small.mean_abs_error + eps
    assert geo_large.mean_relative_error <= geo_small.mean_relative_error + eps


def test_curvature_convergence_sphere(backend):
    small = sample_sphere_points(6, 8, radius=1.0, backend=backend)
    large = sample_sphere_points(8, 10, radius=1.0, backend=backend)
    analytic_curvature = 1.0

    curv_small = curvature_accuracy_report(
        small,
        analytic_curvature=analytic_curvature,
        backend=backend,
    )
    curv_large = curvature_accuracy_report(
        large,
        analytic_curvature=analytic_curvature,
        backend=backend,
    )

    eps = division_epsilon(backend, small)
    assert curv_large.mean_abs_error <= curv_small.mean_abs_error + eps


def test_causal_intervention_report(backend):
    backend.random_seed(7)
    out_dim = 8
    in_dim = 16
    core_samples = 12
    boundary_samples = 4

    target_weight = backend.random_normal((out_dim, in_dim))
    activations_core = backend.random_normal((core_samples, in_dim))
    delta_activations = backend.random_normal((core_samples, out_dim))
    boundary_activations = backend.random_normal((boundary_samples, in_dim))
    backend.eval(target_weight, activations_core, delta_activations, boundary_activations)

    report = causal_intervention_report(
        target_weights=target_weight,
        activations_core=activations_core,
        delta_activations=delta_activations,
        boundary_activations=boundary_activations,
        backend=backend,
    )

    eps = division_epsilon(backend, activations_core)
    assert report.core_mean_shift > eps
    assert report.boundary_max_relative_diff <= report.boundary_tolerance + eps


def test_run_synthetic_evidence_smoke(backend):
    report = run_synthetic_evidence(
        alignment_dim=12,
        alignment_samples=32,
        circle_samples=(24, 48),
        sphere_grid=((5, 6), (6, 8)),
        radius=1.0,
        seed=0,
        backend=backend,
    )

    assert report.alignment_generalization.train_samples > 0
    assert report.geodesic_convergence.small.sample_count > 0
    assert report.curvature_convergence.small.sample_count > 0
    assert report.causal_intervention.core_samples > 0


# Optional real-model coverage using fixtures (skips if not available).
FIXTURES_DIR = Path(__file__).parent / "fixtures" / ".models"
SMOLLM_PATH = FIXTURES_DIR / "HuggingFaceTB--SmolLM-135M"
LFM2_PATH = FIXTURES_DIR / "mlx-community--LFM2-350M-MLX-bf16"


@pytest.mark.skipif(
    not (SMOLLM_PATH / "model.safetensors").exists()
    or not (LFM2_PATH / "model.safetensors").exists(),
    reason="Real model fixtures not found",
)
def test_domain_alignment_real_models(backend):
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from tests.fixtures.models import collect_real_activations

    all_probes = UnifiedAtlasInventory.all_probes()
    if not all_probes:
        pytest.skip("No atlas probes available")

    from modelcypher.core.domain.geometry.atlas_protocols import enum_key

    by_domain: dict[str, list] = {}
    for probe in all_probes:
        by_domain.setdefault(enum_key(probe.domain), []).append(probe)

    domain, probes = max(by_domain.items(), key=lambda item: len(item[1]))
    if len(probes) < 4:
        pytest.skip("Not enough probes in selected domain")

    sample_count = min(len(probes), 12)
    step = max(1, len(probes) // sample_count)
    selected = probes[::step][:sample_count]

    prompts = []
    for probe in selected:
        if probe.support_texts:
            prompts.append(probe.support_texts[0])
        else:
            prompts.append(probe.name)

    try:
        smollm_acts = collect_real_activations(
            SMOLLM_PATH,
            prompts,
            backend=backend,
            layer_indices=[0],
        )
        lfm2_acts = collect_real_activations(
            LFM2_PATH,
            prompts,
            backend=backend,
            layer_indices=[0],
        )
    except Exception as exc:
        pytest.skip(f"Activation collection failed: {exc}")

    source = smollm_acts[0]
    target = lfm2_acts[0]

    report = alignment_generalization_by_domain(
        source=source,
        target=target,
        probes=selected,
        backend=backend,
    )

    assert report.domain_reports
    assert domain in report.domain_reports
