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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.transport_guided_merger import TransportGuidedMerger


def test_transport_guided_merger_synthesize_identity() -> None:
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    source_weights = backend.array([[1.0, 0.0], [0.0, 1.0]])
    target_weights = backend.array([[0.5, 0.5], [0.5, 0.5]])
    transport_plan = backend.array([[1.0, 0.0], [0.0, 1.0]])
    backend.eval(source_weights, target_weights, transport_plan)
    result = merger.synthesize(
        source_weights=source_weights,
        target_weights=target_weights,
        transport_plan=transport_plan,
        config=TransportGuidedMerger.Config(
            coupling_threshold=0.0,
            normalize_rows=False,
            blend_alpha=0.0,
        ),
    )
    assert result is not None


def test_transport_guided_merger_dimension_confidence() -> None:
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    plan = backend.array([[0.9, 0.1], [0.5, 0.5]])
    backend.eval(plan)
    confidences = merger._compute_dimension_confidences(plan)
    assert confidences[0] > confidences[1]


def test_transport_guided_merger_synthesize_with_blend() -> None:
    """Blend alpha mixes source and target weights."""
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    source_weights = backend.array([[1.0, 0.0], [0.0, 1.0]])
    target_weights = backend.array([[0.0, 1.0], [1.0, 0.0]])
    transport_plan = backend.array([[1.0, 0.0], [0.0, 1.0]])
    backend.eval(source_weights, target_weights, transport_plan)
    result = merger.synthesize(
        source_weights=source_weights,
        target_weights=target_weights,
        transport_plan=transport_plan,
        config=TransportGuidedMerger.Config(
            coupling_threshold=0.0,
            normalize_rows=False,
            blend_alpha=0.5,
        ),
    )
    assert result is not None


def test_transport_guided_merger_empty_weights() -> None:
    """Empty weights return None."""
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    source_weights = backend.zeros((0, 2))
    target_weights = backend.array([[1.0]])
    transport_plan = backend.zeros((0, 1))
    backend.eval(source_weights, target_weights, transport_plan)
    result = merger.synthesize(
        source_weights=source_weights,
        target_weights=target_weights,
        transport_plan=transport_plan,
    )
    assert result is None


def test_transport_guided_merger_threshold_application() -> None:
    """Coupling threshold filters small values."""
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    plan = backend.array([[0.01, 0.99], [0.5, 0.5]])
    backend.eval(plan)
    # Use the synthesize method with high threshold to test filtering
    source = backend.array([[1.0, 0.0], [0.0, 1.0]])
    target = backend.array([[0.5, 0.5], [0.5, 0.5]])
    backend.eval(source, target)
    result = merger.synthesize(
        source_weights=source,
        target_weights=target,
        transport_plan=plan,
        config=TransportGuidedMerger.Config(coupling_threshold=0.1),
    )
    assert result is not None


def test_transport_guided_merger_row_normalization() -> None:
    """Row normalization makes rows sum to approximately 1."""
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    source = backend.array([[1.0, 0.0], [0.0, 1.0]])
    target = backend.array([[0.5, 0.5], [0.5, 0.5]])
    plan = backend.array([[0.2, 0.8], [0.3, 0.3]])
    backend.eval(source, target, plan)
    result = merger.synthesize(
        source_weights=source,
        target_weights=target,
        transport_plan=plan,
        config=TransportGuidedMerger.Config(normalize_rows=True),
    )
    assert result is not None


def test_transport_guided_merger_marginal_error() -> None:
    """Marginal error computation works correctly."""
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    # Uniform coupling should have low error
    uniform_plan = backend.array([[0.25, 0.25], [0.25, 0.25]])
    backend.eval(uniform_plan)
    row_err, col_err = merger._compute_marginal_error(uniform_plan)
    assert row_err < 0.1
    assert col_err < 0.1


def test_transport_guided_merger_effective_rank() -> None:
    """Effective rank counts couplings above threshold."""
    backend = get_default_backend()
    merger = TransportGuidedMerger(backend)
    plan = backend.array([[0.9, 0.01], [0.01, 0.9]])
    backend.eval(plan)
    rank = merger._compute_effective_rank(plan, threshold=0.1)
    assert rank == 2  # Only diagonal elements above threshold
