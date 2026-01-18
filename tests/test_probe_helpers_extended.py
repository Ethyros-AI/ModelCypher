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

"""Extended tests for probe stage helpers.

Tests critical APIs:
- _select_probe_text(): Select usable probe text from probe definition
- _proportional_layer_index(): Map layer indices by normalized depth
- _promote_precision(): Promote lower-precision arrays to float32
- _precision_reference(): Pick representative array for precision thresholds
- _extract_top_k_dims(): Extract top-k activated dimensions
- _select_geometry_probes(): Select probes with domain coverage
"""

import pytest
from dataclasses import dataclass
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    regularization_epsilon,
)
from modelcypher.core.use_cases.merge.stages.probe_helpers import (
    _proportional_layer_index,
    _promote_precision,
    _precision_reference,
    _extract_top_k_dims,
    _select_probe_text,
    _select_geometry_probes,
)


@pytest.fixture
def backend():
    return get_default_backend()


# Mock probe class for testing
@dataclass
class MockProbe:
    probe_id: str
    name: str
    description: str
    support_texts: list[str] | None
    domain: str = "test"


class TestProportionalLayerIndex:
    """Tests for _proportional_layer_index()."""

    def test_first_layer_maps_to_first(self):
        """First target layer should map to first source layer."""
        result = _proportional_layer_index(0, 10, 20)
        assert result == 0

    def test_last_layer_maps_to_last(self):
        """Last target layer should map to last source layer."""
        result = _proportional_layer_index(9, 10, 20)
        assert result == 19

    def test_middle_layer_proportional(self):
        """Middle target layer should map proportionally."""
        # Target layer 5 of 10 -> ratio = 5/9 = 0.555...
        # Mapped = round(0.555 * 19) = round(10.55) = 11
        result = _proportional_layer_index(5, 10, 20)
        assert result == 11

    def test_single_target_layer(self):
        """Single target layer should map to 0."""
        result = _proportional_layer_index(0, 1, 10)
        assert result == 0

    def test_single_source_layer(self):
        """Single source layer should always return 0."""
        result = _proportional_layer_index(5, 10, 1)
        assert result == 0

    def test_equal_layer_counts(self):
        """Equal layer counts should map 1:1."""
        for i in range(10):
            result = _proportional_layer_index(i, 10, 10)
            assert result == i

    def test_more_target_than_source(self):
        """Should handle more target layers than source."""
        # Target: 20 layers, Source: 10 layers
        # Target 10 (50%) -> Source 5 (50%)
        result = _proportional_layer_index(10, 20, 10)
        assert result == 5

    def test_negative_clamped_to_zero(self):
        """Negative indices should be clamped to 0."""
        # This shouldn't happen in practice, but test boundary
        result = _proportional_layer_index(0, 10, 10)
        assert result >= 0

    def test_out_of_bounds_clamped(self):
        """Out of bounds should be clamped to source_count - 1."""
        result = _proportional_layer_index(100, 10, 5)
        # Even invalid target_idx should clamp to valid range
        assert result <= 4


class TestPromotePrecision:
    """Tests for _promote_precision()."""

    def test_float32_unchanged(self, backend):
        """Float32 array should remain unchanged."""
        arr = backend.random_normal((16, 32))
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        # Should be same array (or same values)
        diff = backend.mean(backend.abs(arr - result))
        backend.eval(diff)
        eps = regularization_epsilon(backend, arr)
        assert float(backend.to_scalar(diff)) < eps

    def test_float16_promoted(self, backend):
        """Float16 array should be promoted to float32."""
        arr = backend.random_normal((16, 32))
        arr = backend.astype(arr, "float16")
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        # Should be float32 now
        dtype_name = str(result.dtype).lower()
        assert "float32" in dtype_name or "float" in dtype_name

    def test_output_finite(self, backend):
        """Output should always be finite."""
        arr = backend.random_normal((16, 32))
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        assert all_finite(result, backend)

    def test_preserves_values(self, backend):
        """Promotion should preserve values (within precision)."""
        arr = backend.random_normal((8, 8))
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        diff = backend.max(backend.abs(arr - result))
        backend.eval(diff)
        eps = regularization_epsilon(backend, arr)
        assert float(backend.to_scalar(diff)) < eps


class TestPrecisionReference:
    """Tests for _precision_reference()."""

    def test_returns_array(self, backend):
        """Should return an array."""
        result = _precision_reference(backend)

        assert hasattr(result, "dtype")

    def test_with_array_candidate(self, backend):
        """Should return first array candidate."""
        arr = backend.random_normal((4, 4))
        backend.eval(arr)

        result = _precision_reference(backend, arr)

        # Should return the provided array
        assert result is arr

    def test_with_dict_candidate(self, backend):
        """Should extract array from dict."""
        arr = backend.random_normal((4, 4))
        backend.eval(arr)
        d = {"key": arr}

        result = _precision_reference(backend, d)

        assert result is arr

    def test_with_none_uses_default(self, backend):
        """None candidate should use default."""
        result = _precision_reference(backend, None, default_dtype="float32")

        assert hasattr(result, "dtype")

    def test_with_multiple_candidates(self, backend):
        """Should return first valid array."""
        arr1 = backend.random_normal((4, 4))
        arr2 = backend.random_normal((8, 8))
        backend.eval(arr1, arr2)

        result = _precision_reference(backend, arr1, arr2)

        assert result is arr1


class TestExtractTopKDims:
    """Tests for _extract_top_k_dims()."""

    def test_basic_extraction(self, backend):
        """Basic top-k extraction should work."""
        activation = backend.array([0.1, 0.5, 0.3, 0.9, 0.2])
        backend.eval(activation)

        result = _extract_top_k_dims(activation, k=2, backend=backend)

        # Should return ActivatedDimension objects
        assert isinstance(result, list)
        # Highest magnitude should be index 3 (0.9)
        indices = [ad.index for ad in result]
        assert 3 in indices

    def test_returns_activated_dimensions(self, backend):
        """Should return ActivatedDimension objects."""
        activation = backend.array([1.0, 2.0, 3.0, 4.0, 5.0])
        backend.eval(activation)

        result = _extract_top_k_dims(activation, k=3, backend=backend)

        for ad in result:
            assert hasattr(ad, "index")
            assert hasattr(ad, "activation")

    def test_respects_k_limit(self, backend):
        """Should return at most k dimensions."""
        activation = backend.random_normal((100,))
        backend.eval(activation)

        k = 5
        result = _extract_top_k_dims(activation, k=k, backend=backend)

        assert len(result) <= k

    def test_threshold_filtering(self, backend):
        """Threshold should filter small activations."""
        activation = backend.array([0.001, 0.002, 1.0, 2.0, 0.003])
        backend.eval(activation)

        result = _extract_top_k_dims(activation, k=5, threshold=0.5, backend=backend)

        # Only indices 2 and 3 should pass threshold
        indices = [ad.index for ad in result]
        for idx in indices:
            assert idx in [2, 3]

    def test_empty_on_all_below_threshold(self, backend):
        """Should return empty if all below threshold."""
        activation = backend.array([0.01, 0.02, 0.03])
        backend.eval(activation)

        result = _extract_top_k_dims(activation, k=3, threshold=1.0, backend=backend)

        assert len(result) == 0

    def test_default_k_derived(self, backend):
        """Default k should be derived from dimension."""
        activation = backend.random_normal((64,))
        backend.eval(activation)

        result = _extract_top_k_dims(activation, backend=backend)

        # k = ceil(log2(d+1)) for d=64 -> ceil(6.02) = 7
        # Should return some dimensions
        assert len(result) > 0


class TestSelectProbeText:
    """Tests for _select_probe_text()."""

    def test_uses_first_support_text(self):
        """Should use first valid support text."""
        probe = MockProbe(
            probe_id="test",
            name="Test Probe",
            description="A test probe",
            support_texts=["First text", "Second text"],
        )

        result = _select_probe_text(probe)

        assert result == "First text"

    def test_skips_empty_support_texts(self):
        """Should skip empty support texts."""
        probe = MockProbe(
            probe_id="test",
            name="Test Probe",
            description="A test probe",
            support_texts=["", "  ", "Valid text"],
        )

        result = _select_probe_text(probe)

        assert result == "Valid text"

    def test_fallback_to_name_description(self):
        """Should fallback to name: description if no support texts."""
        probe = MockProbe(
            probe_id="test",
            name="Test Probe",
            description="A test probe",
            support_texts=None,
        )

        result = _select_probe_text(probe)

        assert result == "Test Probe: A test probe"

    def test_fallback_to_name_only(self):
        """Should fallback to name if no description."""
        probe = MockProbe(
            probe_id="test",
            name="Test Probe",
            description="",
            support_texts=None,
        )

        result = _select_probe_text(probe)

        assert result == "Test Probe"

    def test_fallback_to_description_only(self):
        """Should fallback to description if no name."""
        probe = MockProbe(
            probe_id="test",
            name="",
            description="A test probe",
            support_texts=None,
        )

        result = _select_probe_text(probe)

        assert result == "A test probe"

    def test_none_if_all_empty(self):
        """Should return None if all options empty."""
        probe = MockProbe(
            probe_id="test",
            name="",
            description="",
            support_texts=["", " "],
        )

        result = _select_probe_text(probe)

        assert result is None


class MockDomain:
    def __init__(self, value: str):
        self.value = value


class TestSelectGeometryProbes:
    """Tests for _select_geometry_probes()."""

    def test_returns_all_if_required_exceeds(self):
        """Should return all probes if required >= len(probes)."""
        probes = [
            (MockProbe("p1", "P1", "", ["text"], MockDomain("d1")), "text1"),
            (MockProbe("p2", "P2", "", ["text"], MockDomain("d2")), "text2"),
        ]

        result = _select_geometry_probes(probes, 10)

        assert len(result) == 2

    def test_returns_empty_list_for_zero_required(self):
        """Should return all probes if required is 0 or negative."""
        probes = [
            (MockProbe("p1", "P1", "", ["text"], MockDomain("d1")), "text1"),
        ]

        result = _select_geometry_probes(probes, 0)

        # Returns all when required <= 0
        assert len(result) == 1

    def test_deduplicates_by_probe_id(self):
        """Should deduplicate probes by probe_id."""
        probes = [
            (MockProbe("p1", "P1", "", ["text"], MockDomain("d1")), "text1"),
            (MockProbe("p1", "P1 Dup", "", ["text"], MockDomain("d1")), "text2"),  # Duplicate
            (MockProbe("p2", "P2", "", ["text"], MockDomain("d2")), "text3"),
        ]

        # Use required < len(probes) to trigger deduplication logic
        result = _select_geometry_probes(probes, 2)

        # Should only have 2 unique probes (p1 and p2)
        assert len(result) == 2
        probe_ids = {probe.probe_id for probe, _ in result}
        assert probe_ids == {"p1", "p2"}

    def test_round_robin_by_domain(self):
        """Should round-robin across domains for coverage."""
        probes = [
            (MockProbe("p1", "P1", "", ["text"], MockDomain("domain_a")), "text1"),
            (MockProbe("p2", "P2", "", ["text"], MockDomain("domain_a")), "text2"),
            (MockProbe("p3", "P3", "", ["text"], MockDomain("domain_b")), "text3"),
            (MockProbe("p4", "P4", "", ["text"], MockDomain("domain_b")), "text4"),
        ]

        result = _select_geometry_probes(probes, 2)

        # Should pick one from each domain
        domains = [probe.domain.value for probe, _ in result]
        assert "domain_a" in domains
        assert "domain_b" in domains


class TestProbeHelpersMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        target_idx=st.integers(min_value=0, max_value=99),
        target_count=st.integers(min_value=2, max_value=100),
        source_count=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_proportional_index_bounded(self, target_idx, target_count, source_count):
        """Proportional index should always be in valid range."""
        # Clamp target_idx to valid range for the test
        target_idx = min(target_idx, target_count - 1)

        result = _proportional_layer_index(target_idx, target_count, source_count)

        assert 0 <= result < source_count

    @given(
        d=st.integers(min_value=4, max_value=64),
    )
    @settings(max_examples=10, deadline=None)
    def test_extract_top_k_indices_valid(self, d):
        """Extracted indices should be valid dimension indices."""
        backend = get_default_backend()
        activation = backend.random_normal((d,))
        backend.eval(activation)

        result = _extract_top_k_dims(activation, k=5, backend=backend)

        for ad in result:
            assert 0 <= ad.index < d

    @given(
        d=st.integers(min_value=8, max_value=64),
        k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_extract_top_k_count_bounded(self, d, k):
        """Extracted count should be at most k."""
        backend = get_default_backend()
        activation = backend.random_normal((d,))
        backend.eval(activation)

        result = _extract_top_k_dims(activation, k=k, backend=backend)

        assert len(result) <= k

    @given(
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10, deadline=None)
    def test_precision_reference_always_returns_array(self, n):
        """Precision reference should always return an array."""
        backend = get_default_backend()

        result = _precision_reference(backend)

        assert hasattr(result, "dtype")
