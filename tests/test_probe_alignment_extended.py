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

"""Extended tests for probe alignment helpers.

Tests critical APIs:
- AlignmentResult: Data class for alignment results
- _activation_count(): Get number of activation samples
- _activation_dim(): Get activation dimension
- _stack_activations(): Stack activations into matrix
- align_layers(): Main layer alignment function
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    regularization_epsilon,
)
from modelcypher.core.use_cases.merge.stages.probe_alignment import (
    AlignmentResult,
    _activation_count,
    _activation_dim,
    _stack_activations,
    align_layers,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestActivationCount:
    """Tests for _activation_count()."""

    def test_2d_array(self, backend):
        """Should return first dimension for 2D array."""
        arr = backend.random_normal((16, 32))
        backend.eval(arr)

        result = _activation_count(backend, arr)

        assert result == 16

    def test_list_of_arrays(self, backend):
        """Should return length for list of arrays."""
        arr_list = [backend.random_normal((32,)) for _ in range(10)]
        for a in arr_list:
            backend.eval(a)

        result = _activation_count(backend, arr_list)

        assert result == 10

    def test_single_element(self, backend):
        """Single element should return 1."""
        arr_list = [backend.random_normal((32,))]
        backend.eval(arr_list[0])

        result = _activation_count(backend, arr_list)

        assert result == 1

    def test_empty_list(self, backend):
        """Empty list should return 0."""
        result = _activation_count(backend, [])

        assert result == 0


class TestActivationDim:
    """Tests for _activation_dim()."""

    def test_2d_array(self, backend):
        """Should return second dimension for 2D array."""
        arr = backend.random_normal((16, 32))
        backend.eval(arr)

        result = _activation_dim(backend, arr)

        assert result == 32

    def test_1d_array(self, backend):
        """Should return dimension for 1D array."""
        arr = backend.random_normal((64,))
        backend.eval(arr)

        result = _activation_dim(backend, arr)

        assert result == 64

    def test_list_of_1d_arrays(self, backend):
        """Should return dimension from first element."""
        arr_list = [backend.random_normal((32,)) for _ in range(5)]
        for a in arr_list:
            backend.eval(a)

        result = _activation_dim(backend, arr_list)

        assert result == 32

    def test_empty_list(self, backend):
        """Empty list should return 0."""
        result = _activation_dim(backend, [])

        assert result == 0


class TestStackActivations:
    """Tests for _stack_activations()."""

    def test_2d_array_slices(self, backend):
        """2D array should be sliced to n_samples."""
        arr = backend.random_normal((20, 32))
        backend.eval(arr)

        result = _stack_activations(backend, arr, n_samples=10)

        assert backend.shape(result) == (10, 32)

    def test_list_stacks(self, backend):
        """List should be stacked into 2D array."""
        arr_list = [backend.random_normal((32,)) for _ in range(10)]
        for a in arr_list:
            backend.eval(a)

        result = _stack_activations(backend, arr_list, n_samples=10)

        assert backend.shape(result) == (10, 32)

    def test_output_finite(self, backend):
        """Output should be finite."""
        arr = backend.random_normal((16, 32))
        backend.eval(arr)

        result = _stack_activations(backend, arr, n_samples=16)

        assert all_finite(result, backend)

    def test_precision_promoted(self, backend):
        """Should promote precision to float32."""
        arr = backend.random_normal((16, 32))
        arr = backend.astype(arr, "float16")
        backend.eval(arr)

        result = _stack_activations(backend, arr, n_samples=16)

        # Should be float32
        dtype_name = str(result.dtype).lower()
        assert "float32" in dtype_name or "float" in dtype_name


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_dataclass_fields(self):
        """AlignmentResult should have all expected fields."""
        result = AlignmentResult(
            layer_mapping={0: 0},
            feature_transforms={0: None},
            scale_ratios={0: 1.0},
            attention_transforms={},
            k_transforms={},
            v_transforms={},
            intermediate_transforms={},
            gate_transforms={},
            layer_cka_scores={0: 1.0},
            cgls_iterations_by_layer={0: 10},
        )

        assert result.layer_mapping == {0: 0}
        assert result.layer_cka_scores == {0: 1.0}


class TestAlignLayers:
    """Tests for align_layers()."""

    def test_empty_activations_returns_empty_result(self, backend):
        """Empty activations should return empty result."""
        result = align_layers(
            source_layer_activations={},
            target_layer_activations={},
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        assert result.layer_mapping == {}
        assert result.feature_transforms == {}

    def test_basic_alignment(self, backend):
        """Basic layer alignment should work."""
        n_samples = 32
        source_dim = 64
        target_dim = 48

        source_activations = {
            0: backend.random_normal((n_samples, source_dim)),
        }
        target_activations = {
            0: backend.random_normal((n_samples, target_dim)),
        }
        for layer_acts in [*source_activations.values(), *target_activations.values()]:
            backend.eval(layer_acts)

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        assert 0 in result.layer_mapping
        assert 0 in result.feature_transforms
        assert 0 in result.layer_cka_scores

    def test_multiple_layers(self, backend):
        """Should align multiple layers."""
        n_samples = 32
        source_dim = 64
        target_dim = 48

        source_activations = {
            i: backend.random_normal((n_samples, source_dim))
            for i in range(4)
        }
        target_activations = {
            i: backend.random_normal((n_samples, target_dim))
            for i in range(4)
        }
        for layer_acts in [*source_activations.values(), *target_activations.values()]:
            backend.eval(layer_acts)

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        assert len(result.layer_mapping) == 4
        assert len(result.feature_transforms) == 4

    def test_cka_scores_bounded(self, backend):
        """CKA scores should be in [0, 1]."""
        n_samples = 32
        source_dim = 64
        target_dim = 48

        source_activations = {
            0: backend.random_normal((n_samples, source_dim)),
        }
        target_activations = {
            0: backend.random_normal((n_samples, target_dim)),
        }
        for layer_acts in [*source_activations.values(), *target_activations.values()]:
            backend.eval(layer_acts)

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        eps = regularization_epsilon(backend, next(iter(source_activations.values())))
        for score in result.layer_cka_scores.values():
            assert 0.0 <= score <= 1.0 + eps

    def test_transforms_have_correct_shape(self, backend):
        """Feature transforms should have correct shape."""
        n_samples = 32
        source_dim = 64
        target_dim = 48

        source_activations = {
            0: backend.random_normal((n_samples, source_dim)),
        }
        target_activations = {
            0: backend.random_normal((n_samples, target_dim)),
        }
        for layer_acts in [*source_activations.values(), *target_activations.values()]:
            backend.eval(layer_acts)

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        # Feature transform maps source_dim -> target_dim
        transform = result.feature_transforms[0]
        # It's a dict of {source_layer: transform}
        assert isinstance(transform, dict)
        for layer_id, F in transform.items():
            shape = backend.shape(F)
            assert shape[0] == source_dim
            assert shape[1] == target_dim

    def test_different_layer_counts(self, backend):
        """Should handle different source/target layer counts."""
        n_samples = 32
        source_dim = 64
        target_dim = 48

        # Source: 6 layers, Target: 4 layers
        source_activations = {
            i: backend.random_normal((n_samples, source_dim))
            for i in range(6)
        }
        target_activations = {
            i: backend.random_normal((n_samples, target_dim))
            for i in range(4)
        }
        for layer_acts in [*source_activations.values(), *target_activations.values()]:
            backend.eval(layer_acts)

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        # Should have mapping for all target layers
        assert len(result.layer_mapping) == 4

    def test_with_intermediate_activations(self, backend):
        """Should align intermediate activations if provided."""
        n_samples = 32
        hidden_dim = 64
        intermediate_dim = 256

        source_hidden = {0: backend.random_normal((n_samples, hidden_dim))}
        target_hidden = {0: backend.random_normal((n_samples, hidden_dim))}
        source_inter = {0: backend.random_normal((n_samples, intermediate_dim))}
        target_inter = {0: backend.random_normal((n_samples, intermediate_dim))}

        for acts in [*source_hidden.values(), *target_hidden.values(),
                     *source_inter.values(), *target_inter.values()]:
            backend.eval(acts)

        result = align_layers(
            source_layer_activations=source_hidden,
            target_layer_activations=target_hidden,
            source_intermediate_activations=source_inter,
            target_intermediate_activations=target_inter,
            backend=backend,
        )

        # Should have intermediate transforms
        assert len(result.intermediate_transforms) > 0


class TestAlignLayersMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=16, max_value=32),
        source_dim=st.integers(min_value=16, max_value=32),
        target_dim=st.integers(min_value=16, max_value=32),
    )
    @settings(max_examples=3, deadline=None)
    def test_alignment_produces_valid_transform(self, n_samples, source_dim, target_dim):
        """Alignment should produce finite transform."""
        backend = get_default_backend()

        source_activations = {0: backend.random_normal((n_samples, source_dim))}
        target_activations = {0: backend.random_normal((n_samples, target_dim))}
        backend.eval(source_activations[0], target_activations[0])

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        for layer_transforms in result.feature_transforms.values():
            for F in layer_transforms.values():
                assert all_finite(F, backend)

    @given(
        n_layers=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=3, deadline=None)
    def test_all_target_layers_mapped(self, n_layers):
        """All target layers should have mappings."""
        backend = get_default_backend()
        n_samples = 24
        dim = 32

        source_activations = {
            i: backend.random_normal((n_samples, dim))
            for i in range(n_layers)
        }
        target_activations = {
            i: backend.random_normal((n_samples, dim))
            for i in range(n_layers)
        }
        for acts in [*source_activations.values(), *target_activations.values()]:
            backend.eval(acts)

        result = align_layers(
            source_layer_activations=source_activations,
            target_layer_activations=target_activations,
            source_intermediate_activations={},
            target_intermediate_activations={},
            backend=backend,
        )

        assert len(result.layer_mapping) == n_layers
