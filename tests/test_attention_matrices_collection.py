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

"""Smoke tests for collect_attention_matrices and collect_attention_matrices_with_values."""


def test_collect_attention_matrices_exists():
    """Smoke test: method exists on the mixin class."""
    from modelcypher.backends._mlx_backend_activation_mixin import (
        _MLXBackendActivationMixin,
    )

    assert hasattr(_MLXBackendActivationMixin, "collect_attention_matrices")
    assert callable(getattr(_MLXBackendActivationMixin, "collect_attention_matrices"))


def test_collect_attention_matrices_with_values_exists():
    """Smoke test: method exists on the mixin class."""
    from modelcypher.backends._mlx_backend_activation_mixin import (
        _MLXBackendActivationMixin,
    )

    assert hasattr(
        _MLXBackendActivationMixin, "collect_attention_matrices_with_values"
    )
    assert callable(
        getattr(_MLXBackendActivationMixin, "collect_attention_matrices_with_values")
    )


def test_collect_attention_layer_results_handles_layernorm():
    """Backend mixin handles q_layernorm/k_layernorm when present."""
    import inspect
    from modelcypher.backends._mlx_backend_activation_mixin import (
        _MLXBackendActivationMixin,
    )

    source = inspect.getsource(
        _MLXBackendActivationMixin._collect_attention_layer_results
    )
    assert "q_layernorm" in source, (
        "_collect_attention_layer_results must handle q_layernorm for LFM2"
    )
    assert "k_layernorm" in source, (
        "_collect_attention_layer_results must handle k_layernorm for LFM2"
    )


def test_adapter_delegates_to_backend():
    """Adapter collect_attention_matrices delegates to backend, not custom code."""
    import inspect
    from modelcypher.adapters.activation_provider import ActivationProviderAdapter

    source = inspect.getsource(ActivationProviderAdapter.collect_attention_matrices)
    assert "_backend.collect_attention_matrices" in source, (
        "Adapter must delegate to backend, not implement its own attention collection"
    )
    assert "_AttnCaptureWrapper" not in source, (
        "Adapter must not have custom wrapper class — delegate to backend"
    )
