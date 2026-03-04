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

"""Tests for attention hook infrastructure."""

import inspect


def test_hook_method_exists():
    """Smoke test: collect_hidden_with_attention_hook exists on the mixin."""
    from modelcypher.backends._mlx_backend_activation_mixin import (
        _MLXBackendActivationMixin,
    )

    assert hasattr(_MLXBackendActivationMixin, "collect_hidden_with_attention_hook")
    assert callable(
        getattr(_MLXBackendActivationMixin, "collect_hidden_with_attention_hook")
    )


def test_hook_signature():
    """Hook method accepts attention_hook parameter."""
    from modelcypher.backends._mlx_backend_activation_mixin import (
        _MLXBackendActivationMixin,
    )

    sig = inspect.signature(
        _MLXBackendActivationMixin.collect_hidden_with_attention_hook
    )
    param_names = list(sig.parameters.keys())
    assert "attention_hook" in param_names
    assert "model" in param_names
    assert "tokenizer" in param_names
    assert "text" in param_names


def test_hook_handles_none_baseline():
    """When attention_hook=None, the method should use standard forward pass."""
    from modelcypher.backends._mlx_backend_activation_mixin import (
        _MLXBackendActivationMixin,
    )

    source = inspect.getsource(
        _MLXBackendActivationMixin.collect_hidden_with_attention_hook
    )
    # Should have a path for hook=None (baseline)
    assert "attention_hook is not None" in source
