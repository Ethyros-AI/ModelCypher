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

import pytest


class _StdModel:
    """Stub: standard layout — model.model has embed_tokens + layers."""
    class _Inner:
        embed_tokens = object()
        layers = [object()]
    model = _Inner()


class _Qwen35Model:
    """Stub: Qwen3.5 layout — model.language_model.model has embed_tokens + layers.

    Mirrors the real hierarchy:
      model.model = None
      model.language_model = TextModel (has .layers, no .embed_tokens)
      model.language_model.model = Qwen3_5TextModel (has both .embed_tokens + .layers)
    """
    class _LMInner:
        embed_tokens = object()
        layers = [object()]

    class _LM:
        layers = [object()]

        def __init__(self, lm_inner):
            self.model = lm_inner

    def __init__(self):
        self._lm_inner = self._LMInner()
        self.language_model = self._LM(self._lm_inner)
        self.model = None


def test_resolve_model_base_standard_layout():
    from modelcypher.backends.mlx_backend import MLXBackend

    backend = MLXBackend()
    m = _StdModel()

    result = backend._resolve_model_base(m)

    assert result is m.model


def test_resolve_model_base_language_model_layout():
    from modelcypher.backends.mlx_backend import MLXBackend

    backend = MLXBackend()
    m = _Qwen35Model()

    # Should return language_model.model (Qwen3_5TextModel), not language_model (TextModel)
    result = backend._resolve_model_base(m)

    assert result is m.language_model.model


def test_mlx_backend_array_accepts_large_int_list():
    import mlx.core as mx

    from modelcypher.backends.mlx_backend import MLXBackend

    backend = MLXBackend()

    # Regression: converting large ints can raise MLX std::bad_cast.
    value = [2**32 - 1]
    arr = backend.array(value)

    assert arr.shape == (1,)
    assert int(backend.to_scalar(arr)) == 2**32 - 1
