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

"""ModelCypher-owned MLX wrapper for lm-eval leaderboard runs."""

from __future__ import annotations

import gc

import transformers
from transformers.utils.import_utils import _LazyModule

from mlx_lm.evaluate import MLXLM

_ORIGINAL_LAZY_GETATTR = _LazyModule.__getattr__


def _patched_lazy_getattr(self, name):
    if name == "AutoModelForVision2Seq":
        return type("_DummyAutoModel", (), {})
    return _ORIGINAL_LAZY_GETATTR(self, name)


if _LazyModule.__getattr__ is not _patched_lazy_getattr:
    _LazyModule.__getattr__ = _patched_lazy_getattr


class MLXModelWrapper(MLXLM):
    """Thin lm-eval wrapper that keeps ModelCypher in control of MLX loading."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(path_or_hf_repo=model_path, batch_size=batch_size)
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.batch_size = batch_size

        if adapter_path:
            from mlx_lm.lora import load_adapters

            load_adapters(self._model, adapter_path)

    def cleanup(self) -> None:
        """Release MLX state after evaluation.

        The repo test harness documents that Python references must be collected
        before clearing the MLX cache to avoid use-after-free issues.
        """
        import mlx.core as mx

        for attr in ("_model", "model", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        mx.clear_cache()
