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

"""Model structure utilities for navigating transformer architectures."""

from __future__ import annotations

from typing import Any


def resolve_model_base(model: Any) -> Any:
    """Return the backbone object that has both ``.embed_tokens`` and ``.layers``.

    Handles:
    - Standard layout: ``model.model`` (has embed_tokens + layers)
    - Qwen3.5 layout: ``model.language_model.model`` (Qwen3_5TextModel has both;
      ``model.language_model`` (TextModel) only has layers, not embed_tokens)
    - Qwen3.5-VL layout: ``model.model.language_model.model``
    """

    def _has_both(obj: Any) -> bool:
        return obj is not None and hasattr(obj, "embed_tokens") and hasattr(obj, "layers")

    inner = getattr(model, "model", None)
    if _has_both(inner):
        return inner

    # Qwen3.5-VL: model.model.language_model(.model) has embed_tokens + layers
    if inner is not None:
        inner_lm = getattr(inner, "language_model", None)
        if inner_lm is not None:
            if _has_both(inner_lm):
                return inner_lm
            inner_lm_inner = getattr(inner_lm, "model", None)
            if _has_both(inner_lm_inner):
                return inner_lm_inner
            if hasattr(inner_lm, "layers"):
                return inner_lm

    lm = getattr(model, "language_model", None)
    if lm is not None:
        if _has_both(lm):
            return lm
        lm_inner = getattr(lm, "model", None)
        if _has_both(lm_inner):
            return lm_inner
        # lm has layers but not embed_tokens — still usable as fallback
        if hasattr(lm, "layers"):
            return lm

    if hasattr(model, "layers"):
        return model
    return model  # best-effort fallback
