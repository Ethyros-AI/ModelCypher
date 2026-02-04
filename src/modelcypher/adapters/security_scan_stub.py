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

from typing import Any, AsyncGenerator

from modelcypher.adapters.inference_engine import InferenceEngine
from modelcypher.core.domain.inference.types import SecurityScanMetrics as _SecurityScanMetrics


class DualPathGenerator:
    """Minimal async generator that delegates to the unified inference engine."""

    def __init__(self, base_model_path: str, adapter_path: str | None = None) -> None:
        self._engine = InferenceEngine()
        self._model_path = base_model_path
        self._adapter_path = adapter_path

    async def generate(self, prompt: str, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        result = self._engine.infer(
            model=self._model_path,
            prompt=prompt,
            adapter=self._adapter_path,
            **kwargs,
        )
        yield {"type": "token", "text": result.get("response", "")}
        yield {"type": "metrics", "metrics": None}


class SecurityScanMetrics(_SecurityScanMetrics):
    """Alias for security scan metrics."""


__all__ = ["DualPathGenerator", "SecurityScanMetrics"]
