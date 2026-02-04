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

from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.bridge.bridge_generator import CrossModalBridge
from modelcypher.ports.bridge_store import BridgeStore

if TYPE_CHECKING:
    from modelcypher.core.domain.bridge.bridge_generator import BridgeGeneratorResult
    from modelcypher.ports.backend import Backend


class SafetensorsBridgeStore(BridgeStore):
    """Persist bridge artifacts using safetensors."""

    def save(
        self,
        path: Path,
        result: "BridgeGeneratorResult",
        backend: "Backend | None" = None,
    ) -> None:
        backend = backend or get_default_backend()

        backend.eval(result.transform, result.transform_inv)
        tensors = {
            "transform": result.transform,
            "transform_inv": result.transform_inv,
        }
        metadata = {
            "scale_ratio": str(result.scale_ratio),
            "source_dim": str(result.source_dim),
            "target_dim": str(result.target_dim),
            "cka_achieved": str(result.cka_achieved),
            "raw_cka": str(result.raw_cka),
            "n_samples": str(result.n_samples),
            "source_name": result.source_name,
            "target_name": result.target_name,
            "created_at": result.created_at.isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend.save_safetensors(str(path), tensors, metadata=metadata)

    def load(
        self,
        path: Path,
        backend: "Backend | None" = None,
    ) -> CrossModalBridge:
        backend = backend or get_default_backend()

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors required for bridge loading. "
                "Install with: pip install safetensors"
            ) from exc

        with safe_open(str(path), framework="numpy") as f:
            metadata = f.metadata() or {}
            scale_ratio = float(metadata.get("scale_ratio", "1.0"))
            source_dim = int(metadata.get("source_dim", "0"))
            target_dim = int(metadata.get("target_dim", "0"))
            source_name = metadata.get("source_name", "source")
            target_name = metadata.get("target_name", "target")

        tensors = backend.load_safetensors(str(path))
        transform = tensors["transform"]
        transform_inv = tensors["transform_inv"]
        backend.eval(transform, transform_inv)

        return CrossModalBridge(
            transform=transform,
            transform_inv=transform_inv,
            scale_ratio=scale_ratio,
            source_dim=source_dim,
            target_dim=target_dim,
            backend=backend,
            source_name=source_name,
            target_name=target_name,
        )


__all__ = ["SafetensorsBridgeStore"]
