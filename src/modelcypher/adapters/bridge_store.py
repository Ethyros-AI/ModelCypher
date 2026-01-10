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
from modelcypher.core.domain.bridge.generator import CrossModalBridge
from modelcypher.ports.bridge_store import BridgeStore

if TYPE_CHECKING:
    from modelcypher.core.domain.bridge.generator import BridgeGeneratorResult
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

        try:
            from safetensors.numpy import save_file
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "safetensors and numpy required for bridge saving. "
                "Install with: pip install safetensors numpy"
            ) from exc

        backend.eval(result.transform, result.transform_inv)

        transform_np = backend.to_numpy(result.transform)
        transform_inv_np = backend.to_numpy(result.transform_inv)

        tensors = {
            "transform": transform_np.astype(np.float32),
            "transform_inv": transform_inv_np.astype(np.float32),
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
        save_file(tensors, str(path), metadata=metadata)

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
            transform_np = f.get_tensor("transform")
            transform_inv_np = f.get_tensor("transform_inv")

            metadata = f.metadata() or {}
            scale_ratio = float(metadata.get("scale_ratio", "1.0"))
            source_dim = int(metadata.get("source_dim", transform_np.shape[0]))
            target_dim = int(metadata.get("target_dim", transform_np.shape[1]))
            source_name = metadata.get("source_name", "source")
            target_name = metadata.get("target_name", "target")

        transform = backend.array(transform_np)
        transform_inv = backend.array(transform_inv_np)
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
