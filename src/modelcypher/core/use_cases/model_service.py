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

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports import ModelStore
    from modelcypher.ports.model_loader import ModelLoaderPort

from modelcypher.core.domain.models import ModelInfo
from modelcypher.utils.paths import expand_path


class ModelService:
    """Service for managing local model registry."""

    def __init__(
        self,
        store: "ModelStore",
        model_loader: "ModelLoaderPort",
    ) -> None:
        self.store = store
        self._model_loader = model_loader

    def list_models(self) -> list[ModelInfo]:
        return self.store.list_models()

    def register_model(
        self,
        alias: str,
        path: str,
        architecture: str,
        parameters: int | None = None,
        default_chat: bool = False,
    ) -> ModelInfo:
        resolved = expand_path(path)
        size_bytes = (
            sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
            if resolved.is_dir()
            else resolved.stat().st_size
        )
        model = ModelInfo(
            id=alias,
            alias=alias,
            architecture=architecture,
            format="safetensors",
            path=str(resolved),
            size_bytes=size_bytes,
            parameter_count=parameters,
            is_default_chat=default_chat,
            created_at=datetime.utcnow(),
        )
        self.store.register_model(model)
        return model

    def delete_model(self, model_id: str) -> None:
        self.store.delete_model(model_id)

    def resolve_model_id(self, model_id: str) -> str:
        model = self.store.get_model(model_id)
        return model.id if model else model_id

    def merge_models(
        self,
        source_model: str,
        target_model: str,
        output_path: str,
        auto_register: bool = False,
        alias: str | None = None,
    ) -> dict[str, Any]:
        """Merge two models using pure geometric alignment."""
        from modelcypher.core.use_cases.merge import UnifiedGeometricMerger

        merger = UnifiedGeometricMerger(model_loader=self._model_loader)
        merge_result = merger.merge(
            source_path=source_model,
            target_path=target_model,
            output_dir=output_path,
        )

        result: dict[str, Any] = {
            "status": "success",
            "outputPath": merge_result.output_path,
            "layerCount": merge_result.layer_count,
            "weightCount": merge_result.weight_count,
            "meanPreservedFraction": merge_result.mean_preserved_fraction,
            "metrics": {
                "meanProcrustesError": merge_result.mean_procrustes_error,
            },
        }

        if auto_register:
            if not alias:
                alias = f"merged-{datetime.utcnow().strftime('%Y%m%d%H%M')}"

            source_info = self.store.get_model(source_model)
            target_info = self.store.get_model(target_model)
            arch = (
                target_info.architecture
                if target_info
                else (source_info.architecture if source_info else "transformer")
            )
            model = self.register_model(alias=alias, path=output_path, architecture=arch)
            result["registeredID"] = model.id

        return result
