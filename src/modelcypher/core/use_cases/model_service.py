from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.hf_hub import HfHubAdapter
from modelcypher.core.domain.models import ModelInfo
from modelcypher.utils.paths import expand_path


class ModelService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()
        self.hub = HfHubAdapter()

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
        size_bytes = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file()) if resolved.is_dir() else resolved.stat().st_size
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

    def fetch_model(
        self,
        repo_id: str,
        revision: str = "main",
        auto_register: bool = False,
        alias: str | None = None,
        architecture: str | None = None,
    ) -> dict:
        local_path = self.hub.fetch(repo_id, revision=revision)
        detected_arch = architecture or self.hub.detect_architecture(local_path) or "unknown"
        registered_id = None
        if auto_register:
            if not alias:
                raise ValueError("Alias required for auto-registration")
            model = self.register_model(alias=alias, path=local_path, architecture=detected_arch)
            registered_id = model.id

        return {
            "repoID": repo_id,
            "localPath": local_path,
            "registeredID": registered_id,
            "detectedArchitecture": detected_arch,
        }

    def resolve_model_id(self, model_id: str) -> str:
        model = self.store.get_model(model_id)
        return model.id if model else model_id
