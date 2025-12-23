from __future__ import annotations

from datetime import datetime

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.hf_hub import HfHubAdapter
from modelcypher.core.domain.models import ModelInfo
from modelcypher.core.domain.merging.rotational_merger import RotationalModelMerger, MergeOptions
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

    def merge_models(
        self,
        source_model: str,
        target_model: str,
        output_path: str,
        options: MergeOptions | None = None,
        auto_register: bool = False,
        alias: str | None = None,
    ) -> dict:
        """Merge two models using rotational alignment."""
        import mlx.core as mx
        from mlx_lm import load, save_tensors
        from pathlib import Path

        options = options or MergeOptions.default()
        
        # Resolve paths
        source_info = self.store.get_model(source_model)
        target_info = self.store.get_model(target_model)
        source_path = Path(source_info.path) if source_info else Path(source_model)
        target_path = Path(target_info.path) if target_info else Path(target_model)
        output_dir = expand_path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        src_model, _ = load(str(source_path))
        tgt_model, tgt_tokenizer = load(str(target_path))
        
        # Extract state dicts (weights only)
        src_weights = dict(src_model.parameters())
        tgt_weights = dict(tgt_model.parameters())
        
        # Merge
        merger = RotationalModelMerger(options)
        merged_weights, analysis = merger.merge_weights(src_weights, tgt_weights)
        
        # Save merged model
        save_tensors(str(output_dir), merged_weights)
        tgt_tokenizer.save_pretrained(str(output_dir))
        
        # Save config
        config_path = output_dir / "config.json"
        if (target_path / "config.json").exists():
             import shutil
             shutil.copy(target_path / "config.json", config_path)
        
        merged_model_id = None
        if auto_register:
             if not alias:
                  alias = f"merged-{datetime.utcnow().strftime('%Y%m%d%H%M')}"
             
             # Determine Architecture (Assume target for now)
             arch = target_info.architecture if target_info else "transformer"
             model = self.register_model(alias=alias, path=str(output_dir), architecture=arch)
             merged_model_id = model.id

        return {
             "mergedPath": str(output_dir),
             "registeredID": merged_model_id,
             "analysis": {
                  "meanProcrustesError": analysis.mean_procrustes_error,
                  "rotationFieldRoughness": analysis.rotation_field_roughness,
                  "layerMetrics": [
                       {
                            "layer": m.layer_index,
                            "error": m.procrustes_error,
                            "roughness": m.rotation_deviation
                       } for m in analysis.layer_metrics
                  ]
             }
        }
