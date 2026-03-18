from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCapabilityManifest:
    """Read-only capability facts that do not alter derived training math."""

    model_path: str
    architecture: str | None = None
    modality: str = "text"
    chat_template_family: str | None = None
    trust_remote_code_required: bool = False
    export_support: dict[str, bool] = field(default_factory=dict)
    deployment_caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "architecture": self.architecture,
            "modality": self.modality,
            "chat_template_family": self.chat_template_family,
            "trust_remote_code_required": self.trust_remote_code_required,
            "export_support": dict(self.export_support),
            "deployment_caveats": list(self.deployment_caveats),
        }


class ModelCapabilityManifestResolver:
    """Resolve model capability facts from local model assets."""

    def resolve(self, model_path: str | Path) -> ModelCapabilityManifest:
        model_dir = Path(model_path).expanduser().resolve()
        config = self._read_json(model_dir / "config.json")
        tokenizer_config = self._read_json(model_dir / "tokenizer_config.json")

        architecture = self._resolve_architecture(config)
        modality = self._resolve_modality(config, architecture)
        chat_template_family = self._resolve_chat_template_family(tokenizer_config)
        trust_remote_code_required = bool(
            config.get("auto_map") or tokenizer_config.get("auto_map")
        )

        deployment_caveats: list[str] = []
        if trust_remote_code_required:
            deployment_caveats.append(
                "Custom runtime classes may require trust_remote_code in downstream runtimes."
            )
        if modality != "text":
            deployment_caveats.append(
                "Non-text models must ship any processor/vision assets alongside exported weights."
            )

        export_support = {
            "adapter": True,
            "merged_fp16": True,
            "deployment_quantized": True,
        }

        return ModelCapabilityManifest(
            model_path=str(model_dir),
            architecture=architecture,
            modality=modality,
            chat_template_family=chat_template_family,
            trust_remote_code_required=trust_remote_code_required,
            export_support=export_support,
            deployment_caveats=deployment_caveats,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _resolve_architecture(config: dict[str, Any]) -> str | None:
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            return str(architectures[0])
        model_type = config.get("model_type")
        return str(model_type) if model_type is not None else None

    @staticmethod
    def _resolve_modality(config: dict[str, Any], architecture: str | None) -> str:
        architecture_text = (architecture or "").lower()
        if config.get("vision_config") is not None or any(
            token in architecture_text
            for token in ("llava", "internvl", "qwen2vl", "qwen2_vl", "mllama", "vision")
        ):
            return "vision_language"
        if config.get("audio_config") is not None or any(
            token in architecture_text for token in ("whisper", "audio")
        ):
            return "audio_text"
        return "text"

    @staticmethod
    def _resolve_chat_template_family(tokenizer_config: dict[str, Any]) -> str | None:
        template = tokenizer_config.get("chat_template")
        if not isinstance(template, str) or not template:
            return None
        if "[INST]" in template:
            return "llama_inst"
        if "<|start_header_id|>" in template:
            return "llama3_header"
        if "<|im_start|>" in template or "<|im_end|>" in template:
            return "chatml"
        if "<|user|>" in template and "<|assistant|>" in template:
            return "role_tokens"
        return "custom"
