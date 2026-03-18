from __future__ import annotations

import json
from pathlib import Path

from modelcypher.core.use_cases.model_capability_manifest import (
    ModelCapabilityManifestResolver,
)


def test_capability_manifest_resolves_text_chatml(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "<|im_start|>user\n{{message}}<|im_end|>"}),
        encoding="utf-8",
    )

    manifest = ModelCapabilityManifestResolver().resolve(model_dir)

    assert manifest.modality == "text"
    assert manifest.chat_template_family == "chatml"
    assert manifest.trust_remote_code_required is False
    assert manifest.export_support["deployment_quantized"] is True


def test_capability_manifest_resolves_multimodal_and_remote_code(tmp_path: Path) -> None:
    model_dir = tmp_path / "vision-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MllamaForConditionalGeneration"],
                "vision_config": {"hidden_size": 1024},
                "auto_map": {"AutoModel": "custom.module.Model"},
            }
        ),
        encoding="utf-8",
    )

    manifest = ModelCapabilityManifestResolver().resolve(model_dir)

    assert manifest.modality == "vision_language"
    assert manifest.trust_remote_code_required is True
    assert manifest.deployment_caveats

