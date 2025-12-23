from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from modelcypher.ports.exporter import Exporter
from modelcypher.utils.paths import expand_path


class LocalExporter(Exporter):
    def export_model(self, model_path: str, output_path: str, export_format: str) -> dict:
        return self._export_any(model_path, output_path, export_format)

    def export_checkpoint(self, checkpoint_path: str, output_path: str, export_format: str) -> dict:
        return self._export_any(checkpoint_path, output_path, export_format)

    def _export_any(self, source_path: str, output_path: str, export_format: str) -> dict:
        source = expand_path(source_path)
        target = expand_path(output_path)
        export_format = export_format.lower()

        if export_format == "npz":
            self._export_npz(source, target)
        elif export_format == "safetensors":
            self._export_safetensors(source, target)
        elif export_format == "gguf":
            self._export_placeholder(source, target, export_format)
        elif export_format == "mlx":
            self._export_mlx(source, target)
        elif export_format == "ollama":
            self._export_ollama_bundle(source, target)
        elif export_format == "lora":
            self._export_lora_bundle(source, target)
        elif export_format == "coreml":
            self._export_coreml_bundle(source, target)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        return {"format": export_format, "outputPath": str(target)}

    def _export_npz(self, source: Path, target: Path) -> None:
        if source.is_dir():
            source = source / "weights.npz"
        if source.suffix != ".npz":
            raise ValueError("NPZ export expects an .npz source")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def _export_safetensors(self, source: Path, target: Path) -> None:
        if source.is_dir():
            # If dir, look for weights.npz or safetensors
            if (source / "weights.npz").exists():
                source = source / "weights.npz"
            elif (source / "model.safetensors").exists():
                 # Already safetensors, copy
                 target.parent.mkdir(parents=True, exist_ok=True)
                 shutil.copy(source / "model.safetensors", target)
                 return

        if source.suffix == ".npz":
            data = np.load(source)
            tensors = {name: data[name] for name in data.files} # Use data.files to iterate
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, target)
        else:
            # Assume it's already a file we can copy if not handled above
             target.parent.mkdir(parents=True, exist_ok=True)
             shutil.copy(source, target)
    
    def _export_mlx(self, source: Path, target: Path) -> None:
        """Export as MLX archive (zip of .npy arrays)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(target, "with_suffix") and target.suffix != ".npz":
            target = target.with_suffix(".npz")
            
        if source.is_dir():
             if (source / "weights.npz").exists():
                 source = source / "weights.npz"
        
        # If source is NPZ, load and re-save (or just copy if format identical)
        # MLX savez produces compressed npz by default
        if source.suffix == ".npz":
             shutil.copy(source, target)
        else:
             # Load and save
             # This part assumes we can load 'source' via numpy
             data = np.load(source)
             np.savez_compressed(target, **data)

    def _export_placeholder(self, source: Path, target: Path, export_format: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "source": str(source),
            "format": export_format,
            "note": "Placeholder export - implement full conversion for production use",
        }
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    def _export_ollama_bundle(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        bundle_path = target if target.suffix == ".zip" else target.with_suffix(".zip")
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            modelfile = "FROM placeholder\nPARAMETER temperature 0.7\n"
            bundle.writestr("Modelfile", modelfile)
            bundle.writestr("model.gguf", "placeholder")

    def _export_lora_bundle(self, source: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "adapter_config.json"
        weights_path = target / "adapters.safetensors"
        config_path.write_text(
            json.dumps({"base_model_name_or_path": str(source), "peft_type": "LORA"}),
            encoding="utf-8",
        )
        save_file({"adapter": np.zeros((1, 1), dtype=np.float32)}, weights_path)

    def _export_coreml_bundle(self, source: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        payload = {"source": str(source), "note": "CoreML export stub"}
        (target / "Info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
