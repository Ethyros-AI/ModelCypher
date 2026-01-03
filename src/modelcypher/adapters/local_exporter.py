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

import shutil
from pathlib import Path

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.exporter import Exporter
from modelcypher.utils.paths import expand_path


class LocalExporter(Exporter):
    def export_model(self, model_path: str, output_path: str, export_format: str) -> dict:
        """Export model to specified format.

        Parameters
        ----------
        model_path : str
            Path to source model directory.
        output_path : str
            Destination path for exported model.
        export_format : str
            Target format (safetensors only - npz is deprecated).

        Returns
        -------
        dict
            Export result with format and output path.
        """
        return self._export_any(model_path, output_path, export_format)

    def export_checkpoint(self, checkpoint_path: str, output_path: str, export_format: str) -> dict:
        """Export training checkpoint to specified format.

        Parameters
        ----------
        checkpoint_path : str
            Path to source checkpoint.
        output_path : str
            Destination path for exported checkpoint.
        export_format : str
            Target format (safetensors only - npz is deprecated).

        Returns
        -------
        dict
            Export result with format and output path.
        """
        return self._export_any(checkpoint_path, output_path, export_format)

    def _export_any(self, source_path: str, output_path: str, export_format: str) -> dict:
        source = expand_path(source_path)
        target = expand_path(output_path)
        export_format = export_format.lower()

        if export_format == "safetensors":
            self._export_safetensors(source, target)
        elif export_format in ("npz", "mlx"):
            raise NotImplementedError(
                f"{export_format.upper()} format is deprecated. Use safetensors format instead."
            )
        elif export_format == "gguf":
            raise NotImplementedError(
                "GGUF export requires llama.cpp conversion tools. "
                "Use 'python -m mlx_lm.convert --hf-path <model> -q' for MLX quantization instead."
            )
        elif export_format == "ollama":
            raise NotImplementedError(
                "Ollama export requires GGUF conversion first. "
                "See https://github.com/ollama/ollama/blob/main/docs/import.md"
            )
        elif export_format == "coreml":
            raise NotImplementedError(
                "CoreML export requires coremltools. "
                "See https://apple.github.io/coremltools/docs-guides/"
            )
        elif export_format == "lora":
            raise NotImplementedError(
                "LoRA export should use the adapter training workflow. "
                "See 'mc train lora --help' for training adapters."
            )
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        return {"format": export_format, "outputPath": str(target)}

    def _export_safetensors(self, source: Path, target: Path) -> None:
        """Export to safetensors format using backend native I/O."""
        backend = get_default_backend()

        if source.is_dir():
            # Look for safetensors files in directory
            if (source / "model.safetensors").exists():
                # Already safetensors, copy
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source / "model.safetensors", target)
                return
            # Try to find any safetensors file
            safetensor_files = list(source.glob("*.safetensors"))
            if safetensor_files:
                # Load and merge all safetensors files
                weights = {}
                for sf in safetensor_files:
                    weights.update(backend.load_safetensors(str(sf)))
                target.parent.mkdir(parents=True, exist_ok=True)
                backend.save_safetensors(str(target), weights)
                return
            raise ValueError(f"No safetensors files found in {source}")

        if source.suffix == ".safetensors":
            # Copy safetensors file directly
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
        else:
            raise ValueError(
                f"Unsupported source format: {source.suffix}. Only .safetensors is supported."
            )
