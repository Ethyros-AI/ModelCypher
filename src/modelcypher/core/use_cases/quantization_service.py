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

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.use_cases.quantization_utils import (
    QuantizationHint,
    requantize_weights,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)

_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt", ".npz", ".gguf"}
_WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")


@dataclass(frozen=True)
class QuantizedModelResult:
    bits: int
    group_size: int
    mode: str
    output_dir: Path
    weights_file: Path
    total_2d_weights: int
    quantized_2d_weights: int
    skipped_2d_weights: int
    skipped_dirs: tuple[str, ...]
    quantization_config: dict[str, int | str]

    def to_dict(self) -> dict[str, object]:
        return {
            "bits": self.bits,
            "groupSize": self.group_size,
            "mode": self.mode,
            "outputDir": str(self.output_dir),
            "weightsFile": str(self.weights_file),
            "weights": {
                "total2d": self.total_2d_weights,
                "quantized2d": self.quantized_2d_weights,
                "skipped2d": self.skipped_2d_weights,
            },
            "skippedDirs": list(self.skipped_dirs),
            "quantizationConfig": self.quantization_config,
        }


class QuantizationService:
    def __init__(self, backend: "Backend", model_loader: "ModelLoaderPort") -> None:
        self._backend = backend
        self._model_loader = model_loader

    def detect_supported_bits(
        self,
        bits: list[int],
        group_size: int,
        mode: str,
    ) -> dict[int, str | None]:
        supported: dict[int, str | None] = {}
        if group_size <= 0:
            raise ValueError("group_size must be > 0")

        dummy = self._backend.zeros((1, group_size))
        for bit in bits:
            if bit <= 0 or 32 % bit != 0:
                supported[bit] = "invalid_bits"
                continue
            try:
                self._backend.quantize(
                    dummy,
                    group_size=group_size,
                    bits=bit,
                    mode=mode,
                )
                supported[bit] = None
            except Exception as exc:  # pragma: no cover - backend dependent
                supported[bit] = str(exc)
        return supported

    def quantize_model(
        self,
        model_path: str | Path,
        output_dir: str | Path,
        bits: int,
        group_size: int,
        mode: str,
        overwrite: bool = False,
    ) -> QuantizedModelResult:
        model_dir = Path(model_path).expanduser().resolve()
        if not model_dir.exists():
            raise ValueError(f"Model path does not exist: {model_dir}")
        if not model_dir.is_dir():
            raise ValueError(f"Model path is not a directory: {model_dir}")

        out_dir = Path(output_dir).expanduser().resolve()
        weights_file = out_dir / "model.safetensors"
        if weights_file.exists() and not overwrite:
            raise FileExistsError(
                f"Weights already exist at {weights_file}. Use overwrite=True to replace."
            )

        out_dir.mkdir(parents=True, exist_ok=True)

        skipped_dirs = self._copy_model_artifacts(model_dir, out_dir)
        quant_config = self._write_quantized_config(model_dir, out_dir, bits, group_size, mode)

        weights = self._model_loader.load_weights(str(model_dir))
        total_2d = self._count_2d_weights(weights)

        hint = QuantizationHint(bits=bits, group_size=group_size, mode=mode)
        quantized = requantize_weights(weights, self._backend, hint)

        quantized_2d = sum(1 for key in quantized if key.endswith(".scales"))
        skipped_2d = max(0, total_2d - quantized_2d)

        self._backend.save_safetensors(str(weights_file), quantized)
        self._backend.clear_cache()

        return QuantizedModelResult(
            bits=bits,
            group_size=group_size,
            mode=mode,
            output_dir=out_dir,
            weights_file=weights_file,
            total_2d_weights=total_2d,
            quantized_2d_weights=quantized_2d,
            skipped_2d_weights=skipped_2d,
            skipped_dirs=tuple(sorted(skipped_dirs)),
            quantization_config=quant_config,
        )

    def _copy_model_artifacts(self, source: Path, dest: Path) -> list[str]:
        skipped_dirs: list[str] = []
        for item in source.iterdir():
            if item.name == ".modelcypher":
                continue
            if item.is_dir():
                skipped_dirs.append(item.name)
                continue
            if self._is_weight_file(item):
                continue
            shutil.copy2(item, dest / item.name)
        return skipped_dirs

    def _write_quantized_config(
        self,
        source: Path,
        dest: Path,
        bits: int,
        group_size: int,
        mode: str,
    ) -> dict[str, int | str]:
        config_path = source / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {source}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quant_config: dict[str, int | str] = {
            "group_size": group_size,
            "bits": bits,
        }
        if mode:
            quant_config["mode"] = mode
        config["quantization"] = dict(quant_config)
        config["quantization_config"] = dict(quant_config)
        (dest / "config.json").write_text(json.dumps(config, indent=2))
        return quant_config

    @staticmethod
    def _is_weight_file(path: Path) -> bool:
        if path.suffix in _WEIGHT_SUFFIXES:
            return True
        name = path.name
        return any(name.endswith(suffix) for suffix in _WEIGHT_INDEX_SUFFIXES)

    @staticmethod
    def _count_2d_weights(weights: dict[str, object]) -> int:
        total = 0
        for key, value in weights.items():
            if not key.endswith(".weight"):
                continue
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) != 2:
                continue
            total += 1
        return total

