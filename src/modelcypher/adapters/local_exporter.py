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
    def export(
        self,
        *,
        model_path: str,
        adapter_path: str | None,
        output_path: str,
        target_kind: str,
    ) -> dict:
        model_dir = expand_path(model_path)
        target = expand_path(output_path)
        normalized_target = target_kind.lower()

        if normalized_target == "adapter":
            if adapter_path is None:
                raise ValueError("adapter target requires adapter_path")
            self._export_adapter(expand_path(adapter_path), target)
        elif normalized_target == "merged_fp16":
            if adapter_path is None:
                raise ValueError("merged_fp16 target requires adapter_path")
            self._export_merged_fp16(model_dir, expand_path(adapter_path), target)
        else:
            raise ValueError(
                f"Unsupported export target: {target_kind}. "
                "Supported targets: adapter, merged_fp16"
            )

        return {"target_kind": normalized_target, "output_path": str(target)}

    def _export_adapter(self, adapter_dir: Path, output_dir: Path) -> None:
        if not adapter_dir.is_dir():
            raise ValueError(f"Adapter path is not a directory: {adapter_dir}")
        if output_dir.exists():
            raise FileExistsError(f"Export destination already exists: {output_dir}")
        shutil.copytree(adapter_dir, output_dir)

    def _export_merged_fp16(self, model_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
        backend = get_default_backend()
        output_dir.mkdir(parents=True, exist_ok=True)

        model, _tokenizer = backend.load_model(
            str(model_dir),
            adapter_path=str(adapter_dir),
        )
        weights = dict(model.parameters())
        backend.save_safetensors(str(output_dir / "model.safetensors"), weights)
        self._copy_model_artifacts(model_dir, output_dir)
        backend.clear_cache()

    def _copy_model_artifacts(self, model_dir: Path, output_dir: Path) -> None:
        skipped_suffixes = {
            ".safetensors",
            ".bin",
            ".pt",
            ".ckpt",
            ".npz",
            ".gguf",
        }
        for item in model_dir.iterdir():
            if item.name == ".modelcypher":
                continue
            if item.is_dir():
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
                continue
            if item.suffix in skipped_suffixes or item.name.endswith(".index.json"):
                continue
            shutil.copy2(item, output_dir / item.name)
