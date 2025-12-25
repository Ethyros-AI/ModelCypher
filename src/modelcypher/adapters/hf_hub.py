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
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from modelcypher.core.domain.models import ModelInfo
from modelcypher.utils.paths import ensure_dir, expand_path


class HfHubAdapter:
    def __init__(self, base_dir: str | None = None) -> None:
        default_base = os.environ.get("HF_HOME", "~/.cache/huggingface")
        self.base_dir = ensure_dir(base_dir or default_base)

    def fetch(
        self,
        repo_id: str,
        revision: str = "main",
        local_dir: str | None = None,
    ) -> str:
        """Download model from Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            Hugging Face repository identifier.
        revision : str
            Git revision (branch, tag, or commit hash).
        local_dir : str or None
            Optional local directory to save the model.

        Returns
        -------
        str
            Path to the downloaded model directory.
        """
        local_path = expand_path(local_dir) if local_dir else None
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )
        return str(Path(path))

    def detect_architecture(self, model_path: str) -> str | None:
        """Detect model architecture from config.json.

        Parameters
        ----------
        model_path : str
            Path to model directory containing config.json.

        Returns
        -------
        str or None
            Model type string if found, None otherwise.
        """
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return None
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        model_type = data.get("model_type")
        if not model_type:
            return None
        return str(model_type)

    @staticmethod
    def build_model_info(
        alias: str,
        path: str,
        architecture: str,
        parameter_count: int | None = None,
    ) -> ModelInfo:
        """Build ModelInfo record from model metadata.

        Parameters
        ----------
        alias : str
            Model alias/identifier.
        path : str
            Path to model directory.
        architecture : str
            Model architecture type.
        parameter_count : int or None
            Number of parameters in the model.

        Returns
        -------
        ModelInfo
            Populated ModelInfo dataclass.
        """
        resolved = expand_path(path)
        size_bytes = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
        return ModelInfo(
            id=alias,
            alias=alias,
            architecture=architecture,
            format="safetensors",
            path=str(resolved),
            size_bytes=size_bytes,
            parameter_count=parameter_count,
            is_default_chat=False,
            created_at=datetime.utcnow(),
        )
