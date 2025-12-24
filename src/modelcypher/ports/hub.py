"""Hub adapter port for model repository operations.

This port abstracts model hub operations (HuggingFace, etc.) for:
- Downloading models from remote repositories
- Detecting model architectures
- Building ModelInfo from local paths
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modelcypher.core.domain.models import ModelInfo

@runtime_checkable
class HubAdapterPort(Protocol):
    """Port for model hub operations (HuggingFace, etc.)."""

    def fetch(
        self,
        repo_id: str,
        revision: str = "main",
        local_dir: str | None = None,
    ) -> str:
        """Download model from hub to local path.

        Args:
            repo_id: Repository identifier (e.g., "meta-llama/Llama-2-7b")
            revision: Git revision (branch, tag, or commit)
            local_dir: Optional local directory to download to

        Returns:
            Local path where model was downloaded
        """
        ...

    def detect_architecture(self, model_path: str) -> str | None:
        """Detect model architecture from config.json.

        Args:
            model_path: Path to model directory

        Returns:
            Architecture string (e.g., "llama", "qwen2") or None
        """
        ...

    def build_model_info(
        self,
        alias: str,
        path: str,
        architecture: str,
        parameter_count: int | None = None,
    ) -> "ModelInfo":
        """Build ModelInfo from local path.

        Args:
            alias: Display name for the model
            path: Local path to model directory
            architecture: Architecture type string
            parameter_count: Optional parameter count

        Returns:
            ModelInfo domain object
        """
        ...
