# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Scientific Validation Protocol - Shared Configuration
#
# This module ensures reproducibility across all experiments by:
# 1. Fixing random seeds
# 2. Recording model hashes
# 3. Logging all hyperparameters
# 4. Capturing hardware specs

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Model paths
LOCAL_FIXTURES = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / ".models"
SMOLLM_PATH = LOCAL_FIXTURES / "HuggingFaceTB--SmolLM-135M"
LFM2_PATH = LOCAL_FIXTURES / "mlx-community--LFM2-350M-MLX-bf16"

# External volume (optional)
CODECYPHER_VOLUME = Path("/Volumes/codecypher/models")


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    experiment_name: str
    experiment_id: str = field(default_factory=lambda: f"{int(time.time())}")
    random_seed: int = RANDOM_SEED
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model info
    source_model_path: str = ""
    target_model_path: str = ""
    source_model_hash: str = ""
    target_model_hash: str = ""

    # Hardware
    platform_info: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    backend_name: str = ""
    backend_version: str = ""

    # Hyperparameters (experiment-specific)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "raw_data": self.raw_data,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
        }

    def save(self, path: Path) -> None:
        """Save result to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model config for reproducibility."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return "unknown"


def get_backend_info(backend: "Backend") -> tuple[str, str]:
    """Get backend name and version."""
    name = type(backend).__name__
    version = getattr(backend, "__version__", "unknown")
    return name, version


def setup_experiment(
    name: str,
    source_path: Path | str,
    target_path: Path | str,
    backend: "Backend",
    hyperparameters: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Set up experiment with full reproducibility info."""
    source_path = Path(source_path)
    target_path = Path(target_path)

    backend_name, backend_version = get_backend_info(backend)

    # Set random seed
    backend.random_seed(RANDOM_SEED)

    config = ExperimentConfig(
        experiment_name=name,
        source_model_path=str(source_path),
        target_model_path=str(target_path),
        source_model_hash=compute_model_hash(source_path),
        target_model_hash=compute_model_hash(target_path),
        backend_name=backend_name,
        backend_version=backend_version,
        hyperparameters=hyperparameters or {},
    )

    logger.info(
        "Experiment %s initialized: source=%s target=%s seed=%d",
        name, source_path.name, target_path.name, RANDOM_SEED
    )

    return config


def ensure_output_dir(experiment_name: str) -> Path:
    """Ensure output directory exists for experiment."""
    output_dir = Path(__file__).parent.parent / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
