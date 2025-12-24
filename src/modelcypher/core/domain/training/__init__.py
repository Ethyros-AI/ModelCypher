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

# Training Domain Package
#
# Platform-Specific Implementations:
# - MLX (macOS): *_mlx.py files
# - CUDA (Linux): *_cuda.py files
# - JAX (TPU/GPU): *_jax.py files
# - Use _platform module for automatic selection

from .types import (
    CheckpointMetadata,
    Hyperparameters,
    LoRAConfig,
    PreflightResult,
    TrainingConfig,
    TrainingStatus,
)
from .validation import TrainingHyperparameterValidator
from .resources import TrainingResourceGuard, ResourceIntensiveOperation
from .checkpoints_mlx import CheckpointManager
from .engine_mlx import TrainingEngine, TrainingError
from .checkpoint_models import (
    CheckpointErrorKind,
    OptimizerStateMetadata,
    FineTunedModelMetadata,
    ModelArchitectureConfig,
    CheckpointMetadataV2,
    RecoveryInfo,
)
from .exceptions import CheckpointError
from .checkpoint_validation import CheckpointValidation
from .checkpoint_recovery import CheckpointRecovery
from .checkpoint_retention import CheckpointRetention
from .checkpoint_persistence import CheckpointPersistence
from .model_architecture_heuristics import ModelArchitectureHeuristics
from .training_benchmark import (
    BenchmarkResults,
    BenchmarkComparison,
    TrainingBenchmark,
)
from .training_notifications import (
    TrainingEventKind,
    TrainingProgress,
    TrainingEvent,
    TrainingEventHandler,
    AsyncTrainingEventHandler,
    TrainingEventBus,
    get_training_event_bus,
    reset_training_event_bus,
)

# Additional training modules (previously not exported)
from .evaluation_mlx import *  # noqa: F401,F403
from .geometric_training_metrics import *  # noqa: F401,F403
from .geometric_metrics_collector import *  # noqa: F401,F403
from .gradient_smoothness_estimator import *  # noqa: F401,F403
from .hessian_estimator import *  # noqa: F401,F403
from .idle_training_scheduler import *  # noqa: F401,F403
from .lora_mlx import *  # noqa: F401,F403
from .loss_landscape_mlx import *  # noqa: F401,F403
from .scheduling import *  # noqa: F401,F403

# Platform selection (auto-detects MLX on macOS, CUDA on Linux, JAX on TPU)
from ._platform import (
    get_training_platform,
    get_training_engine,
    get_checkpoint_manager,
    get_evaluation_engine,
)
