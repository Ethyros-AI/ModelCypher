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

# Platform selection (auto-detects MLX on macOS, CUDA on Linux, JAX on TPU)
from ._platform import (
    get_checkpoint_manager,
    get_evaluation_engine,
    get_training_engine,
    get_training_platform,
)
from .checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
    FineTunedModelMetadata,
    ModelArchitectureConfig,
    OptimizerStateMetadata,
    RecoveryInfo,
)
from .checkpoint_persistence import CheckpointPersistence
from .checkpoint_recovery import CheckpointRecovery
from .checkpoint_retention import CheckpointRetention
from .checkpoint_validation import CheckpointValidation

# Additional training modules (previously not exported)
from .exceptions import CheckpointError
from .geometric_metrics_collector import *  # noqa: F401,F403
from .geometric_training_metrics import *  # noqa: F401,F403
from .gradient_smoothness_estimator import *  # noqa: F401,F403
from .hessian_estimator import *  # noqa: F401,F403
from .idle_training_scheduler import *  # noqa: F401,F403
from .model_architecture_heuristics import ModelArchitectureHeuristics
from .resources import ResourceIntensiveOperation, TrainingResourceGuard
from .scheduling import *  # noqa: F401,F403
from .training_benchmark import (
    BenchmarkComparison,
    BenchmarkResults,
    TrainingBenchmark,
)
from .training_notifications import (
    AsyncTrainingEventHandler,
    TrainingEvent,
    TrainingEventBus,
    TrainingEventHandler,
    TrainingEventKind,
    TrainingProgress,
    get_training_event_bus,
    reset_training_event_bus,
)
from .types import (
    CheckpointMetadata,
    Hyperparameters,
    LoRAConfig,
    PreflightResult,
    TrainingConfig,
    TrainingStatus,
)
from .validation import TrainingHyperparameterValidator

_training_platform = get_training_platform()

if _training_platform == "mlx":
    from .checkpoints_mlx import CheckpointManager
    from .engine_mlx import TrainingEngine, TrainingError
    from .evaluation_mlx import *  # noqa: F401,F403
    from .evaluation_mlx import EvaluationEngine
    from .lora_mlx import *  # noqa: F401,F403
    from .lora_mlx import LoRAConfig as LoRAConfig
    from .loss_landscape_mlx import *  # noqa: F401,F403
    from .loss_landscape_mlx import LossLandscapeComputer
elif _training_platform == "cuda":
    from .checkpoints_cuda import CheckpointManagerCUDA as CheckpointManager
    from .engine_cuda import TrainingEngineCUDA as TrainingEngine
    from .engine_cuda import TrainingErrorCUDA as TrainingError
    from .evaluation_cuda import *  # noqa: F401,F403
    from .evaluation_cuda import EvaluationEngineCUDA as EvaluationEngine
    from .lora_cuda import *  # noqa: F401,F403
    from .lora_cuda import LoRAConfigCUDA as LoRAConfig
    from .loss_landscape_cuda import *  # noqa: F401,F403
    from .loss_landscape_cuda import LossLandscapeComputerCUDA as LossLandscapeComputer
elif _training_platform == "jax":
    from .checkpoints_jax import CheckpointManagerJAX as CheckpointManager
    from .engine_jax import TrainingEngineJAX as TrainingEngine
    from .engine_jax import TrainingErrorJAX as TrainingError
    from .evaluation_jax import *  # noqa: F401,F403
    from .evaluation_jax import EvaluationEngineJAX as EvaluationEngine
    from .lora_jax import *  # noqa: F401,F403
    from .lora_jax import LoRAConfigJAX as LoRAConfig
    from .loss_landscape_jax import *  # noqa: F401,F403
    from .loss_landscape_jax import LossLandscapeComputerJAX as LossLandscapeComputer
else:
    CheckpointManager = None
    TrainingEngine = None
    TrainingError = None
    EvaluationEngine = None
    LossLandscapeComputer = None
