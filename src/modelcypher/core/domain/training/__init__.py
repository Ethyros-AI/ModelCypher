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
# Platform-specific implementations live behind the backend and infrastructure
# factories.

from .checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
    FineTunedModelMetadata,
    ModelArchitectureSpec,
    OptimizerStateMetadata,
    RecoveryInfo,
)
from .checkpoint_persistence import CheckpointPersistence
from .checkpoint_recovery import CheckpointRecovery
from .checkpoint_retention import CheckpointRetention
from .checkpoint_validation import CheckpointValidation

# Additional training modules (previously not exported)
from .exceptions import CheckpointError
from .geometric_early_stopping import check_loss_stable
from .geometric_metrics_collector import *  # noqa: F401,F403
from .geometric_training_metrics import *  # noqa: F401,F403
from .gradient_smoothness_estimator import *  # noqa: F401,F403
from .hessian_estimator import *  # noqa: F401,F403
from .hyperparameter_validation import TrainingHyperparameterValidator
from .mass_step_size import (
    CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    CONTROLLER_MODE_BEHAVIORAL_PROBE,
    CONTROLLER_MODE_STRUCTURAL_OBSERVE,
    OPTIMIZER_MODE_ADAMW_GEOMETRIC,
    OPTIMIZER_MODE_ADAMW_MATCHED_TRACE,
    OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
    BehavioralStateMeasurement,
    ClosedLoopControlDecision,
    ControllerLayerMeasurement,
    ControllerReplayDecision,
    ControllerStepTrace,
    DerivedClosedLoopLaw,
    apply_sqrt_n_epoch_correction,
    apply_validation_backoff,
    compute_closed_loop_trigger_reasons,
    compute_conformal_margin_rate,
    compute_per_step_rates,
    compute_reinforce_budget,
    controller_precision_floor,
    derive_spectral_ceiling,
    evaluate_closed_loop_law,
    replay_controller_trace,
    select_closed_loop_target_layer,
    validate_controller_mode,
    validate_optimizer_research_mode,
    verify_bounded_gain,
)
from .resources import ResourceIntensiveOperation, TrainingResourceGuard
from .scaled_gd import precondition_lora_gradients
from .spectral_budget import compute_budget_ratios, is_budget_exhausted
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
    get_training_event_bus,
    reset_training_event_bus,
)
from .training_notifications import (
    TrainingNotificationProgress as TrainingProgress,
)
from .types import (
    CheckpointMetadata,
    ComputePrecision,
    FineTuneType,
    Hyperparameters,
    LoRASettings,
    PreflightResult,
    TrainingSpec,
    TrainingStatus,
)
