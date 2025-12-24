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

"""Entropy domain models for disagreement and safety metrics."""
from .entropy_tracker import (
    ModelState,
    StateTransition,
    EntropySample,
    EntropyLevel,
    DistressDetection,
    EntropyWindow,
    EntropyWindowStatus,
    ModelStateClassifier,
    ClassifierThresholds,
    LogitEntropyCalculator,
    EntropyPatternDetector,
    PatternConfig,
    EntropyTracker,
    EntropyTrackerConfig,
)
from .hidden_state_extractor import (
    HiddenStateExtractor,
    ExtractorConfig,
    CapturedState,
    ExtractionSummary,
)
from .sep_probe import (
    SEPProbe,
    SEPProbeConfig,
    SEPProbeError,
    LayerProbeWeights,
    ProbeWeightsBundle,
    PredictionResult,
)
from .entropy_pattern_detector import (
    EntropyTrend,
    DistressAction,
    DetectorConfiguration,
    EntropyPattern,
    DistressDetectionResult,
    EntropyPatternAnalyzer,
)
from .baseline_verification_probe import (
    VerificationVerdict,
    EntropyBaseline,
    BaselineComparison,
    PromptResult,
    AdversarialFlag,
    VerificationResult,
    VerificationConfiguration,
    DeltaSample,
    BaselineVerificationProbe,
)
from .entropy_delta_sample import (
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)
from .entropy_delta_tracker import (
    EntropyDeltaTracker,
    EntropyDeltaTrackerConfig,
    PendingEntropyData,
)
from .metrics_ring_buffer import (
    MetricSample,
    MetricEvent,
    EventType,
    MetricsRingBuffer,
    EventMarkerBuffer,
)
from .conflict_score import (
    ConflictScoreCalculator,
    ConflictScoreResult,
    ConflictLevel,
    ConflictAnalysis,
)
from .entropy_window import (
    EntropyWindow as EntropyWindowV2,  # Renamed to avoid conflict with entropy_tracker version
    EntropyWindowConfig,
    EntropyWindowStatus as EntropyWindowStatusV2,
    EntropyLevel as EntropyLevelV2,
)
from .logit_entropy_calculator import (
    LogitEntropyCalculator,
    EntropyThresholds,
    EntropyLevel as EntropyLevelCalculator,
    LogitEntropySample,
)

# Additional entropy modules (previously not exported)
from .chunk_entropy_analyzer import *  # noqa: F401,F403
from .conversation_entropy_tracker import *  # noqa: F401,F403
from .geometric_alignment import *  # noqa: F401,F403
# model_state_classifier excluded - has conflicting ModelState class
from .sep_probe_online_training import *  # noqa: F401,F403
