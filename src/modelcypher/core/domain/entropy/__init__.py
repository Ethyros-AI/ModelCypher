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

from .baseline_verification_probe import (
    AdversarialFlag,
    BaselineComparison,
    BaselineVerificationProbe,
    DeltaSample,
    EntropyBaseline,
    PromptResult,
    VerificationConfiguration,
    VerificationResult,
)

# Additional entropy modules (previously not exported)
from .chunk_entropy_analyzer import *  # noqa: F401,F403
from .conflict_score import (
    ConflictAnalysis,
    ConflictScoreCalculator,
    ConflictScoreResult,
)
from .conversation_entropy_tracker import *  # noqa: F401,F403
from .entropy_delta_sample import (
    BaselineDistribution,
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)
from .entropy_delta_tracker import (
    EntropyDeltaTracker,
    EntropyDeltaTrackerConfig,
    PendingEntropyData,
)
from .entropy_pattern_detector import (
    DetectorConfiguration,
    DistressAction,
    DistressDetectionResult,
    EntropyPattern,
    EntropyPatternAnalyzer,
)
from .entropy_tracker import (
    DistressDetection,
    EntropyPatternDetector,
    EntropySample,
    EntropyTracker,
    EntropyTrackerConfig,
    EntropyTransition,
    EntropyWindow,
    EntropyWindowStatus,
    ModelStateClassifier,
    PatternConfig,
)
from .entropy_window import (
    EntropyWindow as EntropyWindowV2,  # Renamed to avoid conflict with entropy_tracker version
)
from .entropy_window import (
    EntropyWindowConfig,
)
from .entropy_window import (
    EntropyWindowStatus as EntropyWindowStatusV2,
)
from .geometric_alignment import *  # noqa: F401,F403
from .hidden_state_extractor import (
    CapturedState,
    ExtractionSummary,
    ExtractorConfig,
    HiddenStateExtractor,
)
from .layer_entropy_projector import (
    LayerEntropyProjector,
    LayerEntropyResult,
    ModelLayerEntropyProfile,
)
from .logit_entropy_calculator import (
    EntropyThresholds,
    LogitEntropyCalculator,
    LogitEntropySample,
)
from .metrics_ring_buffer import (
    EventMarkerBuffer,
    EventType,
    MetricEvent,
    MetricSample,
    MetricsRingBuffer,
)

# model_state_classifier exports (ModelState enum removed in Pure Geometry refactor)
from .model_state_classifier import (
    CalibratedBaseline,
    ClassificationResult,
    ClassificationSnapshot,
    EntropyStateThresholds,
    ModelStateSignals,
)
from .model_state_classifier import (
    ModelStateClassifier as CalibratedModelStateClassifier,
)
from .sep_probe import (
    LayerProbeWeights,
    PredictionResult,
    ProbeWeightsBundle,
    SEPProbe,
    SEPProbeConfig,
    SEPProbeError,
)
