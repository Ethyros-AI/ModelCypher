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

"""
Agents Package.

Agent infrastructure for tracing, evaluation, and action handling.

Atlas probes are now loaded from JSON files in data/probes/.
Use UnifiedAtlasInventory.all_probes() to get all probes.
"""

# Agent action handling
from .agent_action import (
    ActionClarification,
    ActionDeferral,
    ActionExtraction,
    ActionKind,
    ActionRefusal,
    ActionResponse,
    ActionToolCall,
    AgentActionEnvelope,
    ResponseFormat,
)
from .agent_action_validator import (
    AgentActionValidationResult,
    AgentActionValidator,
)

# Agent evaluation
from .agent_eval_suite_engine import (
    AgentAction,
    AgentActionKind,
    AgentEvalCase,
    AgentEvalCaseCategory,
    AgentEvalCaseProfile,
    AgentEvalScoringEngine,
    AggregateScores,
    CaseResult,
    EvalCaseConstraints,
    EvalRunReport,
    Expected,
    ExpectedOption,
    ExpectedToolSpec,
    ScoredOutput,
    ToolCall,
)

# Agent utilities
from .agent_json_extractor import AgentJSONSnippetExtractor
from .agent_prompt_sanitizer import (
    AgentMessage,
    AgentPromptSanitizationResult,
    AgentPromptSanitizer,
    AgentRole,
    AgentSystemPromptPolicy,
)

# Agent tracing
from .agent_trace import (
    AgentTrace,
    InferenceMetrics,
    PayloadDigest,
    SchemaValidation,
    TraceKind,
    TraceSource,
    TraceSpan,
    TraceStatus,
    TraceStore,
    TraceSummary,
)
from .agent_trace_analytics import (
    ActionCompliance,
    AgentTraceAnalytics,
    EntropyBucket,
    EntropyBuckets,
    MessageCount,
)
from .agent_trace_sanitizer import AgentTraceSanitizer
from .agent_trace_value import (
    AgentTraceValue,
    AgentTraceValueKind,
    ImportOptions,
)
from .monocle_trace_importer import (
    ImportResult,
    MonocleTraceImporter,
    TraceImportError,
)

# Identity and LoRA experts
from .intrinsic_identity_rules import IntrinsicIdentityRules
from .lora_expert import (
    AdapterActivator,
    AgentIntent,
    AgentQuery,
    CompositeAdapterActivator,
    LoRAExpert,
    SkillCategory,
    SkillComplexity,
)
from .task_diversion_detector import (
    LexicalStopWords,
    LexicalTokenizer,
    TaskDiversionAssessment,
    TaskDiversionDetector,
    TaskDiversionMethod,
)

# Base atlas infrastructure
from .atlas_base import (
    AtlasConcept,
    BaseAtlas,
    BaseAtlasSignature,
)

# Unified atlas system (JSON-based probes)
from .unified_atlas import (
    ALL_ATLAS_SOURCES,
    AFFECTIVE_DOMAINS,
    AtlasDomain,
    AtlasProbe,
    AtlasSource,
    COMPUTATIONAL_DOMAINS,
    DEFAULT_ATLAS_SOURCES,
    LINGUISTIC_DOMAINS,
    MATHEMATICAL_DOMAINS,
    MORAL_DOMAINS,
    MultiAtlasTriangulationScore,
    MultiAtlasTriangulationScorer,
    PHILOSOPHICAL_DOMAINS,
    SAFETY_DOMAINS,
    SPATIOTEMPORAL_DOMAINS,
    UnifiedAtlasInventory,
    get_probe_ids,
)

# Probe loader for JSON-based probes
from .probe_loader import (
    get_probe_count_by_domain,
    load_all_probes,
    load_probes_from_file,
)
