"""EXPERIMENTAL: Baranov Sleeping-LLM Replication Framework.

This module implements data models, evaluator interfaces, and artifact
scaffolding for independent replication of Baranov claims under ModelCypher
research constraints.

This is research code -- NOT validated for production use.
No production CLI promotion in this phase.

See:
    docs/research/baranov_sleeping_llm_intake_2026_02.md
    docs/research/baranov_replication_protocol_2026_02.md
"""

from modelcypher.experimental.baranov.artifact_writer import (
    write_manifest_json,
    write_metrics_csv,
    write_summary_stub,
)
from modelcypher.experimental.baranov.consolidation_tracker import (
    FactConsolidationTracker,
)
from modelcypher.experimental.baranov.decision import (
    ModeVerdict,
    ModelVerdict,
    NoiseFloor,
    TrackADecision,
    compute_mode_verdict,
    compute_model_verdict,
    compute_noise_floor,
    compute_track_a_decision,
)
from modelcypher.experimental.baranov.fact_dataset import (
    fact_to_training_text,
    facts_to_training_samples,
    write_fact_training_jsonl,
)
from modelcypher.experimental.baranov.geometry_measurement import (
    CKADriftResult,
    GeometrySnapshot,
    collect_probe_activations,
    compute_cka_drift,
)
from modelcypher.experimental.baranov.edit_applicator import EditApplicator
from modelcypher.experimental.baranov.outer_product_editor import OuterProductEditor
from modelcypher.experimental.baranov.manifest import (
    CodeInfo,
    ControlFlags,
    DataHashes,
    ModelInfo,
    PreRegisteredDecision,
    REQUIRED_METRIC_KEYS,
    ReplicationManifest,
    validate_manifest,
)
from modelcypher.experimental.baranov.models import (
    ConsolidationStage,
    EditState,
    EditStatus,
    FactTriple,
    VALID_TRANSITIONS,
)
from modelcypher.experimental.baranov.simple_recall_evaluator import (
    SimpleRecallEvaluator,
)
from modelcypher.experimental.baranov.recall_evaluator import (
    GenerateFn,
    RecallAggregate,
    RecallEvaluator,
    RecallMode,
    RecallOutcome,
    RecallResult,
    compute_recall_aggregate,
)

__all__ = [
    "CKADriftResult",
    "CodeInfo",
    "ConsolidationStage",
    "ControlFlags",
    "DataHashes",
    "EditApplicator",
    "EditState",
    "EditStatus",
    "FactConsolidationTracker",
    "FactTriple",
    "GenerateFn",
    "GeometrySnapshot",
    "ModeVerdict",
    "ModelInfo",
    "ModelVerdict",
    "NoiseFloor",
    "OuterProductEditor",
    "PreRegisteredDecision",
    "REQUIRED_METRIC_KEYS",
    "RecallAggregate",
    "RecallEvaluator",
    "RecallMode",
    "RecallOutcome",
    "RecallResult",
    "ReplicationManifest",
    "SimpleRecallEvaluator",
    "TrackADecision",
    "VALID_TRANSITIONS",
    "collect_probe_activations",
    "compute_cka_drift",
    "compute_mode_verdict",
    "compute_model_verdict",
    "compute_noise_floor",
    "compute_recall_aggregate",
    "compute_track_a_decision",
    "fact_to_training_text",
    "facts_to_training_samples",
    "validate_manifest",
    "write_fact_training_jsonl",
    "write_manifest_json",
    "write_metrics_csv",
    "write_summary_stub",
]
