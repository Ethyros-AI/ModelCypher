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

"""Regression tests to catch type/API name drift across core modules."""

from __future__ import annotations

import importlib

import pytest

SYMBOL_CONTRACTS: list[tuple[str, tuple[str, ...]]] = [
    (
        "modelcypher.core.domain.models",
        (
            "ModelInfo",
            "CheckpointRecord",
            "TrainingJob",
            "EvaluationResult",
            "CompareCheckpointResult",
            "CompareSession",
        ),
    ),
    (
        "modelcypher.core.domain.geometry.types",
        (
            "PermutationAlignmentResult",
            "RebasinResult",
            "DetectedConcept",
            "DetectionResult",
            "ConceptComparisonResult",
            "CompositionCategory",
            "CompositionProbe",
            "CompositionAnalysis",
            "GeometryConsistencyResult",
            "ProcrustesResult",
            "PairwiseProcrustesResult",
        ),
    ),
    (
        "modelcypher.core.domain.training.types",
        (
            "ComputePrecision",
            "TrainingStatus",
            "FineTuneType",
            "PreflightResult",
            "Hyperparameters",
            "LoRASettings",
            "TrainingSpec",
            "TrainingProgress",
            "CheckpointMetadata",
        ),
    ),
    (
        "modelcypher.core.domain.entropy.metrics_ring_buffer",
        (
            "MetricSample",
            "EventType",
            "MetricEvent",
            "MetricsRingBuffer",
            "EventMarkerBuffer",
        ),
    ),
    (
        "modelcypher.experimental.vocabulary.alignment_map",
        (
            "AlignmentQuality",
            "TokenAlignment",
            "VocabularyAlignmentMap",
            "TokenizerComparisonResult",
        ),
    ),
    (
        "modelcypher.experimental.vocabulary.cross_vocab_merger",
        (
            "AlignmentMethod",
            "CrossVocabMergeResult",
            "CrossVocabMerger",
        ),
    ),
    (
        "modelcypher.ports.storage",
        (
            "ModelStore",
            "JobStore",
            "EvaluationStore",
            "CompareStore",
            "ManifoldProfileStore",
        ),
    ),
    (
        "modelcypher.ports.training",
        (
            "LoRALayerConfig",
            "ParameterInfo",
            "GradientInfo",
            "TrainingPort",
            "TrainingEngine",
        ),
    ),
    (
        "modelcypher.ports.hub",
        (
            "HubAdapterPort",
        ),
    ),
    (
        "modelcypher.cli.presenters",
        (
            "model_payload",
            "evaluation_list_payload",
            "evaluation_detail_payload",
            "compare_list_payload",
            "compare_detail_payload",
            "compare_checkpoint_payload",
            "model_search_payload",
            "model_search_result_payload",
        ),
    ),
]


DRIFT_PRONE_MODULES: tuple[str, ...] = (
    "modelcypher.cli.presenters",
    "modelcypher.core.use_cases.evaluation_service",
    "modelcypher.core.use_cases.geometry_metrics_service",
    "modelcypher.core.use_cases.inference.comparison",
    "modelcypher.core.use_cases.lora_safety_service",
    "modelcypher.core.domain.entropy.entropy_tracker",
    "modelcypher.core.domain.geometry.bilm_probe",
    "modelcypher.core.domain.geometry.fingerprint_cache",
    "modelcypher.core.domain.geometry.gate_detector",
    "modelcypher.core.domain.geometry.geometry_validation_suite",
    "modelcypher.core.domain.geometry.gram_aligner",
    "modelcypher.core.domain.geometry.invariant_layer_mapper",
    "modelcypher.core.domain.geometry.manifold_profile",
    "modelcypher.core.domain.geometry.manifold_stitcher",
    "modelcypher.core.domain.geometry.model_profile",
    "modelcypher.core.domain.geometry.representation_consistency",
    "modelcypher.core.domain.geometry.sparse_region_validator",
    "modelcypher.ports.storage",
    "modelcypher.ports.training",
    "modelcypher.ports.hub",
)


@pytest.mark.parametrize(("module_name", "symbol_names"), SYMBOL_CONTRACTS)
def test_symbol_contracts_stay_stable(module_name: str, symbol_names: tuple[str, ...]):
    """Canonical type names must remain available for downstream imports."""
    module = importlib.import_module(module_name)
    missing = [name for name in symbol_names if not hasattr(module, name)]
    assert not missing, f"{module_name} missing symbols: {missing}"


@pytest.mark.parametrize("module_name", DRIFT_PRONE_MODULES)
def test_drift_prone_modules_import_without_name_errors(module_name: str):
    """Importing historically brittle modules should not raise name errors."""
    module = importlib.import_module(module_name)
    assert module is not None
