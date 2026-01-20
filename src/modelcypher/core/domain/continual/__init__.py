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
Continual Learning Module - Geometric inference with real-time adaptation.

This module implements two fundamental capabilities:

1. **Metacognitive Generation**: The model can observe its own entropy/confidence
   and decide whether to emit a token, think longer, or request clarification.

2. **Inference-Time Adaptation**: The model can encode new knowledge into its
   null-space during inference, enabling continual learning without forgetting.

The core insight: The null-space projection algorithm used for model merging
is the same mathematical operation needed for continual learning. Merging
combines coordinate systems across models; learning extends the coordinate
system into unused dimensions over time.

Architecture:
    EntropyAnalyzer      - Compute entropy and derivatives from logits
    DecisionGate         - Route between emit/think_more/clarify
    ConfidenceEmbedding  - Embed entropy as input to residual stream
    ActivationBuffer     - Rolling buffer with incremental SVD
    NullSpaceTracker     - Track used vs available dimensions
    SurpriseDetector     - Identify novel information for encoding
    KnowledgeEncoder     - Compute and project weight deltas
    GeometricInference   - Unified inference loop with feedback

References:
    - Nested Learning / Hope Architecture (NeurIPS 2025)
    - GNSP: Gradient Null Space Projection (arXiv:2507.19839)
    - Test-Time Training for LLMs (arXiv:2505.20633)
    - Emergent Introspective Awareness (Anthropic, 2025)
"""

from __future__ import annotations

# Lazy loading for all submodules
_SUBMODULE_MAP = {
    "EntropyAnalyzer": "entropy_analyzer",
    "EntropyState": "entropy_analyzer",
    "DecisionGate": "decision_gate",
    "Decision": "decision_gate",
    "DecisionAction": "decision_gate",
    "ConfidenceEmbedding": "confidence_embedding",
    "EmbeddingConfig": "confidence_embedding",
    "ActivationBuffer": "activation_buffer",
    "BufferStats": "activation_buffer",
    "NullSpaceTracker": "null_space_tracker",
    "NullSpaceState": "null_space_tracker",
    "SurpriseDetector": "surprise_detector",
    "SurpriseEvent": "surprise_detector",
    "KnowledgeEncoder": "knowledge_encoder",
    "EncodingResult": "knowledge_encoder",
    "UpdateFrequency": "knowledge_encoder",
    "GeometricInference": "geometric_inference",
    "InferenceConfig": "geometric_inference",
    "InferenceState": "geometric_inference",
    "ManifoldCompletion": "manifold_completion",
    "CompletionConfig": "manifold_completion",
    "CompletionStep": "manifold_completion",
}


def __getattr__(name: str):
    if name in _SUBMODULE_MAP:
        import importlib

        module_name = _SUBMODULE_MAP[name]
        module = importlib.import_module(
            f".{module_name}", "modelcypher.core.domain.continual"
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_SUBMODULE_MAP.keys())
