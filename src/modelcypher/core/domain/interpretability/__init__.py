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
Mechanistic Interpretability Tools.

State-of-the-art interpretability tools for understanding LLM internals:
- Sparse Autoencoders (SAE): Extract monosemantic features from polysemantic neurons
- Transcoders: Cross-layer MLP replacement for circuit tracing
- Activation Patching: Causal intervention to localize computation
- Feature Steering: Modify behavior via activation intervention
- Crosscoders: Model diffing between base and fine-tuned models

All tools are:
- Backend-agnostic (MLX/JAX/CUDA via Backend protocol)
- Geodesic-principled (geodesic distances, not Euclidean)
- Threshold-free (all values derived from data)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = {
    "sae",
    "sae_training",
    "transcoder",
    "activation_patching",
    "feature_steering",
    "crosscoder",
}

_ATTR_TO_MODULE = {
    # Sparse Autoencoders
    "SparseAutoencoder": ("sae", "SparseAutoencoder"),
    "SAEConfig": ("sae", "SAEConfig"),
    "SAEEncodingResult": ("sae", "SAEEncodingResult"),
    "SAEWeights": ("sae", "SAEWeights"),
    # SAE Training
    "SAETrainer": ("sae_training", "SAETrainer"),
    "SAETrainingConfig": ("sae_training", "SAETrainingConfig"),
    "SAETrainingResult": ("sae_training", "SAETrainingResult"),
    # Transcoders
    "Transcoder": ("transcoder", "Transcoder"),
    "TranscoderConfig": ("transcoder", "TranscoderConfig"),
    "TranscoderResult": ("transcoder", "TranscoderResult"),
    # Activation Patching
    "ActivationPatcher": ("activation_patching", "ActivationPatcher"),
    "PatchSpec": ("activation_patching", "PatchSpec"),
    "PatchingResult": ("activation_patching", "PatchingResult"),
    # Feature Steering
    "FeatureSteering": ("feature_steering", "FeatureSteering"),
    "SteeringVector": ("feature_steering", "SteeringVector"),
    "SteeringConfig": ("feature_steering", "SteeringConfig"),
    "SteeringResult": ("feature_steering", "SteeringResult"),
    # Crosscoders
    "Crosscoder": ("crosscoder", "Crosscoder"),
    "CrosscoderConfig": ("crosscoder", "CrosscoderConfig"),
    "ModelDiffResult": ("crosscoder", "ModelDiffResult"),
}


def __getattr__(name: str):
    """Lazy load submodules and commonly used attributes."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _ATTR_TO_MODULE:
        module_name, attr_name = _ATTR_TO_MODULE[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules and attributes."""
    return list(_SUBMODULES) + list(_ATTR_TO_MODULE.keys())


if TYPE_CHECKING:
    from .sae import SAEConfig, SAEEncodingResult, SAEWeights, SparseAutoencoder
