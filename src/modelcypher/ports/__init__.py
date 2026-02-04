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
Ports (Interfaces) for ModelCypher Adapters

Backend is the main abstraction. Other ports exist only where truly needed.
"""

from modelcypher.ports.activation_store import ActivationStore
from modelcypher.ports.adapter_weights import AdapterWeightsLoader
from modelcypher.ports.backend import Array, Backend
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.ports.bridge_store import BridgeStore
from modelcypher.ports.exporter import Exporter
from modelcypher.ports.hub import HubAdapterPort
from modelcypher.ports.inference import HiddenStateEngine, InferenceEngine
from modelcypher.ports.model_loader import ModelLoaderPort
from modelcypher.ports.model_search import ModelSearchService
from modelcypher.ports.multimodal import MultiModalEmbeddingPort
from modelcypher.ports.system_probe import SystemProbePort
from modelcypher.ports.storage import (
    CompareStore,
    EvaluationStore,
    JobStore,
    ManifoldProfileStore,
    ModelStore,
)
from modelcypher.ports.training import TrainingEngine

__all__ = [
    "Backend",
    "Array",
    "InferenceEngine",
    "HiddenStateEngine",
    "EmbeddingProvider",
    "TrainingEngine",
    "Exporter",
    "ModelSearchService",
    "HubAdapterPort",
    "ModelLoaderPort",
    "MultiModalEmbeddingPort",
    "ActivationStore",
    "BridgeStore",
    "AdapterWeightsLoader",
    "SystemProbePort",
    "ModelStore",
    "JobStore",
    "EvaluationStore",
    "CompareStore",
    "ManifoldProfileStore",
]
