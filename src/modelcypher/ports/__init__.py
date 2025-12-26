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

This module defines all abstract interfaces (Protocol classes) that adapters must implement.
Following hexagonal architecture, ports live at the boundary between domain and infrastructure.

PORT CATEGORIES:
================

Compute Backends:
- Backend: Low-level tensor operations (MLX, CUDA, NumPy implementations)

Synchronous Ports (sync-first design):
- InferenceEngine: Basic model inference
- HiddenStateEngine: Inference with hidden state capture
- EmbeddingProvider: Text embedding services
- TrainingEngine: Training job management
- Exporter: Model export to various formats
- ModelSearchService: Model registry search

Storage Ports:
- ModelStore, JobStore, EvaluationStore, CompareStore: Data persistence
- ManifoldProfileStore: Manifold analysis caching

Asynchronous Ports (async-first design for streaming/complex ops):
- InferenceEnginePort: Async dual-path generation with entropy monitoring
- GeometryPort: Async high-dimensional geometry operations
- EmbedderPort: Async embedding interface
- ConceptDiscoveryPort: Async semantic concept detection

USAGE:
======
Import specific ports:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.inference import InferenceEngine
    from modelcypher.ports.async_geometry import GeometryPort as AsyncGeometryPort

Or import from this module:
    from modelcypher.ports import Backend, InferenceEngine
"""

# Compute Backend
from modelcypher.ports.async_embeddings import EmbedderPort
from modelcypher.ports.async_geometry import GeometryPort as AsyncGeometryPort

# Asynchronous Ports
from modelcypher.ports.async_inference import InferenceEnginePort
from modelcypher.ports.backend import Array, Backend
from modelcypher.ports.concept_discovery import ConceptDiscoveryPort
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.ports.exporter import Exporter
from modelcypher.ports.hub import HubAdapterPort

# Synchronous Ports
from modelcypher.ports.inference import HiddenStateEngine, InferenceEngine
from modelcypher.ports.model_loader import ModelLoaderPort
from modelcypher.ports.model_search import ModelSearchService

# Storage Ports
from modelcypher.ports.storage import (
    CompareStore,
    EvaluationStore,
    JobStore,
    ManifoldProfileStore,
    ModelStore,
)
from modelcypher.ports.training import TrainingEngine

__all__ = [
    # Backend
    "Backend",
    "Array",
    # Sync
    "InferenceEngine",
    "HiddenStateEngine",
    "EmbeddingProvider",
    "TrainingEngine",
    "Exporter",
    "ModelSearchService",
    "HubAdapterPort",
    "ModelLoaderPort",
    # Storage
    "ModelStore",
    "JobStore",
    "EvaluationStore",
    "CompareStore",
    "ManifoldProfileStore",
    # Async
    "InferenceEnginePort",
    "AsyncGeometryPort",
    "EmbedderPort",
    "ConceptDiscoveryPort",
]
