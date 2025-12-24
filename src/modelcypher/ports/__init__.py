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
- ModelStore, JobStore, DatasetStore, EvaluationStore, CompareStore: Data persistence
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
from modelcypher.ports.backend import Backend, Array

# Synchronous Ports
from modelcypher.ports.inference import InferenceEngine, HiddenStateEngine
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.ports.training import TrainingEngine
from modelcypher.ports.exporter import Exporter
from modelcypher.ports.model_search import ModelSearchService
from modelcypher.ports.hub import HubAdapterPort
from modelcypher.ports.model_loader import ModelLoaderPort
# NOTE: SyncGeometryPort not imported here to avoid circular dependency with use_cases
# Use: from modelcypher.ports.geometry import GeometryPort as SyncGeometryPort

# Storage Ports
from modelcypher.ports.storage import (
    ModelStore,
    JobStore,
    DatasetStore,
    EvaluationStore,
    CompareStore,
    ManifoldProfileStore,
)

# Asynchronous Ports
from modelcypher.ports.async_inference import InferenceEnginePort
from modelcypher.ports.async_geometry import GeometryPort as AsyncGeometryPort
from modelcypher.ports.async_embeddings import EmbedderPort
from modelcypher.ports.concept_discovery import ConceptDiscoveryPort

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
    # SyncGeometryPort removed due to circular import - import directly from ports.geometry
    # Storage
    "ModelStore",
    "JobStore",
    "DatasetStore",
    "EvaluationStore",
    "CompareStore",
    "ManifoldProfileStore",
    # Async
    "InferenceEnginePort",
    "AsyncGeometryPort",
    "EmbedderPort",
    "ConceptDiscoveryPort",
]
