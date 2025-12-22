"""Domain adapter types for signal routing and coordination."""

from modelcypher.core.domain.adapters.adapter_blender import (
    AdapterBlender,
    BlendResult,
)
from modelcypher.core.domain.adapters.ensemble_orchestrator import (
    AdapterInfo,
    CompositionStrategy,
    Ensemble,
    EnsembleOrchestrator,
    EnsembleOrchestratorError,
    EnsembleResult,
    InvalidAdapterIDError,
    NoActiveEnsembleError,
    NoAdaptersError,
    NoCompatibleAdaptersError,
    OrchestratorConfiguration,
    TooManyAdaptersError,
)

__all__ = [
    "AdapterBlender",
    "AdapterInfo",
    "BlendResult",
    "CompositionStrategy",
    "Ensemble",
    "EnsembleOrchestrator",
    "EnsembleOrchestratorError",
    "EnsembleResult",
    "InvalidAdapterIDError",
    "NoActiveEnsembleError",
    "NoAdaptersError",
    "NoCompatibleAdaptersError",
    "OrchestratorConfiguration",
    "TooManyAdaptersError",
]
