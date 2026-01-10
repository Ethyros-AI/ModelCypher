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

"""PortRegistry - Composition root for all adapter implementations.

This is the ONLY place where concrete adapters are instantiated for production.
All services receive their dependencies from this registry via the ServiceFactory.

Following hexagonal architecture:
- Domain code depends on ports (abstract interfaces)
- This container wires concrete adapters to those ports
- Services receive injected dependencies, never instantiate adapters directly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports import (
        ActivationStore,
        ActivationProvider,
        Backend,
        BridgeStore,
        CompareStore,
        EvaluationStore,
        Exporter,
        HiddenStateEngine,
        HubAdapterPort,
        InferenceEngine,
        JobStore,
        ManifoldProfileStore,
        ModelLoaderPort,
        ModelProbePort,
        ModelSearchService,
        ModelStore,
        TrainingEngine,
    )


@dataclass
class PortRegistry:
    """Composition root for all adapter implementations.

    This container holds all port implementations (adapters) needed by the application.
    All fields are REQUIRED - tests must provide mock implementations.

    The create_production() class method wires the default production adapters.
    """

    # Storage ports
    model_store: "ModelStore"
    job_store: "JobStore"
    evaluation_store: "EvaluationStore"
    compare_store: "CompareStore"
    manifold_profile_store: "ManifoldProfileStore"

    # Engine ports
    inference_engine: "InferenceEngine"
    hidden_state_engine: "HiddenStateEngine"
    training_engine: "TrainingEngine"
    exporter: "Exporter"
    activation_provider: "ActivationProvider"

    # Specialized ports
    model_search: "ModelSearchService"
    model_loader: "ModelLoaderPort"
    model_probe: "ModelProbePort"
    hub_adapter: "HubAdapterPort"
    activation_store: "ActivationStore"
    bridge_store: "BridgeStore"

    # Backend
    backend: "Backend"

    # Paths (for services that need filesystem locations)
    base_dir: Path
    logs_dir: Path

    @classmethod
    def create_production(cls) -> "PortRegistry":
        """Factory for production adapter wiring.

        This method imports and instantiates all concrete adapters.
        It's the single point where adapter dependencies are resolved.
        """
        from modelcypher.adapters.activation_store import NPZActivationStore
        from modelcypher.adapters.bridge_store import SafetensorsBridgeStore
        from modelcypher.adapters.filesystem_storage import FileSystemStore
        from modelcypher.adapters.hf_hub import HfHubAdapter
        from modelcypher.adapters.hf_model_search import HfModelSearchAdapter
        from modelcypher.adapters.local_exporter import LocalExporter
        from modelcypher.adapters.local_manifold_profile_store import (
            LocalManifoldProfileStore,
        )
        from modelcypher.adapters.local_training import LocalTrainingEngine
        from modelcypher.backends import default_backend, initialize_default_backend
        from modelcypher.backends.lazy_backend import LazyBackend
        from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories
        from modelcypher.infrastructure.activation_provider_factory import get_activation_provider
        from modelcypher.infrastructure.inference_engine_factory import get_inference_engine
        from modelcypher.infrastructure.model_loader_factory import get_model_loader
        from modelcypher.infrastructure.model_probe_factory import get_model_probe

        # Initialize the global backend for domain code that calls get_default_backend()
        initialize_default_backend()

        register_default_atlas_inventories()

        # FileSystemStore implements multiple storage protocols
        fs_store = FileSystemStore()

        # Platform-appropriate inference engine (MLX/CUDA/JAX)
        inference_engine = get_inference_engine()

        return cls(
            # Storage - FileSystemStore implements all these protocols
            model_store=fs_store,
            job_store=fs_store,
            evaluation_store=fs_store,
            compare_store=fs_store,
            manifold_profile_store=LocalManifoldProfileStore(),
            # Engines
            inference_engine=inference_engine,
            hidden_state_engine=inference_engine,
            training_engine=LocalTrainingEngine(store=fs_store),
            exporter=LocalExporter(),
            activation_provider=get_activation_provider(),
            # Specialized
            model_search=HfModelSearchAdapter(),
            model_loader=get_model_loader(),
            model_probe=get_model_probe(),
            hub_adapter=HfHubAdapter(),
            activation_store=NPZActivationStore(),
            bridge_store=SafetensorsBridgeStore(),
            # Backend
            backend=LazyBackend(default_backend),
            # Paths
            base_dir=fs_store.paths.base,
            logs_dir=fs_store.paths.logs,
        )
