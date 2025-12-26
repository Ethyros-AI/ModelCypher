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
        Backend,
        CompareStore,
        EvaluationStore,
        Exporter,
        HiddenStateEngine,
        HubAdapterPort,
        InferenceEngine,
        JobStore,
        ManifoldProfileStore,
        ModelLoaderPort,
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

    # Specialized ports
    model_search: "ModelSearchService"
    model_loader: "ModelLoaderPort"
    hub_adapter: "HubAdapterPort"

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
        from modelcypher.adapters.filesystem_storage import FileSystemStore
        from modelcypher.adapters.hf_hub import HfHubAdapter
        from modelcypher.adapters.hf_model_search import HfModelSearchAdapter
        from modelcypher.adapters.local_exporter import LocalExporter
        from modelcypher.adapters.local_inference import LocalInferenceEngine
        from modelcypher.adapters.local_manifold_profile_store import (
            LocalManifoldProfileStore,
        )
        from modelcypher.adapters.local_training import LocalTrainingEngine
        from modelcypher.adapters.mlx_model_loader import MLXModelLoader
        from modelcypher.backends import default_backend

        # FileSystemStore implements multiple storage protocols
        fs_store = FileSystemStore()

        # LocalInferenceEngine implements both InferenceEngine and HiddenStateEngine
        inference_engine = LocalInferenceEngine()

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
            # Specialized
            model_search=HfModelSearchAdapter(),
            model_loader=MLXModelLoader(),
            hub_adapter=HfHubAdapter(),
            # Backend
            backend=default_backend(),
            # Paths
            base_dir=fs_store.paths.base,
            logs_dir=fs_store.paths.logs,
        )
