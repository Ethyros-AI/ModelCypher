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

"""PortRegistry - Composition root for adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.storage import (
        CompareStore,
        EvaluationStore,
        JobStore,
        ManifoldProfileStore,
        ModelStore,
    )
    from modelcypher.ports.inference import HiddenStateEngine, InferenceEngine
    from modelcypher.ports.exporter import Exporter
    from modelcypher.ports.model_loader import ModelLoaderPort
    from modelcypher.adapters.model_probe import ModelProbe


@dataclass
class PortRegistry:
    """Composition root for adapter implementations."""

    # Storage
    model_store: "ModelStore"
    job_store: "JobStore"
    evaluation_store: "EvaluationStore"
    compare_store: "CompareStore"
    manifold_profile_store: "ManifoldProfileStore"

    # Engines
    inference_engine: "InferenceEngine"
    hidden_state_engine: "HiddenStateEngine"
    exporter: "Exporter"
    activation_provider: "ActivationProvider"

    # Specialized
    model_loader: "ModelLoaderPort"
    model_probe: "ModelProbe"

    # Backend
    backend: "Backend"

    # Paths
    base_dir: Path
    logs_dir: Path

    @classmethod
    def create_production(cls) -> "PortRegistry":
        """Factory for production adapter wiring."""
        from modelcypher.adapters.filesystem_storage import FileSystemStore
        from modelcypher.adapters.local_exporter import LocalExporter
        from modelcypher.adapters.local_manifold_profile_store import LocalManifoldProfileStore
        from modelcypher.adapters.model_loader import get_model_loader
        from modelcypher.backends import (
            default_backend,
            get_activation_provider,
            initialize_default_backend,
        )
        from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories
        from modelcypher.infrastructure.inference_engine_factory import get_inference_engine
        from modelcypher.infrastructure.model_probe_factory import get_model_probe

        initialize_default_backend()
        register_default_atlas_inventories()

        fs_store = FileSystemStore()
        inference_engine = get_inference_engine()

        return cls(
            model_store=fs_store,
            job_store=fs_store,
            evaluation_store=fs_store,
            compare_store=fs_store,
            manifold_profile_store=LocalManifoldProfileStore(),
            inference_engine=inference_engine,
            hidden_state_engine=inference_engine,
            exporter=LocalExporter(),
            activation_provider=get_activation_provider(),
            model_loader=get_model_loader(),
            model_probe=get_model_probe(),
            backend=default_backend(),
            base_dir=fs_store.paths.base,
            logs_dir=fs_store.paths.logs,
        )
