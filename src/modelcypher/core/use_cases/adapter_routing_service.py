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

"""Experimental orchestration service for per-layer adapter routing measurements."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.inference.adapter_routing import (
    AdapterIdentity,
    AdapterPool,
    LayerRoutingSnapshot,
    RoutedGenerationResult,
    RoutingTrace,
)
from modelcypher.core.domain.inference.composite_adapter_builder import CompositeAdapterBuilder
from modelcypher.core.domain.inference.divergence_router import LayerDivergenceComputer

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

_SELECTION_NONE = "none"
_SELECTION_MIN_KL = "min_kl"
_SELECTION_MAX_KL = "max_kl"
_VALID_SELECTION_METHODS = {_SELECTION_NONE, _SELECTION_MIN_KL, _SELECTION_MAX_KL}


class AdapterRoutingService:
    """Collect experimental per-layer divergence signals across multiple adapters."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._divergence = LayerDivergenceComputer(self._backend)
        self._composite_builder = CompositeAdapterBuilder(self._backend)

    def load_adapter_pool(
        self,
        base_model_path: str,
        adapter_paths: list[str],
    ) -> AdapterPool:
        """Load base model plus adapter variants for routing measurements."""
        resolved_base_path = str(Path(base_model_path).expanduser().resolve())
        base_model, base_tokenizer = self._backend.load_model(resolved_base_path)

        adapter_models: dict[str, tuple[Any, Any]] = {}
        adapter_identities: list[AdapterIdentity] = []
        used_ids: set[str] = set()

        for adapter_path in adapter_paths:
            resolved_adapter_path = str(Path(adapter_path).expanduser().resolve())
            adapter_id = self._resolve_adapter_id(resolved_adapter_path, used_ids)
            model, tokenizer = self._backend.load_model(
                resolved_base_path,
                adapter_path=resolved_adapter_path,
            )
            adapter_models[adapter_id] = (model, tokenizer)
            adapter_identities.append(
                self._build_adapter_identity(adapter_id, resolved_adapter_path),
            )

        return AdapterPool(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            base_model_path=resolved_base_path,
            adapter_models=adapter_models,
            adapter_identities=tuple(adapter_identities),
        )

    def collect_routing_measurements(
        self,
        pool: AdapterPool,
        prompt: str,
        selection_method: str = _SELECTION_NONE,
    ) -> RoutingTrace:
        """Collect per-layer divergence measurements for one prompt."""
        self._validate_selection_method(selection_method)

        base_activations = self._backend.collect_hidden_activations(
            pool.base_model,
            pool.base_tokenizer,
            [prompt],
        )

        adapter_activations: dict[str, dict[int, "Array"]] = {}
        for adapter_id, (model, tokenizer) in pool.adapter_models.items():
            adapter_activations[adapter_id] = self._backend.collect_hidden_activations(
                model,
                tokenizer,
                [prompt],
            )

        layer_snapshots: list[LayerRoutingSnapshot] = []
        for layer_index in sorted(base_activations):
            adapted_by_id: dict[str, Array] = {}
            for adapter_id, per_layer in adapter_activations.items():
                if layer_index in per_layer:
                    adapted_by_id[adapter_id] = per_layer[layer_index]
            if not adapted_by_id:
                continue
            layer_snapshots.append(
                self._divergence.compute_routing_snapshot(
                    base_activation=base_activations[layer_index],
                    adapted_activations=adapted_by_id,
                    layer_index=layer_index,
                ),
            )

        selected_adapter_per_layer = self._select_adapter_per_layer(
            layer_snapshots,
            selection_method,
        )

        return RoutingTrace(
            prompt=prompt,
            n_adapters=len(pool.adapter_models),
            n_layers=len(layer_snapshots),
            layer_snapshots=tuple(layer_snapshots),
            adapter_identities=pool.adapter_identities,
            selected_adapter_per_layer=selected_adapter_per_layer,
            selection_method=selection_method,
            output_kl_vs_single=None,
        )

    def run_evaluation(
        self,
        pool: AdapterPool,
        prompts: list[str],
        selection_method: str = _SELECTION_NONE,
    ) -> list[RoutingTrace]:
        """Collect routing traces for a prompt set."""
        self._validate_selection_method(selection_method)
        return [
            self.collect_routing_measurements(
                pool=pool,
                prompt=prompt,
                selection_method=selection_method,
            )
            for prompt in prompts
        ]

    def build_composite_adapter(
        self,
        pool: AdapterPool,
        routing_decisions: dict[int, str],
        output_directory: str,
    ) -> str:
        """Build a composite adapter from layer routing decisions."""
        adapter_weight_paths: dict[str, str] = {}
        adapter_configs: dict[str, dict[str, Any]] = {}

        for identity in pool.adapter_identities:
            weights_path = self._find_adapter_weights_path(identity.path)
            if weights_path is None:
                raise ValueError(f"No adapter weights found in {identity.path}")
            adapter_weight_paths[identity.id] = weights_path
            adapter_configs[identity.id] = self._load_adapter_config(identity.path)

        return self._composite_builder.build_composite_adapter(
            adapter_weight_paths=adapter_weight_paths,
            adapter_configs=adapter_configs,
            routing_decisions=routing_decisions,
            output_directory=output_directory,
        )

    def route_and_generate(
        self,
        pool: AdapterPool,
        prompt: str,
        max_tokens: int = 200,
        selection_method: str = _SELECTION_MIN_KL,
        composite_dir: str | None = None,
    ) -> RoutedGenerationResult:
        """Measure, route, build a composite adapter, and generate responses."""
        routing_trace = self.collect_routing_measurements(
            pool=pool,
            prompt=prompt,
            selection_method=selection_method,
        )

        selected = routing_trace.selected_adapter_per_layer
        if composite_dir is None:
            if not selected:
                raise ValueError(
                    "No routing decisions available to build composite adapter. "
                    "Use selection_method='min_kl' or 'max_kl'.",
                )
            composite_dir = tempfile.mkdtemp(prefix="modelcypher-composite-adapter-")
            composite_adapter_path = self.build_composite_adapter(
                pool=pool,
                routing_decisions=selected,
                output_directory=composite_dir,
            )
        else:
            composite_adapter_path = str(Path(composite_dir).expanduser().resolve())

        routed_model, routed_tokenizer = self._backend.load_model(
            pool.base_model_path,
            adapter_path=composite_adapter_path,
        )
        routed_response = self._backend.generate(
            routed_model,
            routed_tokenizer,
            prompt,
            max_tokens=max_tokens,
        )

        base_response = self._backend.generate(
            pool.base_model,
            pool.base_tokenizer,
            prompt,
            max_tokens=max_tokens,
        )

        single_adapter_responses = {
            adapter_id: self._backend.generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
            )
            for adapter_id, (model, tokenizer) in sorted(pool.adapter_models.items())
        }

        return RoutedGenerationResult(
            prompt=prompt,
            routing_trace=routing_trace,
            routed_response=routed_response,
            base_response=base_response,
            single_adapter_responses=single_adapter_responses,
            composite_adapter_path=composite_adapter_path,
        )

    def run_generation_evaluation(
        self,
        pool: AdapterPool,
        prompts: list[str],
        max_tokens: int = 200,
        selection_method: str = _SELECTION_MIN_KL,
    ) -> list[RoutedGenerationResult]:
        """Run routed generation and single/base comparisons for prompt set."""
        self._validate_selection_method(selection_method)
        return [
            self.route_and_generate(
                pool=pool,
                prompt=prompt,
                max_tokens=max_tokens,
                selection_method=selection_method,
            )
            for prompt in prompts
        ]

    def _select_adapter_per_layer(
        self,
        snapshots: list[LayerRoutingSnapshot],
        selection_method: str,
    ) -> dict[int, str]:
        if selection_method == _SELECTION_NONE:
            return {}

        selected: dict[int, str] = {}
        for snapshot in snapshots:
            if not snapshot.measurements:
                continue
            if selection_method == _SELECTION_MIN_KL:
                measurement = min(snapshot.measurements, key=lambda item: item.kl_divergence)
            else:
                measurement = max(snapshot.measurements, key=lambda item: item.kl_divergence)
            selected[snapshot.layer_index] = measurement.adapter_id
        return selected

    def _build_adapter_identity(
        self,
        adapter_id: str,
        adapter_path: str,
    ) -> AdapterIdentity:
        config = self._load_adapter_config(adapter_path)
        rank = int(config.get("r", config.get("rank", 0)))
        alpha = float(config.get("lora_alpha", config.get("alpha", 0.0)))
        scale = alpha / float(rank) if rank > 0 else 0.0
        module_keys = self._load_module_keys(adapter_path, config)

        return AdapterIdentity(
            id=adapter_id,
            path=adapter_path,
            rank=rank,
            scale=scale,
            module_keys=module_keys,
        )

    def _load_adapter_config(self, adapter_path: str) -> dict[str, Any]:
        config_path = Path(adapter_path) / "adapter_config.json"
        if not config_path.exists():
            return {}
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            logger.warning("Failed to parse adapter config at %s: %s", config_path, exc)
        return {}

    def _load_module_keys(
        self,
        adapter_path: str,
        config: dict[str, Any],
    ) -> tuple[str, ...]:
        path = Path(adapter_path)
        module_keys: set[str] = set()

        for safetensors_path in sorted(path.glob("*.safetensors")):
            try:
                weights = self._backend.load_safetensors(str(safetensors_path))
            except Exception as exc:
                logger.warning("Failed to read adapter weights at %s: %s", safetensors_path, exc)
                continue
            module_keys.update(str(key) for key in weights.keys())

        if not module_keys:
            target_modules = config.get("target_modules")
            if isinstance(target_modules, list):
                module_keys.update(str(module_key) for module_key in target_modules)

        return tuple(sorted(module_keys))

    def _resolve_adapter_id(self, adapter_path: str, used_ids: set[str]) -> str:
        base_name = Path(adapter_path).name or "adapter"
        candidate = base_name
        suffix = 2
        while candidate in used_ids:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_ids.add(candidate)
        return candidate

    def _find_adapter_weights_path(self, adapter_path: str) -> str | None:
        """Find adapter safetensors path in directory, if present."""
        base_path = Path(adapter_path).expanduser().resolve()
        for candidate in ("adapters.safetensors", "adapter_model.safetensors"):
            full_path = base_path / candidate
            if full_path.exists():
                return str(full_path)
        return None

    def _validate_selection_method(self, selection_method: str) -> None:
        if selection_method not in _VALID_SELECTION_METHODS:
            raise ValueError(
                "selection_method must be one of "
                f"{sorted(_VALID_SELECTION_METHODS)}, got {selection_method!r}",
            )


__all__ = ["AdapterRoutingService"]
