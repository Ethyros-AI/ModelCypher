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

"""Experimental builder for per-layer composite LoRA adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class CompositeAdapterBuilder:
    """Build a composite LoRA adapter by cherry-picking weights per layer.

    Experimental. The produced adapter is loadable via backend.load_model()
    using the standard mlx_lm adapter directory layout.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def build_composite_adapter(
        self,
        adapter_weight_paths: dict[str, str],
        adapter_configs: dict[str, dict[str, Any]],
        routing_decisions: dict[int, str],
        output_directory: str,
    ) -> str:
        """Construct and persist a composite adapter directory."""
        if not routing_decisions:
            raise ValueError("routing_decisions must not be empty")

        available_ids = set(adapter_weight_paths)
        selected_ids = set(routing_decisions.values())
        unknown_ids = selected_ids - available_ids
        if unknown_ids:
            raise ValueError(
                "routing_decisions reference unknown adapter ids: "
                f"{sorted(unknown_ids)}",
            )

        loaded_weights: dict[str, dict[str, Array]] = {
            adapter_id: self._backend.load_safetensors(path)
            for adapter_id, path in adapter_weight_paths.items()
        }

        composite_weights: dict[str, Array] = {}
        selected_ranks: list[int] = []
        selected_layer_indices: set[int] = set()
        target_modules: set[str] = set()

        for layer_index, adapter_id in sorted(routing_decisions.items()):
            adapter_weights = loaded_weights[adapter_id]
            selected_layer_indices.add(layer_index)
            target_modules.update(self._extract_target_modules(adapter_configs.get(adapter_id, {})))

            bases = self._bases_for_layer(adapter_weights, layer_index)
            if not bases:
                raise ValueError(
                    f"No LoRA weights found for layer {layer_index} in adapter {adapter_id!r}",
                )

            for base_key in sorted(bases):
                key_a = f"{base_key}.lora_a"
                key_b = f"{base_key}.lora_b"
                if key_a not in adapter_weights or key_b not in adapter_weights:
                    raise ValueError(
                        f"Missing LoRA pair for {base_key} in adapter {adapter_id!r}",
                    )

                lora_a = adapter_weights[key_a]
                lora_b = adapter_weights[key_b]
                rank_a = int(lora_a.shape[1])
                rank_b = int(lora_b.shape[0])
                if rank_a != rank_b:
                    raise ValueError(
                        f"Rank mismatch in adapter {adapter_id!r} for {base_key}: "
                        f"lora_a rank={rank_a}, lora_b rank={rank_b}",
                    )

                composite_weights[key_a] = lora_a
                composite_weights[key_b] = lora_b
                selected_ranks.append(rank_a)

        if not composite_weights:
            raise ValueError("No LoRA weights selected for composite adapter")

        global_rank = max(selected_ranks)
        padded_weights = self._pad_to_global_rank(composite_weights, global_rank)

        output_dir = Path(output_directory).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self._backend.save_safetensors(str(output_dir / "adapters.safetensors"), padded_weights)

        config = {
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": int(global_rank),
                "scale": 1.0,
                "dropout": 0.0,
                "keys": sorted(target_modules),
            },
            "num_layers": self._determine_num_layers(adapter_configs, selected_layer_indices),
            "target_modules": sorted(target_modules),
            "rank": int(global_rank),
        }
        (output_dir / "adapter_config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        return str(output_dir)

    def _bases_for_layer(self, weights: dict[str, "Array"], layer_index: int) -> set[str]:
        bases: set[str] = set()
        for key in weights:
            if not (key.endswith(".lora_a") or key.endswith(".lora_b")):
                continue
            if self._extract_layer_index(key) != layer_index:
                continue
            bases.add(key[:-7])
        return bases

    def _pad_to_global_rank(
        self,
        weights: dict[str, "Array"],
        global_rank: int,
    ) -> dict[str, "Array"]:
        b = self._backend
        padded: dict[str, Array] = {}

        for key, value in weights.items():
            result = value
            if key.endswith(".lora_a"):
                rank = int(value.shape[1])
                if rank < global_rank:
                    zeros = b.zeros((int(value.shape[0]), global_rank - rank), dtype=value.dtype)
                    result = b.concatenate([value, zeros], axis=1)
            elif key.endswith(".lora_b"):
                rank = int(value.shape[0])
                if rank < global_rank:
                    zeros = b.zeros((global_rank - rank, int(value.shape[1])), dtype=value.dtype)
                    result = b.concatenate([value, zeros], axis=0)
            padded[key] = result

        b.eval(*padded.values())
        return padded

    def _extract_layer_index(self, key: str) -> int | None:
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            try:
                return int(parts[2])
            except ValueError:
                return None

        if "layers" not in parts:
            return None
        idx = parts.index("layers")
        if idx + 1 >= len(parts):
            return None
        try:
            return int(parts[idx + 1])
        except ValueError:
            return None

    def _extract_target_modules(self, config: dict[str, Any]) -> set[str]:
        target_modules: set[str] = set()

        top_level_targets = config.get("target_modules")
        if isinstance(top_level_targets, list):
            target_modules.update(str(module) for module in top_level_targets)

        lora_parameters = config.get("lora_parameters")
        if isinstance(lora_parameters, dict):
            lora_keys = lora_parameters.get("keys")
            if isinstance(lora_keys, list):
                target_modules.update(str(module) for module in lora_keys)

        return target_modules

    def _determine_num_layers(
        self,
        adapter_configs: dict[str, dict[str, Any]],
        selected_layer_indices: set[int],
    ) -> int:
        config_layers: list[int] = []
        for config in adapter_configs.values():
            raw_num_layers = config.get("num_layers")
            if isinstance(raw_num_layers, int):
                config_layers.append(raw_num_layers)

        if config_layers:
            return max(config_layers)
        if selected_layer_indices:
            return max(selected_layer_indices) + 1
        return 0


__all__ = ["CompositeAdapterBuilder"]

