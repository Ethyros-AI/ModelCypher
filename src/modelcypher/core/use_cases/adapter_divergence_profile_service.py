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

"""Experimental pairwise adapter divergence profiling without generation."""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Any

from modelcypher.core.use_cases.adapter_routing_service import AdapterRoutingService


class AdapterDivergenceProfileService:
    """Compute per-layer and pairwise divergence profiles for adapter sets."""

    def __init__(self, routing_service: AdapterRoutingService | None = None) -> None:
        self._routing_service = routing_service or AdapterRoutingService()
        self._backend = self._routing_service._backend

    def compute_profile(
        self,
        *,
        base_model_path: str,
        adapter_paths: list[str],
        prompts: list[str],
    ) -> dict[str, Any]:
        """Compute pairwise divergence profile across layers for a prompt set."""
        pool = self._routing_service.load_adapter_pool(base_model_path, adapter_paths)
        adapter_ids = [identity.id for identity in pool.adapter_identities]

        per_adapter_layer: dict[str, dict[int, dict[str, list[float]]]] = {
            adapter_id: {} for adapter_id in adapter_ids
        }
        per_adapter_all: dict[str, dict[str, list[float]]] = {
            adapter_id: {"kl": [], "cosine": [], "norm_ratio": []}
            for adapter_id in adapter_ids
        }
        pairwise_state: dict[str, dict[str, Any]] = {}
        unique_layers: set[int] = set()

        for prompt in prompts:
            trace = self._routing_service.collect_routing_measurements(
                pool=pool,
                prompt=prompt,
                selection_method="none",
            )
            for snapshot in trace.layer_snapshots:
                layer_index = snapshot.layer_index
                unique_layers.add(layer_index)
                by_adapter = {m.adapter_id: m for m in snapshot.measurements}

                for adapter_id, measurement in by_adapter.items():
                    layer_bucket = per_adapter_layer[adapter_id].setdefault(
                        layer_index,
                        {"kl": [], "cosine": [], "norm_ratio": []},
                    )
                    layer_bucket["kl"].append(float(measurement.kl_divergence))
                    layer_bucket["cosine"].append(float(measurement.cosine_similarity))
                    layer_bucket["norm_ratio"].append(float(measurement.activation_norm_ratio))

                    per_adapter_all[adapter_id]["kl"].append(float(measurement.kl_divergence))
                    per_adapter_all[adapter_id]["cosine"].append(float(measurement.cosine_similarity))
                    per_adapter_all[adapter_id]["norm_ratio"].append(float(measurement.activation_norm_ratio))

                for left_id, right_id in combinations(sorted(by_adapter), 2):
                    left = by_adapter[left_id]
                    right = by_adapter[right_id]
                    pair_key = f"{left_id}_vs_{right_id}"
                    state = pairwise_state.setdefault(
                        pair_key,
                        {
                            "win_left": 0,
                            "win_right": 0,
                            "tie": 0,
                            "kl_gaps": [],
                            "cosine_gaps": [],
                        },
                    )
                    state["kl_gaps"].append(abs(float(left.kl_divergence) - float(right.kl_divergence)))
                    state["cosine_gaps"].append(
                        abs(float(left.cosine_similarity) - float(right.cosine_similarity)),
                    )

                    if left.kl_divergence < right.kl_divergence:
                        state["win_left"] += 1
                    elif right.kl_divergence < left.kl_divergence:
                        state["win_right"] += 1
                    else:
                        state["tie"] += 1

        per_adapter_payload: dict[str, dict[str, Any]] = {}
        for adapter_id in sorted(adapter_ids):
            layer_payload = []
            for layer_index in sorted(per_adapter_layer[adapter_id]):
                layer_stats = per_adapter_layer[adapter_id][layer_index]
                layer_payload.append(
                    {
                        "layer_index": layer_index,
                        "mean_kl": self._mean(layer_stats["kl"]),
                        "mean_cosine": self._mean(layer_stats["cosine"]),
                        "mean_norm_ratio": self._mean(layer_stats["norm_ratio"]),
                        "n_measurements": len(layer_stats["kl"]),
                    },
                )

            aggregate = per_adapter_all[adapter_id]
            per_adapter_payload[adapter_id] = {
                "per_layer": layer_payload,
                "aggregate": {
                    "mean_kl": self._mean(aggregate["kl"]),
                    "mean_cosine": self._mean(aggregate["cosine"]),
                    "std_kl": self._std(aggregate["kl"]),
                    "std_cosine": self._std(aggregate["cosine"]),
                    "n_measurements": len(aggregate["kl"]),
                },
            }

        pairwise_payload: dict[str, dict[str, float]] = {}
        routing_potential: dict[str, float] = {}
        for pair_key, state in sorted(pairwise_state.items()):
            total = int(state["win_left"] + state["win_right"] + state["tie"])
            if total == 0:
                layer_agreement_rate = 1.0
            else:
                dominant = max(int(state["win_left"]), int(state["win_right"]))
                layer_agreement_rate = float((dominant + int(state["tie"])) / total)

            pairwise_payload[pair_key] = {
                "layer_agreement_rate": layer_agreement_rate,
                "mean_kl_gap": self._mean(state["kl_gaps"]),
                "mean_cosine_gap": self._mean(state["cosine_gaps"]),
            }
            routing_potential[pair_key] = 1.0 - layer_agreement_rate

        return {
            "n_prompts": len(prompts),
            "n_layers": len(unique_layers),
            "adapter_ids": sorted(adapter_ids),
            "per_adapter": per_adapter_payload,
            "pairwise": pairwise_payload,
            "routing_potential": routing_potential,
        }

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _std(self, values: list[float]) -> float:
        if not values:
            return 0.0
        mean = self._mean(values)
        variance = sum((value - mean) * (value - mean) for value in values) / len(values)
        return float(sqrt(variance))


__all__ = ["AdapterDivergenceProfileService"]
