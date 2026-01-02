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

"""Metaphor geometry MCP tools.

Contains tools for:
- Listing CMT (Conceptual Metaphor Theory) mappings
- Collecting metaphor trajectories through layers
- Testing cross-model metaphor invariance (Platonic hypothesis)
"""

from __future__ import annotations

from pathlib import Path

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)
from .safety import _forward_text_backbone, _resolve_text_backbone


def register_geometry_metaphor_tools(ctx: ServiceContext) -> None:
    """Register geometry metaphor tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_metaphor_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_metaphor_list(family: str | None = None) -> dict:
            """List available Conceptual Metaphor Theory (CMT) mappings.

            Lists all 8 CMT metaphors from Lakoff & Johnson (1980) with their
            source and target domains.

            Args:
                family: Optional filter by CMT family (e.g., "time_as_resource").
            """
            from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
                CMTFamily,
                ConceptualMetaphorInventory,
            )

            if family:
                try:
                    family_enum = CMTFamily(family)
                    mappings = ConceptualMetaphorInventory.mappings_by_family(family_enum)
                except ValueError:
                    valid = ", ".join(f.value for f in CMTFamily)
                    raise ValueError(f"Invalid family: {family}. Valid: {valid}")
            else:
                mappings = ConceptualMetaphorInventory.ALL_MAPPINGS

            return {
                "_schema": "mc.geometry.metaphor.list.v1",
                "mappings": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "family": m.family.value,
                        "sourceDomain": m.source_domain,
                        "targetDomain": m.target_domain,
                        "sourceExemplarCount": len(m.source_exemplars),
                        "targetExemplarCount": len(m.target_exemplars),
                        "bridgingExpressionCount": len(m.bridging_expressions),
                    }
                    for m in mappings
                ],
                "count": len(mappings),
            }

    if "mc_geometry_metaphor_trajectory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_metaphor_trajectory(
            modelPath: str,
            metaphorId: str = "cmt_time_is_money",
            layer: int = -1,
        ) -> dict:
            """Collect metaphor trajectory through model layers.

            Tracks CKA between source and target domain activations at each layer.
            The convergence layer is where CKA peaks - where the model maps source
            concepts to target concepts.

            Args:
                modelPath: Path to the model directory.
                metaphorId: CMT mapping ID (default: "cmt_time_is_money").
                layer: Layer to analyze (-1 for sampled layers).
            """
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
                ConceptualMetaphorInventory,
            )
            from modelcypher.core.domain.geometry.metaphor_trajectory import (
                MetaphorTrajectoryCollector,
                compute_convergence_profile,
                convergence_profile_to_dict,
                trajectory_to_dict,
            )

            mapping = ConceptualMetaphorInventory.get_by_id(metaphorId)
            if not mapping:
                raise ValueError(f"Unknown metaphor ID: {metaphorId}")

            model_path = require_existing_directory(modelPath)
            model, tokenizer = load_model_for_training(model_path)
            backend = MLXBackend()

            # Resolve architecture
            resolved = _resolve_text_backbone(model)
            if not resolved:
                raise ValueError("Could not resolve model architecture")
            embed_tokens, layers, norm = resolved
            num_layers = len(layers)

            if layer >= 0:
                target_layers = [layer]
            else:
                # Sample layers: first, 25%, 50%, 75%, last
                target_layers = [
                    0,
                    num_layers // 4,
                    num_layers // 2,
                    3 * num_layers // 4,
                    num_layers - 1,
                ]
                target_layers = sorted(set(target_layers))

            # Collect activations for source domain
            source_acts_by_layer: dict[int, list] = {l: [] for l in target_layers}
            for exemplar in mapping.source_exemplars:
                try:
                    tokens = tokenizer.encode(exemplar)
                    input_ids = backend.array([tokens])
                    for target_layer in target_layers:
                        hidden = _forward_text_backbone(
                            input_ids, embed_tokens, layers, norm, target_layer, backend
                        )
                        activation = backend.mean(hidden[0], axis=0)
                        backend.async_eval(activation)
                        source_acts_by_layer[target_layer].append(activation)
                except Exception:
                    pass  # Skip failed exemplars

            # Collect activations for target domain
            target_acts_by_layer: dict[int, list] = {l: [] for l in target_layers}
            for exemplar in mapping.target_exemplars:
                try:
                    tokens = tokenizer.encode(exemplar)
                    input_ids = backend.array([tokens])
                    for target_layer in target_layers:
                        hidden = _forward_text_backbone(
                            input_ids, embed_tokens, layers, norm, target_layer, backend
                        )
                        activation = backend.mean(hidden[0], axis=0)
                        backend.async_eval(activation)
                        target_acts_by_layer[target_layer].append(activation)
                except Exception:
                    pass  # Skip failed exemplars

            # Stack and sync
            layer_activations = {}
            for target_layer in target_layers:
                if source_acts_by_layer[target_layer] and target_acts_by_layer[target_layer]:
                    all_acts = (
                        source_acts_by_layer[target_layer] + target_acts_by_layer[target_layer]
                    )
                    backend.eval(*all_acts)

                    source_stacked = backend.stack(source_acts_by_layer[target_layer])
                    target_stacked = backend.stack(target_acts_by_layer[target_layer])
                    layer_activations[target_layer] = (source_stacked, target_stacked)

            if not layer_activations:
                raise ValueError("No activations extracted")

            # Collect trajectory
            collector = MetaphorTrajectoryCollector(backend)
            model_id = Path(model_path).name
            trajectory = collector.collect_from_activations(mapping, model_id, layer_activations)
            profile = compute_convergence_profile(trajectory)

            return {
                "_schema": "mc.geometry.metaphor.trajectory.v1",
                "modelPath": model_path,
                "trajectory": trajectory_to_dict(trajectory),
                "profile": convergence_profile_to_dict(profile),
            }

    if "mc_geometry_metaphor_invariance" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_metaphor_invariance(
            modelAPath: str,
            modelBPath: str,
            metaphorId: str = "cmt_time_is_money",
        ) -> dict:
            """Test metaphor geometry invariance between two models.

            Compares metaphor trajectories between models to test the
            Platonic Representation Hypothesis: do models converge to
            similar metaphor geometry?

            Args:
                modelAPath: Path to first model.
                modelBPath: Path to second model.
                metaphorId: CMT mapping ID to test.
            """
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
                ConceptualMetaphorInventory,
            )
            from modelcypher.core.domain.geometry.metaphor_invariance import (
                MetaphorInvarianceAnalyzer,
                invariance_result_to_dict,
            )
            from modelcypher.core.domain.geometry.metaphor_trajectory import (
                MetaphorTrajectoryCollector,
            )

            mapping = ConceptualMetaphorInventory.get_by_id(metaphorId)
            if not mapping:
                raise ValueError(f"Unknown metaphor ID: {metaphorId}")

            backend = MLXBackend()
            collector = MetaphorTrajectoryCollector(backend)

            def collect_trajectory(model_path_str: str):
                model_path = require_existing_directory(model_path_str)
                model, tokenizer = load_model_for_training(model_path)

                resolved = _resolve_text_backbone(model)
                if not resolved:
                    raise ValueError(f"Could not resolve architecture for {model_path}")
                embed_tokens, layers, norm = resolved
                num_layers = len(layers)

                target_layers = [
                    0,
                    num_layers // 4,
                    num_layers // 2,
                    3 * num_layers // 4,
                    num_layers - 1,
                ]
                target_layers = sorted(set(target_layers))

                source_acts_by_layer: dict[int, list] = {l: [] for l in target_layers}
                for exemplar in mapping.source_exemplars:
                    try:
                        tokens = tokenizer.encode(exemplar)
                        input_ids = backend.array([tokens])
                        for target_layer in target_layers:
                            hidden = _forward_text_backbone(
                                input_ids, embed_tokens, layers, norm, target_layer, backend
                            )
                            activation = backend.mean(hidden[0], axis=0)
                            backend.async_eval(activation)
                            source_acts_by_layer[target_layer].append(activation)
                    except Exception:
                        pass

                target_acts_by_layer: dict[int, list] = {l: [] for l in target_layers}
                for exemplar in mapping.target_exemplars:
                    try:
                        tokens = tokenizer.encode(exemplar)
                        input_ids = backend.array([tokens])
                        for target_layer in target_layers:
                            hidden = _forward_text_backbone(
                                input_ids, embed_tokens, layers, norm, target_layer, backend
                            )
                            activation = backend.mean(hidden[0], axis=0)
                            backend.async_eval(activation)
                            target_acts_by_layer[target_layer].append(activation)
                    except Exception:
                        pass

                layer_activations = {}
                for target_layer in target_layers:
                    if source_acts_by_layer[target_layer] and target_acts_by_layer[target_layer]:
                        all_acts = (
                            source_acts_by_layer[target_layer]
                            + target_acts_by_layer[target_layer]
                        )
                        backend.eval(*all_acts)

                        source_stacked = backend.stack(source_acts_by_layer[target_layer])
                        target_stacked = backend.stack(target_acts_by_layer[target_layer])
                        layer_activations[target_layer] = (source_stacked, target_stacked)

                return collector.collect_from_activations(
                    mapping, Path(model_path).name, layer_activations
                )

            trajectory_a = collect_trajectory(modelAPath)
            trajectory_b = collect_trajectory(modelBPath)

            analyzer = MetaphorInvarianceAnalyzer(backend)
            result = analyzer.compare_metaphor_geometry(trajectory_a, trajectory_b)

            return {
                "_schema": "mc.geometry.metaphor.invariance.v1",
                **invariance_result_to_dict(result),
            }

    if "mc_geometry_metaphor_convergence" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_metaphor_convergence(modelPath: str) -> dict:
            """Find convergence layer for all CMT metaphors.

            Analyzes where each metaphor's source→target mapping peaks
            in the model's layer stack.

            Args:
                modelPath: Path to the model directory.
            """
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
                ConceptualMetaphorInventory,
            )
            from modelcypher.core.domain.geometry.metaphor_trajectory import (
                MetaphorTrajectoryCollector,
            )

            model_path = require_existing_directory(modelPath)
            model, tokenizer = load_model_for_training(model_path)
            backend = MLXBackend()

            resolved = _resolve_text_backbone(model)
            if not resolved:
                raise ValueError("Could not resolve model architecture")
            embed_tokens, layers, norm = resolved
            num_layers = len(layers)
            model_id = Path(model_path).name

            target_layers = [
                0,
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4,
                num_layers - 1,
            ]
            target_layers = sorted(set(target_layers))

            collector = MetaphorTrajectoryCollector(backend)
            results = []

            for mapping in ConceptualMetaphorInventory.ALL_MAPPINGS:
                try:
                    source_acts_by_layer: dict[int, list] = {l: [] for l in target_layers}
                    for exemplar in mapping.source_exemplars:
                        try:
                            tokens = tokenizer.encode(exemplar)
                            input_ids = backend.array([tokens])
                            for target_layer in target_layers:
                                hidden = _forward_text_backbone(
                                    input_ids,
                                    embed_tokens,
                                    layers,
                                    norm,
                                    target_layer,
                                    backend,
                                )
                                activation = backend.mean(hidden[0], axis=0)
                                backend.async_eval(activation)
                                source_acts_by_layer[target_layer].append(activation)
                        except Exception:
                            pass

                    target_acts_by_layer: dict[int, list] = {l: [] for l in target_layers}
                    for exemplar in mapping.target_exemplars:
                        try:
                            tokens = tokenizer.encode(exemplar)
                            input_ids = backend.array([tokens])
                            for target_layer in target_layers:
                                hidden = _forward_text_backbone(
                                    input_ids,
                                    embed_tokens,
                                    layers,
                                    norm,
                                    target_layer,
                                    backend,
                                )
                                activation = backend.mean(hidden[0], axis=0)
                                backend.async_eval(activation)
                                target_acts_by_layer[target_layer].append(activation)
                        except Exception:
                            pass

                    layer_activations = {}
                    for target_layer in target_layers:
                        if (
                            source_acts_by_layer[target_layer]
                            and target_acts_by_layer[target_layer]
                        ):
                            all_acts = (
                                source_acts_by_layer[target_layer]
                                + target_acts_by_layer[target_layer]
                            )
                            backend.eval(*all_acts)
                            source_stacked = backend.stack(source_acts_by_layer[target_layer])
                            target_stacked = backend.stack(target_acts_by_layer[target_layer])
                            layer_activations[target_layer] = (source_stacked, target_stacked)

                    if layer_activations:
                        trajectory = collector.collect_from_activations(
                            mapping, model_id, layer_activations
                        )
                        results.append({
                            "metaphorId": mapping.id,
                            "metaphorName": mapping.name,
                            "convergenceLayer": trajectory.convergence_layer,
                            "peakCka": trajectory.peak_cka,
                            "layerCount": trajectory.layer_count,
                        })
                except Exception as e:
                    results.append({
                        "metaphorId": mapping.id,
                        "metaphorName": mapping.name,
                        "error": str(e),
                    })

            # Compute aggregates
            valid = [r for r in results if "convergenceLayer" in r]
            if valid:
                mean_peak_cka = sum(r["peakCka"] for r in valid) / len(valid)
                mean_conv = sum(r["convergenceLayer"] for r in valid) / len(valid)
            else:
                mean_peak_cka = 0.0
                mean_conv = 0.0

            return {
                "_schema": "mc.geometry.metaphor.convergence.v1",
                "modelPath": model_path,
                "modelId": model_id,
                "numLayers": num_layers,
                "results": results,
                "aggregate": {
                    "meanPeakCka": mean_peak_cka,
                    "meanConvergenceLayer": mean_conv,
                    "metaphorsAnalyzed": len(valid),
                    "metaphorsFailed": len(results) - len(valid),
                },
            }
