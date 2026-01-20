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

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.output import write_output
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

from .common import (
    cleanup_memory,
    get_context,
    load_model_and_provider,
)

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("full-profile")
    def full_profile(
        ctx: typer.Context,
        model_path: str = typer.Argument(..., help="Path to the model directory"),
        output_path: str | None = typer.Option(
            None, "--output", "-o", help="Save profile to JSON file"
        ),
        checkpoint_dir: str | None = typer.Option(
            None, "--checkpoint-dir", help="Directory for incremental checkpoints"
        ),
        resume: bool = typer.Option(
            False, "--resume", help="Resume from last checkpoint"
        ),
    ) -> None:
        """Generate comprehensive knowledge density profile for a model.

        Profiles ALL layers and ALL domains to create a complete map of where
        the model is strong (dense) and weak (sparse) across the entire
        representation space.

        This is compute-intensive but produces the data needed for informed
        knowledge transplant decisions. No sampling, no shortcuts.

        Output includes:
        - Per-layer density statistics for each domain
        - Per-concept density scores at every layer
        - Domain strength rankings by layer
        - Overall model capability fingerprint

        Use --checkpoint-dir to save progress incrementally for large models.
        Use --resume to continue from last checkpoint.
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
        )

        # Load model
        model, tokenizer, backend, provider, num_layers = load_model_and_provider(model_path)

        # All layers, no sampling
        all_layers = list(range(num_layers))

        # Get ALL probes (no max_probes limit)
        probes = UnifiedAtlasInventory.all_probes()

        logger.info(
            "Full profile: %d probes x %d layers = %d measurements",
            len(probes),
            num_layers,
            len(probes) * num_layers,
        )

        # Check for checkpoint to resume from
        checkpoint_data: dict = {}
        completed_layers: set[int] = set()
        if checkpoint_dir and resume:
            checkpoint_path = Path(checkpoint_dir) / "full_profile_checkpoint.json"
            if checkpoint_path.exists():
                checkpoint_data = json.loads(checkpoint_path.read_text())
                completed_layers = set(checkpoint_data.get("completedLayers", []))
                logger.info("Resuming from checkpoint: %d layers complete", len(completed_layers))

        analyzer = KnowledgeDensityAnalyzer(backend=backend)

        # Results structure
        layer_profiles: dict[int, dict] = checkpoint_data.get("layerProfiles", {})
        # Convert string keys back to int
        layer_profiles = {int(k): v for k, v in layer_profiles.items()}

        # Process each layer
        for layer_idx in all_layers:
            if layer_idx in completed_layers:
                logger.info("Layer %d already complete, skipping", layer_idx)
                continue

            logger.info("Processing layer %d/%d...", layer_idx + 1, num_layers)

            layer_result = analyzer.analyze_layer(probes, provider, layer_idx)

            # Store results
            layer_profiles[layer_idx] = {
                "layer": layer_idx,
                "totalConcepts": len(layer_result.concept_densities),
                "meanDensity": layer_result.mean_density,
                "medianDensity": layer_result.median_density,
                "concepts": [
                    {
                        "probeID": c.probe_id,
                        "name": c.name,
                        "domain": c.domain,
                        "densityScore": c.density_score,
                        "intrinsicDimension": c.intrinsic_dimension,
                        "activationVariance": c.activation_variance,
                        "clusterTightness": c.cluster_tightness,
                    }
                    for c in layer_result.concept_densities
                ],
            }
            completed_layers.add(layer_idx)

            # Save checkpoint after each layer
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                checkpoint_file = checkpoint_path / "full_profile_checkpoint.json"
                checkpoint_file.write_text(json.dumps({
                    "modelPath": model_path,
                    "completedLayers": sorted(completed_layers),
                    "totalLayers": num_layers,
                    "layerProfiles": {str(k): v for k, v in layer_profiles.items()},
                }, indent=2))
                logger.info(
                    "Checkpoint saved: %d/%d layers",
                    len(completed_layers),
                    num_layers,
                )

        # Compute domain summaries across all layers
        domain_summaries: dict[str, dict] = {}
        for layer_idx, lp in layer_profiles.items():
            for concept in lp.get("concepts", []):
                domain = concept["domain"]
                if domain not in domain_summaries:
                    domain_summaries[domain] = {
                        "domain": domain,
                        "layerDensities": {},
                        "conceptCount": 0,
                        "totalDensitySum": 0.0,
                    }
                if layer_idx not in domain_summaries[domain]["layerDensities"]:
                    domain_summaries[domain]["layerDensities"][layer_idx] = {
                        "densities": [],
                        "meanDensity": 0.0,
                    }
                domain_summaries[domain]["layerDensities"][layer_idx]["densities"].append(
                    concept["densityScore"]
                )
                domain_summaries[domain]["conceptCount"] += 1
                domain_summaries[domain]["totalDensitySum"] += concept["densityScore"]

        # Compute means per domain per layer
        for domain, summary in domain_summaries.items():
            for layer_idx, layer_data in summary["layerDensities"].items():
                densities = layer_data["densities"]
                if densities:
                    layer_data["meanDensity"] = sum(densities) / len(densities)
                del layer_data["densities"]  # Don't need raw list in output
            if summary["conceptCount"] > 0:
                summary["overallMeanDensity"] = (
                    summary["totalDensitySum"] / summary["conceptCount"]
                )
            else:
                summary["overallMeanDensity"] = 0.0
            del summary["totalDensitySum"]

        # Find strongest/weakest layers per domain
        for domain, summary in domain_summaries.items():
            layer_means = [
                (int(layer_idx), data["meanDensity"])
                for layer_idx, data in summary["layerDensities"].items()
            ]
            if layer_means:
                layer_means.sort(key=lambda x: x[1], reverse=True)
                summary["strongestLayers"] = [lm[0] for lm in layer_means[:5]]
                summary["weakestLayers"] = [lm[0] for lm in layer_means[-5:]]

        # Build final payload
        payload = {
            "_schema": "mc.geometry.research.full_profile.v1",
            "modelPath": model_path,
            "totalLayers": num_layers,
            "totalProbes": len(probes),
            "completedLayers": sorted(completed_layers),
            "domainSummaries": list(domain_summaries.values()),
            "layerProfiles": [
                layer_profiles[i] for i in sorted(layer_profiles.keys())
            ],
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(payload, indent=2))
            logger.info("Full profile saved to %s", output_path)

        if context.output_format == "text":
            lines = [
                "FULL MODEL KNOWLEDGE PROFILE",
                f"Model: {model_path}",
                f"Layers: {num_layers}",
                f"Probes: {len(probes)}",
                f"Completed: {len(completed_layers)}/{num_layers} layers",
                "",
                "DOMAIN SUMMARY (strongest -> weakest by overall density):",
                "-" * 60,
            ]

            sorted_domains = sorted(
                domain_summaries.values(),
                key=lambda x: x.get("overallMeanDensity", 0),
                reverse=True,
            )
            for ds in sorted_domains:
                lines.append(
                    f"  {ds['domain']}: mean_density={ds.get('overallMeanDensity', 0):.3f}, "
                    f"strongest_layers={ds.get('strongestLayers', [])[:3]}"
                )

            lines.append("")
            lines.append("LAYER-BY-LAYER SUMMARY:")
            lines.append("-" * 60)
            for layer_idx in sorted(layer_profiles.keys()):
                lp = layer_profiles[layer_idx]
                lines.append(
                    f"  L{layer_idx}: mean={lp['meanDensity']:.3f}, "
                    f"concepts={lp['totalConcepts']}"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    @app.command("batch-profile")
    def batch_profile(
        ctx: typer.Context,
        model_paths: list[str] = typer.Argument(..., help="Paths to model directories"),
        output_dir: str = typer.Option(
            None, "--output-dir", "-o", help="Directory for profile outputs"
        ),
    ) -> None:
        """Profile multiple models SEQUENTIALLY with automatic resource management.

        IMPORTANT: This command profiles models ONE AT A TIME to maximize resource
        utilization and prevent system crashes. Each model gets full CPU/GPU access.

        Memory is aggressively cleaned between models to prevent accumulation.
        Checkpointing is automatic - interrupted profiles can be resumed.

        Example:
            mc geometry research batch-profile /path/to/model1 /path/to/model2 -o ./profiles

        For uber model experiments:
            mc geometry research batch-profile \
                /models/Qwen3-8B-4bit \
                /models/Qwen2.5-Math-7B \
                /models/Mistral-7B \
                -o /experiments/profiles
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
        )

        # Resolve output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = Path.cwd() / "profiles"
        out_path.mkdir(parents=True, exist_ok=True)
        checkpoint_base = out_path / "checkpoints"
        checkpoint_base.mkdir(parents=True, exist_ok=True)

        # Get probes once (same for all models)
        probes = UnifiedAtlasInventory.all_probes()

        logger.info("Batch profiling %d models with %d probes", len(model_paths), len(probes))

        results: list[dict] = []

        for idx, model_path in enumerate(model_paths, 1):
            model_name = Path(model_path).name
            profile_output = out_path / f"{model_name}.json"
            checkpoint_dir = checkpoint_base / model_name

            logger.info("")
            logger.info("=" * 60)
            logger.info("MODEL %d/%d: %s", idx, len(model_paths), model_name)
            logger.info("=" * 60)

            # Check if already complete
            if profile_output.exists():
                try:
                    existing = json.loads(profile_output.read_text())
                    completed = len(existing.get("completedLayers", []))
                    total = existing.get("totalLayers", 0)
                    if completed == total and total > 0:
                        logger.info("Already complete (%d/%d layers)", completed, total)
                        results.append({
                            "model": model_name,
                            "status": "already_complete",
                            "layers": total,
                            "output": str(profile_output),
                        })
                        continue
                    logger.info("Resuming from checkpoint (%d/%d layers)", completed, total)
                except Exception as exc:
                    logger.warning(
                        "Failed to read profile checkpoint %s: %s",
                        profile_output,
                        exc,
                    )

            # Clean memory before loading new model
            logger.info("Cleaning memory before model load...")
            cleanup_memory()

            try:
                # Load model
                logger.info("Loading model: %s", model_path)
                model, tokenizer, backend, provider, num_layers = load_model_and_provider(model_path)

                all_layers = list(range(num_layers))
                logger.info(
                    "Full profile: %d probes x %d layers = %d measurements",
                    len(probes),
                    num_layers,
                    len(probes) * num_layers,
                )

                # Check for checkpoint to resume from
                checkpoint_data: dict = {}
                completed_layers: set[int] = set()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_file = checkpoint_dir / "full_profile_checkpoint.json"

                if checkpoint_file.exists():
                    checkpoint_data = json.loads(checkpoint_file.read_text())
                    completed_layers = set(checkpoint_data.get("completedLayers", []))
                    logger.info("Resuming from checkpoint: %d layers complete", len(completed_layers))

                analyzer = KnowledgeDensityAnalyzer(backend=backend)

                # Results structure
                layer_profiles: dict[int, dict] = checkpoint_data.get("layerProfiles", {})
                layer_profiles = {int(k): v for k, v in layer_profiles.items()}

                # Process each layer
                for layer_idx in all_layers:
                    if layer_idx in completed_layers:
                        continue

                    logger.info("Processing layer %d/%d...", layer_idx + 1, num_layers)

                    layer_result = analyzer.analyze_layer(probes, provider, layer_idx)

                    layer_profiles[layer_idx] = {
                        "layer": layer_idx,
                        "totalConcepts": len(layer_result.concept_densities),
                        "meanDensity": layer_result.mean_density,
                        "medianDensity": layer_result.median_density,
                        "concepts": [
                            {
                                "probeID": c.probe_id,
                                "name": c.name,
                                "domain": c.domain,
                                "densityScore": c.density_score,
                                "intrinsicDimension": c.intrinsic_dimension,
                                "activationVariance": c.activation_variance,
                                "clusterTightness": c.cluster_tightness,
                            }
                            for c in layer_result.concept_densities
                        ],
                    }
                    completed_layers.add(layer_idx)

                    # Save checkpoint after each layer
                    checkpoint_file.write_text(json.dumps({
                        "modelPath": model_path,
                        "completedLayers": sorted(completed_layers),
                        "totalLayers": num_layers,
                        "layerProfiles": {str(k): v for k, v in layer_profiles.items()},
                    }, indent=2))
                    logger.info("Checkpoint saved: %d/%d layers", len(completed_layers), num_layers)

                # Compute domain summaries
                domain_summaries: dict[str, dict] = {}
                for layer_idx, lp in layer_profiles.items():
                    for concept in lp.get("concepts", []):
                        domain = concept["domain"]
                        if domain not in domain_summaries:
                            domain_summaries[domain] = {
                                "domain": domain,
                                "layerDensities": {},
                                "conceptCount": 0,
                                "totalDensitySum": 0.0,
                            }
                        if layer_idx not in domain_summaries[domain]["layerDensities"]:
                            domain_summaries[domain]["layerDensities"][layer_idx] = {
                                "densities": [],
                                "meanDensity": 0.0,
                            }
                        domain_summaries[domain]["layerDensities"][layer_idx]["densities"].append(
                            concept["densityScore"]
                        )
                        domain_summaries[domain]["conceptCount"] += 1
                        domain_summaries[domain]["totalDensitySum"] += concept["densityScore"]

                for domain, summary in domain_summaries.items():
                    for layer_idx, layer_data in summary["layerDensities"].items():
                        densities = layer_data["densities"]
                        if densities:
                            layer_data["meanDensity"] = sum(densities) / len(densities)
                        del layer_data["densities"]
                    if summary["conceptCount"] > 0:
                        summary["overallMeanDensity"] = (
                            summary["totalDensitySum"] / summary["conceptCount"]
                        )
                    else:
                        summary["overallMeanDensity"] = 0.0
                    del summary["totalDensitySum"]

                # Build final payload
                final_payload = {
                    "_schema": "mc.geometry.research.full_profile.v1",
                    "modelPath": model_path,
                    "totalLayers": num_layers,
                    "totalProbes": len(probes),
                    "completedLayers": sorted(completed_layers),
                    "domainSummaries": list(domain_summaries.values()),
                    "layerProfiles": [
                        layer_profiles[i] for i in sorted(layer_profiles.keys())
                    ],
                }

                profile_output.write_text(json.dumps(final_payload, indent=2))
                logger.info("Profile saved to %s", profile_output)

                results.append({
                    "model": model_name,
                    "status": "complete",
                    "layers": num_layers,
                    "output": str(profile_output),
                })

            except Exception as exc:
                logger.error("Failed to profile %s: %s", model_name, exc)
                results.append({
                    "model": model_name,
                    "status": "failed",
                    "error": str(exc),
                })

            # Clean memory after each model (CRITICAL)
            logger.info("Cleaning memory after model completion...")
            cleanup_memory()

        # Final output
        summary = {
            "_schema": "mc.geometry.research.batch_profile.v1",
            "outputDir": str(out_path),
            "totalModels": len(model_paths),
            "completedModels": sum(1 for r in results if r.get("status") in ("complete", "already_complete")),
            "failedModels": sum(1 for r in results if r.get("status") == "failed"),
            "results": results,
        }

        if context.output_format == "text":
            lines = [
                "",
                "=" * 60,
                "BATCH PROFILING COMPLETE",
                "=" * 60,
                f"Output directory: {out_path}",
                f"Models processed: {len(model_paths)}",
                f"Completed: {summary['completedModels']}",
                f"Failed: {summary['failedModels']}",
                "",
                "Results:",
            ]
            for r in results:
                status_icon = (
                    "[done]"
                    if r.get("status") in ("complete", "already_complete")
                    else "[fail]"
                )
                lines.append(f"  {status_icon} {r['model']}: {r.get('status', 'unknown')}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(summary, context.output_format, context.pretty)
