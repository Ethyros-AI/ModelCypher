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

"""CLI for null-space constrained transplant (AlphaEdit-style).

Transplants functional behavior from source to target while preserving
boundary relationships. Uses the mathematical guarantee:
    A_boundary @ W' = A_boundary @ W_target

Reference: AlphaEdit (ICLR 2025 Outstanding Paper)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def validate_transplant_geometry(
    weight_before: "Array",
    weight_after: "Array",
    weight_source: "Array",
    activations_core: "Array",
    activations_boundary: "Array",
    backend: "Backend",
) -> dict:
    """Validate the geometric guarantees of null-space transplant.

    Measures:
    1. Boundary preservation error (should be ~0) - the mathematical guarantee
    2. Core output shift direction (should be toward source)
    3. Functional alignment gain (CKA improvement)

    The geometry has rules. This function measures whether those rules were followed.
    """
    from modelcypher.core.domain.geometry.cka import compute_cka

    # Metric 1: Boundary preservation (the mathematical guarantee)
    # If A_boundary @ (W' - W_t) is not ~0, the null-space projection is wrong
    delta_applied = weight_after - weight_before
    boundary_output_change = backend.matmul(activations_boundary, backend.transpose(delta_applied))
    boundary_baseline = backend.matmul(activations_boundary, backend.transpose(weight_before))
    backend.eval(boundary_output_change, boundary_baseline)

    boundary_error = float(backend.to_numpy(backend.norm(boundary_output_change)))
    boundary_baseline_norm = float(backend.to_numpy(backend.norm(boundary_baseline)))
    boundary_relative_error = boundary_error / max(boundary_baseline_norm, 1e-10)

    # Metric 2: Core output shift direction
    # Did the core outputs move toward source's outputs?
    output_before = backend.matmul(activations_core, backend.transpose(weight_before))
    output_after = backend.matmul(activations_core, backend.transpose(weight_after))
    output_source = backend.matmul(activations_core, backend.transpose(weight_source))
    backend.eval(output_before, output_after, output_source)

    dist_before = float(backend.to_numpy(backend.norm(output_before - output_source)))
    dist_after = float(backend.to_numpy(backend.norm(output_after - output_source)))

    if dist_before > 1e-10:
        alignment_improvement = (dist_before - dist_after) / dist_before
    else:
        alignment_improvement = 0.0  # Already at source

    # Metric 3: Functional alignment (CKA)
    # Did the core behavior become more similar to source?
    cka_before = compute_cka(output_before, output_source, backend=backend)
    cka_after = compute_cka(output_after, output_source, backend=backend)

    functional_alignment_gain = cka_after.best - cka_before.best

    return {
        "boundary_preservation_error": boundary_relative_error,
        "boundary_guarantee_holds": boundary_relative_error < 1e-4,
        "core_dist_to_source_before": dist_before,
        "core_dist_to_source_after": dist_after,
        "alignment_improvement": alignment_improvement,
        "moved_toward_source": alignment_improvement > 0,
        "cka_before": cka_before.best,
        "cka_after": cka_after.best,
        "functional_alignment_gain": functional_alignment_gain,
    }


app = typer.Typer(
    name="transplant",
    help="Null-space constrained knowledge transplant (AlphaEdit-style).",
    no_args_is_help=True,
)


@app.command("run")
def transplant_run(
    source: Annotated[str, typer.Option(help="Source model path (dense knowledge)")],
    target: Annotated[str, typer.Option(help="Target model path (sparse regions)")],
    output_dir: Annotated[str, typer.Option(help="Output directory for transplanted model")],
    core_domain: Annotated[
        str,
        typer.Option(help="Domain of concepts to transplant (e.g., 'mathematical')"),
    ] = "mathematical",
    target_layer: Annotated[
        int,
        typer.Option(help="Target layer for transplant (default: auto-detect)"),
    ] = -1,
    output: Annotated[str, typer.Option(help="Output format")] = "json",
) -> None:
    """Transplant knowledge from source to target model.

    Uses null-space projection to preserve boundary relationships while
    replacing core (sparse) concepts with source's dense representations.

    Example:
        mc geometry transplant run \\
            --source /path/to/qwen2 \\
            --target /path/to/smolm \\
            --output-dir /path/to/output \\
            --core-domain mathematical
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain.agents.unified_atlas import AtlasDomain, UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.cross_dimensional_projection import (
        ProjectionMethod,
        project_cross_dimensional,
    )
    from modelcypher.core.domain.geometry.null_space_filter import NullSpaceFilterConfig
    from modelcypher.core.domain.geometry.transplant import (
        compute_transplant_delta,
        partition_core_boundary,
    )

    backend = get_default_backend()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    domain_map = {domain.value: domain for domain in AtlasDomain}
    core_atlas_domain = domain_map.get(core_domain.lower())
    if core_atlas_domain is None:
        typer.echo(f"Unknown domain: {core_domain}. Valid: {list(domain_map.keys())}")
        raise typer.Exit(1)

    # Get core probe IDs from atlas
    core_probes = UnifiedAtlasInventory.probes_by_domain({core_atlas_domain})
    core_probe_ids = {p.probe_id for p in core_probes}
    typer.echo(f"Core probes ({core_domain}): {len(core_probe_ids)}")

    # Get all probes for boundary computation
    all_probes = UnifiedAtlasInventory.all_probes()
    all_probe_ids = [p.probe_id for p in all_probes]
    # Use first support text, or fallback to name + description
    all_probe_texts = [
        p.support_texts[0] if p.support_texts else f"{p.name}: {p.description}"
        for p in all_probes
    ]
    typer.echo(f"Total probes: {len(all_probes)}")

    # Load models
    from modelcypher.adapters.model_loader import load_model_for_training

    typer.echo(f"Loading source model: {source}")
    source_model, source_tokenizer = load_model_for_training(source)

    typer.echo(f"Loading target model: {target}")
    target_model, target_tokenizer = load_model_for_training(target)

    # Determine target layer
    num_layers = len(target_model.layers) if hasattr(target_model, "layers") else 16
    if target_layer < 0:
        # From research: Layer 5 has 87% of transfer candidates
        target_layer = min(5, num_layers - 1)
    typer.echo(f"Target layer: {target_layer}")

    # Collect activations through both models
    typer.echo("Collecting probe activations...")

    def get_activations(
        model: object, tokenizer: object, texts: list[str], layer: int, use_mlp_intermediate: bool = True
    ) -> list:
        """Collect activations for probe texts at specified layer.

        Args:
            use_mlp_intermediate: If True, collect intermediate MLP activations (after up_proj)
                                  which match down_proj input dimension. If False, collect
                                  layer output activations.
        """
        import mlx.core as mx
        from mlx import nn

        activations = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            elif hasattr(tokens, "ids"):
                token_ids = list(tokens.ids)
            else:
                token_ids = list(tokens)

            x = mx.array([token_ids])
            # Forward through embedding and layers up to target
            if hasattr(model, "model"):
                inner = model.model
            else:
                inner = model

            h = inner.embed_tokens(x)
            for i, layer_module in enumerate(inner.layers):
                if use_mlp_intermediate and i == layer:
                    # Get intermediate MLP activation (after up_proj, before down_proj)
                    residual = h
                    h_norm = layer_module.input_layernorm(h)

                    # Self-attention (skip for now, just add residual)
                    attn_out = layer_module.self_attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h = h + attn_out

                    # Post-attention norm
                    h_post = layer_module.post_attention_layernorm(h)

                    # MLP intermediate: up_proj and gate_proj
                    mlp = layer_module.mlp
                    up = mlp.up_proj(h_post)
                    gate = mlp.gate_proj(h_post)
                    # Intermediate activation: silu(gate) * up
                    intermediate = nn.silu(gate) * up
                    mx.eval(intermediate)

                    # Take mean over sequence length
                    h_mean = mx.mean(intermediate, axis=1).squeeze()
                    mx.eval(h_mean)
                    activations.append(h_mean)
                    break
                else:
                    # Normal forward pass
                    result = layer_module(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result
                    if i == layer and not use_mlp_intermediate:
                        h_mean = mx.mean(h, axis=1).squeeze()
                        mx.eval(h_mean)
                        activations.append(h_mean)
                        break

        return activations

    source_acts = get_activations(source_model, source_tokenizer, all_probe_texts, target_layer)
    target_acts = get_activations(target_model, target_tokenizer, all_probe_texts, target_layer)

    # Stack activations
    import mlx.core as mx

    source_stacked = mx.stack(source_acts, axis=0)
    target_stacked = mx.stack(target_acts, axis=0)
    mx.eval(source_stacked, target_stacked)
    typer.echo(f"Activation shape: {target_stacked.shape}")

    # Partition into core and boundary
    typer.echo("Partitioning probes into core and boundary...")
    partition = partition_core_boundary(
        activations=target_stacked,
        probe_ids=all_probe_ids,
        core_probe_ids=core_probe_ids,
        boundary_k=10,  # 10 nearest neighbors per core probe
        geodesic_k_neighbors=15,
        backend=backend,
    )

    typer.echo(f"Core indices: {len(partition.core_indices)}")
    typer.echo(f"Boundary indices: {len(partition.boundary_indices)}")

    if not partition.core_indices:
        typer.echo("No core probes found in activations. Check domain selection.")
        raise typer.Exit(1)

    # Extract core and boundary activations
    core_target_acts = backend.take(
        target_stacked,
        backend.array(partition.core_indices, dtype="int32"),
        axis=0,
    )
    boundary_target_acts = backend.take(
        target_stacked,
        backend.array(partition.boundary_indices, dtype="int32"),
        axis=0,
    )
    backend.eval(core_target_acts, boundary_target_acts)

    typer.echo(f"Core activations: {core_target_acts.shape}")
    typer.echo(f"Boundary activations: {boundary_target_acts.shape}")

    # Get the weight matrix to transplant
    if hasattr(target_model, "model"):
        target_inner = target_model.model
        source_inner = source_model.model
    else:
        target_inner = target_model
        source_inner = source_model

    target_layer_module = target_inner.layers[target_layer]
    source_layer_module = source_inner.layers[target_layer]

    # Target: MLP down projection (matches hidden dimension from activations)
    target_o_proj = target_layer_module.mlp.down_proj
    source_o_proj = source_layer_module.mlp.down_proj

    # Handle quantized vs non-quantized weights
    def get_dequantized_weight(proj: object) -> mx.array:
        """Extract dequantized weight from linear or quantized linear layer."""
        if hasattr(proj, "scales"):
            # QuantizedLinear - dequantize
            biases = getattr(proj, "biases", None)
            return mx.dequantize(
                proj.weight,
                scales=proj.scales,
                biases=biases,
                group_size=proj.group_size,
                bits=proj.bits,
                mode=getattr(proj, "mode", "affine"),
            )
        else:
            # Regular Linear
            return proj.weight

    target_weight = get_dequantized_weight(target_o_proj)
    source_weight = get_dequantized_weight(source_o_proj)
    mx.eval(target_weight, source_weight)

    typer.echo(f"Target weight shape: {target_weight.shape}")
    typer.echo(f"Source weight shape: {source_weight.shape}")

    # Check dimension compatibility
    if target_weight.shape != source_weight.shape:
        typer.echo("Dimension mismatch - projecting source to target shape.")
        projection = project_cross_dimensional(
            source=source_weight,
            target=target_weight,
            method=ProjectionMethod.GRAM_TRANSPORT,
            backend=backend,
        )
        source_weight_aligned = projection.projected
        target_weight_matched = target_weight
        backend.eval(source_weight_aligned, target_weight_matched)
    else:
        source_weight_aligned = source_weight
        target_weight_matched = target_weight

    # Compute transplant delta
    typer.echo("Computing transplant delta with null-space projection...")
    config = NullSpaceFilterConfig(rank_threshold=1e-6)
    result = compute_transplant_delta(
        weight_target=target_weight_matched,
        weight_source_aligned=source_weight_aligned,
        activations_core=core_target_acts,
        activations_boundary=boundary_target_acts,
        backend=backend,
        nullspace_config=config,
    )

    typer.echo(f"Transplant applied: {result.applied}")
    typer.echo(f"Null dimension: {result.null_dim}")
    typer.echo(f"Delta norm: {result.delta_norm:.4f}")
    typer.echo(f"Filtered norm: {result.filtered_norm:.4f}")
    typer.echo(f"Preserved fraction: {result.preserved_fraction:.4f}")
    typer.echo(f"Projection loss: {result.projection_loss:.4f}")

    # Save results
    result_dict = {
        "source": source,
        "target": target,
        "core_domain": core_domain,
        "target_layer": target_layer,
        "core_probes": len(partition.core_indices),
        "boundary_probes": len(partition.boundary_indices),
        "applied": result.applied,
        "null_dim": result.null_dim,
        "delta_norm": result.delta_norm,
        "filtered_norm": result.filtered_norm,
        "preserved_fraction": result.preserved_fraction,
        "projection_loss": result.projection_loss,
    }

    result_path = output_path / "transplant_result.json"
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    typer.echo(f"Results saved to: {result_path}")

    # Apply transplant and test inference if successful
    if result.applied:
        typer.echo("\nApplying transplanted weight and testing inference...")

        # Get the merged weight from result
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)

        # Update the target model's weight
        # For non-quantized models, directly update the weight parameter
        is_quantized = hasattr(target_o_proj, "scales")
        if not is_quantized:
            try:
                merged_weight = merged_weight.astype(target_o_proj.weight.dtype)
            except Exception:
                pass
            target_o_proj.weight = merged_weight
            typer.echo("Weight updated successfully.")
        else:
            typer.echo("Note: Quantized model - re-quantization required to save weights.")

        # === GEOMETRIC VALIDATION ===
        # Measure what the geometry actually did - no assumptions
        typer.echo("\n=== Geometric Validation ===")
        validation = validate_transplant_geometry(
            weight_before=target_weight_matched,
            weight_after=merged_weight,
            weight_source=source_weight_aligned,
            activations_core=core_target_acts,
            activations_boundary=boundary_target_acts,
            backend=backend,
        )

        # Metric 1: Boundary preservation (the mathematical guarantee)
        typer.echo(f"Boundary preservation error: {validation['boundary_preservation_error']:.2e}")
        if validation["boundary_guarantee_holds"]:
            typer.echo("  [OK] Boundary guarantee holds (null-space projection correct)")
        else:
            typer.echo("  [FAIL] Boundary guarantee VIOLATED - check implementation")

        # Metric 2: Core output shift direction
        typer.echo(f"Core distance to source - before: {validation['core_dist_to_source_before']:.4f}")
        typer.echo(f"Core distance to source - after:  {validation['core_dist_to_source_after']:.4f}")
        typer.echo(f"Alignment improvement: {validation['alignment_improvement'] * 100:.1f}%")
        if validation["moved_toward_source"]:
            typer.echo("  [OK] Core outputs moved TOWARD source (transplant succeeded)")
        else:
            typer.echo("  [--] Core outputs did NOT move toward source")

        # Metric 3: Functional alignment (CKA)
        typer.echo(f"CKA with source - before: {validation['cka_before']:.4f}")
        typer.echo(f"CKA with source - after:  {validation['cka_after']:.4f}")
        typer.echo(f"Functional alignment gain: {validation['functional_alignment_gain']:.4f}")

        # Update result_dict with validation metrics
        result_dict.update({
            "boundary_preservation_error": validation["boundary_preservation_error"],
            "boundary_guarantee_holds": validation["boundary_guarantee_holds"],
            "core_dist_to_source_before": validation["core_dist_to_source_before"],
            "core_dist_to_source_after": validation["core_dist_to_source_after"],
            "alignment_improvement": validation["alignment_improvement"],
            "moved_toward_source": validation["moved_toward_source"],
            "cka_before": validation["cka_before"],
            "cka_after": validation["cka_after"],
            "functional_alignment_gain": validation["functional_alignment_gain"],
        })

        # Quick inference test
        test_prompt = "What is the Fibonacci sequence? Answer briefly:"
        typer.echo(f"\nTest prompt: {test_prompt}")

        tokens = target_tokenizer.encode(test_prompt, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        elif hasattr(tokens, "ids"):
            token_ids = list(tokens.ids)
        else:
            token_ids = list(tokens)

        x = mx.array([token_ids])
        # Simple greedy decode (just a few tokens to test coherence)
        for _ in range(30):
            logits = target_model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            x = mx.concatenate([x, next_token[:, None]], axis=1)
            # Stop on EOS
            if int(next_token[0]) == target_tokenizer.eos_token_id:
                break

        # Decode response
        output_tokens = x[0].tolist()
        response = target_tokenizer.decode(output_tokens)
        typer.echo(f"Response: {response}")

        result_dict["test_response"] = response

        if not is_quantized:
            try:
                loader = MLXModelLoader()
                weights = loader.load_weights(target)

                layer_token = f"layers.{target_layer}."
                candidates = [
                    key for key in weights
                    if layer_token in key and "mlp" in key and "down_proj" in key and key.endswith("weight")
                ]
                weight_key = candidates[0] if candidates else None

                if weight_key is None:
                    typer.echo("Could not locate down_proj weight key in safetensors; skipping save.")
                else:
                    try:
                        merged_to_save = merged_weight.astype(weights[weight_key].dtype)
                    except Exception:
                        merged_to_save = merged_weight
                    weights[weight_key] = merged_to_save
                    output_weights_path = output_path / "model.safetensors"
                    mx.save_safetensors(str(output_weights_path), weights)

                    for config_file in [
                        "config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                    ]:
                        src = Path(target) / config_file
                        if src.exists():
                            shutil.copy(src, output_path / config_file)

                    result_dict["output_model"] = str(output_weights_path)
                    typer.echo(f"Saved transplanted weights to: {output_weights_path}")
            except Exception as e:
                typer.echo(f"Failed to save transplanted model: {e}")

        # Save updated results
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    if output == "json":
        typer.echo(json.dumps(result_dict, indent=2))
