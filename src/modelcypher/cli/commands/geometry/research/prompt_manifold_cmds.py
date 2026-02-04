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

from dataclasses import asdict
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.support.array_utils import array_to_list

from .common import cleanup_memory, get_context


def _select_probes(probe_count: int | None, probes: list):
    if probe_count is None:
        return probes
    if probe_count <= 0:
        raise ValueError("probe-count must be positive.")
    if probe_count >= len(probes):
        return probes
    step = max(1, len(probes) // probe_count)
    return probes[::step][:probe_count]


def _select_indices(count: int, total: int) -> list[int]:
    if count >= total:
        return list(range(total))
    step = max(1, total // count)
    return list(range(0, total, step))[:count]


def _probe_prompt(probe) -> str:
    if probe.support_texts:
        return probe.support_texts[0]
    return probe.name


def register(app: typer.Typer) -> None:
    @app.command("prompt-manifold")
    def prompt_manifold(
        ctx: typer.Context,
        model: Path = typer.Argument(..., help="Path to model directory"),
        layer: int | None = typer.Option(
            None, "--layer", help="Layer index (defaults to middle layer)"
        ),
        probe_count: int | None = typer.Option(
            None, "--probe-count", help="Optional cap on number of probes"
        ),
        basis_rank: int | None = typer.Option(
            None, "--basis-rank", help="Override derived basis rank"
        ),
        sample_count: int | None = typer.Option(
            None,
            "--sample-count",
            help="Number of coefficient samples for Jacobian probes",
        ),
        projection_count: int | None = typer.Option(
            None,
            "--projection-count",
            help="Random projection count for Jacobian probes",
        ),
        base_prompt: str | None = typer.Option(
            None,
            "--base-prompt",
            help="Prompt to anchor manifold perturbations (defaults to first probe)",
        ),
        seed: int | None = typer.Option(
            None, "--seed", help="Random seed for projection vectors"
        ),
        output: Path | None = typer.Option(
            None, "--output-file", help="Path to save evidence JSON"
        ),
    ) -> None:
        """Probe prompt-manifold Jacobian rank using atlas prompts."""
        context = get_context(ctx)

        try:
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.cli.commands.geometry.helpers import (
                forward_through_backbone_embeddings,
                resolve_model_backbone,
            )
            from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory
            from modelcypher.core.domain.geometry.jacobian_rank import estimate_jacobian_rank
            from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
            from modelcypher.core.domain.geometry.prompt_manifold import (
                apply_prompt_basis,
                derive_prompt_manifold_basis,
            )

            probes = UnifiedAtlasInventory.all_probes()
            if not probes:
                raise ValueError("No atlas probes available.")

            selected = _select_probes(probe_count, probes)
            if not selected:
                raise ValueError("No probes selected.")

            if basis_rank is not None and basis_rank <= 0:
                raise ValueError("basis-rank must be positive.")
            if sample_count is not None and sample_count <= 0:
                raise ValueError("sample-count must be positive.")
            if projection_count is not None and projection_count < 0:
                raise ValueError("projection-count must be >= 0.")

            model_obj, tokenizer = load_model_for_training(str(model))
            resolved = resolve_model_backbone(
                model_obj, getattr(model_obj, "model_type", None)
            )
            if not resolved:
                raise ValueError("Could not resolve model architecture.")
            embed_tokens, layers_module, norm = resolved
            num_layers = len(layers_module)
            backend = get_backend()

            if layer is None:
                layer_idx = max(0, num_layers // 2)
            else:
                if layer < 0 or layer >= num_layers:
                    raise ValueError(
                        f"Layer {layer} out of range for model with {num_layers} layers."
                    )
                layer_idx = layer

            used_probes = []
            pooled_embeddings = []
            pending = []

            for probe in selected:
                prompt = _probe_prompt(probe)
                if not prompt:
                    continue
                tokens = tokenizer.encode(prompt)
                if not tokens:
                    continue
                input_ids = backend.array([tokens])
                embedded = embed_tokens(input_ids)
                pooled = backend.mean(embedded[0], axis=0)
                backend.async_eval(pooled)
                pooled_embeddings.append(pooled)
                pending.append(pooled)
                used_probes.append((probe, prompt))

            if pending:
                backend.eval(*pending)

            if not pooled_embeddings:
                raise ValueError("No valid probe prompts to embed.")

            embeddings_matrix = backend.stack(pooled_embeddings, axis=0)
            backend.eval(embeddings_matrix)

            basis = derive_prompt_manifold_basis(
                embeddings_matrix,
                basis_rank=basis_rank,
                backend=backend,
            )

            centered = embeddings_matrix - basis.mean
            if basis.basis_rank > 0:
                coeffs_matrix = backend.matmul(centered, backend.transpose(basis.basis))
            else:
                coeffs_matrix = backend.zeros(
                    (int(embeddings_matrix.shape[0]), 0), dtype=embeddings_matrix.dtype
                )
            backend.eval(coeffs_matrix)

            total_samples = len(used_probes)
            if sample_count is None:
                sample_count = min(total_samples, max(1, basis.basis_rank + 1))
            else:
                sample_count = min(sample_count, total_samples)

            sample_indices = _select_indices(sample_count, total_samples)
            idx_arr = backend.array(sample_indices)
            coeff_samples = backend.take(coeffs_matrix, idx_arr, axis=0)
            backend.eval(coeff_samples)

            if base_prompt is None:
                base_prompt = used_probes[0][1]
            base_tokens = tokenizer.encode(base_prompt)
            if not base_tokens:
                raise ValueError("Base prompt could not be tokenized.")
            base_input = backend.array([base_tokens])
            base_embeddings = embed_tokens(base_input)
            backend.eval(base_embeddings)

            if seed is not None:
                backend.random_seed(seed)

            if projection_count is None:
                projection_count = basis.basis_rank
            if projection_count < 0:
                projection_count = 0

            sample_reports = []
            if projection_count > 0 and basis.basis_rank > 0:
                proj_shape = (projection_count, basis.feature_dim)
                projections = backend.random_normal(proj_shape, dtype=base_embeddings.dtype)
                norms = backend.sqrt(
                    backend.sum(projections * projections, axis=1, keepdims=True)
                )
                eps = division_epsilon(backend, projections)
                eps_arr = backend.full(norms.shape, eps, dtype=norms.dtype)
                norms_safe = backend.where(norms > eps, norms, eps_arr)
                projections = projections / norms_safe
                backend.eval(projections)

                def scalar_projection(alpha, projection):
                    embedded = apply_prompt_basis(
                        base_embeddings,
                        basis.basis,
                        alpha,
                        backend=backend,
                    )
                    hidden = forward_through_backbone_embeddings(
                        embedded,
                        layers_module,
                        norm,
                        target_layer=layer_idx,
                        backend=backend,
                    )
                    pooled = backend.mean(hidden[0], axis=0)
                    return backend.sum(pooled * projection)

                value_and_grad = backend.value_and_grad(scalar_projection, argnums=0)

                for sample_pos, probe_idx in enumerate(sample_indices):
                    coeff_row = backend.take(
                        coeff_samples, backend.array([sample_pos]), axis=0
                    )
                    coeff_row = backend.squeeze(coeff_row)
                    backend.eval(coeff_row)

                    grads = []
                    for proj_idx in range(projection_count):
                        proj_row = backend.take(
                            projections, backend.array([proj_idx]), axis=0
                        )
                        proj_row = backend.squeeze(proj_row)
                        _value, grad = value_and_grad(coeff_row, proj_row)
                        backend.eval(grad)
                        grads.append(grad)

                    grad_matrix = backend.stack(grads, axis=0) if grads else None
                    if grad_matrix is None:
                        jacobian_rank = estimate_jacobian_rank(
                            backend.zeros((0, basis.basis_rank), dtype=coeff_row.dtype),
                            backend=backend,
                        )
                    else:
                        backend.eval(grad_matrix)
                        jacobian_rank = estimate_jacobian_rank(grad_matrix, backend=backend)

                    coeff_list = array_to_list(backend, coeff_row)
                    probe, prompt = used_probes[probe_idx]
                    sample_reports.append(
                        {
                            "probeID": probe.probe_id,
                            "name": probe.name,
                            "domain": probe.domain.value,
                            "prompt": prompt,
                            "coefficients": coeff_list,
                            "jacobianRank": asdict(jacobian_rank),
                        }
                    )
            else:
                for probe_idx in sample_indices:
                    probe, prompt = used_probes[probe_idx]
                    sample_reports.append(
                        {
                            "probeID": probe.probe_id,
                            "name": probe.name,
                            "domain": probe.domain.value,
                            "prompt": prompt,
                            "coefficients": [],
                            "jacobianRank": asdict(
                                estimate_jacobian_rank(
                                    backend.zeros((0, 0)), backend=backend
                                )
                            ),
                        }
                    )

            cleanup_memory()

            payload = {
                "_schema": "mc.geometry.research.prompt_manifold_jacobian.v1",
                "modelPath": str(model),
                "layer": layer_idx,
                "probeCount": len(used_probes),
                "basis": {
                    "rank": basis.basis_rank,
                    "effectiveRank": asdict(basis.effective_rank),
                    "sampleCount": basis.sample_count,
                    "featureDim": basis.feature_dim,
                    "scale": basis.scale,
                },
                "basePrompt": base_prompt,
                "sampleCount": sample_count,
                "projectionCount": projection_count,
                "samples": sample_reports,
            }

            if output:
                from modelcypher.utils.json import dump_json

                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(dump_json(payload, pretty=True))

            if context.output_format == "text":
                lines = [
                    "",
                    "=" * 60,
                    "PROMPT MANIFOLD JACOBIAN REPORT",
                    "=" * 60,
                    f"Model: {model}",
                    f"Layer: {layer_idx}",
                    f"Probe count: {len(used_probes)}",
                    f"Basis rank: {basis.basis_rank}",
                    f"Projection count: {projection_count}",
                    f"Sample count: {sample_count}",
                ]
                for item in sample_reports:
                    rank = item["jacobianRank"]
                    lines.append(
                        f"- {item['name']} "
                        f"(Renyi {rank['renyi_effective_rank']:.4f}, "
                        f"Shannon {rank['shannon_effective_rank']:.4f})"
                    )
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(payload, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Prompt manifold probe failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
