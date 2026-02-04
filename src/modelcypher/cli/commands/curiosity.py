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

"""Curiosity daemon CLI commands.

Commands for controlling the Active Inference curiosity daemon:
    mc curiosity status --model <path>
    mc curiosity weights --model <path>
    mc curiosity analyze --model <path> --activations <path>

All thresholds derived from sqrt(eps) - machine precision.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("status")
def curiosity_status(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
) -> None:
    """Show curiosity policy status for a model.

    Returns the EFE-derived thresholds and current exploration state.
    All values derived from sqrt(eps) - machine precision.

    Examples:
        mc curiosity status --model /path/to/model
    """
    context = _context(ctx)

    from modelcypher.cli.validation import validate_model_path

    validate_model_path(model, context=context)

    from modelcypher.core.domain.continual.curiosity_policy import EFECuriosityPolicy

    backend = get_backend()
    policy = EFECuriosityPolicy(backend=backend)

    payload = {
        "modelPath": model,
        "sqrtEps": policy.sqrt_eps,
        "explorationTemperatureFormula": "mean_eigenscore / sqrt(eps)",
        "consolidationThreshold": 2 * policy.sqrt_eps,
        "capacityThreshold": policy.sqrt_eps,
        "actions": {
            "COMPLETE": f"mean_eigenscore <= {policy.sqrt_eps:.2e}",
            "WAIT": f"mean_capacity <= {policy.sqrt_eps:.2e}",
            "CONSOLIDATE": f"mean_eigenscore > {2 * policy.sqrt_eps:.2e} AND mean_capacity > {policy.sqrt_eps:.2e}",
            "PROBE": "select top candidate by epistemic value",
        },
        "formulas": {
            "epistemic_value": "eigenscore × capacity_fraction",
            "efe": "risk + ambiguity",
            "risk": "(1 - capacity_fraction)²",
            "ambiguity": "eigenscore",
        },
    }

    if context.output_format == "text":
        lines = [
            "CURIOSITY POLICY STATUS (Active Inference / EFE)",
            "",
            f"Model: {model}",
            f"sqrt(eps): {policy.sqrt_eps:.2e}",
            "",
            "DECISION THRESHOLDS (geometry-derived):",
            f"  Consolidation: mean_eigenscore > {2 * policy.sqrt_eps:.2e}",
            f"  Capacity: mean_capacity > {policy.sqrt_eps:.2e}",
            "",
            "EFE FORMULAS:",
            "  epistemic_value = eigenscore × capacity_fraction",
            "  efe = risk + ambiguity",
            "  risk = (1 - capacity_fraction)²",
            "  ambiguity = eigenscore",
            "",
            "EXPLORATION TEMPERATURE:",
            "  T = mean_eigenscore / sqrt(eps)",
            "  T >> 1: explore uniformly",
            "  T << 1: exploit best candidate",
            "  T ~ 1: balanced",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("weights")
def curiosity_weights(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    activations: str = typer.Option(
        None,
        "--activations",
        help="Path to activation corpus (safetensors or numpy)",
    ),
) -> None:
    """Compute geometry-derived acquisition weights.

    Returns the composite acquisition weights derived from:
    - coverage_radius (k-center radius)
    - mean_local_id (mean intrinsic dimension)

    The weight formula:
        w = 1 / (1 + coverage_radius / mean_local_id)
        coreset_weight = 1 - w
        local_weight = w (split between coverage and density)

    Examples:
        mc curiosity weights --model /path/to/model --activations ./corpus.safetensors
    """
    context = _context(ctx)

    from modelcypher.cli.validation import validate_model_path

    validate_model_path(model, context=context)

    from modelcypher.core.domain.geometry.acquisition_composite import CompositeAcquisition

    backend = get_backend()

    # If activations provided, compute actual weights
    if activations:
        activations_path = Path(activations)
        if not activations_path.exists():
            error = ErrorDetail(
                code="MC-1070",
                title="Activations file not found",
                detail=f"File does not exist: {activations}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        try:
            if activations_path.suffix == ".safetensors":
                weights = backend.load_safetensors(str(activations_path))
                keys = list(weights.keys())
                if not keys:
                    raise ValueError("No tensors in safetensors file")
                corpus = weights[keys[0]]
            elif activations_path.suffix in (".npy", ".npz"):
                error = ErrorDetail(
                    code="MC-1071",
                    title="Legacy format not supported",
                    detail=f"NumPy format not supported: {activations}. Please convert to .safetensors format.",
                    hint="Convert using: python -c \"import numpy as np; from safetensors.numpy import save_file; data = np.load('file.npz'); save_file({'data': data['arr_0']}, 'file.safetensors')\"",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)
            else:
                error = ErrorDetail(
                    code="MC-1072",
                    title="Unsupported file format",
                    detail=f"Expected .safetensors: {activations}",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)

            backend.eval(corpus)
            acquisition = CompositeAcquisition(backend=backend)
            weights = acquisition.get_weights(corpus)

            payload = {
                "modelPath": model,
                "activationsPath": activations,
                "corpusSize": int(corpus.shape[0]),
                "hiddenDim": int(corpus.shape[1]) if len(corpus.shape) > 1 else 0,
                "coverageRadius": weights.coverage_radius,
                "meanLocalId": weights.mean_local_id,
                "weights": {
                    "coreset": weights.coreset_weight,
                    "coverage": weights.coverage_weight,
                    "density": weights.density_weight,
                },
                "formula": "w = 1 / (1 + coverage_radius / mean_local_id)",
            }

        except Exception as exc:
            error = ErrorDetail(
                code="MC-1072",
                title="Failed to load activations",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
    else:
        # No activations - show formula only
        acquisition = CompositeAcquisition(backend=backend)

        payload = {
            "modelPath": model,
            "sqrtEps": acquisition.sqrt_eps,
            "formula": "w = 1 / (1 + coverage_radius / mean_local_id)",
            "interpretation": {
                "large_radius": "Sparse corpus → prioritize global coverage (coreset)",
                "small_radius": "Dense corpus → prioritize local exploration (manifold)",
                "high_id": "Complex manifold → more local exploration",
            },
            "note": "Provide --activations to compute actual weights",
        }

    if context.output_format == "text":
        lines = [
            "COMPOSITE ACQUISITION WEIGHTS (geometry-derived)",
            "",
            f"Model: {model}",
        ]
        if "corpusSize" in payload:
            lines.extend(
                [
                    f"Corpus: {payload['corpusSize']} points, {payload['hiddenDim']} dims",
                    f"Coverage Radius: {payload['coverageRadius']:.4f}",
                    f"Mean Local ID: {payload['meanLocalId']:.4f}",
                    "",
                    "WEIGHTS:",
                    f"  Coreset (global): {payload['weights']['coreset']:.4f}",
                    f"  Coverage (local): {payload['weights']['coverage']:.4f}",
                    f"  Density (complexity): {payload['weights']['density']:.4f}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "WEIGHT FORMULA:",
                    "  w = 1 / (1 + coverage_radius / mean_local_id)",
                    "",
                    "INTERPRETATION:",
                    "  Large radius → sparse corpus → prioritize global (coreset)",
                    "  Small radius → dense corpus → prioritize local (manifold)",
                    "  High local ID → complex manifold → more exploration",
                    "",
                    "NOTE: Provide --activations to compute actual weights",
                ]
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("analyze")
def curiosity_analyze(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    candidates: str = typer.Option(
        ..., "--candidates", help="Path to candidate activations (safetensors/numpy)"
    ),
    corpus: str = typer.Option(
        ..., "--corpus", help="Path to corpus activations (safetensors/numpy)"
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of top candidates to show"),
) -> None:
    """Analyze candidates using composite acquisition.

    Computes acquisition scores combining:
    - Coreset contribution (k-center geodesic distance)
    - Coverage contribution (directional gap alignment)
    - Density contribution (local intrinsic dimension factor)

    Examples:
        mc curiosity analyze --model /path/to/model \\
            --candidates ./candidates.safetensors \\
            --corpus ./corpus.safetensors
    """
    context = _context(ctx)

    from modelcypher.cli.validation import validate_model_path

    validate_model_path(model, context=context)

    from modelcypher.core.domain.geometry.acquisition_composite import CompositeAcquisition

    backend = get_backend()

    def load_array(path_str: str, name: str):
        path = Path(path_str)
        if not path.exists():
            error = ErrorDetail(
                code="MC-1073",
                title=f"{name} file not found",
                detail=f"File does not exist: {path_str}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        try:
            if path.suffix == ".safetensors":
                weights = backend.load_safetensors(str(path))
                keys = list(weights.keys())
                if not keys:
                    raise ValueError("No tensors in safetensors file")
                arr = weights[keys[0]]
            elif path.suffix in (".npy", ".npz"):
                raise ValueError(
                    f"NumPy format not supported: {path}. Please convert to .safetensors format."
                )
            else:
                raise ValueError(f"Unsupported format: {path.suffix}. Use .safetensors.")

            backend.eval(arr)
            return arr
        except Exception as exc:
            error = ErrorDetail(
                code="MC-1074",
                title=f"Failed to load {name}",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    candidates_arr = load_array(candidates, "Candidates")
    corpus_arr = load_array(corpus, "Corpus")

    acquisition = CompositeAcquisition(backend=backend)
    result = acquisition.score(candidates_arr, corpus_arr)

    top_scores = result.select_top_k(top_k)

    payload = {
        "modelPath": model,
        "candidatesPath": candidates,
        "corpusPath": corpus,
        "nCandidates": int(candidates_arr.shape[0]),
        "nCorpus": int(corpus_arr.shape[0]),
        "coverageRadius": result.coverage_radius,
        "meanLocalId": result.mean_local_id,
        "sparseFraction": result.sparse_fraction,
        "topCandidates": [
            {
                "rank": i + 1,
                "probeIdx": s.probe_idx,
                "score": s.score,
                "coresetContribution": s.coreset_contribution,
                "coverageContribution": s.coverage_contribution,
                "densityContribution": s.density_contribution,
            }
            for i, s in enumerate(top_scores)
        ],
    }

    if context.output_format == "text":
        lines = [
            "CURIOSITY ACQUISITION ANALYSIS",
            "",
            f"Model: {model}",
            f"Candidates: {int(candidates_arr.shape[0])} points",
            f"Corpus: {int(corpus_arr.shape[0])} points",
            "",
            "MANIFOLD METRICS:",
            f"  Coverage Radius: {result.coverage_radius:.4f}",
            f"  Mean Local ID: {result.mean_local_id:.4f}",
            f"  Sparse Fraction: {result.sparse_fraction:.4f}",
            "",
            f"TOP {len(top_scores)} CANDIDATES:",
        ]
        for i, s in enumerate(top_scores):
            lines.append(
                f"  {i + 1}. idx={s.probe_idx} score={s.score:.4f} "
                f"(coreset={s.coreset_contribution:.3f}, "
                f"coverage={s.coverage_contribution:.3f}, "
                f"density={s.density_contribution:.3f})"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("evaluate")
def curiosity_evaluate(
    ctx: typer.Context,
    eigenscore: float = typer.Option(..., help="Manifold sparsity [0, 1]"),
    capacity: float = typer.Option(..., help="Null-space capacity [0, 1]"),
) -> None:
    """Evaluate EFE scores for a probe candidate.

    Computes:
    - Epistemic value = eigenscore × capacity_fraction
    - EFE = risk + ambiguity
    - Recommended action

    Examples:
        mc curiosity evaluate --eigenscore 0.7 --capacity 0.5
    """
    context = _context(ctx)

    from modelcypher.core.domain.continual.curiosity_policy import (
        EFECuriosityPolicy,
        compute_efe,
        compute_epistemic_value,
    )

    backend = get_backend()
    policy = EFECuriosityPolicy(backend=backend)

    # Compute scores
    epistemic_value = compute_epistemic_value(eigenscore, capacity)
    efe = compute_efe(eigenscore, capacity)
    risk = (1.0 - capacity) ** 2
    ambiguity = eigenscore

    # Compute exploration temperature
    temp = policy.compute_exploration_temperature(eigenscore, policy.sqrt_eps)

    # Determine action
    if eigenscore <= policy.sqrt_eps:
        action = "COMPLETE (manifold dense)"
    elif capacity <= policy.sqrt_eps:
        action = "WAIT (no capacity)"
    elif eigenscore > 2 * policy.sqrt_eps and capacity > policy.sqrt_eps:
        action = "CONSOLIDATE (geometric conditions met)"
    else:
        action = "PROBE (select by epistemic value)"

    payload = {
        "inputs": {
            "eigenscore": eigenscore,
            "capacityFraction": capacity,
        },
        "scores": {
            "epistemicValue": epistemic_value,
            "efe": efe,
            "risk": risk,
            "ambiguity": ambiguity,
        },
        "explorationTemperature": temp,
        "recommendedAction": action,
        "thresholds": {
            "sqrtEps": policy.sqrt_eps,
            "consolidationEigenscore": 2 * policy.sqrt_eps,
        },
    }

    if context.output_format == "text":
        lines = [
            "EFE EVALUATION",
            "",
            "INPUTS:",
            f"  Eigenscore: {eigenscore:.4f}",
            f"  Capacity: {capacity:.4f}",
            "",
            "SCORES:",
            f"  Epistemic Value: {epistemic_value:.4f}",
            f"  EFE (lower = better): {efe:.4f}",
            f"    Risk: {risk:.4f}",
            f"    Ambiguity: {ambiguity:.4f}",
            "",
            f"Exploration Temperature: {temp:.4f}",
            f"Recommended Action: {action}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
