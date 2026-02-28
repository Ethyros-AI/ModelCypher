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

"""Quantize CLI - Tikhonov quantization correction.

Applies Marchenko-Pastur weighted Tikhonov correction to partially reverse
quantization damage using the activation covariance eigenbasis.

Commands:
    mc quantize correct -q QUANTIZED -f FP_REF -o OUTPUT
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

logger = logging.getLogger(__name__)

quantize_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _validate_path(path: Path, label: str, context: CLIContext) -> None:
    """Validate a model path exists, exit with error if not."""
    if not path.exists():
        error = ErrorDetail(
            code="MC-4001",
            title=f"{label} not found",
            detail=f"Path does not exist: {path}",
            hint="Provide a valid path to a model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


def _load_eval_texts(dataset_path: str, n_samples: int) -> list[str]:
    """Load text samples from JSONL dataset."""
    texts: list[str] = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
    return texts[:n_samples]


def _extract_fp_weights(model: Any, backend: Any) -> dict[str, Any]:
    """Extract all projection weights from a model as a flat dict.

    Returns weight tensors keyed by their canonical path
    (e.g., "model.layers.0.self_attn.q_proj.weight").
    """
    from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

    base = getattr(model, "model", model)
    weights: dict[str, Any] = {}
    all_params = dict(model.parameters()) if hasattr(model, "parameters") else {}

    attn_projs = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projs = ("up_proj", "down_proj", "gate_proj")

    for layer_idx, layer in enumerate(base.layers):
        for block_name, proj_names in (
            ("self_attn", attn_projs),
            ("mlp", mlp_projs),
        ):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is None:
                    continue
                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                if hasattr(proj, "weight"):
                    w = proj.weight
                    # Dequantize if needed (integer weights from quantized model)
                    w = dequantize_if_needed(w, key, all_params, backend)
                    weights[key] = w
    return weights


def _dequantize_model(model: Any, backend: Any) -> int:
    """Convert all QuantizedLinear modules to plain Linear.

    Uses the backend's dequantize protocol to avoid framework-specific imports.
    Returns number of projections dequantized.
    """
    from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

    base = getattr(model, "model", model)
    all_params = dict(model.parameters()) if hasattr(model, "parameters") else {}
    n_deq = 0

    attn_projs = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_projs = ("up_proj", "down_proj", "gate_proj")

    for layer_idx, layer in enumerate(base.layers):
        for block_name, proj_names in (
            ("self_attn", attn_projs),
            ("mlp", mlp_projs),
        ):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue

                w = proj.weight
                dtype_name = str(getattr(w, "dtype", ""))
                if "float" in dtype_name.lower():
                    continue  # Already float

                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                dequantized = dequantize_if_needed(w, key, all_params, backend)
                if dequantized is not w:
                    # Replace the quantized module with a plain linear
                    # This is framework-specific but uses duck typing
                    try:
                        import mlx.nn as nn

                        in_features = int(dequantized.shape[1])
                        out_features = int(dequantized.shape[0])
                        new_linear = nn.Linear(
                            in_features, out_features, bias=False
                        )
                        new_linear.weight = dequantized
                        backend.eval(new_linear.weight)
                        if hasattr(proj, "bias") and proj.bias is not None:
                            new_linear.bias = proj.bias
                        setattr(block, proj_name, new_linear)
                        n_deq += 1
                    except ImportError:
                        # Non-MLX backend — set weight directly
                        proj.weight = dequantized
                        n_deq += 1

    gc.collect()
    return n_deq


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert QuantizationCorrectionResult to serializable dict."""
    per_layer = []
    for lr in result.per_layer:
        layer_dict: dict[str, Any] = {
            "layer_idx": lr.layer_idx,
            "n_features": lr.n_features,
            "D_eff": lr.D_eff,
            "mp_edge": lr.mp_edge,
            "sigma_sq": lr.sigma_sq,
            "aspect_ratio": lr.aspect_ratio,
            "effective_rank": lr.effective_rank,
            "top_eigenvalues": lr.top_eigenvalues,
            "top_tikhonov_weights": lr.top_tikhonov_weights,
            "n_projections_corrected": len(lr.projections),
            "n_projections_skipped": len(lr.skipped_keys),
            "correction_fraction": lr.correction_fraction,
            "preserved_fraction": lr.preserved_fraction,
            "time_seconds": lr.time_seconds,
        }
        per_layer.append(layer_dict)

    return {
        "n_layers": result.n_layers,
        "n_projections_corrected": result.n_projections_corrected,
        "aggregate_correction_fraction": result.aggregate_correction_fraction,
        "aggregate_preserved_fraction": result.aggregate_preserved_fraction,
        "per_layer": per_layer,
    }


@quantize_app.command("correct")
def quantize_correct(
    ctx: typer.Context,
    quantized_model: str = typer.Option(
        ...,
        "--quantized-model",
        "-q",
        help="Path to quantized model",
    ),
    fp_model: str = typer.Option(
        ...,
        "--fp-model",
        "-f",
        help="Path to full-precision (bf16) reference model",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for corrected model",
    ),
    eval_dataset: str = typer.Option(
        "data/training/benchmark_val.jsonl",
        "--eval-dataset",
        "-e",
        help="Path to evaluation dataset (JSONL) for calibration",
    ),
    n_calibration: int = typer.Option(
        30,
        "--n-calibration",
        help=(
            "Number of calibration samples for activation covariance. "
            "30 >> D_eff~3-5 (measured). CLI-overridable, not a decision boundary."
        ),
    ),
    max_seq_len: int = typer.Option(
        128,
        "--max-seq-len",
        help="Maximum sequence length for calibration. Memory-compute tradeoff.",
    ),
) -> None:
    """Apply Tikhonov correction to a quantized model using its FP reference.

    Partially reverses quantization damage by projecting the quantization error
    through the activation covariance eigenbasis, weighted by Marchenko-Pastur
    derived Tikhonov regularization. Only corrects in high-variance activation
    directions where the model actually uses the representation.

    Algorithm:
        For each layer l (sequential):
          1. Collect activation covariance from calibration data
          2. Eigendecompose → eigenvectors V, eigenvalues λ
          3. MP noise edge α = σ² × (1 + √(D/N))²
          4. Tikhonov weights: w_i = λ_i / (λ_i + α)
          5. Correction: Delta = E @ V @ diag(w) @ V^T
          6. W_corrected = W_quantized + Delta

    Output fields (when --json):
        n_layers: Number of transformer layers
        n_projections_corrected: Weight matrices corrected
        aggregate_correction_fraction: Overall correction ratio
        per_layer: Per-layer diagnostics (D_eff, mp_edge, effective_rank)

    Example:
        mc quantize correct \\
          -q /path/to/4bit-model \\
          -f /path/to/bf16-reference \\
          -o /path/to/output
    """
    context = _context(ctx)
    q_path = Path(quantized_model)
    fp_path = Path(fp_model)

    _validate_path(q_path, "Quantized model", context)
    _validate_path(fp_path, "FP reference model", context)

    from modelcypher.cli.composition import get_backend

    backend = get_backend()

    # Load evaluation texts
    eval_texts = _load_eval_texts(eval_dataset, n_calibration)
    if not eval_texts:
        error = ErrorDetail(
            code="MC-4002",
            title="No evaluation texts",
            detail=f"No valid text samples in {eval_dataset}",
            hint="Provide a JSONL file with 'text' fields",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    logger.info("Loaded %d calibration texts from %s", len(eval_texts), eval_dataset)

    # Load FP model and extract reference weights
    logger.info("Loading FP reference model: %s", fp_path)
    fp_model_obj, _fp_tokenizer = backend.load_model(str(fp_path))
    fp_weights = _extract_fp_weights(fp_model_obj, backend)
    logger.info("Extracted %d FP weight matrices", len(fp_weights))
    del fp_model_obj, _fp_tokenizer
    gc.collect()

    # Load quantized model
    logger.info("Loading quantized model: %s", q_path)
    q_model, q_tokenizer = backend.load_model(str(q_path))

    # Dequantize (QuantizedLinear → Linear) so weights can be modified
    n_deq = _dequantize_model(q_model, backend)
    logger.info("Dequantized %d projections", n_deq)

    # Run sequential Tikhonov correction
    from modelcypher.core.use_cases.quantization_correction_service import (
        run_sequential_correction,
    )

    correction_result = run_sequential_correction(
        model=q_model,
        fp_weights=fp_weights,
        tokenizer=q_tokenizer,
        eval_texts=eval_texts,
        backend=backend,
        n_calibration=n_calibration,
        max_seq_len=max_seq_len,
    )

    # Save corrected model weights + config files
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving corrected model to %s", output_path)

    # Extract corrected weights from model
    corrected_weights = {}
    if hasattr(q_model, "parameters"):
        for k, v in q_model.parameters().items():
            corrected_weights[k] = v
    else:
        corrected_weights = dict(q_model.state_dict())

    backend.save_safetensors(
        str(output_path / "model.safetensors"), corrected_weights
    )

    # Copy config files from quantized model
    import shutil

    for config_name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src = q_path / config_name
        if src.exists():
            shutil.copy2(src, output_path / config_name)

    # Clean up
    del q_model, q_tokenizer, fp_weights
    gc.collect()

    # Output results
    result_dict = _result_to_dict(correction_result)
    result_dict["output_path"] = str(output_path)
    result_dict["quantized_model"] = str(q_path)
    result_dict["fp_model"] = str(fp_path)

    write_output(result_dict, context.output_format, context.pretty)
