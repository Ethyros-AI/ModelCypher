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

"""Model registry CLI commands.

Commands:
    mc model list     - List registered models
    mc model add      - Register a local model
    mc model delete   - Delete a model
    mc model info     - Inspect a model
    mc model capacity - Analyze per-layer spectral capacity
    mc model moe-profile - Analyze MoE routing + expert capacity
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from modelcypher.cli.composition import (
    get_capacity_analysis_service,
    get_model_probe_service,
    get_model_service,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.presenters import model_payload
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _write_probe_output(result: Any, context: CLIContext, model_path: str) -> None:
    """Write model probe output."""
    def _coerce_value(value: Any) -> Any:
        if type(value).__module__.startswith("unittest.mock"):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_coerce_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _coerce_value(val) for key, val in value.items()}
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item") and hasattr(value, "ndim") and getattr(value, "ndim", 1) == 0:
            return value.item()
        return str(value)

    def _layer_payload(layer: Any) -> dict[str, Any]:
        return {
            "name": _coerce_value(getattr(layer, "name", layer)),
            "type": _coerce_value(getattr(layer, "type", None)),
            "parameters": _coerce_value(getattr(layer, "parameters", None)),
            "shape": _coerce_value(getattr(layer, "shape", None)),
        }

    payload = {
        "architecture": _coerce_value(getattr(result, "architecture", None)),
        "parameterCount": _coerce_value(getattr(result, "parameter_count", None)),
        "vocabSize": _coerce_value(getattr(result, "vocab_size", None)),
        "hiddenSize": _coerce_value(getattr(result, "hidden_size", None)),
        "numAttentionHeads": _coerce_value(getattr(result, "num_attention_heads", None)),
        "quantization": _coerce_value(getattr(result, "quantization", None)),
        "layerCount": len(result.layers),
        "layerCountConfig": _coerce_value(getattr(result, "layer_count_config", None)),
        "layers": [_layer_payload(layer) for layer in result.layers[:20]],
    }

    # Detect MoE topology from config
    moe_topology = None
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            from modelcypher.core.domain.moe.topology import MoETopology

            raw_config = json.loads(config_path.read_text())
            text_cfg = raw_config.get("text_config")
            cfg = {**text_cfg, **raw_config} if isinstance(text_cfg, dict) else raw_config
            moe_topology = MoETopology.from_config(cfg)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "MoE topology detection failed: %s", exc,
            )

    if moe_topology is not None:
        payload["moe"] = {
            "numExperts": moe_topology.num_experts,
            "activePerToken": moe_topology.num_experts_per_tok,
            "moeIntermediateSize": moe_topology.moe_intermediate_size,
            "hasSharedExpert": moe_topology.has_shared_expert,
            "sharedExpertIntermediateSize": moe_topology.shared_expert_intermediate_size,
            "moeLayers": len(moe_topology.moe_layer_indices),
            "totalLayers": moe_topology.num_layers,
        }

    if context.output_format == "text":
        lines = [
            "MODEL INFO",
            f"Architecture: {result.architecture}",
            f"Parameters: {result.parameter_count:,}",
            f"Vocab Size: {result.vocab_size:,}",
            f"Hidden Size: {result.hidden_size}",
            f"Attention Heads: {result.num_attention_heads}",
            f"Layers: {len(result.layers)}",
        ]
        if result.layer_count_config:
            lines.append(f"Layers (config): {result.layer_count_config}")
        if result.quantization:
            lines.append(f"Quantization: {result.quantization}")
        if moe_topology is not None:
            lines.append("")
            lines.append("MoE Topology:")
            lines.append(f"  Experts: {moe_topology.num_experts}")
            lines.append(f"  Active per token: {moe_topology.num_experts_per_tok}")
            lines.append(f"  Expert intermediate: {moe_topology.moe_intermediate_size}")
            lines.append(
                f"  Shared expert: {'yes' if moe_topology.has_shared_expert else 'no'}"
                + (
                    f" (intermediate={moe_topology.shared_expert_intermediate_size})"
                    if moe_topology.shared_expert_intermediate_size
                    else ""
                )
            )
            lines.append(
                f"  MoE layers: {len(moe_topology.moe_layer_indices)}/{moe_topology.num_layers}"
            )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _normalize_target_modules(target_modules: list[str] | None) -> list[str]:
    if not target_modules:
        return []
    normalized: list[str] = []
    for raw in target_modules:
        for token in raw.split(","):
            stripped = token.strip()
            if stripped:
                normalized.append(stripped)
    return sorted(set(normalized))


def _validate_sort_by(sort_by: str) -> str:
    normalized = sort_by.strip().lower()
    allowed = {"null", "effective-rank", "recommended-rank"}
    if normalized not in allowed:
        raise ValueError(
            "Invalid --sort-by value. Use one of: null, effective-rank, recommended-rank."
        )
    return normalized


def _sort_label(sort_by: str) -> str:
    if sort_by == "null":
        return "Available Capacity"
    if sort_by == "effective-rank":
        return "Effective Rank"
    return "Recommended Rank"


def _format_capacity_text(report: Any, top: int, sort_by: str, lora_config_path: str | None) -> str:
    sorted_layers = report.sorted_layers(sort_by)
    top_layers = sorted_layers[:max(0, top)]

    if top_layers:
        name_width = max(
            len("Layer"),
            max(len(layer.layer_name) for layer in top_layers),
        )
    else:
        name_width = len("Layer")
    shape_width = len("Shape")

    lines = [
        f"Model: {report.model_name}",
        f"Total Parameters: {report.total_parameters:,}",
        (
            f"Mean Effective Rank: {report.mean_effective_rank:.1f} / "
            f"{report.reference_rank_dimension} "
            f"({report.mean_capacity_utilization * 100.0:.1f}%)"
        ),
        f"Mean Capacity Utilization: {report.mean_capacity_utilization * 100.0:.1f}%",
        "",
        f"Top {len(top_layers)} Layers by {_sort_label(sort_by)}:",
        (
            f"  {'Layer':<{name_width}}  {'Shape':<{shape_width}}  "
            f"{'Eff.Rank':>8}  {'Null%':>6}  {'Rec.Rank':>8}  {'Gap':>7}"
        ),
    ]

    for layer in top_layers:
        shape_text = f"{layer.weight_shape[0]}x{layer.weight_shape[1]}"
        gap_text = (
            "infx"
            if layer.spectral_gap_at_rank == float("inf")
            else f"{layer.spectral_gap_at_rank:.1f}x"
        )
        lines.append(
            f"  {layer.layer_name:<{name_width}}  {shape_text:<{shape_width}}  "
            f"{layer.effective_rank:>8.1f}  "
            f"{layer.null_space_fraction * 100.0:>5.1f}%  "
            f"{layer.recommended_rank:>8d}  {gap_text:>7}"
        )

    lines.append("")
    lines.append("Per-Layer LoRA Configuration (copy to training config):")
    for layer in report.sorted_layers("recommended-rank"):
        lines.append(f"  {layer.layer_name}: rank={layer.recommended_rank}")

    if lora_config_path is not None:
        lines.append("")
        lines.append(f"LoRA config written: {lora_config_path}")

    if report.failed_layers:
        lines.append("")
        lines.append(f"Skipped layers due to SVD failure: {len(report.failed_layers)}")

    return "\n".join(lines)


def _build_lora_config_payload(report: Any) -> dict[str, Any]:
    per_layer_rank = {
        layer.layer_name: {"rank": layer.recommended_rank}
        for layer in report.sorted_layers("recommended-rank")
    }
    return {
        "model": report.model_name,
        "modelPath": report.model_path,
        "targetModules": list(report.target_modules),
        "minDim": report.min_dim,
        "maxDim": report.max_dim,
        "perLayerRank": per_layer_rank,
    }


def _write_lora_config_file(path: str, payload: dict[str, Any]) -> str:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml

        serialized = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        serialized = json.dumps(payload, indent=2, sort_keys=False)

    output_path.write_text(serialized, encoding="utf-8")
    return str(output_path)


# --- Registry Commands ---


@app.command("list")
def model_list(ctx: typer.Context) -> None:
    """List all registered models.

    Returns an array of registered model entries with their aliases,
    paths, architecture, and parameter counts.

    Output fields (when --json, per entry):
        alias: Model alias
        path: Local filesystem path
        architecture: Detected architecture name
        parameters: Total parameter count

    Example:
        mc model list
    """
    context = _context(ctx)
    service = get_model_service()
    models = [model_payload(model) for model in service.list_models()]
    write_output(models, context.output_format, context.pretty)


@app.command("add")
def model_add(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Local path to model directory"),
    alias: str | None = typer.Option(None, "--alias", "-a", help="Alias for the model"),
    architecture: str | None = typer.Option(None, "--architecture"),
    parameters: int | None = typer.Option(None, "--parameters"),
) -> None:
    """Register a local model.

    Examples:
        mc model add /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
        mc model add ./models/my-model --alias my-model
    """
    context = _context(ctx)
    service = get_model_service()
    probe_service = get_model_probe_service()

    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-1001",
            title="Model not found",
            detail=f"Path does not exist: {model_path}",
            hint="Provide a valid local path to a model directory.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    resolved_alias = alias or model_path.name
    resolved_path = str(model_path)

    # Probe to detect architecture and parameters
    detected_arch = architecture
    detected_params = parameters
    if architecture is None or parameters is None:
        try:
            probe_result = probe_service.probe(resolved_path)
            if architecture is None:
                detected_arch = probe_result.architecture or "unknown"
            if parameters is None and probe_result.parameter_count:
                detected_params = probe_result.parameter_count
        except Exception as exc:
            if architecture is None:
                error = ErrorDetail(
                    code="MC-1002",
                    title="Model probe failed",
                    detail=str(exc),
                    hint="Provide --architecture when probing fails.",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)
            typer.echo(f"Warning: probe failed: {exc}", err=True)

    if detected_arch is None:
        detected_arch = "unknown"

    service.register_model(
        resolved_alias,
        resolved_path,
        detected_arch,
        parameters=detected_params,
    )

    payload = {
        "path": resolved_path,
        "alias": resolved_alias,
        "architecture": detected_arch,
        "parameters": detected_params,
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("delete")
def model_delete(
    ctx: typer.Context,
    model_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a registered model.

    Examples:
        mc model delete my-llama
    """
    context = _context(ctx)
    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete model '{model_id}' from the registry?"):
            raise typer.Exit(code=1)
    service = get_model_service()
    service.delete_model(model_id)
    write_output({"deleted": model_id}, context.output_format, context.pretty)


@app.command("info")
def model_info(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Inspect a model's architecture and structure.

    Probes a model directory to extract architecture, parameter count,
    layer structure, hidden dimensions, attention heads, and quantization
    status. Does not load full weights — reads config and metadata only.

    Output fields (when --json):
        architecture: Model architecture name (e.g., LlamaForCausalLM)
        parameterCount: Total parameter count
        vocabSize: Vocabulary size
        hiddenSize: Hidden dimension
        numAttentionHeads: Number of attention heads
        quantization: Quantization config if present
        layerCount: Number of layers detected
        layers: Per-layer details (name, type, parameters, shape)

    Example:
        mc model info /path/to/model
    """
    context = _context(ctx)
    service = get_model_probe_service()
    try:
        result = service.probe(model_path)
    except FileNotFoundError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model probe failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model probe failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        from modelcypher.cli.exit_codes import EXIT_RUNTIME

        error = ErrorDetail(
            code="MC-1001",
            title="Model probe failed",
            detail=str(exc),
            hint="Check backend runtime status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    _write_probe_output(result, context, model_path)


@app.command("capacity")
def model_capacity(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    top: int = typer.Option(10, "--top", "-n", min=1, help="Top N layers to show in text output"),
    sort_by: str = typer.Option(
        "null",
        "--sort-by",
        help="Sort key: null, effective-rank, recommended-rank",
    ),
    target_modules: list[str] | None = typer.Option(
        None,
        "--target-modules",
        "-m",
        help="Filter layers by module substrings (repeatable or comma-separated).",
    ),
    min_dim: int | None = typer.Option(
        None,
        "--min-dim",
        help="Only include layers with min(weight_shape) >= this value.",
    ),
    max_dim: int | None = typer.Option(
        None,
        "--max-dim",
        help="Only include layers with min(weight_shape) <= this value.",
    ),
    emit_lora_config: str | None = typer.Option(
        None,
        "--emit-lora-config",
        help="Optional path to write per-layer LoRA rank config (yaml/json).",
    ),
) -> None:
    """Analyze per-layer spectral capacity and recommended LoRA ranks.

    Computes SVD of each weight matrix to measure effective rank, null-space
    fraction, and spectral gap. Recommends per-layer LoRA ranks based on
    available capacity. Optionally emits a LoRA config file for training.

    Output fields (when --json):
        modelName: Model directory name
        totalParameters: Total parameter count
        meanEffectiveRank: Average effective rank across layers
        meanCapacityUtilization: Fraction of rank used (0-1)
        topLayers: Top N layers sorted by selected metric
        loraConfiguration: Per-layer recommended LoRA ranks
        loraConfigPath: Path to emitted config file (with --emit-lora-config)

    Example:
        mc model capacity /path/to/model
        mc model capacity /path/to/model --top 20 --sort-by effective-rank
        mc model capacity /path/to/model --emit-lora-config config.yaml
    """
    context = _context(ctx)
    service = get_capacity_analysis_service()
    try:
        normalized_sort_by = _validate_sort_by(sort_by)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Invalid capacity options",
            detail=str(exc),
            hint="Set --sort-by to null, effective-rank, or recommended-rank.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    normalized_targets = _normalize_target_modules(target_modules)

    try:
        report = service.analyze(
            model_path,
            target_modules=normalized_targets,
            min_dim=min_dim,
            max_dim=max_dim,
        )
    except FileNotFoundError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model not found",
            detail=str(exc),
            hint="Ensure the path exists and contains model safetensors.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Capacity analysis failed",
            detail=str(exc),
            hint="Ensure model weights include analyzable 2D tensors.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        from modelcypher.cli.exit_codes import EXIT_RUNTIME

        error = ErrorDetail(
            code="MC-1002",
            title="Capacity analysis failed",
            detail=str(exc),
            hint="Check backend runtime status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    lora_payload = _build_lora_config_payload(report)
    lora_config_path: str | None = None
    if emit_lora_config is not None:
        try:
            lora_config_path = _write_lora_config_file(emit_lora_config, lora_payload)
        except Exception as exc:
            error = ErrorDetail(
                code="MC-1002",
                title="LoRA config export failed",
                detail=str(exc),
                hint="Ensure destination path is writable.",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    if context.output_format == "text":
        write_output(
            _format_capacity_text(
                report=report,
                top=top,
                sort_by=normalized_sort_by,
                lora_config_path=lora_config_path,
            ),
            context.output_format,
            context.pretty,
        )
        return

    payload = report.to_dict()
    top_layers = [
        layer.to_dict()
        for layer in report.sorted_layers(normalized_sort_by)[:top]
    ]
    payload["topLayers"] = top_layers
    payload["topLayersByAvailableCapacity"] = top_layers
    payload["loraConfiguration"] = lora_payload["perLayerRank"]
    payload["sortBy"] = normalized_sort_by
    if lora_config_path is not None:
        payload["loraConfigPath"] = lora_config_path
    write_output(payload, context.output_format, context.pretty)


@app.command("moe-profile")
def model_moe_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to JSONL dataset"),
) -> None:
    """Profile MoE routing and expert capacity for a dataset."""
    context = _context(ctx)
    resolved_model = Path(model_path).expanduser().resolve()
    resolved_dataset = Path(dataset).expanduser().resolve()

    if not resolved_model.exists():
        error = ErrorDetail(
            code="MC-1001",
            title="Model not found",
            detail=f"Path does not exist: {resolved_model}",
            hint="Provide a valid local model path.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not resolved_dataset.exists():
        error = ErrorDetail(
            code="MC-1001",
            title="Dataset not found",
            detail=f"Path does not exist: {resolved_dataset}",
            hint="Provide a valid JSONL dataset path.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    from modelcypher.adapters.model_architecture import load_config
    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.cli.composition import get_activation_provider, get_backend
    from modelcypher.core.domain.dataset_loading import load_jsonl_dataset
    from modelcypher.core.domain.moe.routing_analysis import RoutingProfile
    from modelcypher.core.domain.moe.topology import MoETopology

    try:
        config = load_config(resolved_model)
        topology = MoETopology.from_config(config)
        if topology is None:
            raise ValueError("Model config does not expose MoE fields.")

        samples = load_jsonl_dataset(resolved_dataset)
        texts = [sample["text"] for sample in samples if isinstance(sample.get("text"), str)]
        if not texts:
            raise ValueError("Dataset does not contain any valid 'text' entries.")

        backend = get_backend()
        activation_provider = get_activation_provider()
        model, tokenizer = backend.load_model(str(resolved_model))
        routing_decisions = activation_provider.collect_routing_decisions(
            model, tokenizer, texts,
        )
        routing_profile = RoutingProfile.from_routing_decisions(routing_decisions, topology)

        task_relevant = routing_profile.task_relevant_experts()
        task_set = set(task_relevant)
        candidate_experts: set[tuple[int, int]] = set(task_relevant)
        for layer_idx, _expert_idx in task_relevant:
            candidate_experts.update(routing_profile.underutilized_experts(layer_idx))

        adapter = MLXTrainingAdapter(backend)
        candidate_keys: list[str] = []
        for layer_idx, expert_idx in sorted(candidate_experts):
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                candidate_keys.append(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                    f"{proj_name}.weight"
                )

        geometries = adapter.analyze_weight_geometries_for_keys_streaming(
            model, candidate_keys,
        )

        expert_rows: list[dict[str, Any]] = []
        for layer_idx, expert_idx in sorted(candidate_experts):
            stats = routing_profile.stats.get((layer_idx, expert_idx))
            gate_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            )
            gate_geom = geometries.get(gate_key)
            tail_dims = gate_geom.tail_dims if gate_geom is not None else None
            if tail_dims is None:
                capacity_status = "unprofiled"
            elif tail_dims > 0:
                capacity_status = "available"
            else:
                capacity_status = "saturated"
            expert_rows.append({
                "layer": layer_idx,
                "expert": expert_idx,
                "category": "primary" if (layer_idx, expert_idx) in task_set else "expansion",
                "routingFrequency": stats.frequency if stats is not None else 0.0,
                "tokenCount": stats.token_count if stats is not None else 0,
                "tailDims": tail_dims,
                "effectiveRank": (
                    gate_geom.shannon_effective_rank if gate_geom is not None else None
                ),
                "capacityStatus": capacity_status,
            })

        payload: dict[str, Any] = {
            "modelPath": str(resolved_model),
            "datasetPath": str(resolved_dataset),
            "topology": {
                "numExperts": topology.num_experts,
                "numExpertsPerTok": topology.num_experts_per_tok,
                "moeIntermediateSize": topology.moe_intermediate_size,
                "numLayers": topology.num_layers,
                "moeLayerIndices": topology.moe_layer_indices,
                "hasSharedExpert": topology.has_shared_expert,
                "sharedExpertIntermediateSize": topology.shared_expert_intermediate_size,
            },
            "routing": {
                "totalTokens": routing_profile.total_tokens,
                "uniformFrequency": topology.uniform_routing_frequency,
                "nTaskRelevantExperts": len(task_relevant),
            },
            "experts": expert_rows,
        }

        if context.output_format == "text":
            lines = [
                "MOE PROFILE",
                f"Model: {resolved_model}",
                f"Dataset: {resolved_dataset}",
                (
                    "Topology: "
                    f"layers={len(topology.moe_layer_indices)}/{topology.num_layers}, "
                    f"experts/layer={topology.num_experts}, "
                    f"top_k={topology.num_experts_per_tok}"
                ),
                (
                    "Routing: "
                    f"total_tokens={routing_profile.total_tokens}, "
                    f"uniform_frequency={topology.uniform_routing_frequency:.6f}, "
                    f"task_relevant={len(task_relevant)}"
                ),
                "",
                "Experts:",
            ]
            for row in sorted(
                expert_rows,
                key=lambda item: (float(item["routingFrequency"]), int(item["layer"])),
                reverse=True,
            ):
                lines.append(
                    "  "
                    f"L{row['layer']}.E{row['expert']} "
                    f"{row['category']} "
                    f"freq={row['routingFrequency']:.6f} "
                    f"tokens={row['tokenCount']} "
                    f"tail={row['tailDims']} "
                    f"status={row['capacityStatus']}"
                )
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
    except FileNotFoundError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="MoE profile failed",
            detail=str(exc),
            hint="Check model and dataset paths.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="MoE profile failed",
            detail=str(exc),
            hint="Ensure the model is MoE and dataset is valid JSONL with text.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except Exception as exc:
        from modelcypher.cli.exit_codes import EXIT_RUNTIME

        error = ErrorDetail(
            code="MC-1002",
            title="MoE profile failed",
            detail=str(exc),
            hint="Check backend runtime status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)



# =============================================================================
# MODEL QUANTIZATION
# =============================================================================


@app.command("quantize")
def model_quantize(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model to quantize"),
    output_path: str = typer.Argument(..., help="Output path for quantized model"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (4 or 8)"),
    group_size: int = typer.Option(64, "--group-size", "-g", help="Quantization group size"),
    mode: str = typer.Option("affine", "--mode", "-m", help="Quantization mode (affine, symmetric)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output"),
) -> None:
    """Quantize a model to reduce size.

    Supports 4-bit and 8-bit quantization with configurable group size.

    Examples:
        mc model quantize /path/to/model /path/to/output --bits 4
        mc model quantize /path/to/model /path/to/output --bits 8 --group-size 128
    """
    from modelcypher.cli.composition import get_quantization_service

    context = _context(ctx)

    typer.echo(f"Quantizing model to {bits}-bit with group_size={group_size}...")

    try:
        service = get_quantization_service()

        result = service.quantize_model(
            model_path=model_path,
            output_dir=output_path,
            bits=bits,
            group_size=group_size,
            mode=mode,
            overwrite=overwrite,
        )

        payload = result.to_dict()

        if context.output_format == "text":
            lines = [
                "QUANTIZATION COMPLETE",
                f"Input: {model_path}",
                f"Output: {result.output_dir}",
                "",
                f"Bits: {result.bits}",
                f"Group Size: {result.group_size}",
                f"Mode: {result.mode}",
                "",
                "WEIGHTS:",
                f"  Total 2D: {result.total_2d_weights}",
                f"  Quantized: {result.quantized_2d_weights}",
                f"  Skipped: {result.skipped_2d_weights}",
            ]
            if result.skipped_dirs:
                lines.append("")
                lines.append(f"Skipped Dirs: {', '.join(result.skipped_dirs)}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except FileExistsError as e:
        error = ErrorDetail(
            code="MC-1003",
            title="Output exists",
            detail=str(e),
            hint="Use --overwrite to replace existing output.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except Exception as e:
        from modelcypher.cli.exit_codes import EXIT_RUNTIME

        error = ErrorDetail(
            code="MC-1002",
            title="Quantization failed",
            detail=str(e),
            hint="Ensure model contains valid safetensors weights.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)
