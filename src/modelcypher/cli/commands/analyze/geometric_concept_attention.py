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

"""Concept, chain, and attention geometry analyze subcommands."""

from __future__ import annotations

from modelcypher.cli.exit_codes import EXIT_RUNTIME

from ._common import ErrorDetail, Path, get_context, typer, write_error, write_output


def concept_volume_analysis(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    concepts: str = typer.Option(
        ..., "--concepts", help="Path to concepts JSON file"
    ),
    layer: int = typer.Option(..., "--layer", help="Layer index to analyze"),
) -> None:
    """Analyze concept volumes in activation space using Riemannian density estimation.

    For each named concept (defined by prompts in a JSON file), collects hidden
    activations at the specified layer, estimates a Riemannian ConceptVolume, and
    computes pairwise geometric relations (overlap, distance, curvature divergence).

    Concepts file format (JSON):
        {
          "capital_cities": ["Paris is the capital of France", "Tokyo is ..."],
          "math_operations": ["2+2=4", "The integral of x is x^2/2"]
        }

    Output includes per-concept volume statistics and a pairwise relation table.

    Examples:
        mc analyze concept-volume --model ./my-model --concepts concepts.json --layer 12
    """
    import json as json_mod

    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3080",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    concepts_path = Path(concepts)
    if not concepts_path.exists():
        error = ErrorDetail(
            code="MC-3081",
            title="Concepts file not found",
            detail=f"Concepts file does not exist: {concepts}",
            hint="Provide a valid path to a concepts JSON file.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        concept_data = json_mod.loads(concepts_path.read_text())
    except (json_mod.JSONDecodeError, OSError) as exc:
        error = ErrorDetail(
            code="MC-3082",
            title="Invalid concepts file",
            detail=f"Could not parse concepts JSON: {exc}",
            hint="Ensure the concepts file is valid JSON mapping concept names to prompt lists.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not isinstance(concept_data, dict) or not concept_data:
        error = ErrorDetail(
            code="MC-3083",
            title="Invalid concepts format",
            detail="Concepts file must be a non-empty JSON object mapping concept names to prompt lists.",
            hint='Expected format: {"concept_name": ["prompt1", "prompt2"], ...}.',
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_concept_volume_service

        service = get_concept_volume_service()

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        result = service.analyze_concept_volumes(
            loaded_model, tokenizer, concept_data, layer,
        )

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3084",
            title="Concept volume analysis failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    payload = result.to_dict()
    payload["modelPath"] = str(model_path)
    payload["conceptsPath"] = str(concepts_path)

    if context.output_format == "text":
        lines = [
            "CONCEPT VOLUME ANALYSIS (Riemannian Density)",
            f"Model: {model_path}",
            f"Layer: {layer}",
            f"Concepts: {len(result.concept_stats)}",
            "",
        ]

        if result.concept_stats:
            lines.append("Per-Concept Volumes:")
            for s in result.concept_stats:
                lines.append(f"  {s.concept_id}:")
                lines.append(f"    Samples: {s.num_samples}")
                lines.append(f"    Influence: {s.influence_type}")
                lines.append(f"    Geodesic Radius: {s.geodesic_radius:.6f}")
                lines.append(f"    Effective Radius: {s.effective_radius:.6f}")
                lines.append(f"    Volume: {s.volume:.6e}")
                lines.append(f"    Dimension: {s.dimension}")
                if s.mean_sectional_curvature is not None:
                    lines.append(f"    Mean Curvature: {s.mean_sectional_curvature:.6f}")
            lines.append("")

        if result.pairwise_relations:
            lines.append("Pairwise Relations:")
            for r in result.pairwise_relations:
                lines.append(f"  {r.concept_a} <-> {r.concept_b}:")
                lines.append(f"    Overlap:      {r.overlap_coefficient:.4f}")
                lines.append(f"    Jaccard:      {r.jaccard_index:.4f}")
                lines.append(f"    Bhattacharyya: {r.bhattacharyya_coefficient:.4f}")
                lines.append(f"    Centroid Dist: {r.centroid_distance:.6f}")
                lines.append(f"    Curvature Div: {r.curvature_divergence:.6f}")
                lines.append(f"    Subspace Align: {r.subspace_alignment:.4f}")
            lines.append("")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


# Default probes for chain-profile — 6 categories × 10, covering diverse
# semantic domains to sample the representation space uniformly.
_CHAIN_PROFILE_PROBES: tuple[str, ...] = (
    # Factual recall
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at standard pressure.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "DNA carries genetic information in living organisms.",
    "The periodic table organizes elements by atomic number.",
    "Gravity causes objects to fall toward Earth.",
    "The human body contains 206 bones.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Earth revolves around the Sun in approximately 365 days.",
    "Sound travels faster in water than in air.",
    # Logical reasoning
    "If all mammals are warm-blooded, and dolphins are mammals, then dolphins are warm-blooded.",
    "A number that is divisible by 6 must also be divisible by 2 and 3.",
    "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
    "All squares are rectangles, but not all rectangles are squares.",
    "If A implies B, and B implies C, then A implies C.",
    "The contrapositive of 'if P then Q' is 'if not Q then not P'.",
    "A set with n elements has 2^n subsets.",
    "If no fish can fly and all birds can fly, then no fish are birds.",
    "The negation of 'all cats are black' is 'there exists a cat that is not black'.",
    "If the sum of two numbers is even, they are either both even or both odd.",
    # Creative / abstract
    "Imagine a world where colors have sounds and music has texture.",
    "The meaning of life is a question that has puzzled philosophers for millennia.",
    "Art transcends language barriers and speaks to the human condition.",
    "A poem about autumn leaves falling like memories fading.",
    "The boundary between dreaming and waking is thinner than we think.",
    "Music is mathematics made audible, patterns made beautiful.",
    "Time flows like a river, but rivers can be dammed.",
    "The stars are not just lights — they are furnaces forging the elements.",
    "Consciousness might be the universe experiencing itself.",
    "Language shapes thought as much as thought shapes language.",
    # Technical / mathematical
    "The eigenvalues of a symmetric matrix are always real.",
    "Gradient descent minimizes a function by following the negative gradient.",
    "The Fourier transform decomposes a signal into its frequency components.",
    "A neural network with one hidden layer can approximate any continuous function.",
    "The determinant of a matrix is zero if and only if it is singular.",
    "Convolution in the spatial domain equals multiplication in the frequency domain.",
    "The chain rule computes derivatives of composed functions.",
    "Singular value decomposition factors a matrix into rotation and scaling.",
    "Entropy measures the uncertainty in a probability distribution.",
    "The central limit theorem states that sample means converge to normal distribution.",
    # Conversational / social
    "Hello, how are you doing today?",
    "Could you explain this concept in simpler terms?",
    "I disagree with that assessment. Here is why.",
    "Thank you for your help. I appreciate it.",
    "What would you recommend for someone just starting out?",
    "That is an interesting perspective I had not considered before.",
    "Can you summarize the main points of this discussion?",
    "I think we should approach this problem differently.",
    "Let me know if you need any clarification on my question.",
    "What are the potential risks we should be aware of?",
    # Ambiguous / adversarial
    "The bank was steep and the river was deep.",
    "She saw the man with the telescope.",
    "Flying planes can be dangerous.",
    "Time flies like an arrow; fruit flies like a banana.",
    "The old man the boats.",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "I never said she stole my money.",
    "The chicken is ready to eat.",
    "Visiting relatives can be boring.",
    "The horse raced past the barn fell.",
)


def safety_chain_profile(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with probe texts (one per line)"
    ),
    samples: int = typer.Option(
        60, "--samples", help="Number of probe samples to use"
    ),
) -> None:
    """Compute unified causal chain profile for a model.

    Measures the validated causal chain at every layer:

        Entropy -> Curvature (angular) -> Cumulative curvature -> ID -> Phase

    Per-layer measurements:
    - Entropy: Shannon entropy via unembedding projection (Entropy-Lens)
    - Curvature: Angular change in radians (arccos of cosine similarity)
    - Attn/MLP decomposition: Fraction of curvature from attention vs MLP
    - Intrinsic dimension: TwoNN estimator (Facco et al., 2017)
    - Phase: highway / processing / exit (data-derived boundaries)

    Cross-link correlations:
    - Entropy <-> Curvature: Spearman r (validated range: 0.4-0.6)
    - Cumulative curvature <-> ID: Spearman r (family-dependent)
    - Mean attention fraction: ~0.37 (universal across architectures)

    For LFM2 hybrid models, non-attention layers (ShortConv/SSM) get total
    curvature only -- attn/MLP decomposition is not available.

    Examples:
        mc analyze chain-profile --model ./my-model
        mc analyze chain-profile --model ./my-model --samples 100
        mc analyze chain-profile --model ./my-model --probes ./probes.txt
    """
    import math

    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load probe texts if provided
    probe_texts: list[str] | None = None
    if probes:
        probes_path = Path(probes)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-3031",
                title="Probes file not found",
                detail=f"Probes file does not exist: {probes}",
                hint="Provide a valid path to a probes text file (one probe per line).",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        probe_texts = [
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        ]

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_chain_analysis_service

        service = get_chain_analysis_service()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        # Use default probes if not provided
        if probe_texts is None:
            probe_texts = list(_CHAIN_PROFILE_PROBES)

        # Limit to requested samples
        probe_texts = probe_texts[:samples]

        # Compute chain profile
        profile = service.analyze_chain(loaded_model, tokenizer, probe_texts)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3032",
            title="Chain profile analysis failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    payload = profile.as_dict()

    if context.output_format == "text":
        lines = [
            "CAUSAL CHAIN PROFILE",
            f"Model: {model_path}",
            f"Layers: {profile.num_layers}",
            f"Hidden Dim: {profile.hidden_dim}",
            f"Probes: {profile.probe_count}",
            "",
            "Per-Layer Chain:",
            f"  {'Layer':>5} | {'Entropy':>8} | {'Curv (rad)':>10} | {'Attn frac':>9} | {'ID (TwoNN)':>10} | Phase",
            f"  {'-----':>5}-+-{'--------':>8}-+-{'----------':>10}-+-{'---------':>9}-+-{'----------':>10}-+------",
        ]

        for m in profile.layers:
            entropy_str = f"{m.entropy:8.4f}" if m.entropy > 0 else "     N/A"
            curv_str = f"{m.total_curvature:10.4f}"
            attn_str = f"{m.attn_fraction:9.2%}" if m.attn_fraction is not None else "      N/A"
            id_str = f"{m.intrinsic_dimension:10.2f}" if not math.isnan(m.intrinsic_dimension) else "       N/A"
            phase_str = m.phase.value

            lines.append(
                f"  {m.layer_idx:5d} | {entropy_str} | {curv_str} | {attn_str} | {id_str} | {phase_str}"
            )

        lines.extend([
            "",
            "Cross-Link Correlations:",
            f"  Entropy -> Curvature:       Spearman r = {profile.correlations.entropy_to_curvature:.3f}"
            if not math.isnan(profile.correlations.entropy_to_curvature)
            else "  Entropy -> Curvature:       N/A (insufficient data)",
            f"  Cumulative curv -> ID:      Spearman r = {profile.correlations.cumulative_curvature_to_id:.3f}"
            if not math.isnan(profile.correlations.cumulative_curvature_to_id)
            else "  Cumulative curv -> ID:      N/A (insufficient data)",
        ])

        if profile.correlations.mean_attn_fraction is not None:
            lines.append(
                f"  Mean attention fraction:   {profile.correlations.mean_attn_fraction:.3f}"
            )
        else:
            lines.append("  Mean attention fraction:   N/A (no decomposition available)")

        # Phase summary
        phase_counts: dict[str, int] = {}
        for m in profile.layers:
            phase_counts[m.phase.value] = phase_counts.get(m.phase.value, 0) + 1

        lines.extend([
            "",
            "Phase Summary:",
        ])
        for phase_name in ("highway", "processing", "exit"):
            count = phase_counts.get(phase_name, 0)
            if count > 0:
                layer_idxs = [
                    m.layer_idx for m in profile.layers if m.phase.value == phase_name
                ]
                lines.append(f"  {phase_name:>10}: {count} layers ({layer_idxs})")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


def safety_attention_collapse(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str = typer.Option(
        "The capital of France is",
        "--prompt",
        help="Text input for attention analysis",
    ),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        help="Model dtype for rank-1 threshold (float32, float16, bfloat16)",
    ),
) -> None:
    """Detect attention head collapse via SVD analysis.

    Computes per-head singular value decomposition of attention weight matrices
    to identify rank-1 collapsed heads (gradient-dead) and measure effective rank.

    Metrics per head:
    - Rank-1 ratio: sigma_2 / sigma_1 (collapsed when < sqrt(eps_dtype), IEEE 754)
    - Effective rank: exp(Shannon entropy of normalized sigma^2)  (Roy & Vetterli 2007)
    - Gradient suppression: sigma_2 / sqrt(2T)  (Theorem H.1, Sanyal et al. TMLR 2025)

    Thresholds:
    - bfloat16 (7-bit mantissa): sqrt(2^-7) = 0.0884
    - float16  (10-bit mantissa): sqrt(2^-10) = 0.0312
    - float32  (23-bit mantissa): sqrt(2^-23) = 3.45e-4

    Only softmax attention layers are analyzed. Conv layers (LFM2) and
    linear attention layers (Qwen3.5 GatedDeltaNet) are skipped.

    Examples:
        mc analyze attention-collapse --model ./my-model
        mc analyze attention-collapse --model ./my-model --prompt "Hello world"
        mc analyze attention-collapse --model ./my-model --dtype float32
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3060",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)

    valid_dtypes = ("float32", "float16", "bfloat16")
    if dtype not in valid_dtypes:
        error = ErrorDetail(
            code="MC-3061",
            title="Invalid dtype",
            detail=f"dtype must be one of {valid_dtypes}, got '{dtype}'",
            hint="Use --dtype float32, --dtype float16, or --dtype bfloat16.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.core.domain.geometry.attention_collapse import (
            _DTYPE_SQRT_EPS,
            compute_attention_collapse,
            compute_collapse_profile,
            summarize_layer_collapse,
        )
        from modelcypher.ports.activation_provider import get_activation_provider

        provider = get_activation_provider()
        sqrt_eps = _DTYPE_SQRT_EPS[dtype]

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        attn_matrices = provider.collect_attention_matrices(
            loaded_model, tokenizer, prompt
        )

        layer_results = []
        all_head_results: dict[int, list] = {}

        for layer_idx in sorted(attn_matrices.keys()):
            head_matrices = attn_matrices[layer_idx]
            head_results = []
            for head_mat in head_matrices:
                mat_list = provider._backend.tolist(head_mat)
                result = compute_attention_collapse(mat_list, dtype)
                head_results.append(result)
            all_head_results[layer_idx] = head_results
            layer_summary = summarize_layer_collapse(
                head_results, layer_idx=layer_idx
            )
            layer_results.append(layer_summary)

        profile = compute_collapse_profile(layer_results)

        # Build output payload
        payload = {
            "model": str(model_path),
            "prompt": prompt,
            "dtype": dtype,
            "profile": profile.to_dict(),
            "layers": [lr.to_dict() for lr in layer_results],
        }

        if context.output_format == "text":
            lines = [
                f"Attention Collapse Analysis: {model_path.name}",
                f"Prompt: {prompt!r}",
                f"Dtype: {dtype}  |  Rank-1 threshold: sqrt(eps) = {sqrt_eps:.6f}",
                "",
                f"{'Layer':>5} | {'Collapsed':>9} | {'MaxEffRank':>10} | {'MeanGradSupp':>12}",
                f"{'-----':>5}-+-{'---------':>9}-+-{'----------':>10}-+-{'------------':>12}",
            ]
            for lr in layer_results:
                n_heads = len(all_head_results[lr.layer_idx])
                lines.append(
                    f"{lr.layer_idx:5d} | "
                    f"{lr.collapsed_head_count}/{n_heads:>3}     | "
                    f"{lr.max_effective_rank:10.2f} | "
                    f"{lr.mean_gradient_suppression:12.4f}"
                )
            lines.extend([
                "",
                f"Total layers: {profile.total_layers}  |  "
                f"Layers with any collapse: {profile.collapsed_layer_count}",
            ])
            if profile.collapse_onset_layer is not None:
                lines.append(
                    f"Collapse onset: layer {profile.collapse_onset_layer}"
                )
            else:
                lines.append("Collapse onset: none detected")

            write_output("\n".join(lines), context.output_format, context.pretty)
        else:
            write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-3062",
            title="Attention collapse analysis failed",
            detail=str(e),
            hint="Check model path and ensure model is loadable.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)


def safety_attention_sink(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str = typer.Option(
        "The capital of France is",
        "--prompt",
        help="Text input for attention analysis",
    ),
) -> None:
    """Analyze attention sink positions and value-weighted active sinks.

    Computes per-head sink scores measuring how much attention each token
    position receives from causal successors, then weights by value-vector
    norms to identify geometrically impactful sinks.

    Metrics per token:
    - Sink score: s_i = (1/(T-i)) * sum_{u>=i} A_{u,i}  (Binkowski et al. 2026)
    - Active sink: sink_score * ||V_i||_2  (geometric impact weighting)

    No tuned thresholds. All measurements are continuous.

    Only softmax attention layers are analyzed. Conv layers (LFM2) and
    linear attention layers (Qwen3.5 GatedDeltaNet) are skipped.

    Examples:
        mc analyze attention-sink --model ./my-model
        mc analyze attention-sink --model ./my-model --prompt "Hello world"
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3063",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)

    try:
        import math

        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.core.domain.geometry.attention_sink import (
            compute_active_sinks,
            compute_sink_scores,
            summarize_layer_sinks,
        )
        from modelcypher.ports.activation_provider import get_activation_provider

        provider = get_activation_provider()

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        attn_matrices, value_vectors = provider.collect_attention_matrices_with_values(
            loaded_model, tokenizer, prompt
        )

        layer_results = []
        for layer_idx in sorted(attn_matrices.keys()):
            head_matrices = attn_matrices[layer_idx]
            head_values = value_vectors[layer_idx]

            head_sinks = []
            active_sinks = []

            for head_idx, (head_mat, head_v) in enumerate(
                zip(head_matrices, head_values)
            ):
                mat_list = provider._backend.tolist(head_mat)
                head_sink = compute_sink_scores(mat_list, head_idx=head_idx)
                head_sinks.append(head_sink)

                v_list = provider._backend.tolist(head_v)
                v_norms = [
                    math.sqrt(sum(x * x for x in row)) for row in v_list
                ]
                active = compute_active_sinks(head_sink, v_norms)
                active_sinks.append(active)

            layer_summary = summarize_layer_sinks(
                head_sinks, active_results=active_sinks, layer_idx=layer_idx
            )
            layer_results.append(layer_summary)

        payload = {
            "model": str(model_path),
            "prompt": prompt,
            "layers": [lr.to_dict() for lr in layer_results],
        }

        if context.output_format == "text":
            lines = [
                f"Attention Sink Analysis: {model_path.name}",
                f"Prompt: {prompt!r}",
                "",
                f"{'Layer':>5} | {'DomSinkPos':>10} | {'MeanMaxSink':>11} | {'Heads':>5}",
                f"{'-----':>5}-+-{'----------':>10}-+-{'-----------':>11}-+-{'-----':>5}",
            ]
            for lr in layer_results:
                n_heads = len(lr.head_results)
                lines.append(
                    f"{lr.layer_idx:5d} | "
                    f"{lr.dominant_sink_position:10d} | "
                    f"{lr.mean_max_sink_score:11.4f} | "
                    f"{n_heads:5d}"
                )

            # Summary: how many layers have BOS/early token as dominant
            early_count = sum(
                1 for lr in layer_results if lr.dominant_sink_position <= 1
            )
            total = len(layer_results)
            lines.extend([
                "",
                f"Layers with BOS/early dominant sink: {early_count}/{total}",
            ])

            # Show heads where active differs from raw
            reweighted_count = 0
            for lr in layer_results:
                if lr.active_results:
                    for hs, act in zip(lr.head_results, lr.active_results):
                        if hs.max_sink_position != act.max_active_position:
                            reweighted_count += 1
            total_heads = sum(len(lr.head_results) for lr in layer_results)
            lines.append(
                f"Heads where V-norm reweighting changes max sink: "
                f"{reweighted_count}/{total_heads}"
            )

            write_output("\n".join(lines), context.output_format, context.pretty)
        else:
            write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-3064",
            title="Attention sink analysis failed",
            detail=str(e),
            hint="Check model path and ensure model is loadable.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)
