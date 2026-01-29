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

"""LoRA Geometry Diagnostic: Analyze what LoRA actually changes in the weight space.

This module provides tools to understand whether LoRA training:
1. Activates null space (new directions)
2. Overwrites existing transformations
3. Changes the positive geometry signatures

Key metrics:
- Singular value changes (amplification vs. attenuation)
- New singular vectors (null space activation)
- Rank changes
- Positive Grassmannian signature changes
- Subspace overlap (how much of LoRA lives in existing vs. new directions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerSVDAnalysis:
    """SVD analysis for a single layer's weight matrix."""

    layer_idx: int
    weight_name: str
    shape: tuple[int, int]

    # Rank analysis
    rank_before: int
    rank_after: int
    rank_delta: int  # Positive = null space activation

    # Singular value statistics
    sv_before: list[float]  # Top singular values before
    sv_after: list[float]   # Top singular values after
    sv_delta_norm: float    # ||sv_after - sv_before|| / ||sv_before||

    # Directional analysis
    subspace_overlap: float  # How much of delta lives in existing subspace (0-1)
    null_space_component: float  # How much projects into null space (0-1)

    # Change magnitude
    frobenius_delta: float  # ||W_after - W_before||_F
    relative_change: float  # ||delta||_F / ||W_before||_F


@dataclass
class PositiveGeometryComparison:
    """Compare positive geometry signatures before/after."""

    layer_idx: int

    # Minor sign patterns
    positive_minors_before: float  # % positive minors
    positive_minors_after: float
    sign_flip_count: int  # Number of minors that flipped sign

    # Grassmannian metrics
    grassmannian_distance: float  # Distance between subspaces


@dataclass
class LoRAGeometryReport:
    """Complete diagnostic report for LoRA geometry changes."""

    model_path: str
    adapter_path: str

    # Summary statistics
    total_layers: int
    layers_with_lora: int
    total_params_modified: int

    # Per-layer analysis
    layer_svd: list[LayerSVDAnalysis] = field(default_factory=list)
    positive_geometry: list[PositiveGeometryComparison] = field(default_factory=list)

    # Aggregate metrics
    avg_null_space_activation: float = 0.0
    avg_subspace_overlap: float = 0.0
    avg_relative_change: float = 0.0
    peak_change_layer: int = -1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "LORA GEOMETRY DIAGNOSTIC REPORT",
            "=" * 60,
            f"Model: {self.model_path}",
            f"Adapter: {self.adapter_path}",
            f"Layers with LoRA: {self.layers_with_lora}/{self.total_layers}",
            f"Parameters modified: {self.total_params_modified:,}",
            "",
            "AGGREGATE METRICS:",
            f"  Avg null space activation: {self.avg_null_space_activation:.1%}",
            f"  Avg subspace overlap: {self.avg_subspace_overlap:.1%}",
            f"  Avg relative change: {self.avg_relative_change:.4f}",
            f"  Peak change at layer: {self.peak_change_layer}",
            "",
        ]

        if self.layer_svd:
            lines.append("PER-LAYER SVD ANALYSIS:")
            lines.append("-" * 60)
            for svd in self.layer_svd:
                lines.append(
                    f"  Layer {svd.layer_idx} ({svd.weight_name}):"
                )
                lines.append(
                    f"    Rank: {svd.rank_before} -> {svd.rank_after} "
                    f"(Δ={svd.rank_delta:+d})"
                )
                lines.append(
                    f"    Null space activation: {svd.null_space_component:.1%}"
                )
                lines.append(
                    f"    Subspace overlap: {svd.subspace_overlap:.1%}"
                )
                lines.append(
                    f"    Relative change: {svd.relative_change:.4f}"
                )
            lines.append("")

        if self.positive_geometry:
            lines.append("POSITIVE GEOMETRY CHANGES:")
            lines.append("-" * 60)
            for pg in self.positive_geometry:
                lines.append(
                    f"  Layer {pg.layer_idx}:"
                )
                lines.append(
                    f"    Positive minors: {pg.positive_minors_before:.1%} -> "
                    f"{pg.positive_minors_after:.1%}"
                )
                lines.append(
                    f"    Sign flips: {pg.sign_flip_count}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def _effective_rank(singular_values: mx.array, threshold: float = 1e-6) -> int:
    """Compute effective rank (number of singular values above threshold)."""
    sv_np = np.array(singular_values.tolist())
    max_sv = sv_np.max() if len(sv_np) > 0 else 1.0
    return int(np.sum(sv_np > max_sv * threshold))


def _subspace_overlap(U1: mx.array, U2: mx.array, k: int = None) -> float:
    """Compute overlap between column spaces of U1 and U2.

    Returns value in [0, 1] where 1 means identical subspaces.
    Uses principal angles between subspaces.
    """
    if k is None:
        k = min(U1.shape[1], U2.shape[1])

    U1_k = U1[:, :k].astype(mx.float32)
    U2_k = U2[:, :k].astype(mx.float32)

    # Compute U1^T @ U2
    M = U1_k.T @ U2_k
    mx.eval(M)

    # Singular values of M are cosines of principal angles
    try:
        _, S, _ = mx.linalg.svd(M, stream=mx.cpu)
        mx.eval(S)
        # Average of squared cosines gives overlap measure
        S_np = np.array(S.tolist())
        return float(np.mean(S_np ** 2))
    except Exception:
        return 0.5  # Default if SVD fails


def _null_space_projection(delta: mx.array, U: mx.array, k: int) -> float:
    """Compute what fraction of delta projects into null space of original.

    Args:
        delta: The change matrix (W_after - W_before)
        U: Left singular vectors of original W
        k: Rank of original W

    Returns:
        Fraction in [0, 1] - higher means more null space activation
    """
    U_k = U[:, :k].astype(mx.float32)
    delta_f = delta.astype(mx.float32)

    # Project delta onto column space of U_k
    proj = U_k @ (U_k.T @ delta_f)
    mx.eval(proj)

    # Component in null space
    null_component = delta_f - proj
    mx.eval(null_component)

    delta_norm = float(mx.sqrt(mx.sum(delta_f * delta_f)))
    null_norm = float(mx.sqrt(mx.sum(null_component * null_component)))

    if delta_norm < 1e-10:
        return 0.0

    return null_norm / delta_norm


def _compute_positive_minors(W: mx.array, sample_size: int = 100) -> tuple[float, list[float]]:
    """Sample minors and compute fraction that are positive.

    Returns (positive_fraction, list of minor values for comparison)
    """
    W_np = np.array(W.astype(mx.float32).tolist())
    m, n = W_np.shape
    k = min(m, n, 4)  # Use 4x4 minors max

    if k < 2:
        return 0.5, []

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    minors = []

    for _ in range(sample_size):
        rows = rng.choice(m, size=k, replace=False)
        cols = rng.choice(n, size=k, replace=False)
        submatrix = W_np[np.ix_(rows, cols)]
        det = np.linalg.det(submatrix)
        minors.append(det)

    minors = np.array(minors)
    positive_frac = np.mean(minors > 0)
    return float(positive_frac), minors.tolist()


def analyze_layer_weights(
    W_before: mx.array,
    W_after: mx.array,
    layer_idx: int,
    weight_name: str,
) -> tuple[LayerSVDAnalysis, PositiveGeometryComparison]:
    """Analyze changes in a single weight matrix."""

    # Convert to float32 for numerical stability
    W_before_f = W_before.astype(mx.float32)
    W_after_f = W_after.astype(mx.float32)
    delta = W_after_f - W_before_f
    mx.eval(delta)

    shape = tuple(W_before.shape)

    # SVD of original
    try:
        U_before, S_before, Vt_before = mx.linalg.svd(W_before_f, stream=mx.cpu)
        mx.eval(U_before, S_before, Vt_before)
    except Exception as e:
        logger.warning(f"SVD failed for layer {layer_idx}: {e}")
        # Return default values
        svd_analysis = LayerSVDAnalysis(
            layer_idx=layer_idx,
            weight_name=weight_name,
            shape=shape,
            rank_before=0,
            rank_after=0,
            rank_delta=0,
            sv_before=[],
            sv_after=[],
            sv_delta_norm=0.0,
            subspace_overlap=0.5,
            null_space_component=0.5,
            frobenius_delta=0.0,
            relative_change=0.0,
        )
        pg_analysis = PositiveGeometryComparison(
            layer_idx=layer_idx,
            positive_minors_before=0.5,
            positive_minors_after=0.5,
            sign_flip_count=0,
            grassmannian_distance=0.0,
        )
        return svd_analysis, pg_analysis

    # SVD of modified
    try:
        U_after, S_after, Vt_after = mx.linalg.svd(W_after_f, stream=mx.cpu)
        mx.eval(U_after, S_after, Vt_after)
    except Exception as e:
        logger.warning(f"SVD of modified weights failed for layer {layer_idx}: {e}")
        U_after, S_after = U_before, S_before

    # Rank analysis
    rank_before = _effective_rank(S_before)
    rank_after = _effective_rank(S_after)

    # Singular value comparison (top 10)
    k = min(10, len(S_before), len(S_after))
    sv_before = [float(x) for x in S_before[:k].tolist()]
    sv_after = [float(x) for x in S_after[:k].tolist()]

    sv_before_arr = np.array(sv_before)
    sv_after_arr = np.array(sv_after)
    sv_delta_norm = float(np.linalg.norm(sv_after_arr - sv_before_arr) / (np.linalg.norm(sv_before_arr) + 1e-10))

    # Subspace overlap
    overlap = _subspace_overlap(U_before, U_after, k=min(rank_before, 20))

    # Null space projection
    null_component = _null_space_projection(delta, U_before, rank_before)

    # Change magnitude
    frobenius_delta = float(mx.sqrt(mx.sum(delta * delta)))
    frobenius_before = float(mx.sqrt(mx.sum(W_before_f * W_before_f)))
    relative_change = frobenius_delta / (frobenius_before + 1e-10)

    svd_analysis = LayerSVDAnalysis(
        layer_idx=layer_idx,
        weight_name=weight_name,
        shape=shape,
        rank_before=rank_before,
        rank_after=rank_after,
        rank_delta=rank_after - rank_before,
        sv_before=sv_before,
        sv_after=sv_after,
        sv_delta_norm=sv_delta_norm,
        subspace_overlap=overlap,
        null_space_component=null_component,
        frobenius_delta=frobenius_delta,
        relative_change=relative_change,
    )

    # Positive geometry analysis
    pos_before, minors_before = _compute_positive_minors(W_before_f)
    pos_after, minors_after = _compute_positive_minors(W_after_f)

    # Count sign flips
    sign_flips = 0
    for m1, m2 in zip(minors_before, minors_after):
        if (m1 > 0) != (m2 > 0):
            sign_flips += 1

    # Grassmannian distance (using principal angles)
    grassmannian_dist = 1.0 - overlap

    pg_analysis = PositiveGeometryComparison(
        layer_idx=layer_idx,
        positive_minors_before=pos_before,
        positive_minors_after=pos_after,
        sign_flip_count=sign_flips,
        grassmannian_distance=grassmannian_dist,
    )

    return svd_analysis, pg_analysis


def run_diagnostic(
    model_path: str,
    adapter_path: str,
    target_layers: list[int] | None = None,
    target_weights: list[str] | None = None,
) -> LoRAGeometryReport:
    """Run full LoRA geometry diagnostic.

    Args:
        model_path: Path to base model
        adapter_path: Path to LoRA adapter
        target_layers: Which layers to analyze (default: all)
        target_weights: Which weight types to analyze
            (default: ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"])

    Returns:
        Complete diagnostic report
    """
    from mlx_lm import load as mlx_load
    from modelcypher.core.domain.training.self_reflection import (
        load_self_reflection_adapters,
    )

    logger.info("Loading base model...")
    model_before, _ = mlx_load(model_path)

    logger.info("Loading model with adapter...")
    model_after, tokenizer = load_self_reflection_adapters(model_path, adapter_path)

    # Get base model layers
    base_before = getattr(model_before, "model", model_before)
    base_after = getattr(model_after, "model", model_after)

    layers_before = base_before.layers
    layers_after = base_after.layers

    total_layers = len(layers_before)

    if target_layers is None:
        target_layers = list(range(total_layers))

    if target_weights is None:
        target_weights = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"]

    report = LoRAGeometryReport(
        model_path=model_path,
        adapter_path=adapter_path,
        total_layers=total_layers,
        layers_with_lora=0,
        total_params_modified=0,
    )

    svd_analyses = []
    pg_analyses = []

    for layer_idx in target_layers:
        logger.info(f"Analyzing layer {layer_idx}...")

        layer_before = layers_before[layer_idx]
        layer_after = layers_after[layer_idx]

        # Check different module paths for weights
        # LFM2 models use 'conv' for some layers and 'self_attn' for others
        modules_to_check = [
            ("self_attn", ["q_proj", "k_proj", "v_proj", "out_proj"]),
            ("conv", ["in_proj", "out_proj"]),
            ("feed_forward", ["w1", "w2", "w3"]),
            ("mlp", ["gate_proj", "up_proj", "down_proj"]),
        ]

        for module_name, weight_names in modules_to_check:
            module_before = getattr(layer_before, module_name, None)
            module_after = getattr(layer_after, module_name, None)

            if module_before is None or module_after is None:
                continue

            for weight_name in weight_names:
                proj_before = getattr(module_before, weight_name, None)
                proj_after = getattr(module_after, weight_name, None)

                if proj_before is None or proj_after is None:
                    continue

                # For base model, weight is directly on the module
                W_before = getattr(proj_before, "weight", None)

                # For LoRA model, weight is in proj.linear.weight
                linear_after = getattr(proj_after, "linear", None)
                if linear_after is not None:
                    # This is a LoRALinear wrapper
                    lora_a = getattr(proj_after, "lora_a", None)
                    lora_b = getattr(proj_after, "lora_b", None)
                    scale = getattr(proj_after, "scale", 1.0)

                    if lora_a is None or lora_b is None:
                        continue  # No LoRA matrices

                    if W_before is None:
                        W_before = getattr(linear_after, "weight", None)

                    if W_before is None:
                        continue

                    # Compute LoRA contribution
                    # For forward: output = x @ W.T + x @ lora_a @ lora_b
                    # So effective weight.T = W.T + lora_a @ lora_b
                    # Thus effective weight = W + (lora_a @ lora_b).T
                    lora_delta = (lora_a @ lora_b).T
                    mx.eval(lora_delta)

                    lora_delta = lora_delta * scale
                    mx.eval(lora_delta)

                    W_after = W_before + lora_delta
                    mx.eval(W_after)

                    diff_norm = float(mx.sum(mx.abs(lora_delta)))
                else:
                    continue  # Not a LoRA layer

                if diff_norm < 1e-8:
                    continue  # No change, skip

                report.layers_with_lora += 1
                report.total_params_modified += W_before.size

                full_name = f"{module_name}.{weight_name}"
                svd_analysis, pg_analysis = analyze_layer_weights(
                    W_before, W_after, layer_idx, full_name
                )

                svd_analyses.append(svd_analysis)
                pg_analyses.append(pg_analysis)

    report.layer_svd = svd_analyses
    report.positive_geometry = pg_analyses

    # Compute aggregates
    if svd_analyses:
        report.avg_null_space_activation = np.mean([s.null_space_component for s in svd_analyses])
        report.avg_subspace_overlap = np.mean([s.subspace_overlap for s in svd_analyses])
        report.avg_relative_change = np.mean([s.relative_change for s in svd_analyses])

        # Find peak change layer
        peak_idx = np.argmax([s.relative_change for s in svd_analyses])
        report.peak_change_layer = svd_analyses[peak_idx].layer_idx

    return report


def compare_activation_patterns(
    model_path: str,
    adapter_path: str,
    probe_prompts: list[str],
) -> dict:
    """Compare activation patterns before/after LoRA on specific prompts.

    This shows how the model's internal representations change for
    the same inputs.
    """
    from mlx_lm import load as mlx_load
    from modelcypher.core.domain.training.self_reflection import (
        load_self_reflection_adapters,
    )

    logger.info("Loading models...")
    model_before, tokenizer = mlx_load(model_path)
    model_after, _ = load_self_reflection_adapters(model_path, adapter_path)

    base_before = getattr(model_before, "model", model_before)
    base_after = getattr(model_after, "model", model_after)

    results = []

    for prompt in probe_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Capture hidden states at each layer
        hidden_before = base_before.embed_tokens(input_ids)
        hidden_after = base_after.embed_tokens(input_ids)

        layer_diffs = []

        for idx, (layer_b, layer_a) in enumerate(zip(base_before.layers, base_after.layers)):
            hidden_before = layer_b(hidden_before, mask=None, cache=None)
            hidden_after = layer_a(hidden_after, mask=None, cache=None)

            if isinstance(hidden_before, tuple):
                hidden_before = hidden_before[0]
            if isinstance(hidden_after, tuple):
                hidden_after = hidden_after[0]

            mx.eval(hidden_before, hidden_after)

            # Compute difference
            diff = hidden_after - hidden_before
            diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
            before_norm = float(mx.sqrt(mx.sum(hidden_before * hidden_before)))

            layer_diffs.append({
                "layer": idx,
                "diff_norm": diff_norm,
                "relative_diff": diff_norm / (before_norm + 1e-10),
            })

        results.append({
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "layer_diffs": layer_diffs,
            "peak_diff_layer": max(layer_diffs, key=lambda x: x["relative_diff"])["layer"],
        })

    return {
        "activation_comparisons": results,
        "avg_peak_layer": np.mean([r["peak_diff_layer"] for r in results]),
    }
