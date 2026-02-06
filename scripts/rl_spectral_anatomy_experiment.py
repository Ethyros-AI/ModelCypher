#!/usr/bin/env python3
"""RL Spectral Anatomy Experiment (Experiment 3).

Derives the mathematical chain connecting weight differences to activation-space
effects observed in Experiments 1 and 2:

    Weights → Jacobian singular values → Activation effective rank
            → Intrinsic dimension → Expansion ratio / CKA profile

Five formulas are tested:
1. Weight norms → Jacobian σ_max (submultiplicative bound)
2. Jacobian effective rank → Centroid effective rank
3. Centroid effective rank → Cross-category CKA
4. Centroid effective rank ↔ Intrinsic dimension (TwoNN)
5. Jacobian spectrum → Expansion ratio

Uses existing JSONL data from Experiment 2 for Phase 1 (CPU-only velocity
decomposition). Phases 2-3 collect new model-dependent data and derive formulas.

Usage:
    # Smoke test (~15 min, 2 Jacobian prompts + 10 centroid prompts per model)
    poetry run python scripts/rl_spectral_anatomy_experiment.py \
        --smoke --output results/rl_spectral_anatomy/smoke/

    # Full experiment (~40 min, 5 Jacobian prompts + 30 centroid prompts)
    poetry run python scripts/rl_spectral_anatomy_experiment.py \
        --output results/rl_spectral_anatomy/

    # Skip Jacobian collection (CPU-only phases + centroid collection)
    poetry run python scripts/rl_spectral_anatomy_experiment.py \
        --skip-jacobian --output results/rl_spectral_anatomy/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry (same as experiments 1 & 2)
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY = {
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "architecture": "qwen3",
        "training": "base",
    },
    "DeepSeek-R1-8B": {
        "path": f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        "architecture": "qwen3",
        "training": "rl_grpo",
    },
}


# =============================================================================
# Probe Categories (same 30 prompts as experiments 1 & 2)
# =============================================================================

PROBE_CATEGORIES = {
    "retrieval": [
        "The capital of France is",
        "Who wrote Romeo and Juliet?",
        "The chemical symbol for water is",
        "The largest planet in our solar system is",
        "The speed of light in a vacuum is approximately",
        "The first president of the United States was",
    ],
    "arithmetic": [
        "What is 347 + 528?",
        "What is 15 * 23?",
        "What is 1024 / 16?",
        "What is 99 - 37?",
        "What is 8 * 7 + 13?",
        "What is 256 + 384 - 100?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. At the second stop, 12 get off and 7 get on. How many people are on the bus now?",
        "A lily pad doubles in size every day. If it takes 48 days to cover the entire lake, on what day does it cover half the lake?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Describe a sunset over the mountains in one vivid sentence.",
        "Write a short poem about the passage of time.",
        "Describe the taste of your favorite food using only three words.",
        "Write a one-sentence story with a twist ending.",
        "Describe the sound of rain on a tin roof.",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "Write a Python function that checks if a number is prime.",
        "Write a Python function to compute the Fibonacci sequence up to n terms.",
        "Write a Python function that finds the maximum element in a list without using max().",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python one-liner to flatten a nested list.",
    ],
}

# Jacobian prompts: index 0 from each category
JACOBIAN_PROMPTS = {
    "retrieval": 0,
    "arithmetic": 0,
    "reasoning": 0,
    "creative": 0,
    "code": 0,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VelocityDecomposition:
    """Velocity decomposed into radial (magnitude) and angular (rotation) components."""

    prompt: str
    category: str
    radial_trajectory: list[float]
    angular_trajectory: list[float]
    cos_theta_trajectory: list[float]
    velocity_trajectory: list[float]
    norm_trajectory: list[float]


@dataclass
class LayerWeightProfile:
    """Weight norm profile for a single layer."""

    layer_idx: int
    # Attention weight spectral norms
    q_spectral: float
    k_spectral: float
    v_spectral: float
    o_spectral: float
    # MLP weight spectral norms
    gate_spectral: float
    up_spectral: float
    down_spectral: float
    # Frobenius norms
    q_frobenius: float
    k_frobenius: float
    v_frobenius: float
    o_frobenius: float
    gate_frobenius: float
    up_frobenius: float
    down_frobenius: float
    # Predicted Jacobian norm upper bound
    predicted_jacobian_norm: float


@dataclass
class EmbeddingProfile:
    """Embedding matrix analysis."""

    top_20_singular_values: list[float]
    effective_rank_renyi: float
    effective_rank_shannon: float
    var_top1: float
    var_top5: float
    vocab_size: int
    hidden_dim: int


@dataclass
class CentroidLayerMetrics:
    """Effective rank metrics computed from centroid matrix at a single layer."""

    layer_idx: int
    effective_rank_renyi: float
    effective_rank_shannon: float
    var_top1: float
    var_top5: float
    n_samples: int


@dataclass
class FormulaFit:
    """Result of fitting a formula to data."""

    name: str
    formula: str
    alpha: float
    beta: float
    r_squared: float
    n_points: int
    residuals: list[float] = field(default_factory=list)


@dataclass
class SpectralAnatomyResults:
    """All results from the experiment."""

    # Phase 1
    velocity_decompositions: dict[str, list[dict]]  # model -> list of decomps
    velocity_analysis: dict[str, Any]  # summary statistics

    # Phase 2
    weight_profiles: dict[str, list[dict]]  # model -> per-layer profiles
    embedding_profiles: dict[str, dict]  # model -> embedding analysis
    jacobian_traces: dict[str, list[dict]]  # model -> per-prompt traces
    centroid_metrics: dict[str, list[dict]]  # model -> per-layer centroid ER

    # Phase 3
    formula_fits: dict[str, dict]  # formula_name -> fit results per model
    cross_validation: dict[str, Any]  # predictions vs observed

    # Metadata
    experiment: dict[str, Any]


# =============================================================================
# Statistics Helpers (reused from experiment 2 + new fitting functions)
# =============================================================================


def cohens_d_two_groups(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d between two independent groups (pooled std)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    return (m1 - m2) / math.sqrt(pooled_var)


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _safe_mean(values: list[float]) -> float:
    """Mean of values, filtering NaN."""
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def _safe_std(values: list[float]) -> float:
    """Std of values, filtering NaN."""
    valid = [v for v in values if v == v]
    if len(valid) < 2:
        return 0.0
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def fit_linear(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares: y = alpha * x + beta.

    Returns (alpha, beta, r_squared).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    mx = sum(x) / n
    my = sum(y) / n
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    ss_yy = sum((yi - my) ** 2 for yi in y)

    if ss_xx < 1e-15:
        return 0.0, my, 0.0

    alpha = ss_xy / ss_xx
    beta = my - alpha * mx

    # R²
    ss_res = sum((yi - (alpha * xi + beta)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1.0 - ss_res / ss_yy if ss_yy > 1e-15 else 0.0

    return alpha, beta, r_squared


def fit_exponential(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Fit y = a * exp(-b * x) via log-linear OLS on log(y) = log(a) - b*x.

    Returns (a, b, r_squared) where r_squared is computed on the original scale.
    Only uses positive y values.
    """
    # Filter to positive y values
    pairs = [(xi, yi) for xi, yi in zip(x, y) if yi > 0]
    if len(pairs) < 2:
        return 0.0, 0.0, 0.0

    x_pos = [p[0] for p in pairs]
    y_pos = [p[1] for p in pairs]
    log_y = [math.log(yi) for yi in y_pos]

    # Linear fit: log(y) = log(a) - b*x  →  log(y) = alpha*x + beta
    alpha, beta, _ = fit_linear(x_pos, log_y)
    a = math.exp(beta)
    b = -alpha

    # R² on original scale
    y_pred = [a * math.exp(-b * xi) for xi in x_pos]
    my = sum(y_pos) / len(y_pos)
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y_pos, y_pred))
    ss_tot = sum((yi - my) ** 2 for yi in y_pos)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return a, b, r_squared


def fit_inverse(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Fit y = alpha / x + beta via OLS on y vs 1/x.

    Returns (alpha, beta, r_squared).
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y) if abs(xi) > 1e-15]
    if len(pairs) < 2:
        return 0.0, 0.0, 0.0

    inv_x = [1.0 / p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    return fit_linear(inv_x, y_vals)


# =============================================================================
# Main Experiment Class
# =============================================================================


class RLSpectralAnatomyExperiment:
    """RL Spectral Anatomy Experiment (Experiment 3)."""

    def __init__(
        self,
        output_dir: Path,
        seed: int = 42,
        smoke: bool = False,
        existing_data: str = "results/rl_processing",
        skip_jacobian: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.seed = seed
        self.smoke = smoke
        self.existing_data = Path(existing_data)
        self.skip_jacobian = skip_jacobian
        self._rng = random.Random(seed)
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0

    # ----- Model lifecycle -----

    def _load_model(self, model_name: str) -> None:
        """Load model and tokenizer, detect layer count."""
        from modelcypher.backends import initialize_default_backend

        if self.backend is None:
            self.backend = initialize_default_backend()

        model_info = MODEL_REGISTRY[model_name]
        model_path = model_info["path"]

        logger.info(f"Loading model: {model_name} from {model_path}")
        self.model, self.tokenizer = self.backend.load_model(model_path)

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        self.num_layers = len(layers) if layers else 0

        logger.info(f"Model loaded: {self.num_layers} layers")

    def _unload_model(self) -> None:
        """Unload model and free GPU memory."""
        self.model = None
        self.tokenizer = None
        self.num_layers = 0
        gc.collect()
        gc.collect()
        if self.backend is not None:
            try:
                self.backend.clear_cache()
            except Exception:
                pass

    def _format_prompt(self, raw_prompt: str) -> str:
        """Apply chat template if available."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return raw_prompt

    # =====================================================================
    # Phase 1: CPU Velocity Decomposition
    # =====================================================================

    def _load_existing_data(self) -> dict[str, list[dict]]:
        """Load existing JSONL data from experiment 2."""
        raw_dir = self.existing_data / "raw"
        data: dict[str, list[dict]] = {}

        for model_name in MODEL_REGISTRY:
            jsonl_path = raw_dir / f"{model_name}_geometry.jsonl"
            if not jsonl_path.exists():
                logger.warning(f"Missing JSONL: {jsonl_path}")
                continue

            records = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            data[model_name] = records
            logger.info(f"Loaded {len(records)} prompts from {jsonl_path}")

        return data

    def _load_existing_results(self) -> dict | None:
        """Load raw_results.json from experiment 2 (for CKA data)."""
        results_path = self.existing_data / "raw_results.json"
        if not results_path.exists():
            logger.warning(f"Missing results: {results_path}")
            return None
        with open(results_path) as f:
            return json.load(f)

    def _decompose_velocities(
        self, existing_data: dict[str, list[dict]]
    ) -> dict[str, list[VelocityDecomposition]]:
        """Decompose velocity into radial and angular components via law of cosines."""
        results: dict[str, list[VelocityDecomposition]] = {}

        for model_name, records in existing_data.items():
            decomps = []
            for record in records:
                vel_traj = record.get("velocity_trajectory", [])
                norm_traj = record.get("norm_trajectory", [])

                if not vel_traj or not norm_traj or len(norm_traj) < len(vel_traj) + 1:
                    continue

                radial = []
                angular = []
                cos_theta = []

                for l in range(len(vel_traj)):
                    vel = vel_traj[l]
                    n_l = norm_traj[l]
                    n_l1 = norm_traj[l + 1]

                    if n_l < 1e-15 or n_l1 < 1e-15:
                        radial.append(0.0)
                        angular.append(0.0)
                        cos_theta.append(1.0)
                        continue

                    # ||Δh|| = vel * ||h_l||
                    step = vel * n_l

                    # Law of cosines: ||h_{l+1}||² = ||h_l||² + ||Δh||² + 2·||h_l||·||Δh||·cos(angle)
                    # But velocity = ||h_{l+1} - h_l|| / ||h_l||, so step = ||h_{l+1} - h_l||
                    # ||h_{l+1}||² = ||h_l||² + step² - 2·||h_l||·step·cos(π-θ)
                    # Actually: h_{l+1} · h_l = (||h_{l+1}||² + ||h_l||² - step²) / 2
                    dot = (n_l1**2 + n_l**2 - step**2) / 2.0

                    ct = max(-1.0, min(1.0, dot / (n_l * n_l1)))
                    cos_theta.append(ct)

                    # Radial component: magnitude change along h_l direction
                    rad = ct * n_l1 / n_l - 1.0
                    radial.append(rad)

                    # Angular component: perpendicular change
                    ang_sq = max(0.0, vel**2 - rad**2)
                    angular.append(math.sqrt(ang_sq))

                decomps.append(VelocityDecomposition(
                    prompt=record["prompt"],
                    category=record["category"],
                    radial_trajectory=radial,
                    angular_trajectory=angular,
                    cos_theta_trajectory=cos_theta,
                    velocity_trajectory=vel_traj,
                    norm_trajectory=norm_traj,
                ))
            results[model_name] = decomps
            logger.info(f"  {model_name}: decomposed {len(decomps)} prompts")

        return results

    def _analyze_velocity_decomposition(
        self, decomps: dict[str, list[VelocityDecomposition]]
    ) -> dict:
        """Analyze radial vs angular components across models."""
        analysis: dict[str, Any] = {}

        model_names = list(decomps.keys())
        if len(model_names) < 2:
            return analysis

        base_name = [n for n in model_names if MODEL_REGISTRY[n]["training"] == "base"]
        rl_name = [n for n in model_names if MODEL_REGISTRY[n]["training"] != "base"]

        if not base_name or not rl_name:
            return analysis

        base_name = base_name[0]
        rl_name = rl_name[0]

        base_decomps = decomps[base_name]
        rl_decomps = decomps[rl_name]

        if not base_decomps or not rl_decomps:
            return analysis

        num_layers = len(base_decomps[0].velocity_trajectory)

        per_layer: list[dict] = []
        for l in range(num_layers):
            base_radial = [d.radial_trajectory[l] for d in base_decomps if l < len(d.radial_trajectory)]
            rl_radial = [d.radial_trajectory[l] for d in rl_decomps if l < len(d.radial_trajectory)]
            base_angular = [d.angular_trajectory[l] for d in base_decomps if l < len(d.angular_trajectory)]
            rl_angular = [d.angular_trajectory[l] for d in rl_decomps if l < len(d.angular_trajectory)]
            base_cos = [d.cos_theta_trajectory[l] for d in base_decomps if l < len(d.cos_theta_trajectory)]
            rl_cos = [d.cos_theta_trajectory[l] for d in rl_decomps if l < len(d.cos_theta_trajectory)]

            per_layer.append({
                "layer": l,
                "base_mean_radial": _safe_mean(base_radial),
                "rl_mean_radial": _safe_mean(rl_radial),
                "base_mean_angular": _safe_mean(base_angular),
                "rl_mean_angular": _safe_mean(rl_angular),
                "base_mean_cos_theta": _safe_mean(base_cos),
                "rl_mean_cos_theta": _safe_mean(rl_cos),
                "cohens_d_radial": cohens_d_two_groups(base_radial, rl_radial),
                "cohens_d_angular": cohens_d_two_groups(base_angular, rl_angular),
            })

        analysis["per_layer"] = per_layer
        analysis["base_name"] = base_name
        analysis["rl_name"] = rl_name

        # Overall: is the L0-2 spike radial or angular?
        early_layers = list(range(min(3, num_layers)))
        base_early_radial = [abs(per_layer[l]["base_mean_radial"]) for l in early_layers]
        rl_early_radial = [abs(per_layer[l]["rl_mean_radial"]) for l in early_layers]
        base_early_angular = [per_layer[l]["base_mean_angular"] for l in early_layers]
        rl_early_angular = [per_layer[l]["rl_mean_angular"] for l in early_layers]

        analysis["early_layer_summary"] = {
            "base_mean_abs_radial": _safe_mean(base_early_radial),
            "rl_mean_abs_radial": _safe_mean(rl_early_radial),
            "base_mean_angular": _safe_mean(base_early_angular),
            "rl_mean_angular": _safe_mean(rl_early_angular),
            "dominant_component": (
                "angular" if _safe_mean(rl_early_angular) > _safe_mean(rl_early_radial)
                else "radial"
            ),
        }

        return analysis

    # =====================================================================
    # Phase 2: Model-Dependent Collection
    # =====================================================================

    def _collect_weight_profiles(self, model_name: str) -> list[LayerWeightProfile]:
        """Collect spectral and Frobenius norms of weight matrices per layer."""
        from modelcypher.core.domain.geometry.numerical_stability import geodesic_svd

        b = self.backend
        base_model = getattr(self.model, "model", self.model)
        layers = base_model.layers
        profiles = []

        for l_idx in range(len(layers)):
            layer = layers[l_idx]
            attn = layer.self_attn
            mlp = layer.mlp

            # Weight matrices and their names
            weight_specs = {
                "q": attn.q_proj.weight,
                "k": attn.k_proj.weight,
                "v": attn.v_proj.weight,
                "o": attn.o_proj.weight,
                "gate": mlp.gate_proj.weight,
                "up": mlp.up_proj.weight,
                "down": mlp.down_proj.weight,
            }

            spectral_norms = {}
            frobenius_norms = {}

            for name, w in weight_specs.items():
                w_arr = b.array(w)
                b.eval(w_arr)

                # Frobenius norm
                frob = float(b.to_scalar(b.sqrt(b.sum(w_arr * w_arr))))
                frobenius_norms[name] = frob

                # Spectral norm via SVD k=1
                _, s, _ = geodesic_svd(b, w_arr, k=1)
                b.eval(s)
                spectral_norms[name] = float(b.to_scalar(s[0])) if s.shape[0] > 0 else 0.0

                # Free immediately
                del w_arr, s

            # Predicted Jacobian norm upper bound (submultiplicative on residual stream):
            # J_l = I + dAttn/dh + dMLP/dh
            # ||J_l|| ≤ 1 + σ(O)·σ(V) + σ(down)·[σ(gate) + σ(up)]
            predicted_j = (
                1.0
                + spectral_norms["o"] * spectral_norms["v"]
                + spectral_norms["down"] * (spectral_norms["gate"] + spectral_norms["up"])
            )

            profiles.append(LayerWeightProfile(
                layer_idx=l_idx,
                q_spectral=spectral_norms["q"],
                k_spectral=spectral_norms["k"],
                v_spectral=spectral_norms["v"],
                o_spectral=spectral_norms["o"],
                gate_spectral=spectral_norms["gate"],
                up_spectral=spectral_norms["up"],
                down_spectral=spectral_norms["down"],
                q_frobenius=frobenius_norms["q"],
                k_frobenius=frobenius_norms["k"],
                v_frobenius=frobenius_norms["v"],
                o_frobenius=frobenius_norms["o"],
                gate_frobenius=frobenius_norms["gate"],
                up_frobenius=frobenius_norms["up"],
                down_frobenius=frobenius_norms["down"],
                predicted_jacobian_norm=predicted_j,
            ))

            if (l_idx + 1) % 10 == 0:
                logger.info(f"    Weight profiles: {l_idx + 1}/{len(layers)} layers")

        return profiles

    def _collect_embedding_profile(self, model_name: str) -> EmbeddingProfile:
        """Analyze embedding matrix: SVD, effective rank, variance concentration.

        Uses Gram matrix approach (W^T @ W) to avoid full SVD on the large
        [vocab_size, hidden_dim] matrix which exceeds Metal memory limits.
        """
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
        )
        from modelcypher.core.domain.geometry.variance_concentration import (
            compute_variance_concentration,
        )

        b = self.backend
        base_model = getattr(self.model, "model", self.model)
        embed_weight = base_model.embed_tokens.weight

        w = b.array(embed_weight)
        b.eval(w)

        vocab_size = int(w.shape[0])
        hidden_dim = int(w.shape[1])

        # Use Gram matrix: W^T @ W is [hidden_dim, hidden_dim] — fits in memory
        # Singular values of W = sqrt(eigenvalues of W^T @ W)
        mean = b.mean(w, axis=0, keepdims=True)
        centered = w - mean
        gram = b.matmul(b.transpose(centered), centered)
        eps = machine_epsilon(b, gram)
        gram = gram + eps * b.eye(hidden_dim)
        eigenvalues = b.eigvalsh(gram)
        b.eval(eigenvalues)

        # Sort descending, take sqrt for singular values
        eig_sorted = b.sort(eigenvalues)[::-1]
        eig_sorted = b.maximum(eig_sorted, b.zeros_like(eig_sorted))
        sv_all = b.sqrt(eig_sorted)
        b.eval(sv_all)

        # Top-20 singular values
        n_sv = min(20, int(sv_all.shape[0]))
        top_20_sv = [float(b.to_scalar(sv_all[i])) for i in range(n_sv)]

        # Effective rank from eigenvalues (same formula as EffectiveRank)
        sum_eig = b.sum(eig_sorted)
        sum_eig_sq = b.sum(eig_sorted * eig_sorted)
        b.eval(sum_eig, sum_eig_sq)

        sum_eig_val = float(b.to_scalar(sum_eig))
        sum_eig_sq_val = float(b.to_scalar(sum_eig_sq))

        renyi_rank = (
            (sum_eig_val * sum_eig_val) / sum_eig_sq_val
            if sum_eig_sq_val > float(eps)
            else 0.0
        )

        if sum_eig_val > float(eps):
            p = eig_sorted / sum_eig_val
            log_eps_val = 1e-30
            p_safe = b.maximum(p, b.full(p.shape, log_eps_val))
            entropy_terms = -p * b.log(p_safe)
            spectral_entropy_arr = b.sum(entropy_terms)
            shannon_rank_arr = b.exp(spectral_entropy_arr)
            b.eval(shannon_rank_arr)
            shannon_rank = float(b.to_scalar(shannon_rank_arr))
        else:
            shannon_rank = 0.0

        # Variance concentration (uses Gram internally for large matrices)
        vc_result = compute_variance_concentration(w, backend=b, top_k=[1, 5])

        del w, centered, gram, eigenvalues, eig_sorted, sv_all
        return EmbeddingProfile(
            top_20_singular_values=top_20_sv,
            effective_rank_renyi=renyi_rank,
            effective_rank_shannon=shannon_rank,
            var_top1=vc_result.var_top1,
            var_top5=vc_result.var_top_k.get(5, 0.0),
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )

    def _collect_jacobian_traces(
        self, model_name: str
    ) -> list[dict]:
        """Collect Jacobian spectrum traces for selected prompts."""
        from modelcypher.core.domain.geometry.jacobian_analyzer import (
            trace_jacobian_spectrum,
        )

        model_info = MODEL_REGISTRY[model_name]
        model_path = model_info["path"]

        # Select prompts based on smoke mode
        if self.smoke:
            categories_to_use = ["retrieval", "reasoning"]
        else:
            categories_to_use = list(JACOBIAN_PROMPTS.keys())

        traces = []
        for cat in categories_to_use:
            idx = JACOBIAN_PROMPTS[cat]
            prompt = PROBE_CATEGORIES[cat][idx]
            formatted = self._format_prompt(prompt)

            logger.info(f"    Jacobian trace: {cat} prompt")
            result = trace_jacobian_spectrum(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted,
                model_path=model_path,
                num_probes=32,
                backend=self.backend,
            )
            traces.append(result.as_dict())

            # Clear cache between prompts
            if self.backend is not None:
                try:
                    self.backend.clear_cache()
                except Exception:
                    pass

        return traces

    def _collect_centroid_metrics(
        self, model_name: str
    ) -> tuple[list[CentroidLayerMetrics], list[list[list[float]]]]:
        """Collect centroids for all prompts, then compute effective rank per layer.

        Returns:
            (layer_metrics, all_centroids) where all_centroids is [n_prompts][n_layers][hidden_dim]
        """
        from modelcypher.adapters.activation_provider import ActivationProviderAdapter
        from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
        from modelcypher.core.domain.geometry.variance_concentration import (
            compute_variance_concentration,
        )

        b = self.backend
        provider = ActivationProviderAdapter(backend=b)
        er_calc = EffectiveRank(backend=b)

        # Select prompts
        all_prompts = []
        for cat in PROBE_CATEGORIES:
            prompts = PROBE_CATEGORIES[cat]
            if self.smoke:
                prompts = prompts[:2]
            for p in prompts:
                all_prompts.append((p, cat))

        logger.info(f"    Collecting centroids for {len(all_prompts)} prompts")

        # Collect centroids: [n_prompts][n_layers][hidden_dim] as Python lists
        all_centroids: list[list[list[float]]] = []

        for i, (prompt, cat) in enumerate(all_prompts):
            formatted = self._format_prompt(prompt)

            trajectory = provider.collect_trajectory_batch(
                self.model, self.tokenizer, [formatted]
            )

            prompt_centroids: list[list[float]] = []
            for l_idx in range(self.num_layers):
                if l_idx in trajectory.positions:
                    pos = trajectory.positions[l_idx]
                    centroid = b.mean(pos, axis=0)
                    b.eval(centroid)
                    centroid_list = b.tolist(centroid)
                    if not isinstance(centroid_list, list):
                        centroid_list = [centroid_list]
                    prompt_centroids.append(centroid_list)
                else:
                    prompt_centroids.append([])

            all_centroids.append(prompt_centroids)

            # Free trajectory positions
            del trajectory
            gc.collect()

            if (i + 1) % 5 == 0:
                logger.info(f"      Centroids: {i + 1}/{len(all_prompts)} prompts")
                if self.backend is not None:
                    try:
                        self.backend.clear_cache()
                    except Exception:
                        pass

        # Compute per-layer effective rank from centroid matrix
        layer_metrics: list[CentroidLayerMetrics] = []
        n_prompts = len(all_centroids)

        for l_idx in range(self.num_layers):
            # Build [n_prompts, hidden_dim] centroid matrix
            centroids_at_layer = [
                all_centroids[p][l_idx]
                for p in range(n_prompts)
                if l_idx < len(all_centroids[p]) and all_centroids[p][l_idx]
            ]

            if len(centroids_at_layer) < 2:
                layer_metrics.append(CentroidLayerMetrics(
                    layer_idx=l_idx,
                    effective_rank_renyi=0.0,
                    effective_rank_shannon=0.0,
                    var_top1=0.0,
                    var_top5=0.0,
                    n_samples=0,
                ))
                continue

            centroid_matrix = b.array(centroids_at_layer)
            b.eval(centroid_matrix)

            er_result = er_calc.compute(centroid_matrix)
            vc_result = compute_variance_concentration(
                centroid_matrix, backend=b, top_k=[1, 5]
            )

            layer_metrics.append(CentroidLayerMetrics(
                layer_idx=l_idx,
                effective_rank_renyi=er_result.renyi_effective_rank,
                effective_rank_shannon=er_result.shannon_effective_rank,
                var_top1=vc_result.var_top1,
                var_top5=vc_result.var_top_k.get(5, 0.0),
                n_samples=len(centroids_at_layer),
            ))

            del centroid_matrix

        return layer_metrics, all_centroids

    # =====================================================================
    # Phase 3: Formula Derivation
    # =====================================================================

    def _fit_weight_to_jacobian(
        self,
        weight_profiles: dict[str, list[dict]],
        jacobian_traces: dict[str, list[dict]],
    ) -> dict[str, FormulaFit]:
        """Formula 1: predicted weight norm bound vs measured Jacobian σ_max."""
        fits: dict[str, FormulaFit] = {}

        for model_name in weight_profiles:
            if model_name not in jacobian_traces or not jacobian_traces[model_name]:
                continue

            wp = weight_profiles[model_name]
            # Average measured σ_max across all Jacobian-traced prompts
            jt = jacobian_traces[model_name]

            num_layers = len(wp)
            predicted = [wp[l]["predicted_jacobian_norm"] for l in range(num_layers)]

            # Measured: average norm_amplification per layer across prompts
            measured = [0.0] * num_layers
            count = [0] * num_layers
            for trace in jt:
                for profile in trace["profiles"]:
                    l = profile["layer_idx"]
                    if l < num_layers:
                        measured[l] += profile["norm_amplification"]
                        count[l] += 1

            measured = [
                measured[l] / count[l] if count[l] > 0 else 0.0
                for l in range(num_layers)
            ]

            # Filter layers with valid data
            valid_pred = []
            valid_meas = []
            for l in range(num_layers):
                if predicted[l] > 0 and measured[l] > 0:
                    valid_pred.append(predicted[l])
                    valid_meas.append(measured[l])

            alpha, beta, r2 = fit_linear(valid_pred, valid_meas)

            residuals = [
                valid_meas[i] - (alpha * valid_pred[i] + beta)
                for i in range(len(valid_pred))
            ]

            # Tightness ratio: measured / predicted
            tightness = [
                valid_meas[i] / valid_pred[i] if valid_pred[i] > 0 else 0.0
                for i in range(len(valid_pred))
            ]

            fits[model_name] = FormulaFit(
                name="weight_to_jacobian",
                formula="measured_J_σmax = α · predicted_J_norm + β",
                alpha=alpha,
                beta=beta,
                r_squared=r2,
                n_points=len(valid_pred),
                residuals=residuals,
            )

            mean_tightness = _safe_mean(tightness)
            logger.info(
                f"  Formula 1 ({model_name}): R²={r2:.4f}, "
                f"α={alpha:.4f}, β={beta:.4f}, "
                f"mean tightness={mean_tightness:.4f}"
            )

        return fits

    def _fit_jacobian_to_centroid_rank(
        self,
        jacobian_traces: dict[str, list[dict]],
        centroid_metrics: dict[str, list[dict]],
    ) -> dict[str, FormulaFit]:
        """Formula 2: Jacobian effective rank → Centroid effective rank."""
        fits: dict[str, FormulaFit] = {}

        for model_name in jacobian_traces:
            if model_name not in centroid_metrics:
                continue

            jt = jacobian_traces[model_name]
            cm = centroid_metrics[model_name]

            if not jt or not cm:
                continue

            num_layers = len(cm)

            # Average Jacobian effective rank per layer
            j_er = [0.0] * num_layers
            j_count = [0] * num_layers
            for trace in jt:
                for profile in trace["profiles"]:
                    l = profile["layer_idx"]
                    if l < num_layers:
                        j_er[l] += profile["effective_rank_shannon"]
                        j_count[l] += 1

            j_er = [j_er[l] / j_count[l] if j_count[l] > 0 else 0.0 for l in range(num_layers)]
            c_er = [cm[l]["effective_rank_shannon"] for l in range(num_layers)]

            # Spearman correlation
            valid_j = []
            valid_c = []
            for l in range(num_layers):
                if j_er[l] > 0 and c_er[l] > 0:
                    valid_j.append(j_er[l])
                    valid_c.append(c_er[l])

            if len(valid_j) < 3:
                continue

            rho = spearman_correlation(valid_j, valid_c)

            # Test cumulative model: centroid_ER(l) ≈ α · min(Jacobian_ER(1..l)) + β
            cum_min_j = []
            running_min = float("inf")
            for l in range(num_layers):
                if j_er[l] > 0:
                    running_min = min(running_min, j_er[l])
                cum_min_j.append(running_min if running_min < float("inf") else 0.0)

            valid_min = []
            valid_c2 = []
            for l in range(num_layers):
                if cum_min_j[l] > 0 and c_er[l] > 0:
                    valid_min.append(cum_min_j[l])
                    valid_c2.append(c_er[l])

            alpha, beta, r2 = fit_linear(valid_min, valid_c2)

            fits[model_name] = FormulaFit(
                name="jacobian_to_centroid_rank",
                formula="centroid_ER(l) ≈ α · min(J_ER(1..l)) + β",
                alpha=alpha,
                beta=beta,
                r_squared=r2,
                n_points=len(valid_min),
            )

            logger.info(
                f"  Formula 2 ({model_name}): Spearman ρ={rho:.4f}, "
                f"cumulative min model R²={r2:.4f}"
            )

        return fits

    def _fit_rank_to_cka(
        self,
        centroid_metrics: dict[str, list[dict]],
        existing_results: dict | None,
    ) -> dict[str, FormulaFit]:
        """Formula 3: Centroid effective rank → Cross-category CKA."""
        fits: dict[str, FormulaFit] = {}

        if existing_results is None or "comparison" not in existing_results:
            return fits

        cka_data = existing_results["comparison"].get("cross_category_cka", {})
        base_cka = cka_data.get("base_mean_cross_cka_per_layer", [])
        rl_cka = cka_data.get("rl_mean_cross_cka_per_layer", [])

        model_cka_map = {}
        for model_name in MODEL_REGISTRY:
            if MODEL_REGISTRY[model_name]["training"] == "base":
                model_cka_map[model_name] = base_cka
            else:
                model_cka_map[model_name] = rl_cka

        for model_name in centroid_metrics:
            cm = centroid_metrics[model_name]
            cka_per_layer = model_cka_map.get(model_name, [])

            if not cm or not cka_per_layer:
                continue

            num_layers = min(len(cm), len(cka_per_layer))

            c_er = [cm[l]["effective_rank_shannon"] for l in range(num_layers)]
            cka_vals = [cka_per_layer[l] for l in range(num_layers)]

            # Filter valid pairs
            valid_er = []
            valid_cka = []
            for l in range(num_layers):
                if c_er[l] > 0 and cka_vals[l] == cka_vals[l]:  # not NaN
                    valid_er.append(c_er[l])
                    valid_cka.append(cka_vals[l])

            if len(valid_er) < 3:
                continue

            # Test inverse: CKA = α / ER + β
            alpha_inv, beta_inv, r2_inv = fit_inverse(valid_er, valid_cka)

            # Test exponential: CKA = a * exp(-b * ER)
            a_exp, b_exp, r2_exp = fit_exponential(valid_er, valid_cka)

            # Pick best fit
            if r2_inv >= r2_exp:
                fits[model_name] = FormulaFit(
                    name="rank_to_cka",
                    formula="CKA(l) = α / ER(l) + β",
                    alpha=alpha_inv,
                    beta=beta_inv,
                    r_squared=r2_inv,
                    n_points=len(valid_er),
                )
                logger.info(
                    f"  Formula 3 ({model_name}): INVERSE fit R²={r2_inv:.4f} "
                    f"(exp R²={r2_exp:.4f})"
                )
            else:
                fits[model_name] = FormulaFit(
                    name="rank_to_cka",
                    formula="CKA(l) = a · exp(-b · ER(l))",
                    alpha=a_exp,
                    beta=b_exp,
                    r_squared=r2_exp,
                    n_points=len(valid_er),
                )
                logger.info(
                    f"  Formula 3 ({model_name}): EXPONENTIAL fit R²={r2_exp:.4f} "
                    f"(inv R²={r2_inv:.4f})"
                )

        return fits

    def _fit_rank_to_id(
        self,
        centroid_metrics: dict[str, list[dict]],
        existing_data: dict[str, list[dict]],
    ) -> dict[str, FormulaFit]:
        """Formula 4: Centroid effective rank ↔ TwoNN intrinsic dimension."""
        fits: dict[str, FormulaFit] = {}

        for model_name in centroid_metrics:
            if model_name not in existing_data:
                continue

            cm = centroid_metrics[model_name]
            records = existing_data[model_name]

            if not cm or not records:
                continue

            # Average TwoNN ID per layer across all prompts
            num_layers = len(cm)
            id_sums = [0.0] * num_layers
            id_counts = [0] * num_layers

            for record in records:
                id_traj = record.get("id_trajectory", [])
                for l in range(min(len(id_traj), num_layers)):
                    if id_traj[l] == id_traj[l]:  # not NaN
                        id_sums[l] += id_traj[l]
                        id_counts[l] += 1

            mean_id = [
                id_sums[l] / id_counts[l] if id_counts[l] > 0 else 0.0
                for l in range(num_layers)
            ]
            c_er = [cm[l]["effective_rank_shannon"] for l in range(num_layers)]

            valid_er = []
            valid_id = []
            for l in range(num_layers):
                if c_er[l] > 0 and mean_id[l] > 0:
                    valid_er.append(c_er[l])
                    valid_id.append(mean_id[l])

            if len(valid_er) < 3:
                continue

            rho = spearman_correlation(valid_er, valid_id)
            alpha, beta, r2 = fit_linear(valid_er, valid_id)

            fits[model_name] = FormulaFit(
                name="rank_to_id",
                formula="TwoNN_ID(l) ≈ α · centroid_ER(l) + β",
                alpha=alpha,
                beta=beta,
                r_squared=r2,
                n_points=len(valid_er),
            )

            logger.info(
                f"  Formula 4 ({model_name}): Spearman ρ={rho:.4f}, "
                f"linear R²={r2:.4f}"
            )

        return fits

    def _fit_jacobian_to_expansion(
        self,
        jacobian_traces: dict[str, list[dict]],
        existing_data: dict[str, list[dict]],
    ) -> dict[str, FormulaFit]:
        """Formula 5: Jacobian spectrum → Expansion ratio.

        Test: expansion_ratio ≈ max_l(Jacobian_ER(l)) / Jacobian_ER(L_final)
        """
        fits: dict[str, FormulaFit] = {}

        for model_name in jacobian_traces:
            jt = jacobian_traces[model_name]
            if not jt:
                continue

            # Build prompt → expansion_ratio mapping from existing data
            er_map: dict[str, float] = {}
            if model_name in existing_data:
                for record in existing_data[model_name]:
                    er_val = record.get("expansion_ratio", float("nan"))
                    if er_val == er_val:
                        er_map[record["prompt"]] = er_val

            predicted_er = []
            measured_er = []

            for trace in jt:
                prompt = trace.get("prompt", "")
                profiles = trace.get("profiles", [])

                if not profiles:
                    continue

                # Compute max_l(Jacobian_ER) / Jacobian_ER(final)
                j_ers = [p["effective_rank_shannon"] for p in profiles if p["effective_rank_shannon"] > 0]
                if not j_ers:
                    continue

                max_j_er = max(j_ers)
                final_j_er = j_ers[-1] if j_ers else 1.0

                if final_j_er > 0:
                    pred_expansion = max_j_er / final_j_er
                else:
                    continue

                # Try to find measured expansion_ratio
                # Match by checking if the prompt (formatted) matches any record
                # Since formatted prompts may differ, also try unformatted
                measured = None
                for raw_prompt, er_val in er_map.items():
                    if raw_prompt in prompt or prompt in raw_prompt:
                        measured = er_val
                        break

                if measured is not None:
                    predicted_er.append(pred_expansion)
                    measured_er.append(measured)

            if len(predicted_er) < 2:
                # Not enough matched prompts; report what we have
                fits[model_name] = FormulaFit(
                    name="jacobian_to_expansion",
                    formula="expansion_ratio ≈ max(J_ER) / J_ER(final)",
                    alpha=0.0,
                    beta=0.0,
                    r_squared=0.0,
                    n_points=len(predicted_er),
                )
                if predicted_er:
                    logger.info(
                        f"  Formula 5 ({model_name}): {len(predicted_er)} points, "
                        f"predicted={predicted_er}, measured={measured_er}"
                    )
                continue

            alpha, beta, r2 = fit_linear(predicted_er, measured_er)

            fits[model_name] = FormulaFit(
                name="jacobian_to_expansion",
                formula="expansion_ratio ≈ α · max(J_ER)/J_ER(final) + β",
                alpha=alpha,
                beta=beta,
                r_squared=r2,
                n_points=len(predicted_er),
            )

            logger.info(
                f"  Formula 5 ({model_name}): R²={r2:.4f}, "
                f"α={alpha:.4f}, β={beta:.4f}, "
                f"n={len(predicted_er)}"
            )

        return fits

    def _cross_validate_predictions(
        self,
        formula_fits: dict[str, dict[str, FormulaFit]],
        weight_profiles: dict[str, list[dict]],
        centroid_metrics: dict[str, list[dict]],
        existing_results: dict | None,
        existing_data: dict[str, list[dict]],
    ) -> dict:
        """Apply formulas to both models and check if they predict observed differences."""
        cv: dict[str, Any] = {}

        model_names = list(MODEL_REGISTRY.keys())
        base_name = [n for n in model_names if MODEL_REGISTRY[n]["training"] == "base"]
        rl_name = [n for n in model_names if MODEL_REGISTRY[n]["training"] != "base"]

        if not base_name or not rl_name:
            return cv

        base_name = base_name[0]
        rl_name = rl_name[0]

        # Prediction 1: Expansion ratio difference from weight norms
        if base_name in weight_profiles and rl_name in weight_profiles:
            base_wp = weight_profiles[base_name]
            rl_wp = weight_profiles[rl_name]

            base_predicted_j = [w["predicted_jacobian_norm"] for w in base_wp]
            rl_predicted_j = [w["predicted_jacobian_norm"] for w in rl_wp]

            cv["weight_norm_comparison"] = {
                "base_mean_predicted_j": _safe_mean(base_predicted_j),
                "rl_mean_predicted_j": _safe_mean(rl_predicted_j),
                "ratio": (
                    _safe_mean(rl_predicted_j) / _safe_mean(base_predicted_j)
                    if _safe_mean(base_predicted_j) > 0 else 0.0
                ),
            }

        # Prediction 2: CKA drop from effective rank
        if base_name in centroid_metrics and rl_name in centroid_metrics:
            base_cm = centroid_metrics[base_name]
            rl_cm = centroid_metrics[rl_name]

            if base_cm and rl_cm:
                # Layer 1 CKA prediction via centroid ER
                l1_base_er = base_cm[1]["effective_rank_shannon"] if len(base_cm) > 1 else 0.0
                l1_rl_er = rl_cm[1]["effective_rank_shannon"] if len(rl_cm) > 1 else 0.0

                cv["centroid_er_l1"] = {
                    "base_er": l1_base_er,
                    "rl_er": l1_rl_er,
                    "ratio": l1_rl_er / l1_base_er if l1_base_er > 0 else 0.0,
                }

        # Prediction 3: Velocity inversion from Jacobian
        # (Would need per-layer Jacobian comparison, only available if not skipped)
        if existing_results and "comparison" in existing_results:
            vel_data = existing_results["comparison"].get("velocity", {})
            base_vel = vel_data.get("base_mean_velocity_per_layer", [])
            rl_vel = vel_data.get("rl_mean_velocity_per_layer", [])

            if base_vel and rl_vel:
                # Find velocity inversion layers
                inversion_layers = []
                for l in range(min(len(base_vel), len(rl_vel))):
                    bv = base_vel[l] if base_vel[l] == base_vel[l] else 0.0
                    rv = rl_vel[l] if rl_vel[l] == rl_vel[l] else 0.0
                    if l < 5 and rv > bv:
                        inversion_layers.append(l)

                cv["velocity_inversion"] = {
                    "early_layers_rl_higher": inversion_layers,
                    "n_inversion_layers": len(inversion_layers),
                }

        # Overall expansion ratio comparison
        if existing_results:
            models = existing_results.get("models", {})
            base_er_mean = models.get(base_name, {}).get("overall_expansion_ratio_mean", 0)
            rl_er_mean = models.get(rl_name, {}).get("overall_expansion_ratio_mean", 0)
            cv["observed_expansion_ratio"] = {
                "base": base_er_mean,
                "rl": rl_er_mean,
                "ratio": rl_er_mean / base_er_mean if base_er_mean > 0 else 0.0,
            }

        return cv

    # =====================================================================
    # Main Run
    # =====================================================================

    def run(self) -> None:
        """Run the full experiment."""
        logger.info("=" * 70)
        logger.info("RL Spectral Anatomy Experiment (Experiment 3)")
        logger.info("=" * 70)

        # Verify volume
        if not os.path.isdir(MODELS_BASE):
            logger.error(f"Models volume not mounted: {MODELS_BASE}")
            logger.error("Mount the volume and retry.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Phase 1: CPU Velocity Decomposition ----
        logger.info("")
        logger.info("=" * 50)
        logger.info("Phase 1: CPU Velocity Decomposition")
        logger.info("=" * 50)

        existing_data = self._load_existing_data()
        existing_results = self._load_existing_results()

        decomps = self._decompose_velocities(existing_data)
        velocity_analysis = self._analyze_velocity_decomposition(decomps)

        logger.info("Phase 1 complete.")

        # ---- Phase 2: Model-Dependent Collection ----
        logger.info("")
        logger.info("=" * 50)
        logger.info("Phase 2: Model-Dependent Collection")
        logger.info("=" * 50)

        all_weight_profiles: dict[str, list[LayerWeightProfile]] = {}
        all_embedding_profiles: dict[str, EmbeddingProfile] = {}
        all_jacobian_traces: dict[str, list[dict]] = {}
        all_centroid_metrics: dict[str, list[CentroidLayerMetrics]] = {}
        all_centroids_raw: dict[str, list[list[list[float]]]] = {}

        for model_name in MODEL_REGISTRY:
            logger.info(f"\n--- {model_name} ---")

            self._load_model(model_name)

            # 2A: Weight profiles
            logger.info("  Phase 2A: Weight norm profiles")
            wp = self._collect_weight_profiles(model_name)
            all_weight_profiles[model_name] = wp
            logger.info(f"    Done: {len(wp)} layers")

            # 2B: Embedding profile
            logger.info("  Phase 2B: Embedding matrix analysis")
            ep = self._collect_embedding_profile(model_name)
            all_embedding_profiles[model_name] = ep
            logger.info(
                f"    Done: vocab={ep.vocab_size}, dim={ep.hidden_dim}, "
                f"ER_shannon={ep.effective_rank_shannon:.1f}"
            )

            # 2C: Jacobian traces
            if not self.skip_jacobian:
                logger.info("  Phase 2C: Jacobian spectrum traces")
                jt = self._collect_jacobian_traces(model_name)
                all_jacobian_traces[model_name] = jt
                logger.info(f"    Done: {len(jt)} traces")
            else:
                logger.info("  Phase 2C: SKIPPED (--skip-jacobian)")

            # 2D: Centroid collection + effective rank
            logger.info("  Phase 2D: Centroid collection + effective rank")
            cm, centroids = self._collect_centroid_metrics(model_name)
            all_centroid_metrics[model_name] = cm
            all_centroids_raw[model_name] = centroids
            logger.info(f"    Done: {len(cm)} layers, {len(centroids)} prompts")

            self._unload_model()
            logger.info(f"  Model {model_name} unloaded.")

        # ---- Phase 3: Formula Derivation ----
        logger.info("")
        logger.info("=" * 50)
        logger.info("Phase 3: Formula Derivation (CPU-only)")
        logger.info("=" * 50)

        # Serialize for Phase 3
        wp_serial = {
            name: [
                {
                    "layer_idx": p.layer_idx,
                    "q_spectral": p.q_spectral,
                    "k_spectral": p.k_spectral,
                    "v_spectral": p.v_spectral,
                    "o_spectral": p.o_spectral,
                    "gate_spectral": p.gate_spectral,
                    "up_spectral": p.up_spectral,
                    "down_spectral": p.down_spectral,
                    "q_frobenius": p.q_frobenius,
                    "k_frobenius": p.k_frobenius,
                    "v_frobenius": p.v_frobenius,
                    "o_frobenius": p.o_frobenius,
                    "gate_frobenius": p.gate_frobenius,
                    "up_frobenius": p.up_frobenius,
                    "down_frobenius": p.down_frobenius,
                    "predicted_jacobian_norm": p.predicted_jacobian_norm,
                }
                for p in profiles
            ]
            for name, profiles in all_weight_profiles.items()
        }

        ep_serial = {
            name: {
                "top_20_singular_values": p.top_20_singular_values,
                "effective_rank_renyi": p.effective_rank_renyi,
                "effective_rank_shannon": p.effective_rank_shannon,
                "var_top1": p.var_top1,
                "var_top5": p.var_top5,
                "vocab_size": p.vocab_size,
                "hidden_dim": p.hidden_dim,
            }
            for name, p in all_embedding_profiles.items()
        }

        cm_serial = {
            name: [
                {
                    "layer_idx": m.layer_idx,
                    "effective_rank_renyi": m.effective_rank_renyi,
                    "effective_rank_shannon": m.effective_rank_shannon,
                    "var_top1": m.var_top1,
                    "var_top5": m.var_top5,
                    "n_samples": m.n_samples,
                }
                for m in metrics
            ]
            for name, metrics in all_centroid_metrics.items()
        }

        decomps_serial = {
            name: [
                {
                    "prompt": d.prompt,
                    "category": d.category,
                    "radial_trajectory": d.radial_trajectory,
                    "angular_trajectory": d.angular_trajectory,
                    "cos_theta_trajectory": d.cos_theta_trajectory,
                }
                for d in dlist
            ]
            for name, dlist in decomps.items()
        }

        # Formula fits
        formula_fits: dict[str, dict] = {}

        logger.info("  Formula 1: Weight norms → Jacobian σ_max")
        f1 = self._fit_weight_to_jacobian(wp_serial, all_jacobian_traces)
        formula_fits["f1_weight_to_jacobian"] = {
            name: {
                "name": f.name, "formula": f.formula, "alpha": f.alpha,
                "beta": f.beta, "r_squared": f.r_squared, "n_points": f.n_points,
            }
            for name, f in f1.items()
        }

        logger.info("  Formula 2: Jacobian ER → Centroid ER")
        f2 = self._fit_jacobian_to_centroid_rank(all_jacobian_traces, cm_serial)
        formula_fits["f2_jacobian_to_centroid"] = {
            name: {
                "name": f.name, "formula": f.formula, "alpha": f.alpha,
                "beta": f.beta, "r_squared": f.r_squared, "n_points": f.n_points,
            }
            for name, f in f2.items()
        }

        logger.info("  Formula 3: Centroid ER → CKA")
        f3 = self._fit_rank_to_cka(cm_serial, existing_results)
        formula_fits["f3_rank_to_cka"] = {
            name: {
                "name": f.name, "formula": f.formula, "alpha": f.alpha,
                "beta": f.beta, "r_squared": f.r_squared, "n_points": f.n_points,
            }
            for name, f in f3.items()
        }

        logger.info("  Formula 4: Centroid ER ↔ Intrinsic Dimension")
        f4 = self._fit_rank_to_id(cm_serial, existing_data)
        formula_fits["f4_rank_to_id"] = {
            name: {
                "name": f.name, "formula": f.formula, "alpha": f.alpha,
                "beta": f.beta, "r_squared": f.r_squared, "n_points": f.n_points,
            }
            for name, f in f4.items()
        }

        logger.info("  Formula 5: Jacobian spectrum → Expansion ratio")
        f5 = self._fit_jacobian_to_expansion(all_jacobian_traces, existing_data)
        formula_fits["f5_jacobian_to_expansion"] = {
            name: {
                "name": f.name, "formula": f.formula, "alpha": f.alpha,
                "beta": f.beta, "r_squared": f.r_squared, "n_points": f.n_points,
            }
            for name, f in f5.items()
        }

        logger.info("  Cross-validation: RL vs Base predictions")
        cross_validation = self._cross_validate_predictions(
            formula_fits, wp_serial, cm_serial, existing_results, existing_data
        )

        # ---- Save & Report ----
        results = SpectralAnatomyResults(
            velocity_decompositions=decomps_serial,
            velocity_analysis=velocity_analysis,
            weight_profiles=wp_serial,
            embedding_profiles=ep_serial,
            jacobian_traces=all_jacobian_traces,
            centroid_metrics=cm_serial,
            formula_fits=formula_fits,
            cross_validation=cross_validation,
            experiment={
                "name": "rl_spectral_anatomy",
                "version": 1,
                "seed": self.seed,
                "smoke": self.smoke,
                "skip_jacobian": self.skip_jacobian,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_categories": len(PROBE_CATEGORIES),
                "prompts_per_category": 2 if self.smoke else 6,
                "jacobian_prompts_per_model": 2 if self.smoke else 5,
                "centroid_prompts_per_model": 10 if self.smoke else 30,
            },
        )

        self._save_results(results)
        self._generate_report(results)

        logger.info("")
        logger.info("=" * 50)
        logger.info("Experiment complete!")
        logger.info(f"Results: {self.output_dir / 'raw_results.json'}")
        logger.info(f"Report: {self.output_dir / 'SPECTRAL_ANATOMY_REPORT.md'}")
        logger.info("=" * 50)

    # =====================================================================
    # Save & Report
    # =====================================================================

    def _save_results(self, results: SpectralAnatomyResults) -> None:
        """Save all results as JSON."""
        def _sf(v: float) -> float | None:
            if isinstance(v, float) and v != v:
                return None
            return v

        serializable = {
            "experiment": results.experiment,
            "velocity_decompositions": results.velocity_decompositions,
            "velocity_analysis": results.velocity_analysis,
            "weight_profiles": results.weight_profiles,
            "embedding_profiles": results.embedding_profiles,
            "jacobian_traces": results.jacobian_traces,
            "centroid_metrics": results.centroid_metrics,
            "formula_fits": results.formula_fits,
            "cross_validation": results.cross_validation,
        }

        path = self.output_dir / "raw_results.json"
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    def _generate_report(self, results: SpectralAnatomyResults) -> None:
        """Generate SPECTRAL_ANATOMY_REPORT.md."""
        lines: list[str] = []
        w = lines.append

        w("# RL Spectral Anatomy Report (Experiment 3)")
        w("")
        w(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
        w(f"**Seed**: {self.seed}")
        w(f"**Mode**: {'smoke test' if self.smoke else 'full experiment'}")
        w(f"**Jacobian**: {'skipped' if self.skip_jacobian else 'collected'}")
        w("")

        # Section 1: Summary
        w("## 1. Formula Summary")
        w("")
        w("| Formula | Description | Best R² (Base) | Best R² (RL) |")
        w("|---------|-------------|---------------|-------------|")

        for fname, display in [
            ("f1_weight_to_jacobian", "Weight norms → Jacobian σ_max"),
            ("f2_jacobian_to_centroid", "Jacobian ER → Centroid ER"),
            ("f3_rank_to_cka", "Centroid ER → CKA"),
            ("f4_rank_to_id", "Centroid ER ↔ Intrinsic Dimension"),
            ("f5_jacobian_to_expansion", "Jacobian → Expansion ratio"),
        ]:
            fits = results.formula_fits.get(fname, {})
            base_r2 = "N/A"
            rl_r2 = "N/A"
            for model_name, fit in fits.items():
                r2_str = f"{fit['r_squared']:.4f}" if isinstance(fit, dict) else "N/A"
                if MODEL_REGISTRY.get(model_name, {}).get("training") == "base":
                    base_r2 = r2_str
                else:
                    rl_r2 = r2_str
            w(f"| {fname} | {display} | {base_r2} | {rl_r2} |")
        w("")

        # Section 2: Velocity Decomposition
        w("## 2. Velocity Decomposition")
        w("")

        va = results.velocity_analysis
        if va and "per_layer" in va:
            early = va.get("early_layer_summary", {})
            dom = early.get("dominant_component", "unknown")
            w(f"**L0-2 dominant component**: {dom}")
            w(f"- Base mean |radial| (L0-2): {early.get('base_mean_abs_radial', 0):.6f}")
            w(f"- RL mean |radial| (L0-2): {early.get('rl_mean_abs_radial', 0):.6f}")
            w(f"- Base mean angular (L0-2): {early.get('base_mean_angular', 0):.6f}")
            w(f"- RL mean angular (L0-2): {early.get('rl_mean_angular', 0):.6f}")
            w("")

            w("### Per-Layer Decomposition")
            w("")
            w("| Layer | Base Radial | RL Radial | Base Angular | RL Angular | d(Radial) | d(Angular) |")
            w("|-------|------------|-----------|-------------|------------|-----------|------------|")
            for entry in va["per_layer"]:
                l = entry["layer"]
                br = entry["base_mean_radial"]
                rr = entry["rl_mean_radial"]
                ba = entry["base_mean_angular"]
                ra = entry["rl_mean_angular"]
                dr = entry["cohens_d_radial"]
                da = entry["cohens_d_angular"]
                w(f"| {l} | {br:.6f} | {rr:.6f} | {ba:.6f} | {ra:.6f} | {dr:+.3f} | {da:+.3f} |")
            w("")
        else:
            w("No velocity decomposition data available.")
            w("")

        # Section 3: Weight Norm Profiles
        w("## 3. Weight Norm Profiles")
        w("")

        for model_name, profiles in results.weight_profiles.items():
            w(f"### {model_name}")
            w("")
            w("| Layer | σ(V) | σ(O) | σ(gate) | σ(up) | σ(down) | Predicted J |")
            w("|-------|------|------|---------|-------|---------|------------|")
            for p in profiles:
                w(
                    f"| {p['layer_idx']} | {p['v_spectral']:.3f} | {p['o_spectral']:.3f} | "
                    f"{p['gate_spectral']:.3f} | {p['up_spectral']:.3f} | {p['down_spectral']:.3f} | "
                    f"{p['predicted_jacobian_norm']:.3f} |"
                )
            w("")

        # Section 4: Embedding Profiles
        w("## 4. Embedding Matrix Analysis")
        w("")
        w("| Model | Vocab | Dim | ER (Shannon) | ER (Renyi) | var_top1 | var_top5 |")
        w("|-------|-------|-----|-------------|-----------|---------|---------|")
        for model_name, ep in results.embedding_profiles.items():
            w(
                f"| {model_name} | {ep['vocab_size']} | {ep['hidden_dim']} | "
                f"{ep['effective_rank_shannon']:.1f} | {ep['effective_rank_renyi']:.1f} | "
                f"{ep['var_top1']:.4f} | {ep['var_top5']:.4f} |"
            )
        w("")

        # Section 5: Jacobian Spectrum
        w("## 5. Jacobian Spectrum")
        w("")

        if results.jacobian_traces:
            for model_name, traces in results.jacobian_traces.items():
                w(f"### {model_name}")
                w("")
                for trace in traces:
                    prompt = trace.get("prompt", "")[:60]
                    w(f"**Prompt**: {prompt}...")
                    w("")
                    w("| Layer | σ_max | ER (Shannon) | ER (Renyi) | κ | Spectral Gap |")
                    w("|-------|-------|-------------|-----------|---|-------------|")
                    for p in trace.get("profiles", []):
                        w(
                            f"| {p['layer_idx']} | {p['norm_amplification']:.4f} | "
                            f"{p['effective_rank_shannon']:.2f} | {p['effective_rank_renyi']:.2f} | "
                            f"{p['condition_number']:.2f} | {p['spectral_gap']:.2f} |"
                        )
                    w("")
        else:
            w("Jacobian traces skipped.")
            w("")

        # Section 6: Centroid Effective Rank
        w("## 6. Centroid Effective Rank by Layer")
        w("")

        if results.centroid_metrics:
            # Combined table
            model_names = list(results.centroid_metrics.keys())
            if len(model_names) == 2:
                m1, m2 = model_names
                cm1 = results.centroid_metrics[m1]
                cm2 = results.centroid_metrics[m2]
                num_layers = min(len(cm1), len(cm2))

                w(f"| Layer | {m1} ER | {m2} ER | {m1} var_top1 | {m2} var_top1 |")
                w("|-------|---------|---------|-------------|-------------|")
                for l in range(num_layers):
                    w(
                        f"| {l} | {cm1[l]['effective_rank_shannon']:.2f} | "
                        f"{cm2[l]['effective_rank_shannon']:.2f} | "
                        f"{cm1[l]['var_top1']:.4f} | {cm2[l]['var_top1']:.4f} |"
                    )
                w("")
            else:
                for model_name, cm in results.centroid_metrics.items():
                    w(f"### {model_name}")
                    w("")
                    w("| Layer | ER (Shannon) | ER (Renyi) | var_top1 |")
                    w("|-------|-------------|-----------|---------|")
                    for m in cm:
                        w(
                            f"| {m['layer_idx']} | {m['effective_rank_shannon']:.2f} | "
                            f"{m['effective_rank_renyi']:.2f} | {m['var_top1']:.4f} |"
                        )
                    w("")

        # Section 7: Formula Chain
        w("## 7. Formula Chain: Predicted vs Measured")
        w("")

        for fname, display in [
            ("f1_weight_to_jacobian", "Formula 1: Weight Norms → Jacobian σ_max"),
            ("f2_jacobian_to_centroid", "Formula 2: Jacobian ER → Centroid ER"),
            ("f3_rank_to_cka", "Formula 3: Centroid ER → CKA"),
            ("f4_rank_to_id", "Formula 4: Centroid ER ↔ Intrinsic Dimension"),
            ("f5_jacobian_to_expansion", "Formula 5: Jacobian → Expansion Ratio"),
        ]:
            w(f"### {display}")
            w("")
            fits = results.formula_fits.get(fname, {})
            if not fits:
                w("No data available.")
                w("")
                continue

            for model_name, fit in fits.items():
                formula = fit.get("formula", "")
                alpha = fit.get("alpha", 0)
                beta = fit.get("beta", 0)
                r2 = fit.get("r_squared", 0)
                n = fit.get("n_points", 0)
                w(f"**{model_name}**: {formula}")
                w(f"- α = {alpha:.6f}, β = {beta:.6f}")
                w(f"- R² = {r2:.4f} (n={n})")
                w("")

        # Section 8: Cross-Prediction
        w("## 8. Cross-Prediction: Do Weight Differences Predict Activation Differences?")
        w("")

        cv = results.cross_validation
        if cv:
            if "weight_norm_comparison" in cv:
                wnc = cv["weight_norm_comparison"]
                w("### Weight Norm Comparison")
                w(f"- Base mean predicted Jacobian norm: {wnc.get('base_mean_predicted_j', 0):.4f}")
                w(f"- RL mean predicted Jacobian norm: {wnc.get('rl_mean_predicted_j', 0):.4f}")
                w(f"- RL/Base ratio: {wnc.get('ratio', 0):.4f}")
                w("")

            if "centroid_er_l1" in cv:
                cel = cv["centroid_er_l1"]
                w("### L1 Centroid Effective Rank")
                w(f"- Base ER at L1: {cel.get('base_er', 0):.4f}")
                w(f"- RL ER at L1: {cel.get('rl_er', 0):.4f}")
                w(f"- RL/Base ratio: {cel.get('ratio', 0):.4f}")
                w("")

            if "velocity_inversion" in cv:
                vi = cv["velocity_inversion"]
                w("### Velocity Inversion")
                w(f"- Early layers where RL velocity > Base: {vi.get('early_layers_rl_higher', [])}")
                w("")

            if "observed_expansion_ratio" in cv:
                oer = cv["observed_expansion_ratio"]
                w("### Observed Expansion Ratio (from Experiment 2)")
                w(f"- Base: {oer.get('base', 0):.4f}")
                w(f"- RL: {oer.get('rl', 0):.4f}")
                w(f"- RL/Base: {oer.get('ratio', 0):.4f}")
                w("")
        else:
            w("No cross-validation data available.")
            w("")

        # Section 9: Open Questions
        w("## 9. Open Questions")
        w("")
        w("1. **Generality**: All formulas derived from a single base→RL pair (Qwen3-8B → DeepSeek-R1-8B).")
        w("   Need additional pairs to validate universality vs architecture-dependence.")
        w("")
        w("2. **Causal direction**: Formulas show correlation, not causation.")
        w("   Weight changes during GRPO → Jacobian changes → activation changes is the hypothesized chain,")
        w("   but the exact gradient flow mechanism is not proven here.")
        w("")
        w("3. **LayerNorm effects**: The L5→L6 norm jump (~40x) suggests LayerNorm creates")
        w("   a structural boundary. Velocity decomposition helps separate this from computation,")
        w("   but a full treatment of LayerNorm's effect on the Jacobian is needed.")
        w("")
        w("4. **Centroid ER ceiling**: With 30 centroids, SVD-based ER is capped at 30.")
        w("   This is a measurement artifact, not a model property. More probes would raise the ceiling.")
        w("")
        w("5. **Next model pairs**: Download additional MLX base→RL pairs from HuggingFace")
        w("   to test whether formulas generalize.")
        w("")

        # Write report
        report_path = self.output_dir / "SPECTRAL_ANATOMY_REPORT.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Report saved to {report_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RL Spectral Anatomy Experiment (Experiment 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rl_spectral_anatomy",
        help="Output directory (default: results/rl_spectral_anatomy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 2 Jacobian prompts + 10 centroid prompts per model",
    )
    parser.add_argument(
        "--existing-data",
        type=str,
        default="results/rl_processing",
        help="Path to experiment 2 results (default: results/rl_processing)",
    )
    parser.add_argument(
        "--skip-jacobian",
        action="store_true",
        help="Skip Jacobian collection (Phases 2C, Formulas 1/2/5)",
    )

    args = parser.parse_args()

    experiment = RLSpectralAnatomyExperiment(
        output_dir=Path(args.output),
        seed=args.seed,
        smoke=args.smoke,
        existing_data=args.existing_data,
        skip_jacobian=args.skip_jacobian,
    )
    experiment.run()


if __name__ == "__main__":
    main()
