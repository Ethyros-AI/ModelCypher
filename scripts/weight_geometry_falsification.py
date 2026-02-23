#!/usr/bin/env python3
"""Weight space geometry falsification experiments.

Pre-registered falsification tests for the hypotheses:

    H_null: NB-LoRA weight space is effectively Euclidean under training.
            The pullback metric P = M M^T ≈ I during training. Optimization
            benefit comes from the Stiefel constraint (Cayley transform),
            not from the metric curvature.

    H_alt:  The pullback metric P = M M^T captures non-trivial curvature
            that materially affects optimization beyond the Stiefel constraint.

Six falsification tests with automated verdicts:

F1: PULLBACK METRIC DEVIATION FROM IDENTITY
    Measurement: ||P_hat - I||_F / sqrt(r) at each training step
    H_null supported if: median < 0.1

F2: STIEFEL VS CURVATURE ISOLATION (Critical)
    Two conditions: Cayley+P (current) vs Cayley+I (no preconditioner)
    H_null supported if: Cohen's d between final losses < 0.3

F3: DIRECTION COSINE — NATURAL VS EUCLIDEAN GRADIENT
    Measurement: cos(P@g, g) at each step
    H_null supported if: median > 0.95

F4: METRIC DRIFT ALONG TRAINING TRAJECTORY
    Measurement: ||P(step_t) - P(step_0)|| / ||P(step_0)|| over training
    H_null supported if: max drift < 0.1

F5: FISHER EIGENSPECTRUM ON NB-LoRA PARAMETERS
    Measurement: diagonal FIM eigenvalue distribution
    From theory (Karakida 2021): pathologically degenerate

F6: GEODESIC VS RETRACTION DEVIATION ON STIEFEL
    Measurement: ||Cayley(Z) - expm(-Z_skew)||_F / ||Z_skew||_F
    From theory (Lezcano-Casado 2019): O(||Z||^2) agreement

Usage:
    # Smoke test
    poetry run python scripts/weight_geometry_falsification.py \\
        --model LFM2-350M --steps 10 --output results/weight_geometry/smoke/

    # Full run
    poetry run python scripts/weight_geometry_falsification.py \\
        --model LFM2-350M --steps 200 --output results/weight_geometry/full/

References:
    Amari (1998): Natural gradient, Neural Computation 10(2):251-276
    Watanabe (2009): Algebraic Geometry and Statistical Learning Theory
    Karakida et al. (2021): Pathological FIM spectra, Neural Computation 33(8)
    Martens (2020): Natural gradient insights, JMLR 21(146):1-76
    Edelman et al. (1998): Stiefel manifold geometry, SIAM J. Matrix Anal.
    Lezcano-Casado (2019): Trivializations for Stiefel, NeurIPS
    Absil et al. (2008): Optimization on Matrix Manifolds
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
    },
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
    },
}

DEFAULT_DATASET = "data/training/benchmark_train.jsonl"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExperimentConfig:
    model: str = "LFM2-350M"
    dataset_path: str = DEFAULT_DATASET
    n_steps: int = 200
    batch_size: int = 2
    seq_length: int = 256
    seed: int = 42
    lr: float = 0.01  # Fixed LR for experiments (not MASS — we want identical conditions)
    output_dir: Path = field(
        default_factory=lambda: Path("results/weight_geometry"),
    )


@dataclass
class FalsificationVerdict:
    test_name: str
    prediction: str
    result: str  # "SUPPORTS H_null", "SUPPORTS H_alt", "INCONCLUSIVE"
    details: dict[str, Any]


@dataclass
class PreconditionerSnapshot:
    """Per-layer preconditioner data at a single training step."""

    layer: str
    P: Any  # [r, r] matrix
    Z: Any  # [r, r] skew + PSD
    rank: int


# =============================================================================
# Main Experiment Class
# =============================================================================


class WeightGeometryFalsification:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.backend = None
        self.adapter = None
        self.model = None
        self.tokenizer = None

    # ================================================================
    # Model and Data Loading
    # ================================================================

    def _load_model_and_inject(self):
        """Load model, compute geometry, inject NB-LoRA, freeze base."""
        import mlx.core as mx

        from modelcypher.backends import initialize_default_backend
        from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
        from modelcypher.core.domain.training.geometric_lora import (
            analyze_weight_geometries,
        )

        if self.backend is None:
            self.backend = initialize_default_backend()
        if self.adapter is None:
            self.adapter = MLXTrainingAdapter(self.backend)

        model_path = MODEL_REGISTRY[self.config.model]["path"]
        logger.info("Loading model from %s", model_path)
        self.model, self.tokenizer = self.backend.load_model(model_path)

        # Compute spectral geometry for NB-LoRA injection
        logger.info("Computing weight geometries (SVD)...")
        weights = self.adapter.extract_weight_matrices(self.model)
        geometries = analyze_weight_geometries(weights, self.backend)

        target_modules = [k for k, g in geometries.items() if g.is_targetable]
        logger.info("Found %d targetable layers", len(target_modules))

        # Inject NB-LoRA
        n_injected = self.adapter.inject_nb_lora(
            self.model, geometries, target_modules,
        )
        logger.info("Injected NB-LoRA into %d layers", n_injected)

        # Freeze base, unfreeze only LoRA params
        self.adapter.freeze_and_apply_lora(self.model)

        # Verify trainable parameters
        from mlx.utils import tree_flatten

        n_trainable = sum(
            v.size for _, v in tree_flatten(self.model.trainable_parameters())
        )
        logger.info("Trainable parameters: %d", n_trainable)

        return geometries

    def _load_dataset(self) -> list:
        """Load JSONL dataset and tokenize into iterate_batches format."""
        dataset_path = Path(self.config.dataset_path)
        raw_samples = []
        with open(dataset_path) as f:
            for line in f:
                raw_samples.append(json.loads(line))

        dataset = self.adapter.prepare_dataset(raw_samples, self.tokenizer)
        logger.info("Loaded %d samples from %s", len(dataset), dataset_path)
        return dataset

    def _unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        gc.collect()
        if self.backend is not None:
            try:
                self.backend.clear_cache()
            except Exception:
                pass

    # ================================================================
    # Preconditioner Extraction
    # ================================================================

    def _extract_preconditioner_snapshots(self) -> list[PreconditionerSnapshot]:
        """Extract P, Z from all NBLoRALinear modules.

        Uses _iter_nb_lora_modules (same as production preconditioner code)
        to access NB-LoRA parameters directly, avoiding key-format mismatches.

        Each snapshot stores the module name (with "model." prefix stripped)
        to match gradient key format.
        """
        import mlx.core as mx

        snapshots = []
        for name, nb_lora in self.adapter._iter_nb_lora_modules(self.model):
            # Strip "model." prefix to match gradient key namespace
            prefix = name.replace(".weight", "").replace("model.", "")

            r = nb_lora._rank

            # Replicate the Cayley Z computation (same math as _cayley_transform)
            stacked = mx.concatenate([nb_lora.A_tilde.T, nb_lora.B_tilde.T], axis=0)
            X = stacked[:r, :]
            Y = stacked[r:, :]
            Z = (X - X.T) + Y.T @ Y
            M = mx.eye(r) + Z
            P = M @ M.T
            mx.eval(P, Z)

            snapshots.append(PreconditionerSnapshot(
                layer=prefix, P=P, Z=Z, rank=r,
            ))

        return snapshots

    def _normalize_P(self, P, r: int):
        """Normalize P by spectral radius via power iteration. Returns (P_hat, lambda_max)."""
        import mlx.core as mx

        v = mx.ones((r, 1)) / math.sqrt(r)
        mx.eval(v)
        lam = 1.0
        for _ in range(30):
            u = P @ v
            mx.eval(u)
            norm_u = float(mx.sqrt(mx.sum(u * u)))
            if norm_u <= 0:
                break
            lam_new = float(mx.sum(v * u))
            v = u * (1.0 / norm_u)
            mx.eval(v)
            if abs(lam_new - lam) < 1e-10 * max(1.0, abs(lam)):
                lam = lam_new
                break
            lam = lam_new

        lam_max = max(lam, 1.0)
        P_hat = P * (1.0 / lam_max)
        mx.eval(P_hat)
        return P_hat, lam_max

    # ================================================================
    # Instrumented Training Loop
    # ================================================================

    def _run_instrumented_training(
        self,
        dataset: list,
        n_steps: int,
        use_preconditioner: bool = True,
    ) -> tuple[list[float], list[list[float]], list[list[float]], list[list[float]]]:
        """Run training loop collecting F1/F3/F4 measurements.

        Returns:
            (loss_history, f1_history, f3_history, f4_history)
            Each *_history is a list of per-step lists of per-layer measurements.
        """
        import mlx.core as mx
        import mlx.nn as nn_mlx
        from mlx import optimizers as optim
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_fn = nn_mlx.value_and_grad(self.model, default_loss)
        optimizer = optim.SGD(learning_rate=self.config.lr, momentum=0.0)

        loss_history: list[float] = []
        f1_history: list[list[float]] = []
        f3_history: list[list[float]] = []
        f4_history: list[list[float]] = []

        # Store initial P for F4 drift measurement
        initial_snapshots = self._extract_preconditioner_snapshots()
        initial_P_by_layer = {
            s.layer: (s.P, s.rank) for s in initial_snapshots
        }

        step = 0
        for batch, lengths in iterate_batches(
            dataset, self.config.batch_size, self.config.seq_length, loop=True,
        ):
            if step >= n_steps:
                break

            # Forward + backward
            (loss_val, n_tokens), grad = loss_fn(self.model, batch, lengths)
            mx.eval(loss_val)
            loss_float = float(loss_val)
            loss_history.append(loss_float)

            # Extract current P
            snapshots = self._extract_preconditioner_snapshots()

            # ── F1: ||P_hat - I||_F / sqrt(r) ──
            f1_step = []
            for snap in snapshots:
                P_hat, _ = self._normalize_P(snap.P, snap.rank)
                diff = P_hat - mx.eye(snap.rank)
                mx.eval(diff)
                dev = float(mx.sqrt(mx.sum(diff * diff))) / math.sqrt(snap.rank)
                f1_step.append(dev)
            f1_history.append(f1_step)

            # ── F4: ||P(t) - P(0)|| / ||P(0)|| ──
            f4_step = []
            for snap in snapshots:
                if snap.layer in initial_P_by_layer:
                    P0, _ = initial_P_by_layer[snap.layer]
                    drift = snap.P - P0
                    mx.eval(drift)
                    drift_norm = float(mx.sqrt(mx.sum(drift * drift)))
                    p0_norm = float(mx.sqrt(mx.sum(P0 * P0)))
                    f4_step.append(drift_norm / max(p0_norm, 1e-30))
            f4_history.append(f4_step)

            # ── Apply preconditioner (or not) and measure F3 ──
            grad_flat = dict(tree_flatten(grad))

            if use_preconditioner:
                f3_step = []
                for snap in snapshots:
                    P_hat, _ = self._normalize_P(snap.P, snap.rank)

                    # Measure F3 cosine and apply preconditioner for BOTH A_tilde and B_tilde.
                    # At initialization, A_tilde has zero gradient (B=0 makes d(loss)/d(A)=0).
                    # B_tilde gets gradient from step 0. Both get gradient from step 1+.
                    for suffix in (".A_tilde", ".B_tilde"):
                        for gk in grad_flat:
                            if gk.endswith(suffix) and snap.layer in gk:
                                g_raw = grad_flat[gk]
                                Pg = P_hat @ g_raw

                                # cos(Pg, g)
                                g_flat = mx.reshape(g_raw, (-1,))
                                Pg_flat = mx.reshape(Pg, (-1,))
                                dot = float(mx.sum(g_flat * Pg_flat))
                                g_norm = float(mx.sqrt(mx.sum(g_flat * g_flat)))
                                Pg_norm = float(mx.sqrt(mx.sum(Pg_flat * Pg_flat)))
                                denom = g_norm * Pg_norm
                                if denom > 0:
                                    f3_step.append(dot / denom)

                                # Replace gradient with preconditioned version
                                grad_flat[gk] = Pg
                                mx.eval(grad_flat[gk])

                f3_history.append(f3_step)
                grad = tree_unflatten(list(grad_flat.items()))
            else:
                f3_history.append([])

            # SGD step
            optimizer.apply_gradients(grad, self.model)
            mx.eval(*[v for _, v in tree_flatten(self.model.parameters())])

            # Clamp S_raw
            for _, nb_lora in self.adapter._iter_nb_lora_modules(self.model):
                nb_lora.clamp_scale()
                mx.eval(nb_lora.S_raw)

            if step % max(1, n_steps // 20) == 0:
                f1_med = _median(f1_step) if f1_step else 0.0
                f4_med = _median(f4_step) if f4_step else 0.0
                log_msg = f"  Step {step}/{n_steps}: loss={loss_float:.4f}, F1={f1_med:.2e}, F4={f4_med:.2e}"
                if use_preconditioner and f3_history[-1]:
                    log_msg += f", F3_cos={_median(f3_history[-1]):.6f}"
                logger.info(log_msg)

            step += 1

        return loss_history, f1_history, f3_history, f4_history

    # ================================================================
    # F5: Fisher Eigenspectrum
    # ================================================================

    def _test_f5_fisher_eigenspectrum(self, dataset: list, n_batches: int = 10):
        """Diagonal FIM for NB-LoRA parameters. Tests Karakida (2021) prediction."""
        import mlx.core as mx
        import mlx.nn as nn_mlx
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_fn = nn_mlx.value_and_grad(self.model, default_loss)

        accum: dict[str, Any] | None = None
        count = 0

        for batch, lengths in iterate_batches(
            dataset, self.config.batch_size, self.config.seq_length, loop=False,
        ):
            if count >= n_batches:
                break

            (loss_val, _), grad = loss_fn(self.model, batch, lengths)
            mx.eval(loss_val)

            grad_flat = dict(tree_flatten(grad))

            if accum is None:
                accum = {}
                for k, v in grad_flat.items():
                    if "A_tilde" in k or "B_tilde" in k:
                        accum[k] = v * v
                        mx.eval(accum[k])
            else:
                for k in accum:
                    if k in grad_flat:
                        accum[k] = accum[k] + grad_flat[k] * grad_flat[k]
                        mx.eval(accum[k])
            count += 1

        if accum is None or count == 0:
            return FalsificationVerdict(
                test_name="F5: Fisher Eigenspectrum",
                prediction="Pathologically degenerate (Karakida 2021)",
                result="INCONCLUSIVE",
                details={"error": "No batches processed"},
            )

        # Diagonal FIM ≈ E[grad^2]
        all_vals: list[float] = []
        for k in accum:
            fim_diag = accum[k] * (1.0 / count)
            mx.eval(fim_diag)
            flat = mx.reshape(fim_diag, (-1,))
            mx.eval(flat)
            all_vals.extend(float(v) for v in flat)

        if not all_vals:
            return FalsificationVerdict(
                test_name="F5: Fisher Eigenspectrum",
                prediction="Pathologically degenerate (Karakida 2021)",
                result="INCONCLUSIVE",
                details={"error": "No eigenvalues"},
            )

        all_vals.sort(reverse=True)
        max_eig = all_vals[0]
        if max_eig <= 0:
            return FalsificationVerdict(
                test_name="F5: Fisher Eigenspectrum",
                prediction="Pathologically degenerate (Karakida 2021)",
                result="INCONCLUSIVE",
                details={"error": "All eigenvalues zero"},
            )

        # Fraction < 1% of max
        n_below_1pct = sum(1 for v in all_vals if v < max_eig * 0.01)
        frac_below_1pct = n_below_1pct / len(all_vals)

        # Condition number (p10 as effective min to avoid exact zeros)
        idx_p90 = max(0, int(len(all_vals) * 0.9))
        min_eig_p10 = max(all_vals[idx_p90], 1e-30)
        cond_number = max_eig / min_eig_p10

        is_degenerate = cond_number > 1e4 and frac_below_1pct > 0.9

        return FalsificationVerdict(
            test_name="F5: Fisher Eigenspectrum",
            prediction="Pathologically degenerate (Karakida 2021): cond > 1e4, >90% below 1% of max",
            result="CONFIRMS prediction" if is_degenerate else "CONTRADICTS prediction",
            details={
                "n_values": len(all_vals),
                "max": max_eig,
                "condition_number_p10": cond_number,
                "frac_below_1pct_of_max": frac_below_1pct,
                "is_degenerate": is_degenerate,
                "percentiles": {
                    "p1": all_vals[max(0, int(len(all_vals) * 0.99))],
                    "p10": all_vals[max(0, int(len(all_vals) * 0.9))],
                    "p50": all_vals[len(all_vals) // 2],
                    "p90": all_vals[max(0, int(len(all_vals) * 0.1))],
                },
            },
        )

    # ================================================================
    # F6: Geodesic vs Retraction Deviation
    # ================================================================

    def _test_f6_geodesic_deviation(self):
        """Compare Cayley retraction to Stiefel geodesic (matrix exponential)."""
        import numpy as np
        import scipy.linalg

        snapshots = self._extract_preconditioner_snapshots()

        deviations = []
        for snap in snapshots:
            Z_np = np.array(snap.Z.tolist(), dtype=np.float64)
            r = snap.rank

            # Decompose Z into skew-symmetric + symmetric parts
            Z_skew = (Z_np - Z_np.T) / 2.0
            Z_sym = (Z_np + Z_np.T) / 2.0

            z_skew_norm = np.linalg.norm(Z_skew, "fro")
            if z_skew_norm < 1e-15:
                # Z is purely symmetric (Y^T Y dominates) — both maps ≈ I
                deviations.append({
                    "layer": snap.layer,
                    "z_skew_norm": z_skew_norm,
                    "z_sym_norm": np.linalg.norm(Z_sym, "fro"),
                    "deviation_abs": 0.0,
                    "deviation_rel": 0.0,
                })
                continue

            # Cayley of skew part: (I - Z_skew)(I + Z_skew)^{-1}
            I_r = np.eye(r)
            cayley = np.linalg.solve(I_r + Z_skew, I_r - Z_skew)

            # Geodesic on SO(r): expm(-Z_skew) for skew-symmetric Z_skew
            geodesic = scipy.linalg.expm(-Z_skew)

            diff = np.linalg.norm(cayley - geodesic, "fro")
            rel_dev = diff / z_skew_norm

            deviations.append({
                "layer": snap.layer,
                "z_skew_norm": float(z_skew_norm),
                "z_sym_norm": float(np.linalg.norm(Z_sym, "fro")),
                "deviation_abs": float(diff),
                "deviation_rel": float(rel_dev),
            })

        if not deviations:
            return FalsificationVerdict(
                test_name="F6: Geodesic vs Retraction",
                prediction="O(||Z||^2) deviation (Lezcano-Casado 2019)",
                result="INCONCLUSIVE",
                details={"error": "No layers found"},
            )

        rel_devs = [d["deviation_rel"] for d in deviations]
        med_dev = _median(rel_devs)
        max_dev = max(rel_devs)

        return FalsificationVerdict(
            test_name="F6: Geodesic vs Retraction",
            prediction="O(||Z||^2) deviation (Lezcano-Casado 2019)",
            result=f"Median relative deviation: {med_dev:.6f}, max: {max_dev:.6f}",
            details={
                "n_layers": len(deviations),
                "median_relative_deviation": med_dev,
                "max_relative_deviation": max_dev,
                "per_layer_sample": deviations[:5],
            },
        )

    # ================================================================
    # Report Generation
    # ================================================================

    def _generate_report_md(
        self,
        verdicts: list[FalsificationVerdict],
        loss_a: list[float],
        loss_b: list[float],
        f1_data: list[list[float]],
        f3_data: list[list[float]],
        f4_data: list[list[float]],
    ) -> str:
        lines = [
            "# Weight Space Geometry Falsification Report",
            "",
            f"**Model:** {self.config.model}",
            f"**Steps:** {self.config.n_steps}",
            f"**LR:** {self.config.lr}",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Hypotheses",
            "",
            "**H_null:** Weight space is effectively Euclidean for NB-LoRA training.",
            "The pullback metric P = M M^T is approximately identity. Benefit comes",
            "from the Stiefel constraint (Cayley transform), not from metric curvature.",
            "",
            "**H_alt:** P captures non-trivial curvature that materially affects",
            "optimization beyond the Stiefel constraint alone.",
            "",
            "## Test Results",
            "",
        ]

        for v in verdicts:
            lines.extend([
                f"### {v.test_name}",
                "",
                f"**Prediction:** {v.prediction}",
                f"**Result:** {v.result}",
                "",
                "```json",
                json.dumps(v.details, indent=2, default=_json_default),
                "```",
                "",
            ])

        # F1 distribution
        all_f1 = [v for step_data in f1_data for v in step_data]
        if all_f1:
            all_f1.sort()
            lines.extend([
                "## F1 Distribution: ||P_hat - I||_F / sqrt(r)",
                "",
                f"- N measurements: {len(all_f1)}",
                f"- Median: {_median(all_f1):.6f}",
                f"- p5: {_percentile(all_f1, 0.05):.6f}",
                f"- p95: {_percentile(all_f1, 0.95):.6f}",
                f"- Max: {all_f1[-1]:.6f}",
                "",
            ])

        # F3 distribution
        all_f3 = [v for step_data in f3_data for v in step_data]
        if all_f3:
            all_f3.sort()
            lines.extend([
                "## F3 Distribution: cos(Pg, g)",
                "",
                f"- N measurements: {len(all_f3)}",
                f"- Median: {_median(all_f3):.4f}",
                f"- p5: {_percentile(all_f3, 0.05):.4f}",
                f"- p95: {_percentile(all_f3, 0.95):.4f}",
                f"- Min: {all_f3[0]:.4f}",
                "",
            ])

        # F4 drift evolution
        if f4_data:
            last_f4 = f4_data[-1] if f4_data[-1] else []
            if last_f4:
                last_f4.sort()
                lines.extend([
                    "## F4 Final Metric Drift: ||P(T) - P(0)|| / ||P(0)||",
                    "",
                    f"- Median: {_median(last_f4):.6f}",
                    f"- Max: {max(last_f4):.6f}",
                    "",
                ])

        # Loss curves
        if loss_a and loss_b:
            lines.extend([
                "## F2 Loss Curves",
                "",
                f"- Cayley+P final loss: {loss_a[-1]:.4f}",
                f"- Cayley+I final loss: {loss_b[-1]:.4f}",
                "",
            ])

        lines.extend([
            "## References",
            "",
            "- Amari (1998): Natural gradient, Neural Computation 10(2):251-276",
            "- Watanabe (2009): Algebraic Geometry and Statistical Learning Theory",
            "- Karakida et al. (2021): Pathological FIM spectra, Neural Computation 33(8)",
            "- Martens (2020): Natural gradient insights, JMLR 21(146):1-76",
            "- Edelman et al. (1998): Stiefel manifold geometry, SIAM J. Matrix Anal.",
            "- Lezcano-Casado (2019): Trivializations for Stiefel, NeurIPS",
            "- Absil et al. (2008): Optimization on Matrix Manifolds",
            "",
        ])

        return "\n".join(lines)

    # ================================================================
    # Main Orchestration
    # ================================================================

    def run(self) -> None:
        """Run all falsification tests."""
        output_dir = self.config.output_dir / self.config.model
        output_dir.mkdir(parents=True, exist_ok=True)

        verdicts: list[FalsificationVerdict] = []

        # ── Load model and data ──
        self._load_model_and_inject()
        dataset = self._load_dataset()

        # ── Phase 1: Instrumented training with preconditioner (F1, F3, F4) ──
        logger.info("=" * 60)
        logger.info("Phase 1: Instrumented training — Cayley + P")
        logger.info("=" * 60)

        loss_a, f1_data, f3_data, f4_data = self._run_instrumented_training(
            dataset, self.config.n_steps, use_preconditioner=True,
        )

        # F1 verdict
        all_f1 = [v for step_data in f1_data for v in step_data]
        if all_f1:
            f1_median = _median(all_f1)
            f1_p95 = _percentile(all_f1, 0.95)
            verdicts.append(FalsificationVerdict(
                test_name="F1: Pullback Metric Deviation from Identity",
                prediction="H_null: median ||P_hat - I||_F / sqrt(r) < 0.1",
                result="SUPPORTS H_null" if f1_median < 0.1 else "SUPPORTS H_alt",
                details={
                    "median": f1_median,
                    "p5": _percentile(all_f1, 0.05),
                    "p95": f1_p95,
                    "max": max(all_f1),
                    "n_measurements": len(all_f1),
                },
            ))
            logger.info("F1 median: %.6f (%s)", f1_median,
                        "H_null" if f1_median < 0.1 else "H_alt")

        # F3 verdict
        all_f3 = [v for step_data in f3_data for v in step_data]
        if all_f3:
            f3_median = _median(all_f3)
            verdicts.append(FalsificationVerdict(
                test_name="F3: Direction Cosine (cos(Pg, g))",
                prediction="H_null: median cos > 0.95",
                result="SUPPORTS H_null" if f3_median > 0.95 else "SUPPORTS H_alt",
                details={
                    "median": f3_median,
                    "p5": _percentile(all_f3, 0.05),
                    "p95": _percentile(all_f3, 0.95),
                    "min": min(all_f3),
                    "n_measurements": len(all_f3),
                },
            ))
            logger.info("F3 median cos: %.4f (%s)", f3_median,
                        "H_null" if f3_median > 0.95 else "H_alt")

        # F4 verdict
        if f4_data and f4_data[-1]:
            final_drifts = f4_data[-1]
            f4_max = max(final_drifts)
            f4_median = _median(final_drifts)
            verdicts.append(FalsificationVerdict(
                test_name="F4: Metric Drift Along Training Trajectory",
                prediction="H_null: max ||P(T)-P(0)||/||P(0)|| < 0.1",
                result="SUPPORTS H_null" if f4_max < 0.1 else "SUPPORTS H_alt",
                details={
                    "median_final_drift": f4_median,
                    "max_final_drift": f4_max,
                    "n_layers": len(final_drifts),
                },
            ))
            logger.info("F4 max drift: %.6f (%s)", f4_max,
                        "H_null" if f4_max < 0.1 else "H_alt")

        # ── Phase 2: F2 Ablation — reload model for condition B ──
        logger.info("=" * 60)
        logger.info("Phase 2: Ablation — Cayley + I (no preconditioner)")
        logger.info("=" * 60)

        self._unload_model()
        self._load_model_and_inject()

        loss_b, _, _, _ = self._run_instrumented_training(
            dataset, self.config.n_steps, use_preconditioner=False,
        )

        # F2 verdict: compare loss_a vs loss_b
        if loss_a and loss_b:
            n_compare = max(1, len(loss_a) // 10)
            tail_a = loss_a[-n_compare:]
            tail_b = loss_b[-n_compare:]
            mean_a = sum(tail_a) / len(tail_a)
            mean_b = sum(tail_b) / len(tail_b)
            var_a = sum((x - mean_a) ** 2 for x in tail_a) / max(len(tail_a) - 1, 1)
            var_b = sum((x - mean_b) ** 2 for x in tail_b) / max(len(tail_b) - 1, 1)
            pooled_std = math.sqrt((var_a + var_b) / 2)
            d = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

            verdicts.append(FalsificationVerdict(
                test_name="F2: Stiefel vs Curvature Isolation",
                prediction="H_null: Cohen's d < 0.3 (no significant difference)",
                result="SUPPORTS H_null" if d < 0.3 else "SUPPORTS H_alt",
                details={
                    "mean_loss_with_P": mean_a,
                    "mean_loss_without_P": mean_b,
                    "cohens_d": d,
                    "final_loss_with_P": loss_a[-1],
                    "final_loss_without_P": loss_b[-1],
                    "which_is_better": "Cayley+P" if mean_a < mean_b else "Cayley+I",
                    "n_steps_compared": n_compare,
                },
            ))
            logger.info("F2 Cohen's d: %.4f (%s)", d,
                        "H_null" if d < 0.3 else "H_alt")

        # ── Phase 3: Structural measurements (F5, F6) ──
        logger.info("=" * 60)
        logger.info("Phase 3: Structural — F5 (Fisher), F6 (Geodesic)")
        logger.info("=" * 60)

        f5 = self._test_f5_fisher_eigenspectrum(dataset)
        verdicts.append(f5)
        logger.info("F5: %s", f5.result)

        f6 = self._test_f6_geodesic_deviation()
        verdicts.append(f6)
        logger.info("F6: %s", f6.result)

        # ── Write results ──
        report_md = self._generate_report_md(
            verdicts, loss_a, loss_b, f1_data, f3_data, f4_data,
        )
        (output_dir / "FALSIFICATION_REPORT.md").write_text(report_md)
        logger.info("Report: %s", output_dir / "FALSIFICATION_REPORT.md")

        json_data = {
            "model": self.config.model,
            "config": {
                "n_steps": self.config.n_steps,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "seq_length": self.config.seq_length,
                "seed": self.config.seed,
            },
            "verdicts": [asdict(v) for v in verdicts],
            "loss_with_P": loss_a,
            "loss_without_P": loss_b,
        }
        (output_dir / "results.json").write_text(
            json.dumps(json_data, indent=2, default=_json_default),
        )
        logger.info("Data: %s", output_dir / "results.json")

        # ── Summary ──
        logger.info("")
        logger.info("=" * 60)
        logger.info("FALSIFICATION SUMMARY")
        logger.info("=" * 60)
        for v in verdicts:
            logger.info("  %s: %s", v.test_name, v.result)

        self._unload_model()


# =============================================================================
# Utility Functions
# =============================================================================


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(len(s) * p)))
    return s[idx]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf"
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight space geometry falsification experiments.",
    )
    parser.add_argument(
        "--model",
        default="LFM2-350M",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to test (default: LFM2-350M)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Training steps per condition (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/weight_geometry",
        help="Output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Training dataset path (JSONL)",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        model=args.model,
        dataset_path=args.dataset,
        n_steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        lr=args.lr,
        seed=args.seed,
        output_dir=Path(args.output),
    )

    experiment = WeightGeometryFalsification(config)
    experiment.run()


if __name__ == "__main__":
    main()
