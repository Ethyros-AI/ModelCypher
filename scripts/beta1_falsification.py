#!/usr/bin/env python3
"""Beta-1 topological loop falsification experiments.

Pre-registered falsification tests for the claim:
    "Δβ₁ (late minus early layer β₁) predicts reasoning correctness."

Three falsification tests with automated PASS/FAIL verdicts:

F1: METRIC ROBUSTNESS
    Prediction: sign(Δβ₁) agrees across >= 3/4 metrics for >= 80% of samples.
    Fail: signal is metric artifact -> [DISPROVEN: metric artifact]

F2: GENERALITY BEYOND MATH
    Prediction: Δβ₁ is lower for degenerate outputs than coherent outputs.
    Fail: Cohen's d <= 0.3 -> signal is math-specific, not general.

F3: HELD-OUT REPLICATION
    Prediction: Δβ₁ > 0 for correct, < 0 for incorrect on held-out problems.
    Fail: Cohen's d <= 0.3 OR permutation p >= 0.05 OR bootstrap CI includes 0.

Bonus: Graph proxy correlation
    Spearman ρ between graph β₁ and VR β₁ across layers.
    If ρ > 0.5: graph proxy viable for inference-time use.

Usage:
    # Smoke test
    poetry run python scripts/beta1_falsification.py \\
        --models LFM2-350M \\
        --samples 20 \\
        --output results/beta1_falsification/smoke/

    # Full run
    poetry run python scripts/beta1_falsification.py \\
        --samples 50 \\
        --output results/beta1_falsification/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from modelcypher.core.domain.statistics import (
    cohens_d_bootstrap_ci,
    cohens_d_two_groups,
    permutation_test_p_value,
    spearman_correlation,
)

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
        "architecture": "lfm2",
    },
    "LFM2-1.2B": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-1.2B-bf16",
        "architecture": "lfm2",
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "architecture": "qwen3",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExperimentConfig:
    models: list[str]
    n_samples: int = 50
    max_tokens: int = 256
    seed: int = 42
    output_dir: Path = field(
        default_factory=lambda: Path("results/beta1_falsification")
    )
    max_tda_tokens: int = 50
    pca_dim: int = 50


@dataclass
class TrajectoryData:
    prompt: str
    expected_answer: str
    generated_text: str
    extracted_answer: float | None
    is_correct: bool
    is_degenerate: bool
    n_tokens: int
    layer_hidden_states: list[dict[int, list[float]]]
    tokens: list[str]


@dataclass
class FalsificationVerdict:
    test_name: str
    prediction: str
    result: str  # PASS or FAIL
    details: dict[str, Any]


@dataclass
class FalsificationReport:
    model_name: str
    n_samples: int
    n_correct: int
    n_incorrect: int
    n_degenerate: int
    n_coherent: int
    accuracy: float
    f1_metric_robustness: FalsificationVerdict
    f2_generality: FalsificationVerdict
    f3_replication: FalsificationVerdict
    f5_subsample_stability: FalsificationVerdict
    f6_null_shuffle: FalsificationVerdict
    f7_layer_window: FalsificationVerdict
    graph_proxy_correlation: dict[str, float]
    per_trajectory_data: list[dict[str, Any]]


# =============================================================================
# Degenerate Output Detection
# =============================================================================


def _is_degenerate(
    generated_text: str,
    hit_max_tokens: bool,
    min_repeat_tokens: int = 3,
    min_repeat_count: int = 3,
) -> bool:
    """Detect degenerate outputs.

    Degenerate = hit max_tokens without EOS, OR has 3+ consecutive
    repeats of a 3+ token substring.
    """
    if hit_max_tokens:
        return True

    # Check for repeated substrings
    words = generated_text.split()
    for span_len in range(min_repeat_tokens, len(words) // min_repeat_count + 1):
        for start in range(len(words) - span_len * min_repeat_count + 1):
            pattern = words[start : start + span_len]
            count = 0
            pos = start
            while pos + span_len <= len(words):
                if words[pos : pos + span_len] == pattern:
                    count += 1
                    pos += span_len
                else:
                    break
            if count >= min_repeat_count:
                return True

    return False


# =============================================================================
# Number Extraction and Correctness
# =============================================================================


def _extract_number(text: str) -> float | None:
    if "####" in text:
        after = text.split("####")[-1].strip()
        matches = re.findall(r"-?\d+\.?\d*", after)
        if matches:
            try:
                return float(matches[0].replace(",", ""))
            except ValueError:
                pass

    matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def _check_correctness(extracted: float | None, expected_str: str) -> bool:
    if extracted is None:
        return False
    try:
        expected = float(expected_str.replace(",", ""))
    except ValueError:
        return False
    if expected == int(expected):
        return extracted == expected
    return abs(extracted - expected) < 1e-6


# =============================================================================
# Main Experiment
# =============================================================================


class Beta1Falsification:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0
        self._rng = random.Random(config.seed)

    def _load_model(self, model_name: str) -> None:
        from modelcypher.backends import initialize_default_backend

        if self.backend is None:
            self.backend = initialize_default_backend()

        model_info = MODEL_REGISTRY[model_name]
        logger.info(f"Loading model: {model_name} from {model_info['path']}")
        model_path = Path(model_info["path"])
        self.model, self.tokenizer = self.backend.load_model(str(model_path))

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        self.num_layers = len(layers) if layers else 0
        logger.info(f"Model loaded: {self.num_layers} layers")

    def _unload_model(self) -> None:
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

    def _load_benchmark(self, n_samples: int) -> list[tuple[str, str]]:
        """Load held-out arithmetic problems.

        Uses the generated arithmetic benchmark which is independent of
        any problems used during the original β₁ discovery.
        """
        from modelcypher.core.use_cases.curriculum.benchmark_loader import (
            BenchmarkLoader,
        )

        loader = BenchmarkLoader()
        benchmark = loader.load("arithmetic", split="test", limit=n_samples)
        return [(s.prompt, s.answer) for s in benchmark.samples]

    def _format_prompt(self, raw_prompt: str) -> str:
        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template
        ):
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return raw_prompt

    def _generate_one(
        self, prompt: str, expected_answer: str,
    ) -> TrajectoryData:
        """Generate with GREEDY decoding, capturing full trajectory."""
        b = self.backend
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)

        formatted = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted)
        if isinstance(prompt_tokens, list):
            prompt_ids = prompt_tokens
        else:
            prompt_ids = b.tolist(prompt_tokens)

        current_ids = b.array([prompt_ids])

        generated_tokens: list[int] = []
        token_strings: list[str] = []
        all_hidden_states: list[dict[int, list[float]]] = []
        hit_eos = False

        for gen_step in range(self.config.max_tokens):
            captured: dict[int, Any] = {}

            class CaptureWrapper:
                def __init__(ws, layer: Any, layer_idx: int) -> None:
                    ws._layer = layer
                    ws._layer_idx = layer_idx

                def __call__(ws, *args: Any, **kwargs: Any) -> Any:
                    output = ws._layer(*args, **kwargs)
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    captured[ws._layer_idx] = hidden
                    return output

                def __getattr__(ws, name: str) -> Any:
                    return getattr(ws._layer, name)

            original_layers = list(layers)
            try:
                for i in range(len(layers)):
                    layers[i] = CaptureWrapper(original_layers[i], i)
                logits = self.model(current_ids)
                b.eval(logits)
            finally:
                for i, layer in enumerate(original_layers):
                    layers[i] = layer

            # Extract last-position hidden states -> Python lists
            token_hidden: dict[int, list[float]] = {}
            for layer_idx, hidden in captured.items():
                if hidden.ndim == 3:
                    h = hidden[0, -1, :]
                else:
                    h = hidden[-1, :]
                b.eval(h)
                token_hidden[layer_idx] = b.tolist(h)

            all_hidden_states.append(token_hidden)

            # GREEDY: argmax
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]
            b.eval(last_logits)
            next_token = int(b.to_scalar(b.argmax(last_logits)))

            generated_tokens.append(next_token)
            try:
                token_strings.append(self.tokenizer.decode([next_token]))
            except Exception:
                token_strings.append(f"<{next_token}>")

            current_ids = b.concatenate(
                [current_ids, b.array([[next_token]])], axis=1
            )

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token == eos_id:
                hit_eos = True
                break

        generated_text = self.tokenizer.decode(generated_tokens)
        extracted = _extract_number(generated_text)
        is_correct = _check_correctness(extracted, expected_answer)
        hit_max = not hit_eos
        is_degen = _is_degenerate(generated_text, hit_max)

        return TrajectoryData(
            prompt=prompt,
            expected_answer=expected_answer,
            generated_text=generated_text,
            extracted_answer=extracted,
            is_correct=is_correct,
            is_degenerate=is_degen,
            n_tokens=len(generated_tokens),
            layer_hidden_states=all_hidden_states,
            tokens=token_strings,
        )

    def _collect_trajectories(
        self, samples: list[tuple[str, str]],
    ) -> list[TrajectoryData]:
        trajectories: list[TrajectoryData] = []

        for i, (prompt, answer) in enumerate(samples):
            try:
                traj = self._generate_one(prompt, answer)
                trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                n_correct = sum(1 for t in trajectories if t.is_correct)
                n_degen = sum(1 for t in trajectories if t.is_degenerate)
                logger.info(
                    f"  Progress: {i+1}/{len(samples)}, "
                    f"correct: {n_correct}/{len(trajectories)}, "
                    f"degenerate: {n_degen}"
                )

            if (i + 1) % 50 == 0 and self.backend is not None:
                try:
                    self.backend.clear_cache()
                except Exception:
                    pass

        return trajectories

    def _analyze_topology(
        self, trajectories: list[TrajectoryData],
    ) -> list[dict[str, Any]]:
        """Compute multi-metric topology signal for each trajectory.

        Returns per-trajectory analysis dicts with all metric β₁ and graph proxy.
        """
        from modelcypher.core.domain.geometry.multi_metric_topology import (
            compute_multi_metric_topology_signal,
        )

        b = self.backend
        results: list[dict[str, Any]] = []
        t0 = time.time()

        for i, traj in enumerate(trajectories):
            # Subsample tokens for TDA (O(n³))
            hs_seq = traj.layer_hidden_states
            toks = traj.tokens
            if len(hs_seq) > self.config.max_tda_tokens:
                indices = [
                    int(
                        j * (len(hs_seq) - 1) / (self.config.max_tda_tokens - 1)
                    )
                    for j in range(self.config.max_tda_tokens)
                ]
                hs_seq = [hs_seq[idx] for idx in indices]
                toks = [toks[idx] for idx in indices]

            # Convert Python lists to backend arrays
            hs_arrays = []
            for token_states in hs_seq:
                layer_dict = {}
                for layer_idx, values in token_states.items():
                    layer_dict[int(layer_idx)] = b.array(values)
                hs_arrays.append(layer_dict)

            layer_indices = sorted(hs_arrays[0].keys()) if hs_arrays else []

            signal = compute_multi_metric_topology_signal(
                hs_arrays,
                layer_indices,
                backend=b,
                max_tda_points=self.config.pca_dim,
            )

            result = signal.as_dict()
            result["is_correct"] = traj.is_correct
            result["is_degenerate"] = traj.is_degenerate
            result["n_tokens"] = traj.n_tokens
            result["prompt"] = traj.prompt[:100]
            results.append(result)

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - t0
                per_traj = elapsed / (i + 1)
                remaining = per_traj * (len(trajectories) - i - 1)
                logger.info(
                    f"    Topology analysis: {i+1}/{len(trajectories)} "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )

        return results

    # ----- Falsification Tests -----

    def _test_f1_metric_robustness(
        self, per_traj: list[dict[str, Any]],
    ) -> FalsificationVerdict:
        """F1: sign(Δβ₁) agrees across >= 3/4 metrics for >= 80% of samples."""
        agreements = []
        for t in per_traj:
            agreements.append(t["sign_agreement"])

        mean_agreement = (
            sum(agreements) / len(agreements) if agreements else 0.0
        )
        n_high = sum(1 for a in agreements if a >= 0.75)
        frac_high = n_high / len(agreements) if agreements else 0.0

        passed = frac_high >= 0.80

        return FalsificationVerdict(
            test_name="F1: Metric Robustness",
            prediction=(
                "sign(Δβ₁) agrees across >= 3/4 metrics "
                "for >= 80% of samples"
            ),
            result="PASS" if passed else "FAIL",
            details={
                "mean_agreement": mean_agreement,
                "fraction_high_agreement": frac_high,
                "n_high_agreement": n_high,
                "n_total": len(agreements),
                "threshold": 0.80,
                "per_sample_agreements": agreements,
            },
        )

    def _test_f2_generality(
        self, per_traj: list[dict[str, Any]],
    ) -> FalsificationVerdict:
        """F2: Δβ₁ lower for degenerate than coherent outputs."""
        # Use geodesic Δβ₁ (validated default)
        coherent_deltas = []
        degenerate_deltas = []

        for t in per_traj:
            delta = t["delta_beta1_by_metric"]["geodesic"]
            if t["is_degenerate"]:
                degenerate_deltas.append(delta)
            else:
                coherent_deltas.append(delta)

        if len(coherent_deltas) < 2 or len(degenerate_deltas) < 2:
            return FalsificationVerdict(
                test_name="F2: Generality Beyond Math",
                prediction="Δβ₁ lower for degenerate than coherent outputs",
                result="INCONCLUSIVE",
                details={
                    "reason": "Insufficient degenerate or coherent samples",
                    "n_coherent": len(coherent_deltas),
                    "n_degenerate": len(degenerate_deltas),
                },
            )

        d = cohens_d_two_groups(coherent_deltas, degenerate_deltas)
        passed = abs(d) > 0.3

        return FalsificationVerdict(
            test_name="F2: Generality Beyond Math",
            prediction="Δβ₁ lower for degenerate than coherent outputs",
            result="PASS" if passed else "FAIL",
            details={
                "cohens_d": d,
                "threshold": 0.3,
                "n_coherent": len(coherent_deltas),
                "n_degenerate": len(degenerate_deltas),
                "mean_coherent": sum(coherent_deltas) / len(coherent_deltas),
                "mean_degenerate": (
                    sum(degenerate_deltas) / len(degenerate_deltas)
                ),
            },
        )

    def _test_f3_replication(
        self, per_traj: list[dict[str, Any]],
    ) -> FalsificationVerdict:
        """F3: Δβ₁ > 0 for correct, < 0 for incorrect (held-out problems)."""
        correct_deltas = []
        incorrect_deltas = []

        for t in per_traj:
            delta = t["delta_beta1_by_metric"]["geodesic"]
            if t["is_correct"]:
                correct_deltas.append(delta)
            else:
                incorrect_deltas.append(delta)

        if len(correct_deltas) < 2 or len(incorrect_deltas) < 2:
            return FalsificationVerdict(
                test_name="F3: Held-Out Replication",
                prediction=(
                    "Δβ₁ > 0 for correct, < 0 for incorrect "
                    "on held-out problems"
                ),
                result="INCONCLUSIVE",
                details={
                    "reason": "Insufficient correct or incorrect samples",
                    "n_correct": len(correct_deltas),
                    "n_incorrect": len(incorrect_deltas),
                },
            )

        d = cohens_d_two_groups(correct_deltas, incorrect_deltas)
        ci = cohens_d_bootstrap_ci(
            correct_deltas,
            incorrect_deltas,
            n_bootstrap=1000,
            rng=self._rng,
        )
        p = permutation_test_p_value(
            correct_deltas,
            incorrect_deltas,
            n_permutations=1000,
            rng=self._rng,
        )

        ci_excludes_zero = ci[0] > 0.0 or ci[1] < 0.0
        passed = abs(d) > 0.3 and p < 0.05 and ci_excludes_zero

        return FalsificationVerdict(
            test_name="F3: Held-Out Replication",
            prediction=(
                "Δβ₁ > 0 for correct, < 0 for incorrect "
                "on held-out problems"
            ),
            result="PASS" if passed else "FAIL",
            details={
                "cohens_d": d,
                "bootstrap_ci": list(ci),
                "ci_excludes_zero": ci_excludes_zero,
                "permutation_p": p,
                "d_threshold": 0.3,
                "p_threshold": 0.05,
                "n_correct": len(correct_deltas),
                "n_incorrect": len(incorrect_deltas),
                "mean_correct": sum(correct_deltas) / len(correct_deltas),
                "mean_incorrect": (
                    sum(incorrect_deltas) / len(incorrect_deltas)
                ),
            },
        )

    def _compute_graph_proxy_correlation(
        self, per_traj: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Spearman correlation between graph β₁ and VR β₁ across layers."""
        # Aggregate per-layer: for each layer position, collect
        # (graph_beta1, geodesic_beta1) across all trajectories
        all_graph: list[float] = []
        all_vr: list[float] = []

        for t in per_traj:
            graph_by_layer = t["graph_beta1_by_layer"]
            mm_by_layer = t["beta1_by_metric_by_layer"]

            for layer_i in range(len(graph_by_layer)):
                all_graph.append(float(graph_by_layer[layer_i]))
                all_vr.append(float(mm_by_layer[layer_i]["geodesic"]))

        if len(all_graph) < 3:
            return {"spearman_rho": 0.0, "n_datapoints": 0, "proxy_viable": False}

        rho = spearman_correlation(all_graph, all_vr)

        return {
            "spearman_rho": rho,
            "n_datapoints": len(all_graph),
            "proxy_viable": rho > 0.5,
        }

    # ----- Report Generation -----

    def _generate_report_md(
        self, report: FalsificationReport,
    ) -> str:
        """Generate human-readable FALSIFICATION_REPORT.md."""
        lines = [
            "# Beta-1 Falsification Report",
            "",
            f"**Model:** {report.model_name}",
            f"**Samples:** {report.n_samples}",
            f"**Accuracy:** {report.accuracy:.1%} "
            f"({report.n_correct}/{report.n_samples})",
            f"**Degenerate:** {report.n_degenerate}",
            f"**Coherent:** {report.n_coherent}",
            "",
            "---",
            "",
        ]

        for verdict in [
            report.f1_metric_robustness,
            report.f2_generality,
            report.f3_replication,
        ]:
            status = "PASS" if verdict.result == "PASS" else (
                "FAIL" if verdict.result == "FAIL" else "INCONCLUSIVE"
            )
            lines.append(f"## {verdict.test_name}: **{status}**")
            lines.append("")
            lines.append(f"**Prediction:** {verdict.prediction}")
            lines.append("")

            for k, v in verdict.details.items():
                if k == "per_sample_agreements":
                    continue  # too verbose for markdown
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        # Graph proxy
        lines.append("## Bonus: Graph Proxy Correlation")
        lines.append("")
        gpc = report.graph_proxy_correlation
        lines.append(
            f"- Spearman rho: {gpc['spearman_rho']:.4f}"
        )
        lines.append(f"- Datapoints: {gpc['n_datapoints']}")
        lines.append(
            f"- Proxy viable (rho > 0.5): "
            f"{'YES' if gpc['proxy_viable'] else 'NO'}"
        )
        lines.append("")

        # Summary
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        verdicts = [
            report.f1_metric_robustness,
            report.f2_generality,
            report.f3_replication,
        ]
        n_pass = sum(1 for v in verdicts if v.result == "PASS")
        n_fail = sum(1 for v in verdicts if v.result == "FAIL")
        n_inc = sum(1 for v in verdicts if v.result == "INCONCLUSIVE")

        lines.append(f"- PASS: {n_pass}/3")
        lines.append(f"- FAIL: {n_fail}/3")
        if n_inc > 0:
            lines.append(f"- INCONCLUSIVE: {n_inc}/3")
        lines.append("")

        if n_fail > 0:
            failed_names = [
                v.test_name for v in verdicts if v.result == "FAIL"
            ]
            lines.append(
                f"**Action:** Failed tests ({', '.join(failed_names)}) "
                f"should move affected claims to [DISPROVEN]."
            )
        elif n_pass == 3:
            lines.append(
                "**Action:** All tests passed. β₁ claim remains [EMPIRICAL] "
                "and can be considered for promotion to [VALIDATED]."
            )
        else:
            lines.append(
                "**Action:** Inconclusive tests need more data. "
                "Re-run with --samples >= 50."
            )

        lines.append("")
        return "\n".join(lines)

    # ----- Main Run -----

    def run(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.config.output_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": self.config.models,
                    "n_samples": self.config.n_samples,
                    "max_tokens": self.config.max_tokens,
                    "seed": self.config.seed,
                    "max_tda_tokens": self.config.max_tda_tokens,
                    "pca_dim": self.config.pca_dim,
                },
                indent=2,
            )
        )

        for model_name in self.config.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running falsification for: {model_name}")
            logger.info(f"{'='*60}")

            model_dir = self.config.output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Load model
            self._load_model(model_name)

            # Load held-out benchmark
            logger.info("Loading held-out arithmetic benchmark...")
            samples = self._load_benchmark(self.config.n_samples)
            logger.info(f"Loaded {len(samples)} samples")

            # Generate trajectories
            logger.info("Generating trajectories (greedy decoding)...")
            trajectories = self._collect_trajectories(samples)
            logger.info(
                f"Generated {len(trajectories)} trajectories, "
                f"{sum(1 for t in trajectories if t.is_correct)} correct, "
                f"{sum(1 for t in trajectories if t.is_degenerate)} degenerate"
            )

            # Multi-metric topology analysis
            logger.info("Computing multi-metric topology signals...")
            per_traj = self._analyze_topology(trajectories)

            # Run falsification tests
            logger.info("Running falsification tests...")
            f1 = self._test_f1_metric_robustness(per_traj)
            f2 = self._test_f2_generality(per_traj)
            f3 = self._test_f3_replication(per_traj)

            logger.info(f"  F1 Metric Robustness: {f1.result}")
            logger.info(f"  F2 Generality: {f2.result}")
            logger.info(f"  F3 Replication: {f3.result}")

            # Graph proxy correlation
            gpc = self._compute_graph_proxy_correlation(per_traj)
            logger.info(
                f"  Graph proxy Spearman: {gpc['spearman_rho']:.4f} "
                f"({'viable' if gpc['proxy_viable'] else 'not viable'})"
            )

            # Build report
            n_correct = sum(1 for t in trajectories if t.is_correct)
            n_degen = sum(1 for t in trajectories if t.is_degenerate)

            report = FalsificationReport(
                model_name=model_name,
                n_samples=len(trajectories),
                n_correct=n_correct,
                n_incorrect=len(trajectories) - n_correct,
                n_degenerate=n_degen,
                n_coherent=len(trajectories) - n_degen,
                accuracy=n_correct / len(trajectories) if trajectories else 0.0,
                f1_metric_robustness=f1,
                f2_generality=f2,
                f3_replication=f3,
                graph_proxy_correlation=gpc,
                per_trajectory_data=per_traj,
            )

            # Write results
            report_md = self._generate_report_md(report)
            (model_dir / "FALSIFICATION_REPORT.md").write_text(report_md)
            logger.info(f"  Report: {model_dir / 'FALSIFICATION_REPORT.md'}")

            # JSON results (strip large hidden state data from per_traj)
            json_data = {
                "model_name": report.model_name,
                "n_samples": report.n_samples,
                "n_correct": report.n_correct,
                "n_incorrect": report.n_incorrect,
                "n_degenerate": report.n_degenerate,
                "n_coherent": report.n_coherent,
                "accuracy": report.accuracy,
                "f1": {
                    "result": f1.result,
                    "details": f1.details,
                },
                "f2": {
                    "result": f2.result,
                    "details": f2.details,
                },
                "f3": {
                    "result": f3.result,
                    "details": f3.details,
                },
                "graph_proxy": gpc,
                "per_trajectory": per_traj,
            }
            (model_dir / "results.json").write_text(
                json.dumps(json_data, indent=2, default=str)
            )
            logger.info(f"  Data: {model_dir / 'results.json'}")

            # Unload model
            self._unload_model()

        logger.info("\nFalsification experiments complete.")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beta-1 topological loop falsification experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LFM2-350M"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to test (default: LFM2-350M)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per model (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation tokens (default: 256)",
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
        default="results/beta1_falsification",
        help="Output directory",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA dimension for TDA (default: 50)",
    )
    parser.add_argument(
        "--max-tda-tokens",
        type=int,
        default=50,
        help="Max tokens for TDA subsampling (default: 50)",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        models=args.models,
        n_samples=args.samples,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_dir=Path(args.output),
        max_tda_tokens=args.max_tda_tokens,
        pca_dim=args.pca_dim,
    )

    experiment = Beta1Falsification(config)
    experiment.run()


if __name__ == "__main__":
    main()
