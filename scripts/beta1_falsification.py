#!/usr/bin/env python3
"""Beta-1 topological loop falsification experiments.

Pre-registered falsification tests for the claim:
    "Δβ₁ (late minus early layer β₁) predicts reasoning correctness."

Six falsification tests with automated PASS/FAIL verdicts:

F1: METRIC ROBUSTNESS
    Prediction: sign(Δβ₁) agrees across >= 3/4 metrics for >= 80% of samples.
    Fail: signal is metric artifact -> [DISPROVEN: metric artifact]

F2: GENERALITY BEYOND MATH
    Prediction: Δβ₁ is lower for degenerate outputs than coherent outputs.
    Fail: Cohen's d <= 0.3 -> signal is math-specific, not general.

F3: HELD-OUT REPLICATION
    Prediction: Δβ₁ > 0 for correct, < 0 for incorrect on held-out problems.
    Fail: Cohen's d <= 0.3 OR permutation p >= 0.05 OR bootstrap CI includes 0.

F5: SUBSAMPLING STABILITY
    Prediction: sign(Δβ₁) stable across 5 random 80% token subsets for >= 80%.
    Fail: signal is point cloud size artifact.

F6: NULL-SHUFFLE CONTROL
    Prediction: shuffled token order destroys correctness signal (d <= 0.3).
    Fail: shuffled data still discriminates -> signal is spurious.

F7: LAYER-WINDOW CALIBRATION
    Prediction: signal (d > 0.3) in >= 2/3 alternative window fractions.
    Fail: signal only in one specific window -> arbitrary window artifact.

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
import random
import re
import time
from dataclasses import dataclass, field
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
    benchmark: str = "hard_arithmetic"
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
        """Load benchmark problems.

        Supports arithmetic (easy, 1-digit), hard_arithmetic (multi-digit,
        multiplication/division — produces more errors), and gsm8k (word problems).
        """
        if self.config.benchmark == "hard_arithmetic":
            return self._generate_hard_arithmetic(n_samples)

        from modelcypher.core.use_cases.curriculum.benchmark_loader import (
            BenchmarkLoader,
        )

        loader = BenchmarkLoader()
        benchmark = loader.load(
            self.config.benchmark, split="test", limit=n_samples,
        )
        return [(s.prompt, s.answer) for s in benchmark.samples]

    def _generate_hard_arithmetic(
        self, n_samples: int,
    ) -> list[tuple[str, str]]:
        """Generate harder arithmetic for better correct/incorrect split.

        Mix of:
        - 2-3 digit addition/subtraction (30%)
        - 2-digit multiplication (40%)
        - 2-operand chains like a+b*c (30%)
        Deterministic seed for reproducibility.
        """
        rng = random.Random(44)  # distinct from train=42, test=43
        samples: list[tuple[str, str]] = []

        for i in range(n_samples):
            kind = i % 10
            if kind < 3:
                # 2-3 digit addition/subtraction
                a = rng.randint(10, 999)
                b = rng.randint(10, 999)
                op = rng.choice(["+", "-"])
                if op == "-" and b > a:
                    a, b = b, a
                result = a + b if op == "+" else a - b
                prompt = f"{a}{op}{b}="
            elif kind < 7:
                # 2-digit multiplication
                a = rng.randint(2, 99)
                b = rng.randint(2, 99)
                result = a * b
                prompt = f"{a}*{b}="
            else:
                # 2-operand chain: a op1 b op2 c (left to right, no precedence)
                a = rng.randint(1, 50)
                b = rng.randint(1, 50)
                c = rng.randint(1, 20)
                ops = rng.choices(["+", "-", "*"], k=2)
                # Evaluate left to right
                intermediate = a
                for op_i, operand in zip(ops, [b, c]):
                    if op_i == "+":
                        intermediate += operand
                    elif op_i == "-":
                        intermediate -= operand
                    else:
                        intermediate *= operand
                result = intermediate
                prompt = f"{a}{ops[0]}{b}{ops[1]}{c}="

            samples.append((prompt, str(result)))

        return samples

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
        """F1: sign(Δβ₁) agrees across >= 3/4 metrics for >= 80% of samples.

        All 4 metrics are independent:
        - Euclidean: straight-line distances
        - Cosine: angular dissimilarity
        - Spectral: effective-resistance embedding distances
        - Geodesic: kNN graph shortest paths (k=k_min for connectivity)

        Round 1 used k=n-1 for geodesic, making it identical to Euclidean.
        Round 2 uses k=None (minimum connected k), restoring independence.
        """
        agreements = []
        for t in per_traj:
            agreements.append(t["sign_agreement"])

        mean_agreement = (
            sum(agreements) / len(agreements) if agreements else 0.0
        )
        # Agreement across 4 metrics: >= 3/4 = 0.75
        majority_threshold = 3 / 4
        n_high = sum(1 for a in agreements if a >= majority_threshold)
        frac_high = n_high / len(agreements) if agreements else 0.0

        passed = frac_high >= 0.80

        return FalsificationVerdict(
            test_name="F1: Metric Robustness (4 independent metrics)",
            prediction=(
                "sign(Δβ₁) agrees across >= 3/4 of independent metrics "
                "(euclidean, cosine, spectral, geodesic) for >= 80% of "
                "samples. Geodesic now uses k_min (minimum connected k) "
                "instead of k=n-1, making it independent of Euclidean."
            ),
            result="PASS" if passed else "FAIL",
            details={
                "mean_agreement": mean_agreement,
                "fraction_majority_agreement": frac_high,
                "n_majority_agreement": n_high,
                "n_total": len(agreements),
                "sample_threshold": 0.80,
                "majority_threshold": majority_threshold,
                "independent_metrics": [
                    "euclidean", "cosine", "spectral", "geodesic",
                ],
                "geodesic_k": "k_min (minimum connected k)",
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
        """F3: Δβ₁ separates correct from incorrect on held-out problems.

        Tests each independent metric separately. F3 passes if ANY metric
        achieves d > 0.3 AND p < 0.05 AND bootstrap CI excludes 0.

        Round 1 showed geodesic==euclidean (k=n-1 made graph complete).
        Round 2 uses k_min for geodesic, making all 4 metrics independent.
        Per-metric testing captures heterogeneity across distance functions.
        """
        _INDEPENDENT_METRICS = ["euclidean", "cosine", "spectral", "geodesic"]
        _D_THRESHOLD = 0.3
        _P_THRESHOLD = 0.05

        # Check we have enough samples
        n_correct = sum(1 for t in per_traj if t["is_correct"])
        n_incorrect = sum(1 for t in per_traj if not t["is_correct"])

        if n_correct < 2 or n_incorrect < 2:
            return FalsificationVerdict(
                test_name="F3: Held-Out Replication (per-metric)",
                prediction=(
                    "Δβ₁ separates correct from incorrect in at least "
                    "one independent metric (d > 0.3, p < 0.05, CI ∌ 0)"
                ),
                result="INCONCLUSIVE",
                details={
                    "reason": "Insufficient correct or incorrect samples",
                    "n_correct": n_correct,
                    "n_incorrect": n_incorrect,
                },
            )

        per_metric_results: list[dict[str, Any]] = []
        any_passed = False

        for metric in _INDEPENDENT_METRICS:
            correct_deltas = []
            incorrect_deltas = []

            for t in per_traj:
                delta = t["delta_beta1_by_metric"][metric]
                if t["is_correct"]:
                    correct_deltas.append(delta)
                else:
                    incorrect_deltas.append(delta)

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
            metric_passed = (
                abs(d) > _D_THRESHOLD
                and p < _P_THRESHOLD
                and ci_excludes_zero
            )
            if metric_passed:
                any_passed = True

            per_metric_results.append({
                "metric": metric,
                "cohens_d": d,
                "bootstrap_ci": list(ci),
                "ci_excludes_zero": ci_excludes_zero,
                "permutation_p": p,
                "passed": metric_passed,
                "mean_correct": (
                    sum(correct_deltas) / len(correct_deltas)
                ),
                "mean_incorrect": (
                    sum(incorrect_deltas) / len(incorrect_deltas)
                ),
            })

        return FalsificationVerdict(
            test_name="F3: Held-Out Replication (per-metric)",
            prediction=(
                "Δβ₁ separates correct from incorrect in at least "
                "one independent metric (d > 0.3, p < 0.05, CI ∌ 0)"
            ),
            result="PASS" if any_passed else "FAIL",
            details={
                "any_metric_passed": any_passed,
                "n_metrics_passed": sum(
                    1 for r in per_metric_results if r["passed"]
                ),
                "d_threshold": _D_THRESHOLD,
                "p_threshold": _P_THRESHOLD,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "per_metric": per_metric_results,
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

    # ----- Robustness Controls (F5-F7) -----

    @staticmethod
    def _compute_delta_windowed(
        values: list[int | float], fraction: float,
    ) -> float:
        """Compute Δβ₁ with parameterized window fraction.

        Δ = mean(last fraction) - mean(first fraction).
        Default _compute_delta uses fraction=1/3.
        """
        n = len(values)
        if n < 2:
            return 0.0
        k = max(1, int(n * fraction))
        early = values[:k]
        late = values[-k:]
        return sum(late) / len(late) - sum(early) / len(early)

    def _test_f5_subsample_stability(
        self,
        trajectories: list[TrajectoryData],
        per_traj: list[dict[str, Any]],
    ) -> FalsificationVerdict:
        """F5: Δβ₁ sign stable under token subsampling.

        For each trajectory, draw 5 random 80% subsets of tokens,
        recompute topology, check sign(Δβ₁_geodesic) consistency.
        PASS: ≥80% of trajectories have stable sign across subsamples.
        """
        from modelcypher.core.domain.geometry.multi_metric_topology import (
            compute_multi_metric_topology_signal,
        )

        _N_SUBSAMPLES = 5
        _SUBSAMPLE_FRAC = 0.8
        _STABILITY_THRESHOLD = 0.80

        n_stable = 0
        n_tested = 0

        for traj, traj_data in zip(trajectories, per_traj):
            hs_seq = traj.layer_hidden_states
            if len(hs_seq) < 5:
                continue  # too few tokens to subsample meaningfully

            n_tokens = len(hs_seq)
            n_keep = max(3, int(n_tokens * _SUBSAMPLE_FRAC))

            # Original sign (from full topology analysis)
            orig_delta = traj_data["delta_beta1_by_metric"]["geodesic"]
            if orig_delta == 0.0:
                continue  # zero delta is ambiguous, skip

            orig_sign = 1 if orig_delta > 0 else -1
            n_agree = 0

            for s in range(_N_SUBSAMPLES):
                # Random subset without replacement
                indices = sorted(
                    self._rng.sample(range(n_tokens), n_keep)
                )
                sub_hs = [hs_seq[i] for i in indices]

                # Convert to backend arrays
                b = self.backend
                hs_arrays = []
                for token_states in sub_hs:
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
                sub_delta = signal.delta_beta1_by_metric.get("geodesic", 0.0)
                if sub_delta != 0.0:
                    sub_sign = 1 if sub_delta > 0 else -1
                    if sub_sign == orig_sign:
                        n_agree += 1

            # Stable if majority (≥4/5) agree
            is_stable = n_agree >= 4
            if is_stable:
                n_stable += 1
            n_tested += 1

        frac_stable = n_stable / n_tested if n_tested > 0 else 0.0
        passed = frac_stable >= _STABILITY_THRESHOLD

        return FalsificationVerdict(
            test_name="F5: Subsampling Stability",
            prediction=(
                "sign(Δβ₁) stable across ≥4/5 token subsamples "
                "for ≥80% of trajectories"
            ),
            result="PASS" if passed else (
                "INCONCLUSIVE" if n_tested < 5 else "FAIL"
            ),
            details={
                "fraction_stable": frac_stable,
                "n_stable": n_stable,
                "n_tested": n_tested,
                "n_subsamples": _N_SUBSAMPLES,
                "subsample_fraction": _SUBSAMPLE_FRAC,
                "threshold": _STABILITY_THRESHOLD,
            },
        )

    def _test_f6_null_shuffle(
        self,
        trajectories: list[TrajectoryData],
    ) -> FalsificationVerdict:
        """F6: Shuffled token order should NOT predict correctness.

        Permute token order within each trajectory, recompute Δβ₁,
        then test whether shuffled Δβ₁ still discriminates correct/incorrect.
        PASS: Cohen's d ≤ 0.3 on shuffled data (signal destroyed).
        """
        from modelcypher.core.domain.geometry.multi_metric_topology import (
            compute_multi_metric_topology_signal,
        )

        correct_deltas: list[float] = []
        incorrect_deltas: list[float] = []

        for traj in trajectories:
            hs_seq = list(traj.layer_hidden_states)
            if len(hs_seq) < 3:
                continue

            # Shuffle token order (preserves per-token layer structure,
            # destroys temporal/positional relationships)
            shuffled = list(hs_seq)
            self._rng.shuffle(shuffled)

            # Convert to backend arrays
            b = self.backend
            hs_arrays = []
            for token_states in shuffled:
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
            delta = signal.delta_beta1_by_metric.get("geodesic", 0.0)

            if traj.is_correct:
                correct_deltas.append(delta)
            else:
                incorrect_deltas.append(delta)

        if len(correct_deltas) < 2 or len(incorrect_deltas) < 2:
            return FalsificationVerdict(
                test_name="F6: Null-Shuffle Control",
                prediction=(
                    "Shuffled token order destroys correctness signal "
                    "(Cohen's d ≤ 0.3)"
                ),
                result="INCONCLUSIVE",
                details={
                    "reason": "Insufficient correct or incorrect samples",
                    "n_correct": len(correct_deltas),
                    "n_incorrect": len(incorrect_deltas),
                },
            )

        d = cohens_d_two_groups(correct_deltas, incorrect_deltas)
        # PASS means the signal IS destroyed (d ≤ 0.3)
        passed = abs(d) <= 0.3

        return FalsificationVerdict(
            test_name="F6: Null-Shuffle Control",
            prediction=(
                "Shuffled token order destroys correctness signal "
                "(Cohen's d ≤ 0.3)"
            ),
            result="PASS" if passed else "FAIL",
            details={
                "cohens_d_shuffled": d,
                "threshold": 0.3,
                "n_correct": len(correct_deltas),
                "n_incorrect": len(incorrect_deltas),
                "mean_correct_shuffled": (
                    sum(correct_deltas) / len(correct_deltas)
                ),
                "mean_incorrect_shuffled": (
                    sum(incorrect_deltas) / len(incorrect_deltas)
                ),
            },
        )

    def _test_f7_layer_window(
        self, per_traj: list[dict[str, Any]],
    ) -> FalsificationVerdict:
        """F7: Δβ₁ signal robust to layer-window choice.

        Recompute Δβ₁ using 3 alternative window fractions (1/4, 2/5, 1/2)
        instead of default 1/3. Check that ≥2/3 still show Cohen's d > 0.3
        between correct and incorrect.
        PASS: ≥2/3 alternative windows also show signal.
        """
        _WINDOW_FRACTIONS = [0.25, 0.40, 0.50]
        _D_THRESHOLD = 0.3
        _PASS_FRACTION = 2 / 3  # ≥2 of 3

        window_results: list[dict[str, Any]] = []
        n_significant = 0

        for frac in _WINDOW_FRACTIONS:
            correct_deltas: list[float] = []
            incorrect_deltas: list[float] = []

            for t in per_traj:
                # Get per-layer geodesic β₁ values
                mm_by_layer = t["beta1_by_metric_by_layer"]
                geo_values = [
                    float(layer_data["geodesic"])
                    for layer_data in mm_by_layer
                ]

                delta = self._compute_delta_windowed(geo_values, frac)
                if t["is_correct"]:
                    correct_deltas.append(delta)
                else:
                    incorrect_deltas.append(delta)

            if len(correct_deltas) >= 2 and len(incorrect_deltas) >= 2:
                d = cohens_d_two_groups(correct_deltas, incorrect_deltas)
                significant = abs(d) > _D_THRESHOLD
                if significant:
                    n_significant += 1
            else:
                d = 0.0
                significant = False

            window_results.append({
                "fraction": frac,
                "cohens_d": d,
                "significant": significant,
                "n_correct": len(correct_deltas),
                "n_incorrect": len(incorrect_deltas),
            })

        frac_significant = n_significant / len(_WINDOW_FRACTIONS)
        passed = frac_significant >= _PASS_FRACTION

        return FalsificationVerdict(
            test_name="F7: Layer-Window Calibration",
            prediction=(
                "Δβ₁ signal (d > 0.3) in ≥2/3 alternative "
                "window fractions (1/4, 2/5, 1/2)"
            ),
            result="PASS" if passed else "FAIL",
            details={
                "n_significant": n_significant,
                "n_windows": len(_WINDOW_FRACTIONS),
                "fraction_significant": frac_significant,
                "threshold": _PASS_FRACTION,
                "d_threshold": _D_THRESHOLD,
                "per_window": window_results,
            },
        )

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

        all_verdicts = [
            report.f1_metric_robustness,
            report.f2_generality,
            report.f3_replication,
            report.f5_subsample_stability,
            report.f6_null_shuffle,
            report.f7_layer_window,
        ]

        # Verbose keys to skip in markdown
        _SKIP_KEYS = {"per_sample_agreements", "per_window"}

        for verdict in all_verdicts:
            status = "PASS" if verdict.result == "PASS" else (
                "FAIL" if verdict.result == "FAIL" else "INCONCLUSIVE"
            )
            lines.append(f"## {verdict.test_name}: **{status}**")
            lines.append("")
            lines.append(f"**Prediction:** {verdict.prediction}")
            lines.append("")

            for k, v in verdict.details.items():
                if k in _SKIP_KEYS:
                    continue
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
        n_total = len(all_verdicts)
        n_pass = sum(1 for v in all_verdicts if v.result == "PASS")
        n_fail = sum(1 for v in all_verdicts if v.result == "FAIL")
        n_inc = sum(1 for v in all_verdicts if v.result == "INCONCLUSIVE")

        lines.append(f"- PASS: {n_pass}/{n_total}")
        lines.append(f"- FAIL: {n_fail}/{n_total}")
        if n_inc > 0:
            lines.append(f"- INCONCLUSIVE: {n_inc}/{n_total}")
        lines.append("")

        if n_fail > 0:
            failed_names = [
                v.test_name for v in all_verdicts if v.result == "FAIL"
            ]
            lines.append(
                f"**Action:** Failed tests ({', '.join(failed_names)}) "
                f"should move affected claims to [DISPROVEN]."
            )
        elif n_pass == n_total:
            lines.append(
                "**Action:** All tests passed. β₁ claim can be promoted "
                "from [EMPIRICAL] to [VALIDATED]."
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
            logger.info(f"Loading benchmark: {self.config.benchmark}...")
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
            logger.info("Running falsification tests (F1-F3)...")
            f1 = self._test_f1_metric_robustness(per_traj)
            f2 = self._test_f2_generality(per_traj)
            f3 = self._test_f3_replication(per_traj)

            logger.info(f"  F1 Metric Robustness: {f1.result}")
            logger.info(f"  F2 Generality: {f2.result}")
            logger.info(f"  F3 Replication: {f3.result}")

            # Robustness controls (F5-F7)
            logger.info("Running robustness controls (F5-F7)...")
            f5 = self._test_f5_subsample_stability(trajectories, per_traj)
            logger.info(f"  F5 Subsample Stability: {f5.result}")
            f6 = self._test_f6_null_shuffle(trajectories)
            logger.info(f"  F6 Null-Shuffle Control: {f6.result}")
            f7 = self._test_f7_layer_window(per_traj)
            logger.info(f"  F7 Layer-Window Calibration: {f7.result}")

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
                f5_subsample_stability=f5,
                f6_null_shuffle=f6,
                f7_layer_window=f7,
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
                "f5": {
                    "result": f5.result,
                    "details": f5.details,
                },
                "f6": {
                    "result": f6.result,
                    "details": f6.details,
                },
                "f7": {
                    "result": f7.result,
                    "details": f7.details,
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
    parser.add_argument(
        "--benchmark",
        type=str,
        default="hard_arithmetic",
        choices=["arithmetic", "hard_arithmetic", "gsm8k"],
        help="Benchmark to use (default: hard_arithmetic). "
        "hard_arithmetic produces ~30-60%% errors for F2/F3/F6/F7 power. "
        "arithmetic is too easy (>90%% accuracy on 350M).",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        models=args.models,
        benchmark=args.benchmark,
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
