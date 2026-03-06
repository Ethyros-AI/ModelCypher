#!/usr/bin/env python3
"""Attention topology calibration experiment.

Extracts attention weight matrices during inference, computes topological
features (barcode statistics, cross-head/layer Wasserstein agreement),
and measures discriminative power for predicting answer correctness.

This is the PROVEN approach (TOHA AUROC 0.89, Kostenok +16% AUC-ARC)
replacing the DISPROVEN Betti-1 hidden-state point cloud method.

All measurements are direct from persistence diagrams — no discretization,
no arbitrary parameters.

Usage:
    # Smoke test (10 samples, arithmetic — fast)
    poetry run python scripts/attention_topology_calibration.py \
        --model LFM2-350M --samples 10 --benchmark arithmetic \
        --output results/attention_topology/smoke/

    # Full calibration (50 samples)
    poetry run python scripts/attention_topology_calibration.py \
        --model LFM2-350M --samples 50 \
        --output results/attention_topology/calibration/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelcypher.core.domain.geometry.attention_topology import (
    AttentionTopologySignal,
    compute_attention_topology,
)
from modelcypher.core.domain.statistics import (
    cohens_d_two_groups,
    spearman_correlation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model registry
# =============================================================================

MODEL_PATHS = {
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "LFM2-700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "LFM2-1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
}

# Names of the direct barcode / Wasserstein measurements on the signal.
# Every one is derived from persistence diagram birth/death pairs — no
# discretization, no arbitrary parameters.
METRIC_NAMES = [
    "total_persistence",
    "max_persistence",
    "persistence_entropy",
    "n_bars_h0",
    "n_bars_h1",
    "cross_head_wasserstein_mean",
    "cross_layer_wasserstein_mean",
]


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SampleResult:
    prompt: str
    expected_answer: str
    generated_text: str
    extracted_answer: str
    is_correct: bool
    is_degenerate: bool
    n_tokens: int
    signal: AttentionTopologySignal
    error: str | None = None


def _extract_metrics(signal: AttentionTopologySignal) -> dict[str, float]:
    """Extract the direct measurements from a signal into a flat dict."""
    cross_head_mean = (
        sum(signal.cross_head_wasserstein.values())
        / len(signal.cross_head_wasserstein)
        if signal.cross_head_wasserstein
        else 0.0
    )
    cross_layer_mean = (
        sum(signal.cross_layer_wasserstein) / len(signal.cross_layer_wasserstein)
        if signal.cross_layer_wasserstein
        else 0.0
    )
    return {
        "total_persistence": signal.total_persistence,
        "max_persistence": signal.max_persistence,
        "persistence_entropy": signal.persistence_entropy,
        "n_bars_h0": float(signal.n_bars_h0),
        "n_bars_h1": float(signal.n_bars_h1),
        "cross_head_wasserstein_mean": cross_head_mean,
        "cross_layer_wasserstein_mean": cross_layer_mean,
    }


# =============================================================================
# Answer extraction and correctness checking
# =============================================================================


def _extract_answer(text: str) -> str:
    """Extract numeric answer from generated text."""
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    sep = re.search(r"####\s*(.+)", text)
    if sep:
        return sep.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def _normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans.lower()


def _check_correct(generated: str, expected: str) -> bool:
    return _normalize_answer(generated) == _normalize_answer(expected)


def _is_degenerate(text: str, hit_max: bool) -> bool:
    """Detect degenerate (repetitive) output.

    Definition: text is degenerate if it hit the generation budget OR
    contains any word-span of length >= 3 that repeats >= 3 consecutive
    times. These thresholds define "degenerate" — they are operational
    definitions, not derived parameters.
    """
    if hit_max:
        return True
    words = text.split()
    # Check for any span of 3+ words repeating 3+ consecutive times
    min_span = 3
    min_repeats = 3
    for span_len in range(min_span, len(words) // min_repeats + 1):
        for start in range(len(words) - span_len * min_repeats + 1):
            pattern = words[start : start + span_len]
            count = 0
            pos = start
            while pos + span_len <= len(words):
                if words[pos : pos + span_len] == pattern:
                    count += 1
                    pos += span_len
                else:
                    break
            if count >= min_repeats:
                return True
    return False


# =============================================================================
# Experiment runner
# =============================================================================


class AttentionTopologyExperiment:

    def __init__(self, model_name: str, output_dir: Path, max_tokens: int):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.backend = None
        self.provider = None

    def _load_model(self):
        from modelcypher.adapters.activation_provider import ActivationProviderAdapter
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.backends import initialize_default_backend

        model_path = MODEL_PATHS.get(self.model_name)
        if not model_path:
            raise ValueError(f"Unknown model: {self.model_name}")

        logger.info(f"Loading model from {model_path}")
        self.backend = initialize_default_backend()
        self.model, self.tokenizer = ModelLoader(self.backend).load_model(model_path)
        self.provider = ActivationProviderAdapter(
            backend=self.backend, model_path=model_path,
        )
        logger.info("Model loaded")

    def _load_benchmark(self, n_samples: int, benchmark: str) -> list[tuple[str, str]]:
        from modelcypher.core.use_cases.curriculum.benchmark_loader import (
            BenchmarkLoader,
        )

        loader = BenchmarkLoader()
        bm = loader.load(benchmark, split="test", limit=n_samples)
        return [(s.prompt, s.answer) for s in bm.samples]

    def _format_prompt(self, prompt: str) -> str:
        return f"Q: {prompt}\nA:"

    def _generate_and_capture(
        self, prompt: str, expected_answer: str,
    ) -> SampleResult:
        """Generate answer with greedy decoding, capturing attention matrices.

        Raises on attention extraction failure — no silent fallbacks.
        """
        b = self.backend
        formatted = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted)
        if not isinstance(prompt_tokens, list):
            prompt_tokens = b.tolist(prompt_tokens)

        # Extract attention matrices during the comprehension phase.
        # Hard-fail if extraction fails — a zero-signal sample is a lie.
        attn_matrices_raw = self.provider.collect_attention_matrices(
            self.model, self.tokenizer, formatted,
        )

        if not attn_matrices_raw:
            raise RuntimeError(
                "Attention extraction returned empty — no attention layers captured"
            )

        # Convert backend arrays to Python lists for domain module
        attn_matrices: dict[int, list[list[list[float]]]] = {}
        for layer_idx, head_arrays in attn_matrices_raw.items():
            heads_list = []
            for head_arr in head_arrays:
                b.eval(head_arr)
                heads_list.append(b.tolist(head_arr))
            attn_matrices[layer_idx] = heads_list

        # Compute topology signal
        signal = compute_attention_topology(attn_matrices)

        # Generate the actual answer via greedy decoding
        current_ids = b.array([prompt_tokens])
        generated_tokens: list[int] = []
        hit_eos = False

        for _ in range(self.max_tokens):
            logits = self.model(current_ids)
            b.eval(logits)
            if logits.ndim == 3:
                next_logits = logits[0, -1, :]
            else:
                next_logits = logits[-1, :]
            next_id = int(b.to_scalar(b.argmax(next_logits)))

            if next_id == self.tokenizer.eos_token_id:
                hit_eos = True
                break

            generated_tokens.append(next_id)
            current_ids = b.array([prompt_tokens + generated_tokens])

        generated_text = self.tokenizer.decode(generated_tokens)
        extracted = _extract_answer(generated_text)
        is_correct = _check_correct(extracted, expected_answer)
        is_degen = _is_degenerate(generated_text, not hit_eos)

        return SampleResult(
            prompt=prompt,
            expected_answer=expected_answer,
            generated_text=generated_text,
            extracted_answer=extracted,
            is_correct=is_correct,
            is_degenerate=is_degen,
            n_tokens=len(generated_tokens),
            signal=signal,
        )

    def run(self, n_samples: int, benchmark: str = "gsm8k"):
        self._load_model()
        samples = self._load_benchmark(n_samples, benchmark)
        logger.info(f"Loaded {len(samples)} {benchmark} samples")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: list[SampleResult] = []

        for i, (prompt, answer) in enumerate(samples):
            t0 = time.time()
            try:
                result = self._generate_and_capture(prompt, answer)
                results.append(result)
                elapsed = time.time() - t0
                status = "CORRECT" if result.is_correct else "WRONG"
                if result.is_degenerate:
                    status = "DEGEN"
                logger.info(
                    f"[{i+1}/{len(samples)}] {status} "
                    f"(ans={result.extracted_answer!r}, "
                    f"exp={answer!r}, {elapsed:.1f}s)"
                )
            except Exception as e:
                logger.error(f"[{i+1}/{len(samples)}] HARD FAIL: {e}")
                raise

        self._analyze_and_report(results)

    def _analyze_and_report(self, results: list[SampleResult]):
        correct = [r for r in results if r.is_correct]
        incorrect = [r for r in results if not r.is_correct]
        degenerate = [r for r in results if r.is_degenerate]

        logger.info(f"\n{'='*60}")
        logger.info(
            f"RESULTS: {len(results)} total, {len(correct)} correct, "
            f"{len(incorrect)} incorrect, {len(degenerate)} degenerate"
        )
        if results:
            logger.info(f"Accuracy: {len(correct)/len(results)*100:.1f}%")

        report_lines = [
            f"# Attention Topology Calibration Report",
            f"",
            f"**Model:** {self.model_name}",
            f"**Samples:** {len(results)}",
            f"**Correct:** {len(correct)}",
            f"**Incorrect:** {len(incorrect)}",
            f"**Degenerate:** {len(degenerate)}",
            f"**Accuracy:** {len(correct)/len(results)*100:.1f}%"
            if results else "N/A",
            f"",
            f"---",
            f"",
            f"## Per-Metric Discriminative Power (correct vs incorrect)",
            f"",
            f"All measurements are direct from persistence diagram birth/death pairs.",
            f"Cohen's d is reported without threshold — the data speaks for itself.",
            f"",
        ]

        # Per-metric Cohen's d — report values, no threshold gating
        if len(correct) >= 2 and len(incorrect) >= 2:
            correct_metrics = [_extract_metrics(r.signal) for r in correct]
            incorrect_metrics = [_extract_metrics(r.signal) for r in incorrect]

            metric_stats = []
            for name in METRIC_NAMES:
                c_vals = [m[name] for m in correct_metrics]
                i_vals = [m[name] for m in incorrect_metrics]
                d = cohens_d_two_groups(c_vals, i_vals)
                mean_c = sum(c_vals) / len(c_vals)
                mean_i = sum(i_vals) / len(i_vals)
                metric_stats.append({
                    "name": name,
                    "cohens_d": d,
                    "mean_correct": mean_c,
                    "mean_incorrect": mean_i,
                })

            # Sort by absolute Cohen's d
            metric_stats.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

            report_lines.append("| Metric | d | Mean(correct) | Mean(incorrect) |")
            report_lines.append("|--------|---|---------------|-----------------|")
            for ms in metric_stats:
                report_lines.append(
                    f"| {ms['name']} | {ms['cohens_d']:.4f} "
                    f"| {ms['mean_correct']:.4f} | {ms['mean_incorrect']:.4f} |"
                )
        else:
            report_lines.append(
                f"Insufficient samples for per-metric analysis "
                f"(need >= 2 per group, got {len(correct)} correct "
                f"and {len(incorrect)} incorrect)"
            )

        # Cross-layer evolution correlation with correctness
        report_lines.append(f"")
        report_lines.append(f"## Cross-Layer Wasserstein Evolution")
        report_lines.append(f"")
        if results and results[0].signal.cross_layer_wasserstein:
            n_transitions = len(results[0].signal.cross_layer_wasserstein)
            for t in range(n_transitions):
                vals = [
                    r.signal.cross_layer_wasserstein[t]
                    for r in results
                    if len(r.signal.cross_layer_wasserstein) > t
                ]
                labels = [
                    1.0 if r.is_correct else 0.0
                    for r in results
                    if len(r.signal.cross_layer_wasserstein) > t
                ]
                if len(vals) >= 4:
                    rho = spearman_correlation(vals, labels)
                    report_lines.append(
                        f"- Transition {t}: Spearman rho = {rho:.4f}"
                    )

        # Save report
        report_text = "\n".join(report_lines) + "\n"
        report_path = self.output_dir / "CALIBRATION_REPORT.md"
        report_path.write_text(report_text)
        logger.info(f"Report written to {report_path}")

        # Save raw data
        raw_data = []
        for r in results:
            entry = {
                "prompt": r.prompt,
                "expected_answer": r.expected_answer,
                "generated_text": r.generated_text,
                "extracted_answer": r.extracted_answer,
                "is_correct": r.is_correct,
                "is_degenerate": r.is_degenerate,
                "n_tokens": r.n_tokens,
                "metrics": _extract_metrics(r.signal),
                "cross_head_wasserstein": {
                    str(k): v
                    for k, v in r.signal.cross_head_wasserstein.items()
                },
                "cross_layer_wasserstein": r.signal.cross_layer_wasserstein,
                "error": r.error,
            }
            raw_data.append(entry)

        data_path = self.output_dir / "raw_results.json"
        data_path.write_text(json.dumps(raw_data, indent=2))
        logger.info(f"Raw data written to {data_path}")

        print(f"\n{'='*60}")
        print(report_text)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Attention topology calibration experiment",
    )
    parser.add_argument(
        "--model", default="LFM2-350M", choices=list(MODEL_PATHS.keys()),
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument(
        "--benchmark", default="gsm8k", choices=["gsm8k", "arithmetic"],
    )
    parser.add_argument(
        "--max-tokens", type=int, required=True,
        help="Maximum tokens to generate per sample (user must specify)",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/attention_topology/calibration/",
    )

    args = parser.parse_args()
    output_dir = Path(args.output) / args.model

    experiment = AttentionTopologyExperiment(
        model_name=args.model,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
    )
    experiment.run(n_samples=args.samples, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
