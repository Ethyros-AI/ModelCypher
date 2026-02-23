#!/usr/bin/env python3
"""Attention topology calibration experiment.

Extracts attention weight matrices during inference, computes topological
features (Betti curves, barcode statistics, cross-head/layer agreement),
and calibrates a classifier to predict answer correctness.

This is the PROVEN approach (TOHA AUROC 0.89, Kostenok +16% AUC-ARC)
replacing the DISPROVEN Δβ₁ hidden-state point cloud method.

Usage:
    # Smoke test (10 samples)
    poetry run python scripts/attention_topology_calibration.py \
        --model LFM2-350M --samples 10 \
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
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
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
    signal: AttentionTopologySignal | None = None
    feature_vector: list[float] = field(default_factory=list)
    error: str | None = None


# =============================================================================
# Answer extraction and correctness checking
# =============================================================================


def _extract_answer(text: str) -> str:
    """Extract numeric answer from generated text."""
    # Try boxed answer first
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    # Try "#### answer" format (GSM8K)
    sep = re.search(r"####\s*(.+)", text)
    if sep:
        return sep.group(1).strip()
    # Try last number
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
    if hit_max:
        return True
    words = text.split()
    for span_len in range(3, len(words) // 3 + 1):
        for start in range(len(words) - span_len * 3 + 1):
            pattern = words[start : start + span_len]
            count = 0
            pos = start
            while pos + span_len <= len(words):
                if words[pos : pos + span_len] == pattern:
                    count += 1
                    pos += span_len
                else:
                    break
            if count >= 3:
                return True
    return False


# =============================================================================
# Experiment runner
# =============================================================================


class AttentionTopologyExperiment:

    def __init__(self, model_name: str, output_dir: Path, max_tokens: int = 256):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.backend = None
        self.provider = None

    def _load_model(self):
        from modelcypher.adapters.activation_provider import ActivationProviderAdapter
        from modelcypher.adapters.model_loader import load_model_for_training
        from modelcypher.backends import initialize_default_backend

        model_path = MODEL_PATHS.get(self.model_name)
        if not model_path:
            raise ValueError(f"Unknown model: {self.model_name}")

        logger.info(f"Loading model from {model_path}")
        self.backend = initialize_default_backend()
        self.model, self.tokenizer = load_model_for_training(model_path)
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
        """Generate answer with greedy decoding, capturing attention matrices."""
        b = self.backend
        formatted = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted)
        if not isinstance(prompt_tokens, list):
            prompt_tokens = b.tolist(prompt_tokens)

        # For attention topology: run on the FULL prompt to get attention patterns
        # during the model's initial processing. This is the "comprehension" phase
        # where the model builds its internal representation.
        try:
            attn_matrices_raw = self.provider.collect_attention_matrices(
                self.model, self.tokenizer, formatted,
            )
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            attn_matrices_raw = {}

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

        # Now generate the actual answer via greedy decoding
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
            feature_vector=signal.feature_vector(),
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
                logger.warning(f"[{i+1}/{len(samples)}] FAILED: {e}")
                results.append(SampleResult(
                    prompt=prompt, expected_answer=answer,
                    generated_text="", extracted_answer="",
                    is_correct=False, is_degenerate=True, n_tokens=0,
                    error=str(e),
                ))

        self._analyze_and_report(results)

    def _analyze_and_report(self, results: list[SampleResult]):
        valid = [r for r in results if r.signal is not None and r.error is None]
        correct = [r for r in valid if r.is_correct]
        incorrect = [r for r in valid if not r.is_correct]
        degenerate = [r for r in valid if r.is_degenerate]
        coherent = [r for r in valid if not r.is_degenerate]

        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS: {len(valid)} valid, {len(correct)} correct, "
                     f"{len(incorrect)} incorrect, {len(degenerate)} degenerate")
        logger.info(f"Accuracy: {len(correct)/len(valid)*100:.1f}%" if valid else "N/A")

        # Per-feature discriminative analysis
        feature_names = AttentionTopologySignal.feature_names()
        report_lines = [
            f"# Attention Topology Calibration Report",
            f"",
            f"**Model:** {self.model_name}",
            f"**Samples:** {len(valid)}",
            f"**Correct:** {len(correct)}",
            f"**Incorrect:** {len(incorrect)}",
            f"**Degenerate:** {len(degenerate)}",
            f"**Accuracy:** {len(correct)/len(valid)*100:.1f}%" if valid else "N/A",
            f"",
            f"---",
            f"",
            f"## Per-Feature Discriminative Power (correct vs incorrect)",
            f"",
        ]

        feature_stats = []
        if len(correct) >= 2 and len(incorrect) >= 2:
            for fi, fname in enumerate(feature_names):
                correct_vals = [r.feature_vector[fi] for r in correct]
                incorrect_vals = [r.feature_vector[fi] for r in incorrect]
                d = cohens_d_two_groups(correct_vals, incorrect_vals)
                mean_c = sum(correct_vals) / len(correct_vals)
                mean_i = sum(incorrect_vals) / len(incorrect_vals)
                feature_stats.append({
                    "feature": fname,
                    "cohens_d": d,
                    "mean_correct": mean_c,
                    "mean_incorrect": mean_i,
                    "direction": "correct > incorrect" if mean_c > mean_i else "incorrect > correct",
                })
                report_lines.append(
                    f"| {fname} | d={d:.4f} | "
                    f"correct={mean_c:.4f} | incorrect={mean_i:.4f} |"
                )

            # Sort by absolute Cohen's d
            feature_stats.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
            report_lines.append("")
            report_lines.append("## Features Ranked by |d|")
            report_lines.append("")
            for fs in feature_stats:
                marker = "**" if abs(fs["cohens_d"]) >= 0.3 else ""
                report_lines.append(
                    f"- {marker}{fs['feature']}: d={fs['cohens_d']:.4f} "
                    f"({fs['direction']}){marker}"
                )

            # Count features with d >= 0.3
            n_sig = sum(1 for fs in feature_stats if abs(fs["cohens_d"]) >= 0.3)
            report_lines.append(f"")
            report_lines.append(
                f"**{n_sig}/{len(feature_names)} features with |d| >= 0.3**"
            )
        else:
            report_lines.append(
                "Insufficient samples for per-feature analysis "
                f"(need >= 2 correct AND 2 incorrect, got {len(correct)} and {len(incorrect)})"
            )

        # Degenerate vs coherent analysis
        report_lines.append(f"")
        report_lines.append(f"## Degenerate vs Coherent Separation")
        report_lines.append(f"")
        if len(degenerate) >= 2 and len(coherent) >= 2:
            for fi, fname in enumerate(feature_names):
                degen_vals = [r.feature_vector[fi] for r in degenerate]
                coher_vals = [r.feature_vector[fi] for r in coherent]
                d = cohens_d_two_groups(coher_vals, degen_vals)
                report_lines.append(f"| {fname} | d={d:.4f} |")
        else:
            report_lines.append(
                f"Insufficient (degenerate={len(degenerate)}, coherent={len(coherent)})"
            )

        # Cross-layer evolution correlation with correctness
        report_lines.append(f"")
        report_lines.append(f"## Cross-Layer Wasserstein Evolution")
        report_lines.append(f"")
        if valid and valid[0].signal and valid[0].signal.cross_layer_wasserstein:
            n_transitions = len(valid[0].signal.cross_layer_wasserstein)
            for t in range(n_transitions):
                vals = [r.signal.cross_layer_wasserstein[t] for r in valid
                        if r.signal and len(r.signal.cross_layer_wasserstein) > t]
                labels = [1.0 if r.is_correct else 0.0 for r in valid
                          if r.signal and len(r.signal.cross_layer_wasserstein) > t]
                if len(vals) >= 4:
                    rho = spearman_correlation(vals, labels)
                    report_lines.append(
                        f"- Transition {t}: Spearman rho = {rho:.4f}"
                    )

        # Save results
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
                "feature_vector": r.feature_vector,
                "error": r.error,
            }
            if r.signal:
                entry["signal_summary"] = {
                    "betti_curve_width": r.signal.betti_curve_width,
                    "betti_curve_centroid": r.signal.betti_curve_centroid,
                    "betti_curve_peak": r.signal.betti_curve_peak,
                    "betti_curve_spread": r.signal.betti_curve_spread,
                    "total_persistence": r.signal.total_persistence,
                    "max_persistence": r.signal.max_persistence,
                    "persistence_entropy": r.signal.persistence_entropy,
                    "n_bars_h0": r.signal.n_bars_h0,
                    "n_bars_h1": r.signal.n_bars_h1,
                    "cross_head_wasserstein": {
                        str(k): v
                        for k, v in r.signal.cross_head_wasserstein.items()
                    },
                    "cross_layer_wasserstein": r.signal.cross_layer_wasserstein,
                }
            raw_data.append(entry)

        data_path = self.output_dir / "raw_results.json"
        data_path.write_text(json.dumps(raw_data, indent=2))
        logger.info(f"Raw data written to {data_path}")

        # Print summary to console
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
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "arithmetic"])
    parser.add_argument("--max-tokens", type=int, default=256)
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
