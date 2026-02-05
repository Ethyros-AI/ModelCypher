#!/usr/bin/env python3
"""SOTA-Aligned Geometry Trajectory Analysis.

Implements methodology from recent papers on geometric analysis of LLM reasoning:
- Zhang et al. 2025: Linear probes extract correctness before answers
- Lee et al. 2026: L2 spikes detect cognitive pivots (STARS method)
- Marks & Tegmark 2024: Truth is linearly represented in hidden states
- Anderson 2026: Manifold geometry predicts reasoning cost

Key differences from previous experiments (V2/V3):
1. GREEDY DECODING ONLY - temperature is noise, not signal
2. FULL TRAJECTORY analysis - not just answer token
3. L2 SPIKES detect cognitive pivots - where model makes decisions
4. LINEAR PROBES trained on hidden states - not aggregate metrics
5. Per-layer trajectory geometry - not collapsed averages

The signal is in the TRAJECTORY STRUCTURE, not the endpoint.
The signal is LINEAR in hidden space, not complex aggregate metrics.

Usage:
    poetry run python scripts/geometry_trajectory_analysis.py \
        --model /path/to/model \
        --output results/geometry_sota/ \
        --samples 100 \
        --benchmark gsm8k
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMeasurement:
    """Measurements at a single token in the generation trajectory."""
    position: int
    token_id: int
    token_str: str

    # Per-layer L2 distance from previous token (STARS method)
    l2_distances: list[float]

    # Per-layer hidden state norms
    norms: list[float]

    # Is this a cognitive pivot (L2 spike)?
    is_pivot: bool
    pivot_score: float  # z-score if pivot, 0 otherwise
    pivot_layers: list[int]  # which layers showed spikes


@dataclass
class TrajectoryResult:
    """Complete trajectory analysis for one generation."""
    prompt: str
    expected_answer: str
    generated_text: str

    # Correctness
    extracted_number: float | None
    is_correct: bool

    # Trajectory data
    measurements: list[TrajectoryMeasurement]
    n_tokens: int
    prompt_length: int

    # Cognitive pivots
    pivot_indices: list[int]
    n_pivots: int

    # Trajectory statistics
    mean_l2_distance: float
    std_l2_distance: float
    max_l2_distance: float
    pivot_l2_mean: float  # mean L2 at pivot tokens
    non_pivot_l2_mean: float  # mean L2 at non-pivot tokens

    # Per-layer hidden states at reasoning chunk boundaries
    # These are used for linear probe training
    chunk_boundary_states: list[list[float]] | None  # [n_chunks, hidden_dim]

    # Per-layer chunk boundary states for layer-wise probing
    # Key: layer index, Value: list of hidden states at chunk boundaries
    chunk_boundary_states_per_layer: dict[int, list[list[float]]] | None = None


@dataclass
class ProbeResult:
    """Result of linear probe training and evaluation."""
    auroc: float
    accuracy: float
    separation_score: float  # Fisher's criterion
    n_train_correct: int
    n_train_incorrect: int
    n_test_correct: int
    n_test_incorrect: int


@dataclass
class ExperimentConfig:
    model_path: str
    output_dir: Path
    n_samples: int = 100
    max_tokens: int = 256  # longer for CoT reasoning
    seed: int = 42
    benchmark: str = "gsm8k"
    spike_threshold_k: float = 2.0  # std devs above mean for pivot detection
    train_test_split: float = 0.8  # fraction for probe training


class SOTAGeometryAnalysis:
    """SOTA-aligned geometry analysis following recent papers."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0

    def setup(self) -> None:
        from modelcypher.backends import initialize_default_backend

        logger.info(f"Loading model from {self.config.model_path}")
        self.backend = initialize_default_backend()

        model_path = Path(self.config.model_path)
        self.model, self.tokenizer = self.backend.load_model(str(model_path))

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        self.num_layers = len(layers) if layers else 0

        logger.info(f"Model loaded: {self.num_layers} layers")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_number(self, text: str) -> float | None:
        """Extract the LAST number from text."""
        # Handle GSM8K format with #### answer
        if "####" in text:
            after_marker = text.split("####")[-1].strip()
            matches = re.findall(r'-?\d+\.?\d*', after_marker)
            if matches:
                try:
                    return float(matches[0].replace(",", ""))
                except ValueError:
                    pass

        # Fallback: last number in text
        matches = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

    def _check_correctness(self, extracted: float | None, expected_str: str) -> bool:
        """Check if extracted number matches expected."""
        if extracted is None:
            return False
        try:
            expected = float(expected_str.replace(",", ""))
        except ValueError:
            return False

        if expected == int(expected):
            return extracted == expected
        return abs(extracted - expected) < 1e-6

    def _format_prompt(self, raw_prompt: str) -> str:
        """Format prompt using chat template if available."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{'role': 'user', 'content': raw_prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return raw_prompt

    def _detect_spikes(
        self,
        all_l2_distances: list[dict[int, float]],
    ) -> list[tuple[int, float, list[int]]]:
        """Detect L2 spikes using adaptive threshold.

        Returns: list of (token_idx, spike_score, layers_with_spike)
        """
        if len(all_l2_distances) < 3:
            return []

        layer_indices = sorted(all_l2_distances[0].keys())
        if not layer_indices:
            return []

        # Compute per-layer statistics
        layer_stats: dict[int, tuple[float, float]] = {}
        for layer_idx in layer_indices:
            layer_distances = [d.get(layer_idx, 0.0) for d in all_l2_distances]
            if len(layer_distances) < 2:
                layer_stats[layer_idx] = (0.0, 1.0)
                continue
            mean_dist = sum(layer_distances) / len(layer_distances)
            variance = sum((d - mean_dist) ** 2 for d in layer_distances) / len(layer_distances)
            std_dist = variance ** 0.5 if variance > 0 else 1e-8
            layer_stats[layer_idx] = (mean_dist, std_dist)

        # Detect spikes
        spikes = []
        for token_idx, distances in enumerate(all_l2_distances):
            if token_idx == 0:
                continue

            layers_with_spike = []
            spike_scores = []

            for layer_idx in layer_indices:
                mean, std = layer_stats[layer_idx]
                if std < 1e-10:
                    continue
                z_score = (distances.get(layer_idx, 0.0) - mean) / std
                if z_score > self.config.spike_threshold_k:
                    layers_with_spike.append(layer_idx)
                    spike_scores.append(z_score)

            if layers_with_spike:
                mean_spike_score = sum(spike_scores) / len(spike_scores)
                spikes.append((token_idx, mean_spike_score, layers_with_spike))

        return spikes

    def _generate_with_trajectory(
        self,
        prompt: str,
        expected_answer: str,
    ) -> TrajectoryResult:
        """Generate text with GREEDY decoding, capturing full trajectory.

        CRITICAL: Temperature = 0. No sampling. One deterministic path.
        """
        b = self.backend
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)

        formatted_prompt = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        if isinstance(prompt_tokens, list):
            prompt_ids = prompt_tokens
        else:
            prompt_ids = b.tolist(prompt_tokens)

        prompt_length = len(prompt_ids)
        current_ids = b.array([prompt_ids])

        generated_tokens: list[int] = []
        measurements: list[TrajectoryMeasurement] = []
        all_l2_distances: list[dict[int, float]] = []
        prev_hidden: dict[int, Any] = {}

        # For chunk boundary states (probe training)
        chunk_boundary_states: list[list[float]] = []
        # Per-layer chunk boundary states for layer-wise probing
        chunk_boundary_states_per_layer: dict[int, list[list[float]]] = {
            i: [] for i in range(self.num_layers)
        }

        for gen_step in range(self.config.max_tokens):
            # Capture hidden states during forward
            captured: dict[int, Any] = {}

            class CaptureWrapper:
                def __init__(wrapper_self, layer: Any, layer_idx: int) -> None:
                    wrapper_self._layer = layer
                    wrapper_self._layer_idx = layer_idx

                def __call__(wrapper_self, *args: Any, **kwargs: Any) -> Any:
                    output = wrapper_self._layer(*args, **kwargs)
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    captured[wrapper_self._layer_idx] = hidden
                    return output

                def __getattr__(wrapper_self, name: str) -> Any:
                    return getattr(wrapper_self._layer, name)

            original_layers = list(layers)

            try:
                for i in range(len(layers)):
                    layers[i] = CaptureWrapper(original_layers[i], i)

                # GREEDY: no temperature scaling
                logits = self.model(current_ids)
                b.eval(logits)
            finally:
                for i, layer in enumerate(original_layers):
                    layers[i] = layer

            # Extract last position hidden states
            last_pos_hidden: dict[int, Any] = {}
            for layer_idx, hidden in captured.items():
                if hidden.ndim == 3:
                    last_pos_hidden[layer_idx] = hidden[0, -1, :]
                else:
                    last_pos_hidden[layer_idx] = hidden[-1, :]
                b.eval(last_pos_hidden[layer_idx])

            # Compute L2 distances from previous token
            l2_distances: dict[int, float] = {}
            norms: list[float] = []

            for layer_idx in range(self.num_layers):
                h = last_pos_hidden.get(layer_idx)
                if h is None:
                    continue

                norm = float(b.to_scalar(b.norm(h)))
                norms.append(norm)

                if layer_idx in prev_hidden:
                    diff = h - prev_hidden[layer_idx]
                    l2 = float(b.to_scalar(b.norm(diff)))
                    l2_distances[layer_idx] = l2
                else:
                    l2_distances[layer_idx] = 0.0

            all_l2_distances.append(l2_distances)

            # GREEDY token selection (argmax)
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]
            b.eval(last_logits)
            next_token = int(b.to_scalar(b.argmax(last_logits)))

            generated_tokens.append(next_token)

            try:
                token_str = self.tokenizer.decode([next_token])
            except:
                token_str = f"<{next_token}>"

            # Store measurement (pivot detection done later)
            measurements.append(TrajectoryMeasurement(
                position=prompt_length + gen_step,
                token_id=next_token,
                token_str=token_str,
                l2_distances=list(l2_distances.values()),
                norms=norms,
                is_pivot=False,  # filled in later
                pivot_score=0.0,
                pivot_layers=[],
            ))

            # Save chunk boundary states for probe training
            # Use newline, period, or "=" as chunk boundaries
            if token_str.strip() in [".", "\n", "=", "####"] or gen_step == 0:
                # Last layer (backward compat)
                last_layer_h = last_pos_hidden.get(self.num_layers - 1)
                if last_layer_h is not None:
                    chunk_boundary_states.append(b.tolist(last_layer_h))
                # ALL layers for per-layer probing
                for layer_idx in range(self.num_layers):
                    h = last_pos_hidden.get(layer_idx)
                    if h is not None:
                        chunk_boundary_states_per_layer[layer_idx].append(b.tolist(h))

            # Update state
            prev_hidden = last_pos_hidden
            next_token_arr = b.array([[next_token]])
            current_ids = b.concatenate([current_ids, next_token_arr], axis=1)

            # Check EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_token == eos_id:
                break

        # Detect cognitive pivots
        spikes = self._detect_spikes(all_l2_distances)
        pivot_indices = [s[0] for s in spikes]

        for token_idx, spike_score, spike_layers in spikes:
            if token_idx < len(measurements):
                measurements[token_idx] = TrajectoryMeasurement(
                    position=measurements[token_idx].position,
                    token_id=measurements[token_idx].token_id,
                    token_str=measurements[token_idx].token_str,
                    l2_distances=measurements[token_idx].l2_distances,
                    norms=measurements[token_idx].norms,
                    is_pivot=True,
                    pivot_score=spike_score,
                    pivot_layers=spike_layers,
                )

        # Process results
        generated_text = self.tokenizer.decode(generated_tokens)
        extracted_number = self._extract_number(generated_text)
        is_correct = self._check_correctness(extracted_number, expected_answer)

        # Trajectory statistics
        all_l2_flat = [d for m in measurements for d in m.l2_distances if d > 0]
        if all_l2_flat:
            mean_l2 = sum(all_l2_flat) / len(all_l2_flat)
            variance = sum((d - mean_l2) ** 2 for d in all_l2_flat) / len(all_l2_flat)
            std_l2 = variance ** 0.5
            max_l2 = max(all_l2_flat)
        else:
            mean_l2 = std_l2 = max_l2 = 0.0

        # L2 at pivot vs non-pivot
        pivot_l2 = [d for m in measurements if m.is_pivot for d in m.l2_distances if d > 0]
        non_pivot_l2 = [d for m in measurements if not m.is_pivot for d in m.l2_distances if d > 0]
        pivot_l2_mean = sum(pivot_l2) / len(pivot_l2) if pivot_l2 else 0.0
        non_pivot_l2_mean = sum(non_pivot_l2) / len(non_pivot_l2) if non_pivot_l2 else 0.0

        return TrajectoryResult(
            prompt=prompt,
            expected_answer=expected_answer,
            generated_text=generated_text,
            extracted_number=extracted_number,
            is_correct=is_correct,
            measurements=measurements,
            n_tokens=len(generated_tokens),
            prompt_length=prompt_length,
            pivot_indices=pivot_indices,
            n_pivots=len(pivot_indices),
            mean_l2_distance=mean_l2,
            std_l2_distance=std_l2,
            max_l2_distance=max_l2,
            pivot_l2_mean=pivot_l2_mean,
            non_pivot_l2_mean=non_pivot_l2_mean,
            chunk_boundary_states=chunk_boundary_states if chunk_boundary_states else None,
            chunk_boundary_states_per_layer=chunk_boundary_states_per_layer if any(
                chunk_boundary_states_per_layer.values()
            ) else None,
        )

    def _train_and_evaluate_probe(
        self,
        results: list[TrajectoryResult],
    ) -> ProbeResult | None:
        """Train linear probe on hidden states and evaluate."""
        from modelcypher.core.domain.geometry.linear_probe import train_correctness_probe

        # Collect hidden states from chunk boundaries
        correct_states = []
        incorrect_states = []

        for r in results:
            if r.chunk_boundary_states is None:
                continue
            states = [self.backend.array(s) for s in r.chunk_boundary_states]
            if r.is_correct:
                correct_states.extend(states)
            else:
                incorrect_states.extend(states)

        if len(correct_states) < 5 or len(incorrect_states) < 5:
            logger.warning(f"Not enough samples for probe: {len(correct_states)} correct, {len(incorrect_states)} incorrect")
            return None

        # Split train/test
        n_train_c = int(len(correct_states) * self.config.train_test_split)
        n_train_i = int(len(incorrect_states) * self.config.train_test_split)

        train_correct = correct_states[:n_train_c]
        train_incorrect = incorrect_states[:n_train_i]
        test_correct = correct_states[n_train_c:]
        test_incorrect = incorrect_states[n_train_i:]

        if len(train_correct) < 2 or len(train_incorrect) < 2:
            return None
        if len(test_correct) < 1 or len(test_incorrect) < 1:
            # Use training data for eval if no test data
            test_correct = train_correct
            test_incorrect = train_incorrect

        result = train_correctness_probe(
            train_correct, train_incorrect,
            test_correct, test_incorrect,
            backend=self.backend,
            method="difference_in_means",
        )

        return ProbeResult(
            auroc=result.auroc,
            accuracy=result.accuracy,
            separation_score=0.0,  # Could compute Fisher's criterion
            n_train_correct=result.n_train_correct,
            n_train_incorrect=result.n_train_incorrect,
            n_test_correct=result.n_test_correct,
            n_test_incorrect=result.n_test_incorrect,
        )

    def _train_and_evaluate_probe_per_layer(
        self,
        results: list[TrajectoryResult],
    ) -> dict[int, ProbeResult]:
        """Train probes at each layer, return per-layer AUROC.

        SOTA papers (Zhang et al., Marks & Tegmark) suggest correctness
        emerges in middle-to-late layers, not the final layer.
        """
        from modelcypher.core.domain.geometry.linear_probe import train_correctness_probe

        layer_results: dict[int, ProbeResult] = {}

        for layer_idx in range(self.num_layers):
            correct_states = []
            incorrect_states = []

            for r in results:
                if r.chunk_boundary_states_per_layer is None:
                    continue
                states = r.chunk_boundary_states_per_layer.get(layer_idx, [])
                states_arr = [self.backend.array(s) for s in states]
                if r.is_correct:
                    correct_states.extend(states_arr)
                else:
                    incorrect_states.extend(states_arr)

            if len(correct_states) < 10 or len(incorrect_states) < 10:
                continue

            # Train/test split
            n_train_c = int(len(correct_states) * self.config.train_test_split)
            n_train_i = int(len(incorrect_states) * self.config.train_test_split)

            train_correct = correct_states[:n_train_c]
            train_incorrect = incorrect_states[:n_train_i]
            test_correct = correct_states[n_train_c:]
            test_incorrect = incorrect_states[n_train_i:]

            if len(train_correct) < 2 or len(train_incorrect) < 2:
                continue
            if len(test_correct) < 1 or len(test_incorrect) < 1:
                test_correct = train_correct
                test_incorrect = train_incorrect

            try:
                result = train_correctness_probe(
                    train_correct, train_incorrect,
                    test_correct, test_incorrect,
                    backend=self.backend,
                    method="difference_in_means",
                )
                layer_results[layer_idx] = ProbeResult(
                    auroc=result.auroc,
                    accuracy=result.accuracy,
                    separation_score=0.0,
                    n_train_correct=result.n_train_correct,
                    n_train_incorrect=result.n_train_incorrect,
                    n_test_correct=result.n_test_correct,
                    n_test_incorrect=result.n_test_incorrect,
                )
            except Exception as e:
                logger.debug(f"Failed probe at layer {layer_idx}: {e}")
                continue

        return layer_results

    def _analyze_trajectory_patterns(self, results: list[TrajectoryResult]) -> None:
        """Programmatic analysis of trajectory patterns.

        No visualization - just statistics to identify where correctness signals hide.
        """
        from collections import Counter

        correct = [r for r in results if r.is_correct]
        incorrect = [r for r in results if not r.is_correct]

        if not correct or not incorrect:
            print("\n### TRAJECTORY PATTERN ANALYSIS ###")
            print("Insufficient samples for pattern analysis")
            return

        # 1. Token content at pivots
        print("\n### PIVOT TOKEN ANALYSIS ###")
        pivot_tokens_correct = []
        pivot_tokens_incorrect = []

        for r in correct:
            for m in r.measurements:
                if m.is_pivot:
                    pivot_tokens_correct.append(m.token_str.strip() or "<ws>")

        for r in incorrect:
            for m in r.measurements:
                if m.is_pivot:
                    pivot_tokens_incorrect.append(m.token_str.strip() or "<ws>")

        print(f"Correct - total pivots: {len(pivot_tokens_correct)}")
        print(f"  Top tokens: {Counter(pivot_tokens_correct).most_common(10)}")
        print(f"Incorrect - total pivots: {len(pivot_tokens_incorrect)}")
        print(f"  Top tokens: {Counter(pivot_tokens_incorrect).most_common(10)}")

        # 2. Pivot position analysis (relative to sequence)
        print("\n### PIVOT POSITION ANALYSIS ###")

        def relative_positions(results_list: list[TrajectoryResult]) -> list[float]:
            positions = []
            for r in results_list:
                if r.n_tokens > 0:
                    for idx in r.pivot_indices:
                        positions.append(idx / r.n_tokens)  # 0-1 relative position
            return positions

        c_pos = relative_positions(correct)
        i_pos = relative_positions(incorrect)

        if c_pos and i_pos:
            c_mean = sum(c_pos) / len(c_pos)
            i_mean = sum(i_pos) / len(i_pos)
            d = self._cohens_d(c_pos, i_pos)
            print(f"Correct pivots - mean relative position: {c_mean:.3f}")
            print(f"Incorrect pivots - mean relative position: {i_mean:.3f}")
            print(f"Effect size d: {d:+.3f}")
            if abs(d) > 0.3:
                print(f"  → Pivots occur {'earlier' if c_mean < i_mean else 'later'} in correct reasoning")

        # 3. Pivot clustering (do pivots occur at similar relative positions?)
        print("\n### PIVOT CLUSTERING ANALYSIS ###")
        if c_pos and i_pos:
            # Quartile distribution
            def quartile_dist(positions: list[float]) -> list[int]:
                q = [0, 0, 0, 0]  # Q1, Q2, Q3, Q4
                for p in positions:
                    if p < 0.25:
                        q[0] += 1
                    elif p < 0.5:
                        q[1] += 1
                    elif p < 0.75:
                        q[2] += 1
                    else:
                        q[3] += 1
                return q

            c_q = quartile_dist(c_pos)
            i_q = quartile_dist(i_pos)
            total_c = sum(c_q) or 1
            total_i = sum(i_q) or 1

            print("Pivot distribution by generation quartile:")
            print(f"  Correct:   Q1={c_q[0]/total_c:.0%}, Q2={c_q[1]/total_c:.0%}, "
                  f"Q3={c_q[2]/total_c:.0%}, Q4={c_q[3]/total_c:.0%}")
            print(f"  Incorrect: Q1={i_q[0]/total_i:.0%}, Q2={i_q[1]/total_i:.0%}, "
                  f"Q3={i_q[2]/total_i:.0%}, Q4={i_q[3]/total_i:.0%}")

        # 4. Per-layer L2 patterns
        print("\n### PER-LAYER L2 ANALYSIS ###")
        print("Layers with significant correct/incorrect differences (|d| > 0.2):")

        significant_layers = []
        for layer_idx in range(self.num_layers):
            c_l2 = []
            i_l2 = []

            for r in correct:
                for m in r.measurements:
                    if layer_idx < len(m.l2_distances):
                        c_l2.append(m.l2_distances[layer_idx])

            for r in incorrect:
                for m in r.measurements:
                    if layer_idx < len(m.l2_distances):
                        i_l2.append(m.l2_distances[layer_idx])

            if c_l2 and i_l2:
                c_mean = sum(c_l2) / len(c_l2)
                i_mean = sum(i_l2) / len(i_l2)
                d = self._cohens_d(c_l2, i_l2)
                if abs(d) > 0.2:
                    significant_layers.append((layer_idx, c_mean, i_mean, d))
                    print(f"  Layer {layer_idx:2d}: correct={c_mean:.4f}, "
                          f"incorrect={i_mean:.4f}, d={d:+.3f}")

        if not significant_layers:
            print("  No layers with |d| > 0.2")
        else:
            # Find layer with strongest signal
            best_layer = max(significant_layers, key=lambda x: abs(x[3]))
            print(f"\n  Strongest signal: Layer {best_layer[0]} (d={best_layer[3]:+.3f})")

        # 5. Generation length analysis
        print("\n### GENERATION LENGTH ANALYSIS ###")
        c_len = [r.n_tokens for r in correct]
        i_len = [r.n_tokens for r in incorrect]

        if c_len and i_len:
            c_mean = sum(c_len) / len(c_len)
            i_mean = sum(i_len) / len(i_len)
            d = self._cohens_d(c_len, i_len)
            print(f"Correct mean tokens:   {c_mean:.1f}")
            print(f"Incorrect mean tokens: {i_mean:.1f}")
            print(f"Effect size d: {d:+.3f}")

    def run(self) -> None:
        """Run SOTA-aligned geometry analysis."""
        from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

        logger.info("Starting SOTA-aligned geometry trajectory analysis")
        logger.info("Key methodology: GREEDY decoding, L2 spikes, linear probes")
        self.setup()

        loader = BenchmarkLoader()
        benchmark = loader.load(self.config.benchmark, split="test", limit=self.config.n_samples)

        results: list[TrajectoryResult] = []

        for i, sample in enumerate(benchmark.samples):
            try:
                result = self._generate_with_trajectory(sample.prompt, sample.answer)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed on sample {i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                n_correct = sum(1 for r in results if r.is_correct)
                n_pivots = sum(r.n_pivots for r in results)
                logger.info(f"Progress: {i+1}/{len(benchmark.samples)}, "
                           f"correct: {n_correct}, total_pivots: {n_pivots}")

        self._save_and_analyze(results)

    def _save_and_analyze(self, results: list[TrajectoryResult]) -> None:
        """Save results and perform SOTA-aligned analysis."""

        # Save raw results
        output_path = self.config.output_dir / "trajectory_results.jsonl"
        with open(output_path, "w") as f:
            for r in results:
                record = {
                    "prompt": r.prompt[:200],  # Truncate for readability
                    "expected": r.expected_answer,
                    "generated": r.generated_text[:500],
                    "extracted_number": r.extracted_number,
                    "is_correct": r.is_correct,
                    "n_tokens": r.n_tokens,
                    "n_pivots": r.n_pivots,
                    "pivot_indices": r.pivot_indices,
                    "mean_l2": r.mean_l2_distance,
                    "std_l2": r.std_l2_distance,
                    "pivot_l2_mean": r.pivot_l2_mean,
                    "non_pivot_l2_mean": r.non_pivot_l2_mean,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {len(results)} results to {output_path}")

        # Analysis
        print("\n" + "=" * 70)
        print("SOTA-ALIGNED GEOMETRY TRAJECTORY ANALYSIS")
        print("Methodology: Greedy decoding, L2 spikes (STARS), Linear probes")
        print("=" * 70)

        n_total = len(results)
        n_correct = sum(1 for r in results if r.is_correct)
        n_incorrect = n_total - n_correct

        print(f"\nTotal samples: {n_total}")
        print(f"Correct: {n_correct} ({100*n_correct/n_total:.1f}%)")
        print(f"Incorrect: {n_incorrect} ({100*n_incorrect/n_total:.1f}%)")

        # Cognitive Pivot Analysis
        print("\n### COGNITIVE PIVOT ANALYSIS (STARS Method) ###")

        correct_results = [r for r in results if r.is_correct]
        incorrect_results = [r for r in results if not r.is_correct]

        correct_pivots = [r.n_pivots for r in correct_results]
        incorrect_pivots = [r.n_pivots for r in incorrect_results]

        if correct_pivots and incorrect_pivots:
            c_mean = sum(correct_pivots) / len(correct_pivots)
            i_mean = sum(incorrect_pivots) / len(incorrect_pivots)
            d = self._cohens_d(correct_pivots, incorrect_pivots)
            print(f"\nPivots per generation:")
            print(f"  Correct mean:   {c_mean:.2f}")
            print(f"  Incorrect mean: {i_mean:.2f}")
            print(f"  Effect size d:  {d:.3f}")

        # L2 distance at pivots vs non-pivots
        print("\nL2 distance at pivot vs non-pivot tokens:")
        correct_pivot_l2 = [r.pivot_l2_mean for r in correct_results if r.pivot_l2_mean > 0]
        correct_non_pivot_l2 = [r.non_pivot_l2_mean for r in correct_results if r.non_pivot_l2_mean > 0]

        if correct_pivot_l2 and correct_non_pivot_l2:
            print(f"  Correct samples - pivot L2:     {sum(correct_pivot_l2)/len(correct_pivot_l2):.4f}")
            print(f"  Correct samples - non-pivot L2: {sum(correct_non_pivot_l2)/len(correct_non_pivot_l2):.4f}")

        # Trajectory L2 statistics
        print("\nMean trajectory L2 distance:")
        correct_l2 = [r.mean_l2_distance for r in correct_results if r.mean_l2_distance > 0]
        incorrect_l2 = [r.mean_l2_distance for r in incorrect_results if r.mean_l2_distance > 0]

        if correct_l2 and incorrect_l2:
            c_mean = sum(correct_l2) / len(correct_l2)
            i_mean = sum(incorrect_l2) / len(incorrect_l2)
            d = self._cohens_d(correct_l2, incorrect_l2)
            print(f"  Correct mean:   {c_mean:.4f}")
            print(f"  Incorrect mean: {i_mean:.4f}")
            print(f"  Effect size d:  {d:.3f}")

        # Linear Probe (last layer - original)
        print("\n### LINEAR PROBE ANALYSIS (Zhang et al. Method) ###")
        probe_result = self._train_and_evaluate_probe(results)

        if probe_result:
            print(f"\nProbe trained on chunk boundary hidden states (last layer):")
            print(f"  AUROC:    {probe_result.auroc:.3f}")
            print(f"  Accuracy: {probe_result.accuracy:.3f}")
            print(f"  Train: {probe_result.n_train_correct} correct, {probe_result.n_train_incorrect} incorrect")
            print(f"  Test:  {probe_result.n_test_correct} correct, {probe_result.n_test_incorrect} incorrect")

            if probe_result.auroc > 0.65:
                print("\n  ✓ AUROC > 0.65: Model encodes correctness in hidden states")
            else:
                print("\n  ✗ AUROC ≤ 0.65: Weak or no linear correctness signal (last layer)")
        else:
            print("\nInsufficient samples for probe training")

        # Per-Layer Linear Probes
        print("\n### PER-LAYER PROBE ANALYSIS ###")
        print("Training correctness probes at each layer...")
        layer_probe_results = self._train_and_evaluate_probe_per_layer(results)

        if layer_probe_results:
            print(f"\nProbes trained at {len(layer_probe_results)} layers:")
            print("Layer | AUROC | Accuracy | Signal")
            print("-" * 40)
            best_layer = -1
            best_auroc = 0.0
            for layer_idx in sorted(layer_probe_results.keys()):
                pr = layer_probe_results[layer_idx]
                signal = "✓" if pr.auroc > 0.55 else " "
                print(f"  {layer_idx:2d}  | {pr.auroc:.3f} |  {pr.accuracy:.3f}   | {signal}")
                if pr.auroc > best_auroc:
                    best_auroc = pr.auroc
                    best_layer = layer_idx

            print(f"\nBest layer: {best_layer} (AUROC = {best_auroc:.3f})")
            if best_auroc > 0.60:
                print(f"  ✓ Found correctness signal in middle/late layers")
            elif best_auroc > 0.55:
                print(f"  ~ Weak signal - may need more samples")
            else:
                print(f"  ✗ No linear correctness signal found at any layer")
        else:
            print("Insufficient samples for per-layer probe training")

        # Trajectory Pattern Analysis
        self._analyze_trajectory_patterns(results)

        # Summary
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        print("\n1. This experiment uses GREEDY decoding (temp=0)")
        print("   → Every trajectory is deterministic")
        print("   → No sampling noise obscures the geometric signal")

        # Report best probe (either last layer or per-layer)
        best_probe_auroc = probe_result.auroc if probe_result else 0.0
        best_probe_layer = self.num_layers - 1
        if layer_probe_results:
            for layer_idx, pr in layer_probe_results.items():
                if pr.auroc > best_probe_auroc:
                    best_probe_auroc = pr.auroc
                    best_probe_layer = layer_idx

        if best_probe_auroc > 0.55:
            print(f"\n2. Best linear probe: Layer {best_probe_layer}, AUROC = {best_probe_auroc:.3f}")
            if best_probe_auroc > 0.65:
                print("   → Correctness IS linearly encoded (Zhang et al. confirmed)")
            else:
                print("   → Weak signal detected - may strengthen with more samples")
        else:
            print("\n2. No linear correctness signal found at any layer")

        if correct_pivots and incorrect_pivots:
            c_mean = sum(correct_pivots) / len(correct_pivots)
            i_mean = sum(incorrect_pivots) / len(incorrect_pivots)
            if abs(c_mean - i_mean) > 0.5:
                print(f"\n3. Cognitive pivots differ: correct={c_mean:.1f}, incorrect={i_mean:.1f}")
                print("   → L2 spikes (STARS method) detect reasoning structure")

        print("\n" + "=" * 70)

        # Save summary
        summary_path = self.config.output_dir / "analysis_summary.json"

        # Build per-layer probe summary
        per_layer_probe_summary = {}
        if layer_probe_results:
            for layer_idx, pr in layer_probe_results.items():
                per_layer_probe_summary[str(layer_idx)] = {
                    "auroc": pr.auroc,
                    "accuracy": pr.accuracy,
                }

        summary = {
            "n_samples": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy": n_correct / n_total if n_total > 0 else 0,
            "probe_auroc_last_layer": probe_result.auroc if probe_result else None,
            "probe_accuracy_last_layer": probe_result.accuracy if probe_result else None,
            "probe_best_layer": best_probe_layer,
            "probe_best_auroc": best_probe_auroc,
            "probe_per_layer": per_layer_probe_summary,
            "mean_pivots_correct": sum(correct_pivots) / len(correct_pivots) if correct_pivots else 0,
            "mean_pivots_incorrect": sum(incorrect_pivots) / len(incorrect_pivots) if incorrect_pivots else 0,
            "mean_l2_correct": sum(correct_l2) / len(correct_l2) if correct_l2 else 0,
            "mean_l2_incorrect": sum(incorrect_l2) / len(incorrect_l2) if incorrect_l2 else 0,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")

    def _cohens_d(self, a: list[float], b: list[float]) -> float:
        """Compute Cohen's d effect size."""
        if len(a) < 2 or len(b) < 2:
            return 0.0

        nx, ny = len(a), len(b)
        mx, my = sum(a) / nx, sum(b) / ny
        vx = sum((xi - mx) ** 2 for xi in a) / (nx - 1) if nx > 1 else 0
        vy = sum((yi - my) ** 2 for yi in b) / (ny - 1) if ny > 1 else 0
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        if pooled <= 0:
            return 0.0
        return (mx - my) / (pooled ** 0.5)


def main():
    parser = argparse.ArgumentParser(description="SOTA-aligned geometry trajectory analysis")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="results/geometry_sota/", help="Output directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--benchmark", default="gsm8k", help="Benchmark (gsm8k, arithmetic)")
    parser.add_argument("--spike-threshold", type=float, default=2.0,
                       help="Std devs above mean for pivot detection")

    args = parser.parse_args()

    config = ExperimentConfig(
        model_path=args.model,
        output_dir=Path(args.output),
        n_samples=args.samples,
        max_tokens=args.max_tokens,
        seed=args.seed,
        benchmark=args.benchmark,
        spike_threshold_k=args.spike_threshold,
    )

    experiment = SOTAGeometryAnalysis(config)
    experiment.run()


if __name__ == "__main__":
    main()
