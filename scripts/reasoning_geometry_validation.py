#!/usr/bin/env python3
"""Cross-model validation of unified reasoning geometry signals.

Validates three signals from the reasoning_geometry module across multiple
models and benchmarks:

1. LINEAR PROBES (Zhang et al. 2025, Marks & Tegmark 2024):
   Correctness is linearly encoded in hidden states at middle layers.

2. COGNITIVE PIVOTS (Lee et al. 2026, STARS method):
   L2 spikes between consecutive token hidden states detect decision points.
   Incorrect answers have more pivots.

3. BETA-1 TOPOLOGY (persistent homology):
   Reasoning creates topological loops (beta_1 > 0) in the token manifold.
   Delta-beta_1 (late minus early) may predict correctness.

Methodology:
- GREEDY decoding only (temp=0) -- temperature is noise, not signal
- Full trajectory hidden state capture at every layer for every token
- Per-layer analysis with bootstrap CIs and permutation tests
- Cross-model replication across architectures and scales

Usage:
    # Smoke test (tiny model, small sample)
    poetry run python scripts/reasoning_geometry_validation.py \\
        --models LFM2-350M \\
        --benchmark arithmetic \\
        --samples 20 \\
        --output results/reasoning_geometry_validation/smoke/

    # Full validation (all models, both benchmarks)
    poetry run python scripts/reasoning_geometry_validation.py \\
        --output results/reasoning_geometry_validation/

References:
- Zhang et al. (2025) arXiv:2504.05419
- Marks & Tegmark (2024) arXiv:2310.06824
- Lee et al. (2026) arXiv:2601.22484
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
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
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
    "DeepSeek-R1-8B": {
        "path": f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        "architecture": "qwen3",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelConfig:
    """Model metadata for experiment."""

    name: str
    path: str
    architecture: str


@dataclass
class ExperimentConfig:
    """Global experiment configuration."""

    models: list[ModelConfig]
    benchmarks: list[str]
    n_samples: int = 500
    max_tokens: int = 256
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/reasoning_geometry_validation"))
    spike_threshold_k: float = 2.0  # Lee et al. 2026 arXiv:2601.22484 Section 3.2
    pca_dims: list[int] = field(default_factory=lambda: [20, 50, 100])
    batch_size: int = 50  # Trajectories per batch (memory management)
    max_tda_tokens: int = 50  # Subsample tokens for TDA (pure-Python Vietoris-Rips is O(n^3))


@dataclass
class TrajectoryData:
    """Raw trajectory capture for one sample."""

    prompt: str
    expected_answer: str
    generated_text: str
    extracted_answer: float | None
    is_correct: bool
    n_tokens: int
    # Per-token, per-layer hidden states as Python lists (GPU memory freed)
    # Structure: list[dict[int, list[float]]] = [token_0: {layer_0: [...], ...}, ...]
    layer_hidden_states: list[dict[int, list[float]]]
    tokens: list[str]


@dataclass
class SignalStatistics:
    """Statistics for one signal on one model x benchmark."""

    # Probe signal
    probe_auroc_by_layer: dict[int, float] = field(default_factory=dict)
    probe_auroc_ci_by_layer: dict[int, tuple[float, float]] = field(default_factory=dict)
    probe_best_layer: int = -1
    probe_best_auroc: float = 0.5

    # Pivot signal
    pivot_count_correct: list[int] = field(default_factory=list)
    pivot_count_incorrect: list[int] = field(default_factory=list)
    pivot_cohens_d: float = 0.0
    pivot_d_ci: tuple[float, float] = (0.0, 0.0)
    pivot_permutation_p: float = 1.0

    # Topology signal
    delta_beta1_correct: list[float] = field(default_factory=list)
    delta_beta1_incorrect: list[float] = field(default_factory=list)
    beta1_cohens_d: float = 0.0
    beta1_d_ci: tuple[float, float] = (0.0, 0.0)
    beta1_permutation_p: float = 1.0

    # Confound
    gen_length_correct: list[int] = field(default_factory=list)
    gen_length_incorrect: list[int] = field(default_factory=list)
    length_cohens_d: float = 0.0


@dataclass
class ModelBenchmarkResult:
    """Results for one model x benchmark combination."""

    model_name: str
    benchmark_name: str
    n_samples: int
    n_correct: int
    n_incorrect: int
    accuracy: float
    stats: SignalStatistics
    # Beta-1 PCA sensitivity sweep
    beta1_pca_sensitivity: dict[int, float] = field(default_factory=dict)  # pca_dim -> Cohen's d


# =============================================================================
# Statistics Helpers
# =============================================================================


def auroc_bootstrap_ci(
    auroc_point: float,
    correct_scores: list[float],
    incorrect_scores: list[float],
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Bootstrap CI on AUROC given per-sample probe scores.

    Simple percentile bootstrap: resample correct + incorrect, recompute AUROC.
    """
    if len(correct_scores) < 2 or len(incorrect_scores) < 2:
        return (auroc_point, auroc_point)

    if rng is None:
        rng = random.Random(42)

    aurocs = []
    for _ in range(n_bootstrap):
        c = rng.choices(correct_scores, k=len(correct_scores))
        i = rng.choices(incorrect_scores, k=len(incorrect_scores))
        aurocs.append(_compute_auroc(c, i))

    aurocs.sort()
    lo_idx = max(0, int(alpha / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap) - 1)
    return (aurocs[lo_idx], aurocs[hi_idx])


def _compute_auroc(correct_scores: list[float], incorrect_scores: list[float]) -> float:
    """Mann-Whitney U AUROC."""
    n_c, n_i = len(correct_scores), len(incorrect_scores)
    if n_c == 0 or n_i == 0:
        return 0.5

    u = 0
    for c in correct_scores:
        for i in incorrect_scores:
            if c > i:
                u += 1
            elif c == i:
                u += 0.5
    return u / (n_c * n_i)


# =============================================================================
# Main Experiment Class
# =============================================================================


class ReasoningGeometryValidation:
    """Cross-model validation of reasoning geometry signals."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0
        self._rng = random.Random(config.seed)

    # ----- Model loading / unloading -----

    def _load_model(self, model_config: ModelConfig) -> None:
        """Load model and tokenizer, detect layer count."""
        from modelcypher.backends import initialize_default_backend

        if self.backend is None:
            self.backend = initialize_default_backend()

        logger.info(f"Loading model: {model_config.name} from {model_config.path}")
        model_path = Path(model_config.path)
        self.model, self.tokenizer = self.backend.load_model(str(model_path))

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

    # ----- Benchmark loading -----

    def _load_benchmark(self, benchmark_name: str) -> list[tuple[str, str]]:
        """Load benchmark samples as (prompt, expected_answer) pairs."""
        from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

        loader = BenchmarkLoader()
        benchmark = loader.load(benchmark_name, split="test", limit=self.config.n_samples)
        return [(s.prompt, s.answer) for s in benchmark.samples]

    # ----- Trajectory generation -----

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

    def _extract_number(self, text: str) -> float | None:
        """Extract the last number from generated text."""
        # Handle GSM8K #### format
        if "####" in text:
            after = text.split("####")[-1].strip()
            matches = re.findall(r"-?\d+\.?\d*", after)
            if matches:
                try:
                    return float(matches[0].replace(",", ""))
                except ValueError:
                    pass

        # Fallback: last number
        matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
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

    def _generate_one(self, prompt: str, expected_answer: str) -> TrajectoryData:
        """Generate text with GREEDY decoding, capturing full trajectory.

        Returns TrajectoryData with hidden states converted to Python lists
        (GPU memory freed after capture).
        """
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

        for gen_step in range(self.config.max_tokens):
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
                logits = self.model(current_ids)
                b.eval(logits)
            finally:
                for i, layer in enumerate(original_layers):
                    layers[i] = layer

            # Extract last-position hidden states and convert to Python lists
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

            # Update state
            next_token_arr = b.array([[next_token]])
            current_ids = b.concatenate([current_ids, next_token_arr], axis=1)

            # Check EOS
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token == eos_id:
                break

        generated_text = self.tokenizer.decode(generated_tokens)
        extracted = self._extract_number(generated_text)
        is_correct = self._check_correctness(extracted, expected_answer)

        return TrajectoryData(
            prompt=prompt,
            expected_answer=expected_answer,
            generated_text=generated_text,
            extracted_answer=extracted,
            is_correct=is_correct,
            n_tokens=len(generated_tokens),
            layer_hidden_states=all_hidden_states,
            tokens=token_strings,
        )

    def _collect_trajectories(
        self, samples: list[tuple[str, str]]
    ) -> list[TrajectoryData]:
        """Generate trajectories in batches with intermediate saves."""
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
                logger.info(
                    f"  Progress: {i+1}/{len(samples)}, "
                    f"accuracy: {n_correct}/{len(trajectories)} "
                    f"({100*n_correct/len(trajectories):.1f}%)"
                )

            # Memory management: clear cache periodically
            if (i + 1) % self.config.batch_size == 0 and self.backend is not None:
                try:
                    self.backend.clear_cache()
                except Exception:
                    pass

        return trajectories

    # ----- Signal analysis -----

    def _analyze_signals(
        self,
        trajectories: list[TrajectoryData],
        pca_dim: int = 50,
    ) -> tuple[list[Any], Any]:
        """Run ReasoningGeometryAnalyzer on trajectories.

        Converts Python-list hidden states back to backend arrays for analysis.
        Processes trajectories individually with progress logging since TDA is slow.

        Returns (per-trajectory results, probe_signal).
        """
        from modelcypher.core.domain.geometry.reasoning_geometry import (
            ReasoningGeometryAnalyzer,
        )

        b = self.backend
        analyzer = ReasoningGeometryAnalyzer(
            backend=b,
            spike_threshold_k=self.config.spike_threshold_k,
            max_tda_points=pca_dim,
        )

        def _to_arrays(hs_seq: list[dict], toks: list[str]) -> tuple[list[dict], list[str]]:
            """Convert Python-list hidden states to backend arrays."""
            hs_arrays = []
            for token_states in hs_seq:
                layer_dict = {}
                for layer_idx, values in token_states.items():
                    layer_dict[int(layer_idx)] = b.array(values)
                hs_arrays.append(layer_dict)
            return hs_arrays, toks

        def _subsample(hs_seq: list, toks: list[str]) -> tuple[list, list[str]]:
            """Evenly subsample to max_tda_tokens for topology (O(n^3) TDA)."""
            if len(hs_seq) <= self.config.max_tda_tokens:
                return hs_seq, toks
            indices = [
                int(i * (len(hs_seq) - 1) / (self.config.max_tda_tokens - 1))
                for i in range(self.config.max_tda_tokens)
            ]
            return [hs_seq[idx] for idx in indices], [toks[idx] for idx in indices]

        # Analyze trajectories individually with progress logging.
        # CRITICAL: Pivots need the FULL consecutive token sequence (L2 between
        # consecutive tokens). Topology needs subsampled data (O(n^3)).
        # So we run them separately.
        from modelcypher.core.domain.geometry.reasoning_geometry import (
            ReasoningGeometryResult,
            TopologySignal,
            PivotSignal,
        )

        results = []
        t0 = time.time()
        for i, traj in enumerate(trajectories):
            # Pivots: full sequence (L2 distances need consecutive tokens)
            full_arrays, full_toks = _to_arrays(traj.layer_hidden_states, traj.tokens)
            pivot_signal = analyzer._compute_pivot_signal(full_arrays, full_toks)

            # Topology: subsampled sequence (Vietoris-Rips is O(n^3))
            sub_hs, sub_toks = _subsample(traj.layer_hidden_states, traj.tokens)
            sub_arrays, _ = _to_arrays(sub_hs, sub_toks)
            layer_indices = sorted(sub_arrays[0].keys()) if sub_arrays else []
            topology_signal = analyzer._compute_topology_signal(sub_arrays, layer_indices)

            n_layers = len(layer_indices)
            n_tokens = len(traj.tokens)

            results.append(ReasoningGeometryResult(
                topology=topology_signal,
                probe=None,
                pivots=pivot_signal,
                n_layers=n_layers,
                n_tokens=n_tokens,
            ))

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                per_traj = elapsed / (i + 1)
                remaining = per_traj * (len(trajectories) - i - 1)
                logger.info(
                    f"    Signal analysis: {i+1}/{len(trajectories)} "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )

        # Train per-layer probes on labeled data (fast -- no TDA)
        correct_inputs = [
            _to_arrays(t.layer_hidden_states, t.tokens)
            for t in trajectories if t.is_correct
        ]
        incorrect_inputs = [
            _to_arrays(t.layer_hidden_states, t.tokens)
            for t in trajectories if not t.is_correct
        ]

        probe_signal = None
        if correct_inputs and incorrect_inputs:
            probe_signal = analyzer._compute_probe_signal(
                correct_inputs, incorrect_inputs
            )
            # Attach probe signal to all results
            results = [
                ReasoningGeometryResult(
                    topology=r.topology,
                    probe=probe_signal,
                    pivots=r.pivots,
                    n_layers=r.n_layers,
                    n_tokens=r.n_tokens,
                )
                for r in results
            ]

        # Free arrays
        try:
            b.clear_cache()
        except Exception:
            pass

        return results, probe_signal

    # ----- Statistical analysis -----

    def _compute_statistics(
        self,
        trajectories: list[TrajectoryData],
        signal_results: list[Any],
        probe_signal: Any,
    ) -> SignalStatistics:
        """Compute full statistics: Cohen's d, bootstrap CI, permutation tests."""
        stats = SignalStatistics()

        correct = [t for t in trajectories if t.is_correct]
        incorrect = [t for t in trajectories if not t.is_correct]

        # === Probe statistics ===
        if probe_signal is not None:
            for i, layer_idx in enumerate(probe_signal.layer_indices):
                stats.probe_auroc_by_layer[layer_idx] = probe_signal.auroc_by_layer[i]
            stats.probe_best_layer = probe_signal.best_layer
            stats.probe_best_auroc = probe_signal.best_auroc

        # === Pivot statistics ===
        correct_pivots = []
        incorrect_pivots = []
        for traj, result in zip(trajectories, signal_results):
            if traj.is_correct:
                correct_pivots.append(result.pivots.pivot_count)
            else:
                incorrect_pivots.append(result.pivots.pivot_count)

        stats.pivot_count_correct = correct_pivots
        stats.pivot_count_incorrect = incorrect_pivots

        if len(correct_pivots) >= 2 and len(incorrect_pivots) >= 2:
            cp_float = [float(x) for x in correct_pivots]
            ip_float = [float(x) for x in incorrect_pivots]

            stats.pivot_cohens_d = cohens_d_two_groups(cp_float, ip_float)
            stats.pivot_d_ci = cohens_d_bootstrap_ci(
                cp_float, ip_float, n_bootstrap=1000, rng=self._rng
            )
            stats.pivot_permutation_p = permutation_test_p_value(
                cp_float, ip_float, n_permutations=1000, rng=self._rng
            )

        # === Topology statistics ===
        correct_delta_b1 = []
        incorrect_delta_b1 = []
        for traj, result in zip(trajectories, signal_results):
            if traj.is_correct:
                correct_delta_b1.append(result.topology.delta_beta1)
            else:
                incorrect_delta_b1.append(result.topology.delta_beta1)

        stats.delta_beta1_correct = correct_delta_b1
        stats.delta_beta1_incorrect = incorrect_delta_b1

        if len(correct_delta_b1) >= 2 and len(incorrect_delta_b1) >= 2:
            stats.beta1_cohens_d = cohens_d_two_groups(
                correct_delta_b1, incorrect_delta_b1
            )
            stats.beta1_d_ci = cohens_d_bootstrap_ci(
                correct_delta_b1, incorrect_delta_b1, n_bootstrap=1000, rng=self._rng
            )
            stats.beta1_permutation_p = permutation_test_p_value(
                correct_delta_b1, incorrect_delta_b1, n_permutations=1000, rng=self._rng
            )

        # === Confound: generation length ===
        stats.gen_length_correct = [t.n_tokens for t in correct]
        stats.gen_length_incorrect = [t.n_tokens for t in incorrect]

        if len(stats.gen_length_correct) >= 2 and len(stats.gen_length_incorrect) >= 2:
            stats.length_cohens_d = cohens_d_two_groups(
                [float(x) for x in stats.gen_length_correct],
                [float(x) for x in stats.gen_length_incorrect],
            )

        return stats

    def _run_beta1_pca_sensitivity(
        self,
        trajectories: list[TrajectoryData],
    ) -> dict[int, float]:
        """Sweep PCA dimensions and report Cohen's d on delta-beta1.

        Tests whether the topology signal is robust to PCA dimension choice.
        Only re-runs topology (not pivots/probes) at each PCA dimension.
        """
        from modelcypher.core.domain.geometry.reasoning_geometry import (
            ReasoningGeometryAnalyzer,
        )

        sensitivity: dict[int, float] = {}
        b = self.backend

        for pca_dim in self.config.pca_dims:
            logger.info(f"    PCA sensitivity sweep: dim={pca_dim}")
            try:
                analyzer = ReasoningGeometryAnalyzer(
                    backend=b,
                    spike_threshold_k=self.config.spike_threshold_k,
                    max_tda_points=pca_dim,
                )

                correct_db1 = []
                incorrect_db1 = []

                for i, traj in enumerate(trajectories):
                    # Subsample for TDA
                    hs_seq = traj.layer_hidden_states
                    toks = traj.tokens
                    if len(hs_seq) > self.config.max_tda_tokens:
                        indices = [
                            int(j * (len(hs_seq) - 1) / (self.config.max_tda_tokens - 1))
                            for j in range(self.config.max_tda_tokens)
                        ]
                        hs_seq = [hs_seq[idx] for idx in indices]

                    # Convert to arrays
                    hs_arrays = []
                    for token_states in hs_seq:
                        layer_dict = {}
                        for layer_idx, values in token_states.items():
                            layer_dict[int(layer_idx)] = b.array(values)
                        hs_arrays.append(layer_dict)

                    layer_indices = sorted(hs_arrays[0].keys()) if hs_arrays else []
                    topo = analyzer._compute_topology_signal(hs_arrays, layer_indices)

                    if traj.is_correct:
                        correct_db1.append(topo.delta_beta1)
                    else:
                        incorrect_db1.append(topo.delta_beta1)

                    if (i + 1) % 10 == 0 or i == 0:
                        logger.info(f"      PCA dim={pca_dim}: {i+1}/{len(trajectories)}")

                if len(correct_db1) >= 2 and len(incorrect_db1) >= 2:
                    d = cohens_d_two_groups(correct_db1, incorrect_db1)
                    sensitivity[pca_dim] = d
                else:
                    sensitivity[pca_dim] = 0.0

            except Exception as e:
                logger.warning(f"    PCA dim={pca_dim} failed: {e}")
                sensitivity[pca_dim] = 0.0

        return sensitivity

    # ----- Intermediate save/load -----

    def _save_intermediate(
        self,
        model_name: str,
        benchmark: str,
        trajectories: list[TrajectoryData],
    ) -> None:
        """Save raw trajectory data for crash recovery."""
        raw_dir = self.config.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{model_name}_{benchmark}_trajectories.jsonl"

        with open(path, "w") as f:
            for t in trajectories:
                record = {
                    "prompt": t.prompt[:200],
                    "expected": t.expected_answer,
                    "generated": t.generated_text[:500],
                    "extracted": t.extracted_answer,
                    "is_correct": t.is_correct,
                    "n_tokens": t.n_tokens,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"    Saved {len(trajectories)} trajectories to {path}")

    # ----- Main experiment loop -----

    def _run_model_benchmark(
        self,
        model_config: ModelConfig,
        benchmark_name: str,
    ) -> ModelBenchmarkResult:
        """Run experiment for one model x benchmark combination."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_config.name}, Benchmark: {benchmark_name}")
        logger.info(f"{'='*60}")

        # Load benchmark
        samples = self._load_benchmark(benchmark_name)
        logger.info(f"  Loaded {len(samples)} samples")

        # Generate trajectories
        t0 = time.time()
        trajectories = self._collect_trajectories(samples)
        elapsed = time.time() - t0
        logger.info(f"  Generated {len(trajectories)} trajectories in {elapsed:.1f}s")

        n_correct = sum(1 for t in trajectories if t.is_correct)
        n_incorrect = len(trajectories) - n_correct
        accuracy = n_correct / len(trajectories) if trajectories else 0.0
        logger.info(f"  Accuracy: {n_correct}/{len(trajectories)} ({100*accuracy:.1f}%)")

        # Save intermediate
        self._save_intermediate(model_config.name, benchmark_name, trajectories)

        # Analyze signals (default PCA dim)
        logger.info("  Analyzing signals...")
        t0 = time.time()
        signal_results, probe_signal = self._analyze_signals(
            trajectories, pca_dim=self.config.pca_dims[1]  # default: 50
        )
        elapsed = time.time() - t0
        logger.info(f"  Signal analysis complete in {elapsed:.1f}s")

        # Compute statistics
        logger.info("  Computing statistics...")
        stats = self._compute_statistics(trajectories, signal_results, probe_signal)

        # PCA sensitivity sweep for beta-1
        beta1_sensitivity = {}
        if n_correct >= 5 and n_incorrect >= 5:
            logger.info("  Running beta-1 PCA sensitivity sweep...")
            beta1_sensitivity = self._run_beta1_pca_sensitivity(trajectories)
        else:
            logger.info("  Skipping PCA sensitivity (insufficient correct/incorrect split)")

        # Log key findings
        logger.info(f"  --- Key Results ---")
        if probe_signal is not None:
            logger.info(f"  Probe: best AUROC={stats.probe_best_auroc:.3f} at layer {stats.probe_best_layer}")
        else:
            logger.info(f"  Probe: N/A (insufficient correct/incorrect split)")
        logger.info(f"  Pivots: d={stats.pivot_cohens_d:.3f}, p={stats.pivot_permutation_p:.4f}")
        logger.info(f"  Beta-1: d={stats.beta1_cohens_d:.3f}, p={stats.beta1_permutation_p:.4f}")
        logger.info(f"  Length confound: d={stats.length_cohens_d:.3f}")

        return ModelBenchmarkResult(
            model_name=model_config.name,
            benchmark_name=benchmark_name,
            n_samples=len(trajectories),
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            accuracy=accuracy,
            stats=stats,
            beta1_pca_sensitivity=beta1_sensitivity,
        )

    def run(self) -> dict[str, ModelBenchmarkResult]:
        """Run full cross-model validation experiment."""
        logger.info("=" * 70)
        logger.info("REASONING GEOMETRY CROSS-MODEL VALIDATION")
        logger.info("Signals: linear probes, cognitive pivots, beta-1 topology")
        logger.info("Methodology: greedy decoding, bootstrap CI, permutation tests")
        logger.info("=" * 70)
        logger.info(f"Models: {[m.name for m in self.config.models]}")
        logger.info(f"Benchmarks: {self.config.benchmarks}")
        logger.info(f"Samples: {self.config.n_samples}")
        logger.info(f"Output: {self.config.output_dir}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        all_results: dict[str, ModelBenchmarkResult] = {}

        for model_config in self.config.models:
            self._load_model(model_config)

            for benchmark in self.config.benchmarks:
                key = f"{model_config.name}_{benchmark}"
                try:
                    result = self._run_model_benchmark(model_config, benchmark)
                    all_results[key] = result
                except Exception as e:
                    logger.error(f"FAILED: {key}: {e}")
                    import traceback
                    traceback.print_exc()

            self._unload_model()

        # Save results and generate report
        self._save_final_results(all_results)
        self._generate_report(all_results)

        return all_results

    # ----- Results output -----

    def _save_final_results(self, results: dict[str, ModelBenchmarkResult]) -> None:
        """Save all results as JSON."""
        analysis_dir = self.config.output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for key, r in results.items():
            serializable[key] = {
                "model_name": r.model_name,
                "benchmark_name": r.benchmark_name,
                "n_samples": r.n_samples,
                "n_correct": r.n_correct,
                "n_incorrect": r.n_incorrect,
                "accuracy": r.accuracy,
                "probe": {
                    "auroc_by_layer": {str(k): v for k, v in r.stats.probe_auroc_by_layer.items()},
                    "best_layer": r.stats.probe_best_layer,
                    "best_auroc": r.stats.probe_best_auroc,
                },
                "pivots": {
                    "mean_correct": (
                        sum(r.stats.pivot_count_correct) / len(r.stats.pivot_count_correct)
                        if r.stats.pivot_count_correct else None
                    ),
                    "mean_incorrect": (
                        sum(r.stats.pivot_count_incorrect) / len(r.stats.pivot_count_incorrect)
                        if r.stats.pivot_count_incorrect else None
                    ),
                    "cohens_d": r.stats.pivot_cohens_d,
                    "d_ci": list(r.stats.pivot_d_ci),
                    "permutation_p": r.stats.pivot_permutation_p,
                },
                "topology": {
                    "mean_delta_b1_correct": (
                        sum(r.stats.delta_beta1_correct) / len(r.stats.delta_beta1_correct)
                        if r.stats.delta_beta1_correct else None
                    ),
                    "mean_delta_b1_incorrect": (
                        sum(r.stats.delta_beta1_incorrect) / len(r.stats.delta_beta1_incorrect)
                        if r.stats.delta_beta1_incorrect else None
                    ),
                    "cohens_d": r.stats.beta1_cohens_d,
                    "d_ci": list(r.stats.beta1_d_ci),
                    "permutation_p": r.stats.beta1_permutation_p,
                    "pca_sensitivity": {str(k): v for k, v in r.beta1_pca_sensitivity.items()},
                },
                "confounds": {
                    "gen_length_cohens_d": r.stats.length_cohens_d,
                },
            }

        path = analysis_dir / "per_model_results.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "experiment_config": {
                        "n_samples": self.config.n_samples,
                        "seed": self.config.seed,
                        "spike_threshold_k": self.config.spike_threshold_k,
                        "pca_dims_tested": self.config.pca_dims,
                        "max_tda_tokens": self.config.max_tda_tokens,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    "results": serializable,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {path}")

    # ----- Report generation -----

    def _generate_report(self, results: dict[str, ModelBenchmarkResult]) -> None:
        """Generate VALIDATION_REPORT.md with pass/fail verdicts."""
        lines: list[str] = []
        w = lines.append

        w("# Reasoning Geometry Validation Report")
        w("")
        w(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
        w(f"**Samples per model**: {self.config.n_samples}")
        w(f"**Seed**: {self.config.seed}")
        w(f"**Spike threshold**: {self.config.spike_threshold_k} (Lee et al. 2026)")
        w("")

        # Executive summary table
        w("## Executive Summary")
        w("")
        w("| Model | Benchmark | Accuracy | Best Probe AUROC | Pivot d | Beta-1 d | Length d |")
        w("|-------|-----------|----------|------------------|---------|----------|---------|")

        for key, r in results.items():
            probe_str = f"{r.stats.probe_best_auroc:.3f} (L{r.stats.probe_best_layer})" if r.stats.probe_best_layer >= 0 else "N/A"
            w(
                f"| {r.model_name} | {r.benchmark_name} | {r.accuracy:.1%} | "
                f"{probe_str} | {r.stats.pivot_cohens_d:+.3f} | "
                f"{r.stats.beta1_cohens_d:+.3f} | {r.stats.length_cohens_d:+.3f} |"
            )

        w("")

        # Signal 1: Linear Probes
        w("## Signal 1: Linear Probes")
        w("")
        w("**Pass criteria**: Best AUROC > 0.60, CI lower > 0.55, peaks in middle layers.")
        w("")

        probe_pass_count = 0
        for key, r in results.items():
            w(f"### {r.model_name} / {r.benchmark_name}")
            w("")
            if r.stats.probe_best_layer < 0:
                w("N/A -- insufficient correct/incorrect split for probe training.")
            else:
                w(f"- Best layer: **{r.stats.probe_best_layer}** (AUROC = {r.stats.probe_best_auroc:.3f})")

                if r.stats.probe_auroc_by_layer:
                    w("- Per-layer AUROC:")
                    w("")
                    w("  | Layer | AUROC |")
                    w("  |-------|-------|")
                    for layer_idx in sorted(r.stats.probe_auroc_by_layer.keys()):
                        auroc = r.stats.probe_auroc_by_layer[layer_idx]
                        marker = " <-- best" if layer_idx == r.stats.probe_best_layer else ""
                        w(f"  | {layer_idx} | {auroc:.3f}{marker} |")
                    w("")

                if r.stats.probe_best_auroc > 0.60:
                    w("- **PASS**")
                    probe_pass_count += 1
                else:
                    w("- **FAIL** (AUROC <= 0.60)")
            w("")

        w(f"**Probe verdict**: {'PASS' if probe_pass_count >= 2 else 'FAIL'} "
          f"({probe_pass_count} models passed, need >= 2)")
        w("")

        # Signal 2: Cognitive Pivots
        w("## Signal 2: Cognitive Pivots")
        w("")
        w("**Pass criteria**: Cohen's d < -0.5, CI excludes 0, permutation p < 0.05.")
        w("")

        pivot_pass_count = 0
        for key, r in results.items():
            w(f"### {r.model_name} / {r.benchmark_name}")
            w("")

            if not r.stats.pivot_count_correct or not r.stats.pivot_count_incorrect:
                w("N/A -- no correct/incorrect split.")
            else:
                c_mean = sum(r.stats.pivot_count_correct) / len(r.stats.pivot_count_correct)
                i_mean = sum(r.stats.pivot_count_incorrect) / len(r.stats.pivot_count_incorrect)
                w(f"- Correct mean pivots: {c_mean:.1f}")
                w(f"- Incorrect mean pivots: {i_mean:.1f}")
                w(f"- Cohen's d: **{r.stats.pivot_cohens_d:+.3f}** "
                  f"(95% CI: [{r.stats.pivot_d_ci[0]:+.3f}, {r.stats.pivot_d_ci[1]:+.3f}])")
                w(f"- Permutation p: {r.stats.pivot_permutation_p:.4f}")

                ci_excludes_zero = (
                    r.stats.pivot_d_ci[0] > 0 or r.stats.pivot_d_ci[1] < 0
                )
                passed = (
                    r.stats.pivot_cohens_d < -0.5
                    and ci_excludes_zero
                    and r.stats.pivot_permutation_p < 0.05
                )
                if passed:
                    w("- **PASS**")
                    pivot_pass_count += 1
                else:
                    reasons = []
                    if r.stats.pivot_cohens_d >= -0.5:
                        reasons.append(f"|d| too small ({r.stats.pivot_cohens_d:+.3f})")
                    if not ci_excludes_zero:
                        reasons.append("CI includes 0")
                    if r.stats.pivot_permutation_p >= 0.05:
                        reasons.append(f"p={r.stats.pivot_permutation_p:.4f}")
                    w(f"- **FAIL** ({'; '.join(reasons)})")
            w("")

        w(f"**Pivot verdict**: {'PASS' if pivot_pass_count >= 2 else 'FAIL'} "
          f"({pivot_pass_count} models passed, need >= 2)")
        w("")

        # Signal 3: Beta-1 Topology
        w("## Signal 3: Beta-1 Topology")
        w("")
        w("**Pass criteria**: |d| > 0.3, CI excludes 0, consistent direction, robust to PCA dim.")
        w("")

        beta1_pass_count = 0
        for key, r in results.items():
            w(f"### {r.model_name} / {r.benchmark_name}")
            w("")

            if not r.stats.delta_beta1_correct or not r.stats.delta_beta1_incorrect:
                w("N/A -- no correct/incorrect split.")
            else:
                c_mean = sum(r.stats.delta_beta1_correct) / len(r.stats.delta_beta1_correct)
                i_mean = sum(r.stats.delta_beta1_incorrect) / len(r.stats.delta_beta1_incorrect)
                w(f"- Correct mean delta-beta1: {c_mean:+.4f}")
                w(f"- Incorrect mean delta-beta1: {i_mean:+.4f}")
                w(f"- Cohen's d: **{r.stats.beta1_cohens_d:+.3f}** "
                  f"(95% CI: [{r.stats.beta1_d_ci[0]:+.3f}, {r.stats.beta1_d_ci[1]:+.3f}])")
                w(f"- Permutation p: {r.stats.beta1_permutation_p:.4f}")

                if r.beta1_pca_sensitivity:
                    w("- PCA sensitivity:")
                    w("")
                    w("  | PCA dim | Cohen's d |")
                    w("  |---------|-----------|")
                    for dim in sorted(r.beta1_pca_sensitivity.keys()):
                        w(f"  | {dim} | {r.beta1_pca_sensitivity[dim]:+.3f} |")
                    w("")

                ci_excludes_zero = (
                    r.stats.beta1_d_ci[0] > 0 or r.stats.beta1_d_ci[1] < 0
                )
                # Check PCA robustness: signal present at >= 2 dims
                n_robust = sum(
                    1 for d_val in r.beta1_pca_sensitivity.values() if abs(d_val) > 0.3
                )
                pca_robust = n_robust >= 2

                passed = (
                    abs(r.stats.beta1_cohens_d) > 0.3
                    and ci_excludes_zero
                    and r.stats.beta1_permutation_p < 0.05
                    and (pca_robust or not r.beta1_pca_sensitivity)
                )
                if passed:
                    w("- **PASS**")
                    beta1_pass_count += 1
                else:
                    reasons = []
                    if abs(r.stats.beta1_cohens_d) <= 0.3:
                        reasons.append(f"|d| too small ({r.stats.beta1_cohens_d:+.3f})")
                    if not ci_excludes_zero:
                        reasons.append("CI includes 0")
                    if r.stats.beta1_permutation_p >= 0.05:
                        reasons.append(f"p={r.stats.beta1_permutation_p:.4f}")
                    if r.beta1_pca_sensitivity and not pca_robust:
                        reasons.append(f"PCA fragile ({n_robust}/3 dims show |d|>0.3)")
                    w(f"- **FAIL** ({'; '.join(reasons)})")
            w("")

        w(f"**Beta-1 verdict**: {'PASS' if beta1_pass_count >= 2 else 'FAIL'} "
          f"({beta1_pass_count} models passed, need >= 2)")
        w("")

        # Confound analysis
        w("## Confound Analysis: Generation Length")
        w("")
        w("| Model | Benchmark | Length d | Concern |")
        w("|-------|-----------|---------|---------|")
        for key, r in results.items():
            concern = "YES" if abs(r.stats.length_cohens_d) > 0.3 else "no"
            w(f"| {r.model_name} | {r.benchmark_name} | {r.stats.length_cohens_d:+.3f} | {concern} |")
        w("")

        # Cross-model analysis
        w("## Cross-Model Analysis")
        w("")

        # Group by benchmark
        benchmarks_seen = set()
        for r in results.values():
            benchmarks_seen.add(r.benchmark_name)

        for bm in sorted(benchmarks_seen):
            w(f"### {bm}")
            w("")
            bm_results = {k: v for k, v in results.items() if v.benchmark_name == bm}

            # Architecture replication
            architectures: dict[str, list[ModelBenchmarkResult]] = {}
            for r in bm_results.values():
                arch = MODEL_REGISTRY.get(r.model_name, {}).get("architecture", "unknown")
                architectures.setdefault(arch, []).append(r)

            if len(architectures) > 1:
                w("**Cross-architecture replication:**")
                for arch, arch_results in architectures.items():
                    names = [r.model_name for r in arch_results]
                    d_vals = [r.stats.pivot_cohens_d for r in arch_results]
                    w(f"- {arch} ({', '.join(names)}): pivot d = {d_vals}")
                w("")

            # Size scaling (LFM2 family)
            lfm2_results = sorted(
                [r for r in bm_results.values() if "LFM2" in r.model_name],
                key=lambda r: r.model_name,
            )
            if len(lfm2_results) >= 2:
                w("**Size scaling (LFM2 family):**")
                w("")
                w("| Model | Accuracy | Probe AUROC | Pivot d | Beta-1 d |")
                w("|-------|----------|-------------|---------|----------|")
                for r in lfm2_results:
                    probe_str = f"{r.stats.probe_best_auroc:.3f}" if r.stats.probe_best_layer >= 0 else "N/A"
                    w(f"| {r.model_name} | {r.accuracy:.1%} | {probe_str} | "
                      f"{r.stats.pivot_cohens_d:+.3f} | {r.stats.beta1_cohens_d:+.3f} |")
                w("")

            # RL training effect
            qwen_results = [r for r in bm_results.values() if "Qwen3" in r.model_name]
            ds_results = [r for r in bm_results.values() if "DeepSeek" in r.model_name]
            if qwen_results and ds_results:
                q = qwen_results[0]
                d = ds_results[0]
                w("**RL training effect (Qwen3-8B vs DeepSeek-R1-8B):**")
                w("")
                w(f"- Qwen3-8B: pivot d={q.stats.pivot_cohens_d:+.3f}, "
                  f"probe={q.stats.probe_best_auroc:.3f}")
                w(f"- DeepSeek-R1-8B: pivot d={d.stats.pivot_cohens_d:+.3f}, "
                  f"probe={d.stats.probe_best_auroc:.3f}")
                w(f"- Hypothesis: RL training may stabilize geometry, "
                  f"reducing pivot count differences.")
                w("")

        # Final verdict
        w("## CLI Promotion Decision")
        w("")

        promote = (
            probe_pass_count >= 3
            and pivot_pass_count >= 3
            and beta1_pass_count >= 2
        )
        partial_promote = (
            probe_pass_count >= 2
            and pivot_pass_count >= 2
        )

        if promote:
            w("### PROMOTE -- all signals validated")
            w("")
            w(f"- Probes: {probe_pass_count} models passed")
            w(f"- Pivots: {pivot_pass_count} models passed")
            w(f"- Beta-1: {beta1_pass_count} models passed")
        elif partial_promote:
            w("### PARTIAL PROMOTE -- probes + pivots validated, beta-1 experimental")
            w("")
            w(f"- Probes: {probe_pass_count} models passed -- PROMOTE")
            w(f"- Pivots: {pivot_pass_count} models passed -- PROMOTE")
            w(f"- Beta-1: {beta1_pass_count} models passed -- mark as experimental in CLI")
        else:
            w("### DO NOT PROMOTE")
            w("")
            w(f"- Probes: {probe_pass_count} models passed (need >= 2)")
            w(f"- Pivots: {pivot_pass_count} models passed (need >= 2)")
            w(f"- Beta-1: {beta1_pass_count} models passed (need >= 2)")

        w("")

        # Write report
        report_path = self.config.output_dir / "VALIDATION_REPORT.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Report saved to {report_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model validation of reasoning geometry signals"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help=f"Models to test (default: all). Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=["gsm8k", "arithmetic"],
        dest="benchmarks",
        help="Benchmarks to run (default: gsm8k arithmetic)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples per benchmark (default: 500)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
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
        default="results/reasoning_geometry_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Trajectories per batch for memory management (default: 50)",
    )

    args = parser.parse_args()

    # Build model configs
    model_configs = []
    for name in args.models:
        if name not in MODEL_REGISTRY:
            parser.error(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
        info = MODEL_REGISTRY[name]
        model_configs.append(
            ModelConfig(name=name, path=info["path"], architecture=info["architecture"])
        )

    config = ExperimentConfig(
        models=model_configs,
        benchmarks=args.benchmarks,
        n_samples=args.samples,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_dir=Path(args.output),
        batch_size=args.batch_size,
    )

    experiment = ReasoningGeometryValidation(config)
    experiment.run()


if __name__ == "__main__":
    main()
