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

"""Linguistic calorimeter for entropy measurement from model inference.

Orchestrates entropy measurement from actual model inference, replacing
simulated entropy computation with real logit-based Shannon entropy.

Notes
-----
The calorimeter measures:
- First-token entropy (decision point uncertainty)
- Mean generation entropy (overall confidence)
- Entropy trajectory (dynamics over generation)
- Top-K variance (distribution sharpness)

Real inference mode has infrastructure dependencies (mlx_lm for model
loading) that cannot be fully abstracted via the Backend protocol. Simulated
mode works without any MLX dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

from modelcypher.core.domain.entropy.entropy_math import EntropyMath
from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    BehavioralOutcome,
    EntropyDirection,
    LinguisticModifier,
    LocalizedModifiers,
    PerturbedPrompt,
    PromptLanguage,
    ThermoMeasurement,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class EntropyMeasurement:
    """Raw entropy measurement from model inference."""

    prompt: str
    first_token_entropy: float
    mean_entropy: float
    entropy_variance: float
    entropy_trajectory: list[float]
    top_k_concentration: float
    token_count: int
    generated_text: str
    stop_reason: str
    temperature: float
    measurement_time: float  # seconds
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BaselineMeasurements:
    """Baseline entropy statistics from a reference corpus."""

    corpus_size: int
    mean_first_token_entropy: float
    std_first_token_entropy: float
    mean_generation_entropy: float
    std_generation_entropy: float
    percentiles: dict[int, float]  # p25, p50, p75, p95
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EntropyTrajectory:
    """Token-level entropy tracking during generation."""

    prompt: str
    per_token_entropy: list[float]
    per_token_variance: list[float]
    tokens: list[str]
    cumulative_entropy: list[float]
    entropy_trend: EntropyDirection
    inflection_points: list[int]  # Token indices where trend changes
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Linguistic Calorimeter
# =============================================================================


class LinguisticCalorimeter:
    """Orchestrates entropy measurement from model inference.

    Parameters
    ----------
    model_path : str | None
        Path to the model directory.
    adapter_path : str | None
        Optional path to adapter weights.
    simulated : bool
        If True, use simulated entropy (no model needed).
    top_k : int
        Number of top logits for variance calculation.
    epsilon : float
        Numerical stability constant.
    backend : Backend | None
        Optional backend for array operations.
    model : object | None
        Optional pre-loaded model instance.
    tokenizer : object | None
        Optional pre-loaded tokenizer instance.

    Notes
    -----
    The calorimeter operates in two modes:
    1. Real mode: Uses MLX backend to run actual model inference
    2. Simulated mode: Uses heuristics for testing without a model
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
        simulated: bool = False,
        top_k: int = 10,
        epsilon: float = 1e-10,
        backend: "Backend | None" = None,
        model: object | None = None,
        tokenizer: object | None = None,
    ):
        """Initialize the calorimeter.

        Args:
            model_path: Path to the model directory.
            adapter_path: Optional path to adapter weights.
            simulated: If True, use simulated entropy (no model needed).
            top_k: Number of top logits for variance calculation.
            epsilon: Numerical stability constant.
            backend: Optional backend for array operations.
            model: Optional pre-loaded model instance.
            tokenizer: Optional pre-loaded tokenizer instance.
        """
        self.model_path = Path(model_path).expanduser().resolve() if model_path else None
        self.adapter_path = Path(adapter_path).expanduser().resolve() if adapter_path else None
        self.simulated = simulated or (model_path is None and model is None)
        self.top_k = top_k
        self.epsilon = epsilon
        self._backend = backend or get_default_backend()

        # Lazy-loaded components (or pre-loaded)
        self._model = model
        self._tokenizer = tokenizer
        self._entropy_calculator: object | None = None

        # Cache for baseline measurements
        self._baseline_cache: dict[str, BaselineMeasurements] = {}

    def _ensure_model(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None or self.simulated:
            return

        if self.model_path is None:
            raise ValueError("model_path required for real inference")

        try:
            # Infrastructure dependency: MLX-LM for model loading
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError("mlx-lm required for real inference") from exc

        # Load model
        logger.info(f"Loading model from {self.model_path}")
        adapter = str(self.adapter_path) if self.adapter_path else None
        self._model, self._tokenizer = load(str(self.model_path), adapter_path=adapter)

        # Load entropy calculator
        from modelcypher.core.domain.entropy.logit_entropy_calculator import (
            LogitEntropyCalculator,
        )

        self._entropy_calculator = LogitEntropyCalculator(top_k=self.top_k, epsilon=self.epsilon)

    def measure_entropy(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 64,
    ) -> EntropyMeasurement:
        """Compute entropy from model output distribution.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            EntropyMeasurement with all entropy metrics.
        """
        start_time = time.time()

        if self.simulated:
            return self._measure_simulated(prompt, temperature, max_tokens, start_time)

        return self._measure_real(prompt, temperature, max_tokens, start_time)

    def _measure_real(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        start_time: float,
    ) -> EntropyMeasurement:
        """Measure entropy using real model inference."""
        self._ensure_model()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._entropy_calculator is not None

        b = self._backend

        # Tokenize prompt
        tokens = self._tokenizer.encode(prompt)
        input_ids = b.array([tokens])

        # Forward pass to get logits for first token
        logits = self._model(input_ids)
        b.eval(logits)

        # Compute first-token entropy
        first_entropy, first_variance = self._entropy_calculator.compute(logits)

        # Generate tokens and track entropy
        entropy_trajectory = [first_entropy]
        variance_trajectory = [first_variance]
        generated_tokens = []

        # Simple greedy/sampling generation with entropy tracking
        current_tokens = list(tokens)
        stop_reason = "length"

        for _ in range(max_tokens - 1):
            input_ids = b.array([current_tokens])
            logits = self._model(input_ids)
            b.eval(logits)

            # Get entropy for current position
            entropy, variance = self._entropy_calculator.compute(logits)
            entropy_trajectory.append(entropy)
            variance_trajectory.append(variance)

            # Sample next token
            if temperature <= 0:
                # Greedy
                next_token = int(b.to_numpy(b.argmax(logits[0, -1, :], axis=-1)).item())
            else:
                # Temperature sampling
                scaled_logits = logits[0, -1, :] / temperature
                probs = b.softmax(scaled_logits, axis=-1)
                b.eval(probs)
                # Use random_categorical if available, else argmax
                if hasattr(b, "random_categorical"):
                    next_token = int(b.to_numpy(b.random_categorical(b.log(probs))).item())
                else:
                    next_token = int(b.to_numpy(b.argmax(probs, axis=-1)).item())

            generated_tokens.append(next_token)
            current_tokens.append(next_token)

            # Check for EOS
            if hasattr(self._tokenizer, "eos_token_id"):
                if next_token == self._tokenizer.eos_token_id:
                    stop_reason = "stop"
                    break

        # Decode generated text
        generated_text = self._tokenizer.decode(generated_tokens) if generated_tokens else ""

        # Compute statistics using consolidated EntropyMath
        stats = EntropyMath.calculate_trajectory_stats(entropy_trajectory)

        measurement_time = time.time() - start_time

        return EntropyMeasurement(
            prompt=prompt,
            first_token_entropy=stats.first_token_entropy,
            mean_entropy=stats.mean_entropy,
            entropy_variance=stats.entropy_variance,
            entropy_trajectory=entropy_trajectory,
            top_k_concentration=first_variance,
            token_count=len(generated_tokens),
            generated_text=generated_text,
            stop_reason=stop_reason,
            temperature=temperature,
            measurement_time=measurement_time,
        )

    def _measure_simulated(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        start_time: float,
    ) -> EntropyMeasurement:
        """Simulate entropy measurement for testing."""
        # Generate deterministic but varied entropy based on prompt
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        base_entropy = 2.0 + (prompt_hash % 100) / 50.0  # Range 2.0-4.0

        # Temperature effect
        temp_effect = (temperature - 1.0) * 0.5
        base_entropy += temp_effect

        # Length effect
        length_effect = min(len(prompt) / 200.0, 0.5)
        base_entropy += length_effect

        # Generate trajectory
        trajectory_len = min(max_tokens, 20)
        entropy_trajectory = []
        for i in range(trajectory_len):
            # Slight decay over generation (cooling effect)
            decay = i * 0.02
            noise = (hash((prompt_hash, i)) % 100 - 50) / 200.0
            entropy_trajectory.append(max(0.5, base_entropy - decay + noise))

        # Compute statistics using consolidated EntropyMath
        stats = EntropyMath.calculate_trajectory_stats(
            entropy_trajectory, fallback_entropy=base_entropy
        )

        # Simulate generated text
        word_count = min(trajectory_len * 2, 40)
        generated_text = " ".join(["word"] * word_count)

        measurement_time = time.time() - start_time

        return EntropyMeasurement(
            prompt=prompt,
            first_token_entropy=stats.first_token_entropy,
            mean_entropy=stats.mean_entropy,
            entropy_variance=stats.entropy_variance,
            entropy_trajectory=entropy_trajectory,
            top_k_concentration=0.3 + (prompt_hash % 50) / 100.0,
            token_count=trajectory_len,
            generated_text=generated_text,
            stop_reason="length",
            temperature=temperature,
            measurement_time=measurement_time,
        )

    def measure_with_modifiers(
        self,
        prompt: str,
        modifiers: list[LinguisticModifier] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 64,
        language: PromptLanguage = PromptLanguage.ENGLISH,
    ) -> list[ThermoMeasurement]:
        """Batch measurement across modifiers with baseline comparison.

        Args:
            prompt: Base prompt content.
            modifiers: List of modifiers to apply. Defaults to all.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            language: Language for localized modifiers.

        Returns:
            List of ThermoMeasurement, one per modifier.
        """
        if modifiers is None:
            modifiers = list(LinguisticModifier)

        measurements = []
        baseline_entropy: float | None = None

        for modifier in modifiers:
            # Create perturbed prompt
            if language == PromptLanguage.ENGLISH:
                perturbed = PerturbedPrompt.create(prompt, modifier)
            else:
                full_prompt = LocalizedModifiers.apply(modifier, prompt, language)
                perturbed = PerturbedPrompt(
                    base_content=prompt,
                    modifier=modifier,
                    full_prompt=full_prompt,
                )

            # Measure entropy
            raw = self.measure_entropy(perturbed.full_prompt, temperature, max_tokens)

            # Track baseline
            if modifier == LinguisticModifier.BASELINE:
                baseline_entropy = raw.mean_entropy

            # Compute delta_h
            delta_h = None
            if baseline_entropy is not None and modifier != LinguisticModifier.BASELINE:
                delta_h = raw.mean_entropy - baseline_entropy

            # Classify outcome
            outcome = self._classify_outcome(raw.mean_entropy, raw.entropy_variance)

            # Create ThermoMeasurement
            measurement = ThermoMeasurement(
                id=uuid4(),
                prompt=perturbed,
                first_token_entropy=raw.first_token_entropy,
                mean_entropy=raw.mean_entropy,
                entropy_variance=raw.entropy_variance,
                entropy_trajectory=raw.entropy_trajectory,
                top_k_concentration=raw.top_k_concentration,
                model_state=self._classify_model_state(raw.mean_entropy),
                behavioral_outcome=outcome,
                delta_h=delta_h,
                generated_text=raw.generated_text,
                token_count=raw.token_count,
                stop_reason=raw.stop_reason,
            )
            measurements.append(measurement)

        return measurements

    def establish_baseline(
        self,
        corpus: list[str],
        temperature: float = 1.0,
        max_tokens: int = 32,
    ) -> BaselineMeasurements:
        """Compute baseline entropy statistics from reference corpus.

        Args:
            corpus: List of reference prompts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per measurement.

        Returns:
            BaselineMeasurements with statistics.
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty")

        # Check cache
        cache_key = hashlib.md5("".join(corpus[:10]).encode()).hexdigest()
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        first_entropies = []
        mean_entropies = []

        for prompt in corpus:
            measurement = self.measure_entropy(prompt, temperature, max_tokens)
            first_entropies.append(measurement.first_token_entropy)
            mean_entropies.append(measurement.mean_entropy)

        # Compute statistics
        mean_first = sum(first_entropies) / len(first_entropies)
        mean_gen = sum(mean_entropies) / len(mean_entropies)

        std_first = math.sqrt(
            sum((e - mean_first) ** 2 for e in first_entropies) / len(first_entropies)
        )
        std_gen = math.sqrt(sum((e - mean_gen) ** 2 for e in mean_entropies) / len(mean_entropies))

        # Compute percentiles
        sorted_gen = sorted(mean_entropies)
        n = len(sorted_gen)
        percentiles = {
            25: sorted_gen[int(n * 0.25)] if n > 0 else 0.0,
            50: sorted_gen[int(n * 0.50)] if n > 0 else 0.0,
            75: sorted_gen[int(n * 0.75)] if n > 0 else 0.0,
            95: sorted_gen[int(n * 0.95)] if n > 0 else 0.0,
        }

        baseline = BaselineMeasurements(
            corpus_size=len(corpus),
            mean_first_token_entropy=mean_first,
            std_first_token_entropy=std_first,
            mean_generation_entropy=mean_gen,
            std_generation_entropy=std_gen,
            percentiles=percentiles,
        )

        self._baseline_cache[cache_key] = baseline
        return baseline

    def track_generation_entropy(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> EntropyTrajectory:
        """Token-level entropy tracking during generation.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            EntropyTrajectory with per-token metrics.
        """
        measurement = self.measure_entropy(prompt, temperature, max_tokens)

        # Compute cumulative entropy
        cumulative = []
        running_sum = 0.0
        for i, e in enumerate(measurement.entropy_trajectory):
            running_sum += e
            cumulative.append(running_sum / (i + 1))

        # Detect inflection points (where trend changes)
        inflection_points = []
        if len(measurement.entropy_trajectory) >= 3:
            for i in range(1, len(measurement.entropy_trajectory) - 1):
                prev_delta = (
                    measurement.entropy_trajectory[i] - measurement.entropy_trajectory[i - 1]
                )
                next_delta = (
                    measurement.entropy_trajectory[i + 1] - measurement.entropy_trajectory[i]
                )
                # Sign change indicates inflection
                if prev_delta * next_delta < 0 and abs(prev_delta) > 0.1:
                    inflection_points.append(i)

        # Determine overall trend
        if len(measurement.entropy_trajectory) >= 2:
            first_half = measurement.entropy_trajectory[: len(measurement.entropy_trajectory) // 2]
            second_half = measurement.entropy_trajectory[len(measurement.entropy_trajectory) // 2 :]
            first_mean = sum(first_half) / len(first_half) if first_half else 0
            second_mean = sum(second_half) / len(second_half) if second_half else 0
            delta = second_mean - first_mean
            if delta > 0.1:
                trend = EntropyDirection.INCREASE
            elif delta < -0.1:
                trend = EntropyDirection.DECREASE
            else:
                trend = EntropyDirection.NEUTRAL
        else:
            trend = EntropyDirection.NEUTRAL

        # Generate token placeholders for simulated mode
        tokens = [f"token_{i}" for i in range(len(measurement.entropy_trajectory))]

        # Compute per-token variance (sliding window)
        per_token_variance = []
        window_size = 3
        for i in range(len(measurement.entropy_trajectory)):
            start = max(0, i - window_size + 1)
            window = measurement.entropy_trajectory[start : i + 1]
            if len(window) > 1:
                mean_w = sum(window) / len(window)
                var = sum((x - mean_w) ** 2 for x in window) / len(window)
            else:
                var = 0.0
            per_token_variance.append(var)

        return EntropyTrajectory(
            prompt=prompt,
            per_token_entropy=measurement.entropy_trajectory,
            per_token_variance=per_token_variance,
            tokens=tokens,
            cumulative_entropy=cumulative,
            entropy_trend=trend,
            inflection_points=inflection_points,
        )

    def _classify_outcome(
        self,
        entropy: float,
        variance: float,
    ) -> BehavioralOutcome:
        """Classify behavioral outcome from entropy metrics."""
        # High entropy + low variance = distress (stuck in uncertainty)
        if entropy >= 4.0:
            if variance < 0.1:
                return BehavioralOutcome.REFUSED
            return BehavioralOutcome.HEDGED
        elif entropy >= 3.0:
            return BehavioralOutcome.ATTEMPTED
        else:
            return BehavioralOutcome.SOLVED

    def _classify_model_state(self, entropy: float) -> str:
        """Classify model state from entropy."""
        if entropy < 1.5:
            return "confident"
        elif entropy < 3.0:
            return "normal"
        elif entropy < 4.0:
            return "uncertain"
        else:
            return "distressed"
