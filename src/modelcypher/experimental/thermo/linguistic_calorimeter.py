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

Orchestrates entropy measurement from actual model inference using
logit-based Shannon entropy.
Entropy differentials (delta_H) are the primary comparison signal; hidden-state
geometry is captured alongside entropy when available.

Notes
-----
The calorimeter measures:
- First-token entropy (decision point uncertainty)
- Mean generation entropy (overall confidence)
- Entropy trajectory (dynamics over generation)
- Top-K variance (distribution sharpness)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.refusal_direction_detector import RefusalDirection
    from modelcypher.core.domain.inference.activation_stream import ActivationFrame
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

from modelcypher.core.domain.entropy.entropy_math import EntropyMath
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature
from modelcypher.core.domain.inference.activation_stream import ActivationStream
from modelcypher.experimental.thermo.linguistic_thermodynamics import (
    BehavioralOutcome,
    EntropyDirection,
    LinguisticModifier,
    LocalizedModifiers,
    PerturbedPrompt,
    PromptLanguage,
    ThermoGeometryMetrics,
    ThermoMeasurement,
)
from modelcypher.experimental.thermo.measured_thermodynamics import (
    ThermoCalibration,
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
    temperature: float  # Derived from logit statistics (critical temperature)
    measurement_time: float  # seconds
    geometry_metrics: ThermoGeometryMetrics | None = None
    refusal_direction_distance: float | None = None
    refusal_projection_magnitude: float | None = None
    is_approaching_refusal: bool | None = None
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
    backend : Backend | None
        Optional backend for array operations.
    model : object | None
        Optional pre-loaded model instance.
    tokenizer : object | None
        Optional pre-loaded tokenizer instance.

    Notes
    -----
    The calorimeter operates in real mode only, using model inference to
    measure entropy and derived thermodynamic quantities.
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
        backend: "Backend | None" = None,
        model: object | None = None,
        tokenizer: object | None = None,
        calibration: ThermoCalibration | None = None,
        refusal_direction: "RefusalDirection | None" = None,
        model_loader: "ModelLoaderPort | None" = None,
    ):
        """Initialize the calorimeter.

        Args:
            model_path: Path to the model directory.
            adapter_path: Optional path to adapter weights.
            backend: Optional backend for array operations.
            model: Optional pre-loaded model instance.
            tokenizer: Optional pre-loaded tokenizer instance.
            calibration: Optional thermodynamic calibration for this model.
                If provided, classification thresholds will be derived from
                calibrated measurements instead of hardcoded values.
            refusal_direction: Optional precomputed refusal direction for geometry-first
                assessment. If omitted, a cached direction will be used when available.
            model_loader: Optional model loader for loading models. Required for
                real inference when no pre-loaded model is provided.
        """
        self.model_path = Path(model_path).expanduser().resolve() if model_path else None
        self.adapter_path = Path(adapter_path).expanduser().resolve() if adapter_path else None
        self._backend = backend or get_default_backend()
        self._calibration = calibration
        self._refusal_direction = refusal_direction
        self._refusal_direction_checked = False
        self._model_loader = model_loader

        # Lazy-loaded components (or pre-loaded)
        self._model = model
        self._tokenizer = tokenizer
        self._entropy_calculator: object | None = None

        # Cache for baseline measurements
        self._baseline_cache: dict[str, BaselineMeasurements] = {}
        self._context_length: int | None = None
        self._context_length_loaded: bool = False

    def _ensure_model(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None and self._tokenizer is not None:
            if self._entropy_calculator is None:
                from modelcypher.core.domain.entropy.logit_entropy_calculator import (
                    LogitEntropyCalculator,
                )

                self._entropy_calculator = LogitEntropyCalculator(backend=self._backend)
            return

        if self.model_path is None:
            raise ValueError("model_path required for real inference")

        # Get model loader (must be injected for real inference)
        if self._model_loader is None:
            raise RuntimeError("Model loader required for real inference")

        # Load model via port (hexagonal architecture)
        logger.info(f"Loading model from {self.model_path}")
        self._model, self._tokenizer = self._model_loader.load_model(
            str(self.model_path),
            adapter_path=str(self.adapter_path) if self.adapter_path else None,
        )

        # Load entropy calculator
        from modelcypher.core.domain.entropy.logit_entropy_calculator import (
            LogitEntropyCalculator,
        )

        self._entropy_calculator = LogitEntropyCalculator(backend=self._backend)

    def _resolve_context_length(self) -> int | None:
        """Resolve model context length from config.json if available."""
        if self._context_length_loaded:
            return self._context_length

        self._context_length_loaded = True
        candidates: list[int] = []

        if self.model_path is not None:
            config_path = self.model_path / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    logger.warning("Invalid config.json at %s", config_path)
                    config = {}

                for key in (
                    "max_position_embeddings",
                    "max_seq_len",
                    "max_sequence_length",
                    "n_ctx",
                    "context_length",
                    "seq_length",
                ):
                    value = config.get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        candidates.append(int(value))

        if self._tokenizer is not None:
            for key in (
                "model_max_length",
                "max_length",
                "max_seq_len",
                "max_sequence_length",
                "n_ctx",
                "context_length",
                "max_context_length",
            ):
                value = getattr(self._tokenizer, key, None)
                if isinstance(value, (int, float)) and value > 0:
                    candidates.append(int(value))

        if not candidates:
            return None

        self._context_length = min(candidates)
        return self._context_length

    def _prompt_token_count(self, prompt: str) -> int:
        """Estimate prompt length in tokens."""
        if not prompt:
            return 0
        if self._tokenizer is None:
            self._ensure_model()
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer unavailable for prompt tokenization")
        return len(self._tokenizer.encode(prompt))

    def _derive_max_tokens(self, prompt: str) -> int:
        """Derive max tokens from prompt length and model context."""
        prompt_tokens = self._prompt_token_count(prompt)
        context_len = self._resolve_context_length()

        if context_len is None:
            return 0

        remaining = context_len - prompt_tokens
        return max(0, remaining)

    def measure_entropy(
        self,
        prompt: str,
    ) -> EntropyMeasurement:
        """Compute entropy from model output distribution.

        Args:
            prompt: The input prompt.

        Returns:
            EntropyMeasurement with all entropy metrics.
        """
        start_time = time.time()

        max_tokens_val = self._derive_max_tokens(prompt)
        return self._measure_real(prompt, max_tokens_val, start_time)

    def _measure_real(
        self,
        prompt: str,
        max_tokens: int,
        start_time: float,
    ) -> EntropyMeasurement:
        """Measure entropy using real model inference."""
        self._ensure_model()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._entropy_calculator is not None

        b = self._backend

        geometry_stream: ActivationStream | None = None
        capture_ctx = nullcontext()
        geometry_enabled = False
        try:
            geometry_stream = ActivationStream(self._model, backend=b)
            capture_ctx = geometry_stream.capture()
            geometry_enabled = True
        except Exception as exc:
            logger.warning("Geometry capture unavailable: %s", exc)

        # Tokenize prompt
        tokens = self._tokenizer.encode(prompt)
        input_ids = b.array([tokens])

        # Forward pass to get logits for first token
        with capture_ctx:
            if geometry_enabled and geometry_stream is not None:
                geometry_stream.advance_token()
            logits = self._model(input_ids)
            b.eval(logits)

            # Compute first-token entropy
            first_entropy, first_variance = self._entropy_calculator.compute(logits)

            # Derive critical temperature from raw logit statistics.
            seq_len = int(logits.shape[1])
            row = b.take(logits, b.array([0]), axis=0)
            row = b.squeeze(row, axis=0)
            last_logits = b.take(row, b.array([seq_len - 1]), axis=0)
            last_logits = b.squeeze(last_logits, axis=0)
            b.eval(last_logits)

            from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
            from modelcypher.experimental.thermo.phase_transition_theory import (
                PhaseTransitionTheory,
            )

            logit_std = sqrt_scalar(first_variance, b)
            logits_list = b.tolist(last_logits)
            effective_vocab_size = PhaseTransitionTheory.effective_vocabulary_size(
                logits_list, temperature=1.0
            )
            derived_temperature = PhaseTransitionTheory.estimate_critical_temperature(
                logit_std,
                effective_vocab_size,
            )

            # Generate tokens and track entropy
            entropy_trajectory = [first_entropy]
            variance_trajectory = [first_variance]
            generated_tokens = []

            # Simple greedy/sampling generation with entropy tracking
            current_tokens = list(tokens)
            stop_reason = "length"

            for _ in range(max_tokens - 1):
                input_ids = b.array([current_tokens])
                if geometry_enabled and geometry_stream is not None:
                    geometry_stream.advance_token()
                logits = self._model(input_ids)
                b.eval(logits)

                # Get entropy for current position
                entropy, variance = self._entropy_calculator.compute(logits)
                entropy_trajectory.append(entropy)
                variance_trajectory.append(variance)

                # Pull the last position logits via backend indexing.
                seq_len = int(logits.shape[1])
                row = b.take(logits, b.array([0]), axis=0)
                row = b.squeeze(row, axis=0)
                last_logits = b.take(row, b.array([seq_len - 1]), axis=0)
                last_logits = b.squeeze(last_logits, axis=0)
                b.eval(last_logits)

                # Sample next token deterministically
                next_token_arr = b.argmax(last_logits, axis=-1)
                b.eval(next_token_arr)
                next_token = int(b.to_scalar(next_token_arr))

                generated_tokens.append(next_token)
                current_tokens.append(next_token)

                # Check for EOS (record but continue to measure full trajectory)
                if hasattr(self._tokenizer, "eos_token_id"):
                    if next_token == self._tokenizer.eos_token_id:
                        stop_reason = "stop"

        # Decode generated text
        generated_text = self._tokenizer.decode(generated_tokens) if generated_tokens else ""

        # Compute statistics using consolidated EntropyMath
        stats = EntropyMath.calculate_trajectory_stats(entropy_trajectory)

        geometry_metrics = None
        refusal_direction_distance = None
        refusal_projection_magnitude = None
        is_approaching_refusal = None
        if geometry_enabled and geometry_stream is not None:
            geometry_metrics = self._compute_geometry_metrics(geometry_stream.frames)
            (
                refusal_direction_distance,
                refusal_projection_magnitude,
                is_approaching_refusal,
            ) = self._compute_refusal_metrics(geometry_stream.frames)

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
            temperature=derived_temperature,
            measurement_time=measurement_time,
            geometry_metrics=geometry_metrics,
            refusal_direction_distance=refusal_direction_distance,
            refusal_projection_magnitude=refusal_projection_magnitude,
            is_approaching_refusal=is_approaching_refusal,
        )

    def _get_refusal_direction(self) -> "RefusalDirection | None":
        """Resolve refusal direction from cache or injected value."""
        if self._refusal_direction_checked:
            return self._refusal_direction

        self._refusal_direction_checked = True
        if self._refusal_direction is not None:
            return self._refusal_direction

        if self.model_path is None:
            return None

        try:
            from modelcypher.core.domain.geometry.refusal_direction_cache import (
                RefusalDirectionCache,
            )
        except Exception as exc:
            logger.debug("RefusalDirection cache unavailable: %s", exc)
            return None

        cache = RefusalDirectionCache.shared()
        self._refusal_direction = cache.load(self.model_path)
        return self._refusal_direction

    def _compute_refusal_metrics(
        self,
        frames: list["ActivationFrame"],
    ) -> tuple[float | None, float | None, bool | None]:
        """Compute refusal-direction metrics from captured activations."""
        if not frames:
            return (None, None, None)

        refusal_direction = self._get_refusal_direction()
        if refusal_direction is None:
            return (None, None, None)

        try:
            from modelcypher.core.domain.geometry.refusal_direction_detector import (
                RefusalDirectionDetector,
            )
        except Exception as exc:
            logger.debug("RefusalDirection detector unavailable: %s", exc)
            return (None, None, None)

        previous_projection: float | None = None
        last_metrics = None

        for frame in frames:
            if frame.layer_id != refusal_direction.layer_index:
                continue
            metrics = RefusalDirectionDetector.measure_distance(
                frame.hidden_state,
                refusal_direction,
                previous_projection,
                frame.token_idx,
            )
            if metrics is None:
                continue
            previous_projection = metrics.projection_magnitude
            last_metrics = metrics

        if last_metrics is None:
            return (None, None, None)

        return (
            last_metrics.distance_to_refusal,
            last_metrics.projection_magnitude,
            last_metrics.is_approaching_refusal,
        )

    def _compute_geometry_metrics(
        self,
        frames: list["ActivationFrame"],
    ) -> ThermoGeometryMetrics | None:
        """Compute hidden-state geometry metrics from captured activations."""
        if not frames:
            return None

        b = self._backend
        by_layer: dict[int, list] = {}
        for frame in frames:
            by_layer.setdefault(frame.layer_id, []).append(frame.hidden_state)

        id_estimator = IntrinsicDimension(b)
        ricci_estimator = OllivierRicciCurvature(backend=b)

        intrinsic_dimensions: dict[int, float] = {}
        ricci_curvatures: dict[int, float] = {}
        ricci_stds: dict[int, float] = {}
        sample_counts: dict[int, int] = {}

        for layer_id, states in by_layer.items():
            sample_counts[layer_id] = len(states)
            if len(states) < 2:
                continue

            activations = b.stack(states, axis=0)
            b.eval(activations)

            if len(states) >= 3:
                try:
                    id_result = id_estimator.compute(activations)
                    intrinsic_dimensions[layer_id] = id_result.intrinsic_dimension
                except Exception as exc:
                    logger.debug(
                        "Geometry ID failed for layer %d: %s",
                        layer_id,
                        exc,
                    )

            try:
                ricci_result = ricci_estimator.compute(activations)
                ricci_curvatures[layer_id] = ricci_result.mean_edge_curvature
                ricci_stds[layer_id] = ricci_result.std_edge_curvature
            except Exception as exc:
                logger.debug(
                    "Geometry Ricci failed for layer %d: %s",
                    layer_id,
                    exc,
                )

        mean_id = (
            sum(intrinsic_dimensions.values()) / len(intrinsic_dimensions)
            if intrinsic_dimensions
            else None
        )
        mean_ricci = (
            sum(ricci_curvatures.values()) / len(ricci_curvatures)
            if ricci_curvatures
            else None
        )
        mean_ricci_std = (
            sum(ricci_stds.values()) / len(ricci_stds) if ricci_stds else None
        )

        return ThermoGeometryMetrics(
            intrinsic_dimensions=intrinsic_dimensions,
            ricci_curvatures=ricci_curvatures,
            ricci_stds=ricci_stds,
            sample_counts=sample_counts,
            mean_intrinsic_dimension=mean_id,
            mean_ricci_curvature=mean_ricci,
            mean_ricci_std=mean_ricci_std,
        )

    @staticmethod
    def _build_perturbed_prompt(
        prompt: str,
        modifier: LinguisticModifier,
        language: PromptLanguage,
    ) -> PerturbedPrompt:
        """Build a prompt variant for the modifier/language pair."""
        if language == PromptLanguage.ENGLISH:
            return PerturbedPrompt.create(prompt, modifier)

        full_prompt = LocalizedModifiers.apply(modifier, prompt, language)
        return PerturbedPrompt(
            base_content=prompt,
            modifier=modifier,
            full_prompt=full_prompt,
        )

    @staticmethod
    def _compute_geometry_deltas(
        baseline: ThermoGeometryMetrics | None,
        current: ThermoGeometryMetrics | None,
    ) -> tuple[
        dict[int, float] | None,
        dict[int, float] | None,
        dict[int, float] | None,
        float | None,
        float | None,
        float | None,
    ]:
        """Compute per-layer and mean deltas for geometry metrics."""
        if baseline is None or current is None:
            return (None, None, None, None, None, None)

        def _delta_map(
            current_map: dict[int, float],
            baseline_map: dict[int, float],
        ) -> dict[int, float] | None:
            if not current_map or not baseline_map:
                return None
            common = set(current_map).intersection(baseline_map)
            if not common:
                return None
            return {layer: current_map[layer] - baseline_map[layer] for layer in common}

        delta_intrinsic_dimensions = _delta_map(
            current.intrinsic_dimensions,
            baseline.intrinsic_dimensions,
        )
        delta_ricci_curvatures = _delta_map(
            current.ricci_curvatures,
            baseline.ricci_curvatures,
        )
        delta_ricci_stds = _delta_map(
            current.ricci_stds,
            baseline.ricci_stds,
        )

        delta_mean_id = None
        if (
            baseline.mean_intrinsic_dimension is not None
            and current.mean_intrinsic_dimension is not None
        ):
            delta_mean_id = current.mean_intrinsic_dimension - baseline.mean_intrinsic_dimension

        delta_mean_ricci = None
        if (
            baseline.mean_ricci_curvature is not None
            and current.mean_ricci_curvature is not None
        ):
            delta_mean_ricci = current.mean_ricci_curvature - baseline.mean_ricci_curvature

        delta_mean_ricci_std = None
        if baseline.mean_ricci_std is not None and current.mean_ricci_std is not None:
            delta_mean_ricci_std = current.mean_ricci_std - baseline.mean_ricci_std

        return (
            delta_intrinsic_dimensions,
            delta_ricci_curvatures,
            delta_ricci_stds,
            delta_mean_id,
            delta_mean_ricci,
            delta_mean_ricci_std,
        )

    def measure_with_modifiers(
        self,
        prompt: str,
        modifiers: list[LinguisticModifier] | None = None,
        language: PromptLanguage = PromptLanguage.ENGLISH,
    ) -> list[ThermoMeasurement]:
        """Batch measurement across modifiers with baseline comparison.

        Args:
            prompt: Base prompt content.
            modifiers: List of modifiers to apply. Defaults to all.
            language: Language for localized modifiers.

        Returns:
            List of ThermoMeasurement, one per modifier.
        """
        if modifiers is None:
            modifiers = list(LinguisticModifier)

        measurements = []

        baseline_prompt = self._build_perturbed_prompt(
            prompt,
            LinguisticModifier.BASELINE,
            language,
        )
        baseline_raw = self.measure_entropy(
            baseline_prompt.full_prompt,
        )
        baseline_geometry = baseline_raw.geometry_metrics
        baseline_outcome = self._classify_outcome(
            baseline_raw.mean_entropy,
            baseline_raw.entropy_variance,
        )
        baseline_measurement = ThermoMeasurement(
            id=uuid4(),
            prompt=baseline_prompt,
            first_token_entropy=baseline_raw.first_token_entropy,
            mean_entropy=baseline_raw.mean_entropy,
            entropy_variance=baseline_raw.entropy_variance,
            entropy_trajectory=baseline_raw.entropy_trajectory,
            top_k_concentration=baseline_raw.top_k_concentration,
            geometry_metrics=baseline_raw.geometry_metrics,
            model_state=self._classify_model_state(baseline_raw.mean_entropy),
            behavioral_outcome=baseline_outcome,
            delta_h=None,
            refusal_direction_distance=baseline_raw.refusal_direction_distance,
            refusal_projection_magnitude=baseline_raw.refusal_projection_magnitude,
            is_approaching_refusal=baseline_raw.is_approaching_refusal,
            temperature=baseline_raw.temperature,
            generated_text=baseline_raw.generated_text,
            token_count=baseline_raw.token_count,
            stop_reason=baseline_raw.stop_reason,
        )

        for modifier in modifiers:
            if modifier == LinguisticModifier.BASELINE:
                measurements.append(baseline_measurement)
                continue

            perturbed = self._build_perturbed_prompt(prompt, modifier, language)

            # Measure entropy
            raw = self.measure_entropy(perturbed.full_prompt)

            # Compute delta_h
            delta_h = EntropyMath.compute_delta_h(
                raw.mean_entropy,
                baseline_raw.mean_entropy,
            )
            delta_refusal_distance = None
            if (
                baseline_raw.refusal_direction_distance is not None
                and raw.refusal_direction_distance is not None
            ):
                delta_refusal_distance = (
                    raw.refusal_direction_distance
                    - baseline_raw.refusal_direction_distance
                )
            delta_refusal_projection = None
            if (
                baseline_raw.refusal_projection_magnitude is not None
                and raw.refusal_projection_magnitude is not None
            ):
                delta_refusal_projection = (
                    raw.refusal_projection_magnitude
                    - baseline_raw.refusal_projection_magnitude
                )
            (
                delta_intrinsic_dimensions,
                delta_ricci_curvatures,
                delta_ricci_stds,
                delta_intrinsic_dimension,
                delta_ricci_curvature,
                delta_ricci_std,
            ) = self._compute_geometry_deltas(
                baseline_geometry,
                raw.geometry_metrics,
            )

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
                geometry_metrics=raw.geometry_metrics,
                delta_intrinsic_dimension_mean=delta_intrinsic_dimension,
                delta_ricci_curvature_mean=delta_ricci_curvature,
                delta_ricci_std_mean=delta_ricci_std,
                delta_intrinsic_dimensions=delta_intrinsic_dimensions,
                delta_ricci_curvatures=delta_ricci_curvatures,
                delta_ricci_stds=delta_ricci_stds,
                model_state=self._classify_model_state(raw.mean_entropy),
                behavioral_outcome=outcome,
                delta_h=delta_h,
                refusal_direction_distance=raw.refusal_direction_distance,
                refusal_projection_magnitude=raw.refusal_projection_magnitude,
                is_approaching_refusal=raw.is_approaching_refusal,
                delta_refusal_direction_distance=delta_refusal_distance,
                delta_refusal_projection_magnitude=delta_refusal_projection,
                temperature=raw.temperature,
                generated_text=raw.generated_text,
                token_count=raw.token_count,
                stop_reason=raw.stop_reason,
            )
            measurements.append(measurement)

        return measurements

    def establish_baseline(
        self,
        corpus: list[str],
    ) -> BaselineMeasurements:
        """Compute baseline entropy statistics from reference corpus.

        Args:
            corpus: List of reference prompts.

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
            measurement = self.measure_entropy(prompt)
            first_entropies.append(measurement.first_token_entropy)
            mean_entropies.append(measurement.mean_entropy)

        # Compute statistics
        mean_first = sum(first_entropies) / len(first_entropies)
        mean_gen = sum(mean_entropies) / len(mean_entropies)

        _b = get_default_backend()
        var_first = sum((e - mean_first) ** 2 for e in first_entropies) / len(first_entropies)
        std_first = sqrt_scalar(var_first, _b)
        var_gen = sum((e - mean_gen) ** 2 for e in mean_entropies) / len(mean_entropies)
        std_gen = sqrt_scalar(var_gen, _b)

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
    ) -> EntropyTrajectory:
        """Token-level entropy tracking during generation.

        Args:
            prompt: Input prompt.

        Returns:
            EntropyTrajectory with per-token metrics.
        """
        measurement = self.measure_entropy(prompt)

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
                # Sign change indicates inflection (any magnitude)
                if prev_delta * next_delta < 0:
                    inflection_points.append(i)

        # Determine overall trend using boundary values (> 0, < 0)
        if len(measurement.entropy_trajectory) >= 2:
            first_half = measurement.entropy_trajectory[: len(measurement.entropy_trajectory) // 2]
            second_half = measurement.entropy_trajectory[len(measurement.entropy_trajectory) // 2 :]
            first_mean = sum(first_half) / len(first_half) if first_half else 0
            second_mean = sum(second_half) / len(second_half) if second_half else 0
            delta = second_mean - first_mean
            if delta > 0:
                trend = EntropyDirection.INCREASE
            elif delta < 0:
                trend = EntropyDirection.DECREASE
            else:
                trend = EntropyDirection.NEUTRAL
        else:
            trend = EntropyDirection.NEUTRAL

        # Token labels are best-effort when per-token strings are unavailable.
        tokens = [f"token_{i}" for i in range(len(measurement.entropy_trajectory))]

        # Compute per-token variance (sliding window derived from trajectory length)
        per_token_variance = []
        traj_len = len(measurement.entropy_trajectory)
        _b = get_default_backend()
        window_size = max(1, int(sqrt_scalar(float(traj_len), _b))) if traj_len else 1
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
        """Classify behavioral outcome from entropy metrics.

        Requires calibration. Without calibrated thresholds, returns UNKNOWN.
        """
        if self._calibration and self._calibration.thresholds:
            # Use calibrated thresholds
            outcome_str = self._calibration.thresholds.classify_outcome(entropy, variance)
            return BehavioralOutcome(outcome_str)

        # No calibration = no classification. Return UNKNOWN.
        return BehavioralOutcome.UNKNOWN

    def _classify_model_state(self, entropy: float) -> str:
        """Classify model state from entropy.

        Requires calibration. Without calibrated percentiles, returns "uncalibrated".
        """
        if self._calibration and self._calibration.thresholds:
            # Use calibrated percentiles for state classification
            percentiles = self._calibration.thresholds.percentiles
            # Require all percentiles to be present
            if 25 not in percentiles or 50 not in percentiles or 75 not in percentiles:
                return "uncalibrated"
            if entropy < percentiles[25]:
                return "confident"
            elif entropy < percentiles[50]:
                return "normal"
            elif entropy < percentiles[75]:
                return "uncertain"
            else:
                return "distressed"

        # No calibration = no classification
        return "uncalibrated"
