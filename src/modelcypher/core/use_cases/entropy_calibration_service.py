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

"""
Entropy Calibration Service.

Measures actual entropy distributions from model inference to derive
empirically-grounded thresholds. No magic numbers - only measured data.

Usage:
    service = EntropyCalibrationService()
    result = service.calibrate(model_path="/path/to/model")
    # Use result.entropy_values and percentiles to derive thresholds explicitly
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    is_finite,
    log_scalar,
    sqrt_scalar,
)

# Machine epsilon for float64 (native Python float)
_MACHINE_EPS = sys.float_info.epsilon

# Smallest positive float for log safety (prevents log(0))
_LOG_SAFE_MIN = sys.float_info.min

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntropyCalibrationResult:
    """Result of entropy calibration for a model.

    Contains measured entropy statistics that can be used to derive thresholds.
    All values are empirically measured, not assumed.
    """

    model_id: str
    """Model identifier or path."""

    vocab_size: int
    """Model's vocabulary size."""

    max_theoretical_entropy: float
    """ln(vocab_size) - theoretical maximum."""

    entropy_values: list[float]
    """All measured entropy values from calibration."""

    mean: float
    """Mean of measured entropy values."""

    std_dev: float
    """Standard deviation of measured entropy values."""

    min_value: float
    """Minimum observed entropy."""

    max_value: float
    """Maximum observed entropy."""

    percentile_10: float
    """10th percentile - very low entropy responses."""

    percentile_25: float
    """25th percentile - low entropy threshold."""

    percentile_50: float
    """Median entropy."""

    percentile_75: float
    """75th percentile - high entropy threshold."""

    percentile_90: float
    """90th percentile - very high entropy responses."""

    percentile_95: float
    """95th percentile - circuit breaker candidate."""

    percentile_99: float
    """99th percentile - extreme uncertainty."""

    sample_count: int
    """Total number of entropy samples collected."""

    prompt_count: int
    """Number of prompts used for calibration."""

    tokens_per_prompt: list[int]
    """Number of tokens generated per prompt."""

    calibration_duration_seconds: float
    """Total time for calibration."""

    calibrated_at: str
    """ISO timestamp of calibration."""

    calibration_prompts: tuple[str, ...]
    """Prompts used for calibration."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "modelId": self.model_id,
            "vocabSize": self.vocab_size,
            "maxTheoreticalEntropy": self.max_theoretical_entropy,
            "entropyValues": self.entropy_values,
            "statistics": {
                "mean": self.mean,
                "stdDev": self.std_dev,
                "min": self.min_value,
                "max": self.max_value,
                "percentile10": self.percentile_10,
                "percentile25": self.percentile_25,
                "percentile50": self.percentile_50,
                "percentile75": self.percentile_75,
                "percentile90": self.percentile_90,
                "percentile95": self.percentile_95,
                "percentile99": self.percentile_99,
            },
            "sampleCount": self.sample_count,
            "promptCount": self.prompt_count,
            "tokensPerPrompt": self.tokens_per_prompt,
            "calibrationDurationSeconds": self.calibration_duration_seconds,
            "calibratedAt": self.calibrated_at,
            "calibrationPrompts": list(self.calibration_prompts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntropyCalibrationResult":
        """Deserialize from dictionary."""
        stats = data["statistics"]
        return cls(
            model_id=data["modelId"],
            vocab_size=data["vocabSize"],
            max_theoretical_entropy=data["maxTheoreticalEntropy"],
            entropy_values=data["entropyValues"],
            mean=stats["mean"],
            std_dev=stats["stdDev"],
            min_value=stats["min"],
            max_value=stats["max"],
            percentile_10=stats["percentile10"],
            percentile_25=stats["percentile25"],
            percentile_50=stats["percentile50"],
            percentile_75=stats["percentile75"],
            percentile_90=stats["percentile90"],
            percentile_95=stats["percentile95"],
            percentile_99=stats["percentile99"],
            sample_count=data["sampleCount"],
            prompt_count=data["promptCount"],
            tokens_per_prompt=data["tokensPerPrompt"],
            calibration_duration_seconds=data["calibrationDurationSeconds"],
            calibrated_at=data["calibratedAt"],
            calibration_prompts=tuple(data["calibrationPrompts"]),
        )

    def z_score(self, entropy: float) -> float:
        """Compute z-score for an entropy value relative to this baseline.

        Args:
            entropy: Entropy value to evaluate.

        Returns:
            Number of standard deviations from the mean.
        """
        if self.std_dev < _MACHINE_EPS:
            return 0.0 if abs(entropy - self.mean) < _MACHINE_EPS else float("inf")
        return (entropy - self.mean) / self.std_dev

    def is_outlier(self, entropy: float, sigma: float) -> bool:
        """Check if entropy is an outlier (beyond sigma standard deviations).

        Args:
            entropy: Entropy value to check.
            sigma: Number of standard deviations for outlier threshold.

        Returns:
            True if entropy is beyond sigma standard deviations from mean.
        """
        return abs(self.z_score(entropy)) > sigma


class EntropyCalibrationService:
    """
    Service for measuring empirical entropy distributions from model inference.

    This service loads a model, runs calibration prompts, captures logits,
    and computes actual entropy statistics. No guessing, no magic numbers.

    Usage:
        service = EntropyCalibrationService()
        result = service.calibrate(model_path="/path/to/model")

        # Save calibration for later use
        service.save_calibration(result, "/path/to/calibration.json")

        # Load and use
        loaded = service.load_calibration("/path/to/calibration.json")
        z = loaded.z_score(measured_entropy)
    """

    def __init__(self, model_loader: Any = None) -> None:
        """Initialize entropy calibration service.

        Args:
            model_loader: Optional model loader (uses default if None).
        """
        self._model_loader = model_loader
        self._backend = None

    def _ensure_backend(self) -> None:
        """Ensure backend is initialized."""
        if self._backend is not None:
            return

        from modelcypher.core.domain._backend import get_default_backend

        self._backend = get_default_backend()

    def _ensure_model_loader(self) -> Any:
        """Ensure model loader is available."""
        if self._model_loader is None:
            from modelcypher.infrastructure.model_loader_factory import get_model_loader

            self._model_loader = get_model_loader()
        return self._model_loader

    def calibrate(
        self,
        model_path: str,
        prompts: tuple[str, ...],
    ) -> EntropyCalibrationResult:
        """
        Calibrate entropy thresholds for a model by measuring actual distributions.

        Runs calibration prompts through the model, captures logits at each
        generation step, computes Shannon entropy, and derives statistics.

        Args:
            model_path: Path to model directory.
            prompts: Calibration prompts (required).

        Returns:
            EntropyCalibrationResult with measured statistics.

        Raises:
            ValueError: If model path is invalid.
            RuntimeError: If model loader is not available.
        """
        self._ensure_backend()
        model_loader = self._ensure_model_loader()

        model_dir = Path(model_path).expanduser().resolve()
        if not model_dir.exists():
            raise ValueError(f"Model path does not exist: {model_dir}")

        if not prompts:
            raise ValueError("Calibration prompts are required")
        start_time = time.time()

        logger.info("Starting entropy calibration for %s with %d prompts", model_dir, len(prompts))

        # Load model via ModelLoaderPort (hexagonal architecture)
        model, tokenizer = model_loader.load_model_for_training(str(model_dir))
        max_tokens_per_prompt, temperature = self._derive_generation_params(
            model_dir=model_dir,
            tokenizer=tokenizer,
            prompts=prompts,
        )

        # Get vocab size from model config or tokenizer
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            # Try to get from model config
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    vocab_size = config.get("vocab_size")
            if vocab_size is None:
                raise ValueError("Unable to determine vocab_size for calibration")

        _b = get_default_backend()
        max_entropy = log_scalar(float(vocab_size), _b)

        # Collect entropy values from all prompts
        all_entropy_values: list[float] = []
        tokens_per_prompt: list[int] = []

        for prompt_idx, prompt in enumerate(prompts):
            logger.debug("Calibrating prompt %d/%d: %s...", prompt_idx + 1, len(prompts), prompt[:30])

            prompt_entropies = self._measure_prompt_entropy(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens_per_prompt,
                temperature=temperature,
            )

            all_entropy_values.extend(prompt_entropies)
            tokens_per_prompt.append(len(prompt_entropies))

        duration = time.time() - start_time

        if not all_entropy_values:
            raise RuntimeError("No entropy values collected during calibration")

        # Filter out invalid values (NaN, inf)
        valid_values = [v for v in all_entropy_values if is_finite(v, _b)]
        invalid_count = len(all_entropy_values) - len(valid_values)
        if invalid_count > 0:
            logger.warning(
                "Filtered %d invalid entropy values (NaN/inf) - may indicate numerical issues with quantized model",
                invalid_count
            )

        if not valid_values:
            raise RuntimeError("All entropy values were invalid (NaN/inf) - model may have numerical issues")

        # Compute statistics (backend only)
        values_arr = _b.array(valid_values)
        sorted_arr = _b.sort(values_arr)
        n = int(sorted_arr.shape[0])

        mean_arr = _b.mean(values_arr)
        var_arr = _b.var(values_arr)
        std_arr = _b.sqrt(var_arr)
        min_arr = _b.min(sorted_arr)
        max_arr = _b.max(sorted_arr)
        _b.eval(mean_arr, std_arr, min_arr, max_arr)

        mean = float(_b.to_scalar(mean_arr))
        std_dev = float(_b.to_scalar(std_arr))
        min_value = float(_b.to_scalar(min_arr))
        max_value = float(_b.to_scalar(max_arr))

        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            idx_arr = _b.array([idx])
            value_arr = _b.take(sorted_arr, idx_arr, axis=0)
            _b.eval(value_arr)
            return float(_b.to_scalar(value_arr))

        result = EntropyCalibrationResult(
            model_id=str(model_dir),
            vocab_size=vocab_size,
            max_theoretical_entropy=max_entropy,
            entropy_values=valid_values,
            mean=mean,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            percentile_10=percentile(0.10),
            percentile_25=percentile(0.25),
            percentile_50=percentile(0.50),
            percentile_75=percentile(0.75),
            percentile_90=percentile(0.90),
            percentile_95=percentile(0.95),
            percentile_99=percentile(0.99),
            sample_count=n,
            prompt_count=len(prompts),
            tokens_per_prompt=tokens_per_prompt,
            calibration_duration_seconds=duration,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
            calibration_prompts=prompts,
        )

        logger.info(
            "Entropy calibration complete: %d samples, mean=%.3f, std=%.3f, duration=%.1fs",
            n, mean, std_dev, duration
        )

        return result

    def _derive_generation_params(
        self,
        *,
        model_dir: Path,
        tokenizer: Any,
        prompts: tuple[str, ...],
    ) -> tuple[int, float]:
        """Derive generation parameters from model geometry and prompt lengths."""
        # Temperature is fixed at 0.0 for deterministic calibration paths.
        temperature = 0.0

        prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts] if prompts else []
        max_prompt_len = max(prompt_lengths, default=1)

        max_context = self._resolve_context_length(model_dir)
        if max_context is None:
            max_tokens = max(1, max_prompt_len)
        else:
            max_tokens = max(1, max_context - max_prompt_len)

        return max_tokens, temperature

    @staticmethod
    def _resolve_context_length(model_dir: Path) -> int | None:
        """Resolve max context length from model config (if available)."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return None

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
                return int(value)

        return None

    def _measure_prompt_entropy(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> list[float]:
        """Measure entropy values for a single prompt.

        Args:
            model: Loaded model (via ModelLoaderPort).
            tokenizer: Model tokenizer.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of entropy values, one per generated token.
        """
        backend = self._backend

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt)
        tokens = backend.array([input_ids])  # Shape: [1, seq_len]

        # Get initial logits and cache
        # MLX models return (logits, cache) tuple
        cache = None
        result = model(tokens, cache=cache)

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            logits, cache = result
        else:
            # Some models might just return logits
            logits = result
            cache = None

        entropy_values: list[float] = []

        for _ in range(max_tokens):
            # Get logits for the last position
            # logits shape is typically [batch, seq_len, vocab_size]
            if logits.ndim == 3:
                curr_logits = logits[:, -1, :]  # [1, vocab_size]
                flat_logits = backend.reshape(curr_logits, (-1,))  # [vocab_size]
            elif logits.ndim == 2:
                flat_logits = logits[-1, :]  # Last position
            else:
                flat_logits = backend.reshape(logits, (-1,))

            # Compute entropy using Backend
            entropy = self._compute_entropy(flat_logits)
            entropy_values.append(entropy)

            # Sample next token using Backend
            if temperature == 0:
                next_token = backend.argmax(flat_logits)
            else:
                scaled = flat_logits / temperature
                probs = backend.softmax(scaled, axis=-1)
                # Use random categorical sampling via Backend
                next_token = backend.random_categorical(probs)

            backend.eval(next_token)
            next_token_id = int(backend.to_scalar(next_token))

            # Check for EOS
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token_id == eos_id:
                break

            # Generate next logits
            next_input = backend.array([[next_token_id]])
            result = model(next_input, cache=cache)

            if isinstance(result, tuple) and len(result) == 2:
                logits, cache = result
            else:
                logits = result

        return entropy_values

    def _compute_entropy(self, logits: Any) -> float:
        """Compute Shannon entropy from logits.

        Args:
            logits: 1D array of logits.

        Returns:
            Shannon entropy value.
        """
        backend = self._backend

        # Stable softmax
        max_logit = backend.max(logits)
        shifted = logits - max_logit
        exp_logits = backend.exp(shifted)
        sum_exp = backend.sum(exp_logits)
        probs = exp_logits / sum_exp

        # Shannon entropy: -sum(p * log(p))
        # Add smallest positive float to avoid log(0)
        log_probs = backend.log(probs + _LOG_SAFE_MIN)
        entropy = -backend.sum(probs * log_probs)

        backend.eval(entropy)
        return float(backend.to_scalar(entropy))

    def save_calibration(
        self,
        result: EntropyCalibrationResult,
        output_path: str,
    ) -> None:
        """Save calibration result to JSON file.

        Args:
            result: Calibration result to save.
            output_path: Path to save JSON file.
        """
        output_file = Path(output_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Saved entropy calibration to %s", output_file)

    def load_calibration(self, calibration_path: str) -> EntropyCalibrationResult:
        """Load calibration result from JSON file.

        Args:
            calibration_path: Path to calibration JSON file.

        Returns:
            Loaded EntropyCalibrationResult.

        Raises:
            ValueError: If file doesn't exist or is invalid.
        """
        cal_file = Path(calibration_path).expanduser().resolve()
        if not cal_file.exists():
            raise ValueError(f"Calibration file does not exist: {cal_file}")

        with open(cal_file) as f:
            data = json.load(f)

        return EntropyCalibrationResult.from_dict(data)
