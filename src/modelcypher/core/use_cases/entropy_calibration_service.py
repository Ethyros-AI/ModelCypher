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

    # Use calibrated thresholds
    thresholds = EntropyThresholds.from_calibration_data(result.entropy_values)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Calibration Prompts
# =============================================================================

# Standard prompts covering diverse use cases for baseline measurement
CALIBRATION_PROMPTS: tuple[str, ...] = (
    # Factual Q&A - should have lower entropy (confident)
    "What is the capital of France?",
    "What is 2 + 2?",
    "How many days are in a week?",
    "What color is the sky on a clear day?",
    "What is the chemical symbol for water?",
    # Explanation - moderate entropy
    "Explain the concept of photosynthesis in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe how a refrigerator works.",
    "Explain the water cycle for a middle school student.",
    "What is gravity?",
    # Creative - higher entropy expected
    "Write a short poem about the ocean.",
    "Write a professional email requesting a meeting.",
    "Describe a sunset in three sentences.",
    "What might happen if cats could fly?",
    "Write a haiku about coffee.",
    # Ambiguous - should show uncertainty
    "What is the meaning of life?",
    "Is artificial intelligence dangerous?",
    "What makes art beautiful?",
    "Should people eat meat?",
    "What is consciousness?",
)


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

    calibration_prompts: tuple[str, ...] = field(default=CALIBRATION_PROMPTS)
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntropyCalibrationResult":
        """Deserialize from dictionary."""
        stats = data.get("statistics", {})
        return cls(
            model_id=data["modelId"],
            vocab_size=data["vocabSize"],
            max_theoretical_entropy=data["maxTheoreticalEntropy"],
            entropy_values=data.get("entropyValues", []),
            mean=stats.get("mean", 0.0),
            std_dev=stats.get("stdDev", 0.0),
            min_value=stats.get("min", 0.0),
            max_value=stats.get("max", 0.0),
            percentile_10=stats.get("percentile10", 0.0),
            percentile_25=stats.get("percentile25", 0.0),
            percentile_50=stats.get("percentile50", 0.0),
            percentile_75=stats.get("percentile75", 0.0),
            percentile_90=stats.get("percentile90", 0.0),
            percentile_95=stats.get("percentile95", 0.0),
            percentile_99=stats.get("percentile99", 0.0),
            sample_count=data.get("sampleCount", 0),
            prompt_count=data.get("promptCount", 0),
            tokens_per_prompt=data.get("tokensPerPrompt", []),
            calibration_duration_seconds=data.get("calibrationDurationSeconds", 0.0),
            calibrated_at=data.get("calibratedAt", ""),
        )

    def z_score(self, entropy: float) -> float:
        """Compute z-score for an entropy value relative to this baseline.

        Args:
            entropy: Entropy value to evaluate.

        Returns:
            Number of standard deviations from the mean.
        """
        if self.std_dev < 1e-10:
            return 0.0 if abs(entropy - self.mean) < 1e-10 else float("inf")
        return (entropy - self.mean) / self.std_dev

    def is_outlier(self, entropy: float, sigma: float = 3.0) -> bool:
        """Check if entropy is an outlier (beyond sigma standard deviations).

        3-sigma is the geometry of normal distributions (99.7% within).

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

    def __init__(self) -> None:
        """Initialize entropy calibration service."""
        self._mx = None
        self._mlx_load = None
        self._backend = None

    def _ensure_mlx(self) -> None:
        """Ensure MLX is available for inference."""
        if self._mx is not None:
            return

        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("MLX is required for entropy calibration") from exc

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError("mlx-lm is required for entropy calibration") from exc

        from modelcypher.core.domain._backend import get_default_backend

        self._mx = mx
        self._mlx_load = load
        self._backend = get_default_backend()

    def calibrate(
        self,
        model_path: str,
        prompts: tuple[str, ...] | None = None,
        max_tokens_per_prompt: int = 50,
        temperature: float = 0.7,
    ) -> EntropyCalibrationResult:
        """
        Calibrate entropy thresholds for a model by measuring actual distributions.

        Runs calibration prompts through the model, captures logits at each
        generation step, computes Shannon entropy, and derives statistics.

        Args:
            model_path: Path to model directory.
            prompts: Optional custom prompts. Defaults to CALIBRATION_PROMPTS.
            max_tokens_per_prompt: Maximum tokens to generate per prompt.
            temperature: Sampling temperature.

        Returns:
            EntropyCalibrationResult with measured statistics.

        Raises:
            ValueError: If model path is invalid.
            RuntimeError: If MLX is not available.
        """
        self._ensure_mlx()

        model_dir = Path(model_path).expanduser().resolve()
        if not model_dir.exists():
            raise ValueError(f"Model path does not exist: {model_dir}")

        prompts = prompts or CALIBRATION_PROMPTS
        start_time = time.time()

        logger.info("Starting entropy calibration for %s with %d prompts", model_dir, len(prompts))

        # Load model
        model, tokenizer = self._mlx_load(str(model_dir))
        mx = self._mx
        b = self._backend

        # Get vocab size from model config or tokenizer
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            # Try to get from model config
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    vocab_size = config.get("vocab_size", 32000)
            else:
                vocab_size = 32000  # Reasonable default

        max_entropy = math.log(vocab_size)

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
        valid_values = [v for v in all_entropy_values if math.isfinite(v)]
        invalid_count = len(all_entropy_values) - len(valid_values)
        if invalid_count > 0:
            logger.warning(
                "Filtered %d invalid entropy values (NaN/inf) - may indicate numerical issues with quantized model",
                invalid_count
            )

        if not valid_values:
            raise RuntimeError("All entropy values were invalid (NaN/inf) - model may have numerical issues")

        # Compute statistics
        sorted_values = sorted(valid_values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            return sorted_values[idx]

        mean = sum(valid_values) / n
        variance = sum((v - mean) ** 2 for v in valid_values) / n
        std_dev = math.sqrt(variance)

        result = EntropyCalibrationResult(
            model_id=str(model_dir),
            vocab_size=vocab_size,
            max_theoretical_entropy=max_entropy,
            entropy_values=valid_values,
            mean=mean,
            std_dev=std_dev,
            min_value=sorted_values[0],
            max_value=sorted_values[-1],
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
            model: Loaded MLX model.
            tokenizer: Model tokenizer.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of entropy values, one per generated token.
        """
        mx = self._mx

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt)
        tokens = mx.array([input_ids])  # Shape: [1, seq_len]

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
                flat_logits = curr_logits.reshape(-1)  # [vocab_size]
            elif logits.ndim == 2:
                flat_logits = logits[-1, :]  # Last position
            else:
                flat_logits = logits.reshape(-1)

            # Compute entropy
            entropy = self._compute_entropy(flat_logits)
            entropy_values.append(entropy)

            # Sample next token
            if temperature == 0:
                next_token = mx.argmax(flat_logits)
            else:
                scaled = flat_logits / temperature
                probs = mx.softmax(scaled)
                next_token = mx.random.categorical(probs)

            mx.eval(next_token)
            next_token_id = int(next_token.item())

            # Check for EOS
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token_id == eos_id:
                break

            # Generate next logits
            next_input = mx.array([[next_token_id]])
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
        mx = self._mx

        # Stable softmax
        max_logit = mx.max(logits)
        shifted = logits - max_logit
        exp_logits = mx.exp(shifted)
        sum_exp = mx.sum(exp_logits)
        probs = exp_logits / sum_exp

        # Shannon entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        log_probs = mx.log(probs + 1e-10)
        entropy = -mx.sum(probs * log_probs)

        mx.eval(entropy)
        return float(entropy.item())

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
