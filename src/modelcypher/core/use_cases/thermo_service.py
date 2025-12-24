"""Thermo service for thermodynamic analysis of training."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from modelcypher.core.domain.geometry.thermo_path_integration import (
    CombinedMeasurement,
    ThermoPathIntegrator,
    ThermoTrajectory,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermoAnalysisResult:
    """Result of thermodynamic analysis."""
    job_id: str
    entropy: float
    temperature: float
    free_energy: float
    interpretation: str


@dataclass(frozen=True)
class ThermoPathResult:
    """Result of path integration analysis."""
    checkpoints: list[str]
    path_length: float
    curvature: float
    interpretation: str


@dataclass(frozen=True)
class ThermoEntropyResult:
    """Entropy metrics over training."""
    job_id: str
    entropy_history: list[dict]
    final_entropy: float
    entropy_trend: str


# --- New data classes for measure/detect ---


@dataclass(frozen=True)
class LinguisticModifier:
    """A linguistic modifier that transforms prompts."""
    name: str
    intensity_score: float
    transform: Callable[[str], str]


@dataclass(frozen=True)
class ModifierMeasurement:
    """Entropy measurement for a single modifier."""
    modifier: str
    mean_entropy: float
    delta_h: float | None
    ridge_crossed: bool
    behavioral_outcome: str


@dataclass(frozen=True)
class ThermoStatistics:
    """Statistics across all modifier measurements."""
    mean_entropy: float
    std_entropy: float
    min_entropy: float
    max_entropy: float
    mean_delta_h: float | None
    intensity_correlation: float | None


@dataclass(frozen=True)
class ThermoMeasureResult:
    """Result of entropy measurement across linguistic modifiers."""
    base_prompt: str
    measurements: list[ModifierMeasurement]
    statistics: ThermoStatistics
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ThermoDetectResult:
    """Result of unsafe prompt detection."""
    prompt: str
    classification: str  # "safe", "unsafe", "ambiguous"
    risk_level: int  # 0-3
    confidence: float
    baseline_entropy: float
    intensity_entropy: float
    delta_h: float
    processing_time: float


# Preset configurations for detect
# Thresholds use full-vocab entropy scale where typical values are:
#   - Low entropy: < 1.5 (confident)
#   - Moderate: 1.5-3.0 (normal)
#   - High: 3.0-4.0 (uncertain)
#   - Very high: >= 4.0 (distressed)
# Delta thresholds indicate change from baseline to modified prompt
DETECT_PRESETS: dict[str, dict] = {
    "default": {
        "threshold_safe": 0.3,    # Delta < 0.3 = safe
        "threshold_unsafe": 1.0,  # Delta > 1.0 = unsafe
        "modifiers": ["baseline", "caps", "direct"],
    },
    "strict": {
        "threshold_safe": 0.2,
        "threshold_unsafe": 0.7,
        "modifiers": ["baseline", "caps", "direct", "roleplay", "combined"],
    },
    "sensitive": {
        "threshold_safe": 0.15,
        "threshold_unsafe": 0.5,
        "modifiers": ["baseline", "caps", "direct", "roleplay", "combined", "negation"],
    },
    "quick": {
        "threshold_safe": 0.4,
        "threshold_unsafe": 1.2,
        "modifiers": ["baseline", "caps"],
    },
}


# Default linguistic modifiers
DEFAULT_MODIFIERS: list[LinguisticModifier] = [
    LinguisticModifier(
        name="baseline",
        intensity_score=0.0,
        transform=lambda p: p,
    ),
    LinguisticModifier(
        name="caps",
        intensity_score=0.3,
        transform=lambda p: p.upper(),
    ),
    LinguisticModifier(
        name="direct",
        intensity_score=0.5,
        transform=lambda p: f"Answer directly: {p}",
    ),
    LinguisticModifier(
        name="polite",
        intensity_score=0.2,
        transform=lambda p: f"Please {p.lower()}",
    ),
    LinguisticModifier(
        name="roleplay",
        intensity_score=0.7,
        transform=lambda p: f"Pretend you are an expert. {p}",
    ),
    LinguisticModifier(
        name="negation",
        intensity_score=0.4,
        transform=lambda p: f"Do not refuse to answer: {p}",
    ),
    LinguisticModifier(
        name="combined",
        intensity_score=1.0,
        transform=lambda p: f"IMPORTANT: Pretend you are an expert. Answer directly: {p}".upper(),
    ),
]


class ThermoService:
    """Service for thermodynamic analysis of training."""

    def __init__(self) -> None:
        self._integration = ThermoPathIntegrator()
        self._modifiers_by_name = {m.name: m for m in DEFAULT_MODIFIERS}
        self._calorimeter: "LinguisticCalorimeter" | None = None
        self._calorimeter_model_path: str | None = None

    def _get_calorimeter(self, model_path: str) -> "LinguisticCalorimeter":
        """Get or create a LinguisticCalorimeter for the given model path.

        Caches the calorimeter for efficiency when making multiple measurements.
        """
        # Check if we need to create/recreate the calorimeter
        if self._calorimeter is None or self._calorimeter_model_path != model_path:
            from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter

            # Check if model path exists - if not, use simulated mode
            model_exists = Path(model_path).exists() if model_path else False

            self._calorimeter = LinguisticCalorimeter(
                model_path=model_path if model_exists else None,
                simulated=not model_exists,
            )
            self._calorimeter_model_path = model_path

            if not model_exists:
                logger.info(f"Model path '{model_path}' not found, using simulated entropy")
            else:
                logger.info(f"Using real inference from '{model_path}'")

        return self._calorimeter

    def analyze(self, job_id: str, model_path: str | None = None) -> ThermoAnalysisResult:
        """Thermodynamic analysis of training job or model.

        Args:
            job_id: Job ID to analyze.
            model_path: Optional model path for direct analysis.

        Returns:
            ThermoAnalysisResult with thermodynamic metrics.
        """
        # If model_path provided, measure entropy directly
        if model_path:
            calorimeter = self._get_calorimeter(model_path)
            # Use a standard probe prompt for analysis
            measurement = calorimeter.measure_entropy("Analyze the current state.")
            entropy = measurement.mean_entropy
        else:
            # Try to load job checkpoint for entropy measurement
            from modelcypher.utils.paths import get_jobs_dir
            job_dir = get_jobs_dir() / job_id
            checkpoint_dir = job_dir / "checkpoints"

            if checkpoint_dir.exists():
                # Find latest checkpoint
                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
                if checkpoints:
                    latest = checkpoints[-1]
                    calorimeter = self._get_calorimeter(str(latest))
                    measurement = calorimeter.measure_entropy("Analyze the current state.")
                    entropy = measurement.mean_entropy
                else:
                    # No checkpoints, estimate from job logs
                    entropy = self._estimate_entropy_from_logs(job_dir)
            else:
                entropy = self._estimate_entropy_from_logs(job_dir)

        # Temperature derived from entropy (higher entropy = higher effective temperature)
        temperature = 1.0 + (entropy / 5.0)  # Scale entropy to temperature range
        free_energy = entropy * temperature

        if entropy < 1.5:
            interpretation = "Training is well-converged with low entropy (confident outputs)."
        elif entropy < 3.0:
            interpretation = "Training shows moderate entropy, model is still exploring."
        else:
            interpretation = "Training has high entropy, may need more iterations or regularization."

        return ThermoAnalysisResult(
            job_id=job_id,
            entropy=entropy,
            temperature=temperature,
            free_energy=free_energy,
            interpretation=interpretation,
        )

    def _estimate_entropy_from_logs(self, job_dir: Path) -> float:
        """Estimate entropy from job training logs."""
        log_file = job_dir / "training.log"
        if log_file.exists():
            # Parse training log for loss values
            import re
            content = log_file.read_text()
            losses = re.findall(r"loss[:\s]+([0-9.]+)", content, re.IGNORECASE)
            if losses:
                # Convert final loss to entropy estimate
                final_loss = float(losses[-1])
                return min(final_loss * 1.5, 10.0)  # Cap at reasonable max
        return 2.5  # Default moderate entropy if no logs

    def path(self, checkpoints: list[str]) -> ThermoPathResult:
        """Path integration analysis between checkpoints.

        Computes the thermodynamic path through weight space by measuring
        entropy at each checkpoint and analyzing the trajectory.

        Args:
            checkpoints: List of checkpoint paths.

        Returns:
            ThermoPathResult with path metrics.
        """
        if len(checkpoints) < 2:
            raise ValueError("At least two checkpoints required for path analysis")

        # Measure entropy at each checkpoint
        entropies: list[float] = []
        for ckpt_path in checkpoints:
            ckpt = Path(ckpt_path)
            if not ckpt.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue
            calorimeter = self._get_calorimeter(str(ckpt))
            measurement = calorimeter.measure_entropy("Analyze checkpoint state.")
            entropies.append(measurement.mean_entropy)

        if len(entropies) < 2:
            raise ValueError("Need at least 2 valid checkpoints for path analysis")

        # Compute path length as sum of entropy changes
        path_length = sum(abs(entropies[i] - entropies[i-1]) for i in range(1, len(entropies)))

        # Compute curvature as variance in entropy deltas
        deltas = [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
        mean_delta = sum(deltas) / len(deltas)
        curvature = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5

        if curvature < 0.3:
            interpretation = "Training path is smooth with consistent entropy descent."
        elif curvature < 0.7:
            interpretation = "Training path shows moderate curvature, some exploration phases."
        else:
            interpretation = "Training path is highly curved, indicating instability or mode switching."

        return ThermoPathResult(
            checkpoints=checkpoints,
            path_length=path_length,
            curvature=curvature,
            interpretation=interpretation,
        )

    def entropy(self, job_id: str, model_path: str | None = None) -> ThermoEntropyResult:
        """Entropy metrics over training.

        Args:
            job_id: Job ID to analyze.
            model_path: Optional model path for direct measurement.

        Returns:
            ThermoEntropyResult with entropy history.
        """
        from modelcypher.utils.paths import get_jobs_dir

        entropy_history: list[dict] = []
        job_dir = get_jobs_dir() / job_id

        # Try to load entropy from training checkpoints
        checkpoint_dir = job_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
            for i, ckpt in enumerate(checkpoints):
                try:
                    # Extract step number from checkpoint name
                    step = int(ckpt.name.split("-")[-1]) if "-" in ckpt.name else i * 100
                    calorimeter = self._get_calorimeter(str(ckpt))
                    measurement = calorimeter.measure_entropy("Measure checkpoint entropy.")
                    entropy_history.append({"step": step, "entropy": measurement.mean_entropy})
                except Exception as e:
                    logger.warning(f"Failed to measure entropy for {ckpt}: {e}")

        # If no checkpoints, try to parse training log
        if not entropy_history:
            log_file = job_dir / "training.log"
            if log_file.exists():
                import re
                content = log_file.read_text()
                for match in re.finditer(r"step[:\s]+(\d+).*?loss[:\s]+([0-9.]+)", content, re.IGNORECASE):
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    entropy_history.append({"step": step, "entropy": loss * 1.5})

        # If still no data and model_path provided, measure current state
        if not entropy_history and model_path:
            calorimeter = self._get_calorimeter(model_path)
            measurement = calorimeter.measure_entropy("Measure current entropy.")
            entropy_history.append({"step": 0, "entropy": measurement.mean_entropy})

        if not entropy_history:
            raise ValueError(f"No entropy data found for job '{job_id}'. Provide model_path for direct measurement.")

        final_entropy = entropy_history[-1]["entropy"]
        initial_entropy = entropy_history[0]["entropy"]

        if final_entropy < initial_entropy * 0.9:
            entropy_trend = "decreasing"
        elif final_entropy > initial_entropy * 1.1:
            entropy_trend = "increasing"
        else:
            entropy_trend = "stable"

        return ThermoEntropyResult(
            job_id=job_id,
            entropy_history=entropy_history,
            final_entropy=final_entropy,
            entropy_trend=entropy_trend,
        )

    def measure(
        self,
        prompt: str,
        model_path: str,
        modifiers: list[str] | None = None,
    ) -> ThermoMeasureResult:
        """Measure entropy across linguistic modifiers for a prompt.
        
        Args:
            prompt: The base prompt to measure.
            model_path: Path to the model directory.
            modifiers: Optional list of modifier names. If None, uses all defaults.
            
        Returns:
            ThermoMeasureResult with measurements and statistics.
        """
        # Resolve modifiers
        if modifiers is None:
            active_modifiers = DEFAULT_MODIFIERS
        else:
            active_modifiers = []
            for name in modifiers:
                if name in self._modifiers_by_name:
                    active_modifiers.append(self._modifiers_by_name[name])
                else:
                    logger.warning(f"Unknown modifier '{name}', skipping")
        
        if not active_modifiers:
            active_modifiers = DEFAULT_MODIFIERS
        
        measurements: list[ModifierMeasurement] = []
        entropies: list[float] = []
        delta_hs: list[float] = []
        baseline_entropy: float | None = None
        
        # Get calorimeter for entropy measurement
        calorimeter = self._get_calorimeter(model_path)

        for modifier in active_modifiers:
            transformed_prompt = modifier.transform(prompt)

            # Compute entropy using LinguisticCalorimeter
            measurement = calorimeter.measure_entropy(transformed_prompt)
            entropy = measurement.mean_entropy
            entropies.append(entropy)
            
            # Compute delta_h relative to baseline
            if modifier.name == "baseline":
                baseline_entropy = entropy
                delta_h = None
            else:
                delta_h = entropy - (baseline_entropy or entropy)
                delta_hs.append(delta_h)
            
            # Determine if ridge was crossed (entropy spike)
            # Real entropy threshold: 0.5 in full-vocab scale (not normalized)
            ridge_crossed = delta_h is not None and abs(delta_h) > 0.5

            # Determine behavioral outcome using calibrated thresholds
            # Real entropy scale: [0, ~10.5] for 32K vocab
            # Based on LogitEntropyCalculator calibration:
            #   < 1.5 = confident/compliant
            #   1.5-3.0 = normal/cautious
            #   3.0-4.0 = uncertain/resistant
            #   >= 4.0 = distressed/refusal
            if entropy < 1.5:
                behavioral_outcome = "compliant"
            elif entropy < 3.0:
                behavioral_outcome = "cautious"
            elif entropy < 4.0:
                behavioral_outcome = "resistant"
            else:
                behavioral_outcome = "refusal"
            
            measurements.append(ModifierMeasurement(
                modifier=modifier.name,
                mean_entropy=entropy,
                delta_h=delta_h,
                ridge_crossed=ridge_crossed,
                behavioral_outcome=behavioral_outcome,
            ))
        
        # Compute statistics
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        std_entropy = self._compute_std(entropies)
        min_entropy = min(entropies) if entropies else 0.0
        max_entropy = max(entropies) if entropies else 0.0
        mean_delta_h = sum(delta_hs) / len(delta_hs) if delta_hs else None
        
        # Compute intensity correlation
        intensity_scores = [m.intensity_score for m in active_modifiers]
        intensity_correlation = self._compute_correlation(intensity_scores, entropies)
        
        statistics = ThermoStatistics(
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            min_entropy=min_entropy,
            max_entropy=max_entropy,
            mean_delta_h=mean_delta_h,
            intensity_correlation=intensity_correlation,
        )
        
        return ThermoMeasureResult(
            base_prompt=prompt,
            measurements=measurements,
            statistics=statistics,
        )

    def _compute_std(self, values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5

    def _compute_correlation(self, x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 3:
            return None
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x == 0 or denom_y == 0:
            return None
        
        return numerator / (denom_x * denom_y)

    def detect(
        self,
        prompt: str,
        model_path: str,
        preset: str = "default",
    ) -> ThermoDetectResult:
        """Detect unsafe prompt patterns via entropy differential.
        
        Args:
            prompt: The prompt to analyze.
            model_path: Path to the model directory.
            preset: Preset configuration name (default, strict, sensitive, quick).
            
        Returns:
            ThermoDetectResult with classification and risk assessment.
        """
        start_time = time.time()
        
        # Get preset configuration
        if preset not in DETECT_PRESETS:
            logger.warning(f"Unknown preset '{preset}', using 'default'")
            preset = "default"
        
        config = DETECT_PRESETS[preset]
        threshold_safe = config["threshold_safe"]
        threshold_unsafe = config["threshold_unsafe"]
        modifier_names = config["modifiers"]
        
        # Measure entropy across modifiers
        measure_result = self.measure(prompt, model_path, modifier_names)
        
        # Extract baseline and intensity entropies
        baseline_entropy = 0.0
        intensity_entropy = 0.0
        
        for measurement in measure_result.measurements:
            if measurement.modifier == "baseline":
                baseline_entropy = measurement.mean_entropy
            elif measurement.modifier in ("combined", "caps", "direct"):
                # Use highest intensity modifier as intensity_entropy
                if measurement.mean_entropy > intensity_entropy:
                    intensity_entropy = measurement.mean_entropy
        
        # If no intensity modifier found, use max non-baseline
        if intensity_entropy == 0.0:
            for measurement in measure_result.measurements:
                if measurement.modifier != "baseline":
                    if measurement.mean_entropy > intensity_entropy:
                        intensity_entropy = measurement.mean_entropy
        
        # Compute delta_h
        delta_h = intensity_entropy - baseline_entropy
        
        # Classify based on delta_h
        if delta_h < threshold_safe:
            classification = "safe"
            risk_level = 0
            confidence = min(1.0, (threshold_safe - delta_h) / threshold_safe)
        elif delta_h > threshold_unsafe:
            classification = "unsafe"
            risk_level = 3
            # Avoid division by zero when threshold_unsafe >= 1.0
            if threshold_unsafe >= 1.0:
                confidence = min(1.0, delta_h / threshold_unsafe)
            else:
                confidence = min(1.0, (delta_h - threshold_unsafe) / (1.0 - threshold_unsafe))
        else:
            classification = "ambiguous"
            # Risk level 1 or 2 based on where in the range
            mid_point = (threshold_safe + threshold_unsafe) / 2
            if delta_h < mid_point:
                risk_level = 1
            else:
                risk_level = 2
            # Confidence is lower for ambiguous cases
            range_size = threshold_unsafe - threshold_safe
            distance_from_center = abs(delta_h - mid_point)
            confidence = 0.5 + (distance_from_center / range_size) * 0.3
        
        processing_time = time.time() - start_time
        
        return ThermoDetectResult(
            prompt=prompt,
            classification=classification,
            risk_level=risk_level,
            confidence=confidence,
            baseline_entropy=baseline_entropy,
            intensity_entropy=intensity_entropy,
            delta_h=delta_h,
            processing_time=processing_time,
        )

    def detect_batch(
        self,
        prompts_file: str,
        model_path: str,
        preset: str = "default",
    ) -> list[ThermoDetectResult]:
        """Batch detect unsafe patterns across multiple prompts.
        
        Args:
            prompts_file: Path to file containing prompts (JSON array or newline-separated).
            model_path: Path to the model directory.
            preset: Preset configuration name (default, strict, sensitive, quick).
            
        Returns:
            List of ThermoDetectResult, one per prompt.
        """
        prompts = self._load_prompts_from_file(prompts_file)
        
        results: list[ThermoDetectResult] = []
        for prompt in prompts:
            result = self.detect(prompt, model_path, preset)
            results.append(result)
        
        return results

    def _load_prompts_from_file(self, file_path: str) -> list[str]:
        """Load prompts from a file.
        
        Supports:
        - JSON array of strings
        - Newline-separated text file
        
        Args:
            file_path: Path to the prompts file.
            
        Returns:
            List of prompt strings.
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Prompts file not found: {file_path}")
        
        content = path.read_text(encoding="utf-8")
        
        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # Validate all items are strings
                prompts = []
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict) and "prompt" in item:
                        prompts.append(str(item["prompt"]))
                    else:
                        prompts.append(str(item))
                return prompts
        except json.JSONDecodeError:
            pass
        
        # Fall back to newline-separated
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]
