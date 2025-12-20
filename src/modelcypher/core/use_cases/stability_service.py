"""Stability service for model stability testing.

Provides stability suite execution and reporting functionality for
assessing model robustness and consistency.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    """Configuration for stability testing."""

    num_runs: int = 10
    temperature_range: tuple[float, float] = (0.1, 1.0)
    prompt_variations: int = 5
    seed: int | None = None


@dataclass
class StabilityRunResult:
    """Result of a stability test run."""

    suite_id: str
    model_path: str
    status: str
    started_at: str
    config: dict[str, Any]
    summary: dict[str, float] = field(default_factory=dict)


@dataclass
class StabilityReport:
    """Detailed stability report."""

    suite_id: str
    model_path: str
    status: str
    started_at: str
    completed_at: str | None
    config: dict[str, Any]
    metrics: dict[str, float]
    per_prompt_results: list[dict[str, Any]]
    interpretation: str
    recommendations: list[str]


class StabilityService:
    """Service for model stability testing.

    Runs stability suites to assess model robustness across:
    - Temperature variations
    - Prompt perturbations
    - Repeated sampling
    """

    def __init__(self) -> None:
        """Initialize stability service."""
        self._suites: dict[str, dict[str, Any]] = {}

    def run(
        self,
        model: str,
        config: StabilityConfig | None = None,
    ) -> StabilityRunResult:
        """Execute stability suite on a model.

        Args:
            model: Path to model directory
            config: Optional stability configuration

        Returns:
            StabilityRunResult with suite_id and initial status

        Raises:
            ValueError: If model path is invalid
        """
        model_path = Path(model).expanduser().resolve()

        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        config = config or StabilityConfig()
        suite_id = f"stab-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc).isoformat()

        config_dict = {
            "num_runs": config.num_runs,
            "temperature_range": list(config.temperature_range),
            "prompt_variations": config.prompt_variations,
            "seed": config.seed,
        }

        # Store suite state
        self._suites[suite_id] = {
            "model_path": str(model_path),
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "config": config_dict,
            "metrics": {},
            "per_prompt_results": [],
        }

        logger.info(
            "Started stability suite %s for model %s",
            suite_id,
            model_path,
        )

        # Simulate stability testing
        # In production, this would run actual inference tests
        self._run_stability_tests(suite_id, config)

        return StabilityRunResult(
            suite_id=suite_id,
            model_path=str(model_path),
            status=self._suites[suite_id]["status"],
            started_at=started_at,
            config=config_dict,
            summary=self._suites[suite_id]["metrics"],
        )

    def report(self, suite_id: str) -> StabilityReport:
        """Get detailed stability report for a suite.

        Args:
            suite_id: ID of the stability suite

        Returns:
            StabilityReport with detailed metrics and recommendations

        Raises:
            ValueError: If suite_id is not found
        """
        if suite_id not in self._suites:
            raise ValueError(f"Stability suite not found: {suite_id}")

        suite = self._suites[suite_id]
        metrics = suite["metrics"]

        # Generate interpretation
        consistency = metrics.get("consistency_score", 0.0)
        if consistency >= 0.9:
            interpretation = "Model shows excellent stability across test conditions."
        elif consistency >= 0.7:
            interpretation = "Model shows good stability with minor variations."
        elif consistency >= 0.5:
            interpretation = "Model shows moderate stability. Some inconsistencies detected."
        else:
            interpretation = "Model shows poor stability. Significant inconsistencies detected."

        # Generate recommendations
        recommendations = []
        if consistency < 0.9:
            recommendations.append("Consider fine-tuning with more diverse data")
        if metrics.get("temperature_sensitivity", 0.0) > 0.3:
            recommendations.append("Model is sensitive to temperature changes")
        if metrics.get("prompt_sensitivity", 0.0) > 0.3:
            recommendations.append("Model is sensitive to prompt variations")
        if not recommendations:
            recommendations.append("Model is stable and ready for deployment")

        return StabilityReport(
            suite_id=suite_id,
            model_path=suite["model_path"],
            status=suite["status"],
            started_at=suite["started_at"],
            completed_at=suite["completed_at"],
            config=suite["config"],
            metrics=metrics,
            per_prompt_results=suite["per_prompt_results"],
            interpretation=interpretation,
            recommendations=recommendations,
        )

    def _run_stability_tests(
        self,
        suite_id: str,
        config: StabilityConfig,
    ) -> None:
        """Run stability tests (simulated).

        In production, this would:
        1. Load the model
        2. Run inference with various temperatures
        3. Test prompt variations
        4. Compute consistency metrics
        """
        suite = self._suites[suite_id]

        # Simulated metrics
        suite["metrics"] = {
            "consistency_score": 0.85,
            "temperature_sensitivity": 0.15,
            "prompt_sensitivity": 0.12,
            "output_variance": 0.08,
            "semantic_stability": 0.92,
            "runs_completed": config.num_runs,
        }

        # Simulated per-prompt results
        suite["per_prompt_results"] = [
            {
                "prompt_id": f"prompt-{i}",
                "consistency": 0.8 + (i * 0.02),
                "variance": 0.1 - (i * 0.01),
                "runs": config.num_runs,
            }
            for i in range(config.prompt_variations)
        ]

        suite["status"] = "completed"
        suite["completed_at"] = datetime.now(timezone.utc).isoformat()
