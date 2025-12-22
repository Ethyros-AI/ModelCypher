"""Ensemble service for managing model ensembles."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.local_inference import LocalInferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for a model ensemble."""

    ensemble_id: str
    models: list[str]
    routing_strategy: str
    weights: list[float] | None = None
    created_at: str = ""
    config_path: str = ""


@dataclass
class ModelContribution:
    """Contribution from a single model in ensemble inference."""

    model: str
    response: str
    weight: float
    token_count: int
    duration: float


@dataclass
class EnsembleInferenceResult:
    """Result of ensemble inference."""

    ensemble_id: str
    prompt: str
    response: str
    model_contributions: dict[str, float]
    total_duration: float
    strategy: str
    models_used: int
    aggregation_method: str


class EnsembleService:
    """Service for creating and running model ensembles."""

    DEFAULT_LIST_LIMIT = 50
    ENSEMBLE_CONFIG_SUFFIX = ".json"

    VALID_STRATEGIES = {"weighted", "routing", "voting", "cascade"}

    def __init__(
        self,
        store: FileSystemStore | None = None,
        inference_engine: LocalInferenceEngine | None = None,
    ) -> None:
        self._store = store or FileSystemStore()
        self._inference_engine = inference_engine or LocalInferenceEngine()
        self._ensembles_dir = self._store.paths.base / "ensembles"
        self._ensembles_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        model_paths: list[str],
        strategy: str = "weighted",
        weights: list[float] | None = None,
    ) -> EnsembleConfig:
        """Create an ensemble configuration.

        Args:
            model_paths: List of paths to models to include in ensemble
            strategy: Routing strategy - "weighted", "routing", "voting", or "cascade"
            weights: Optional weights for weighted strategy (must sum to 1.0)

        Returns:
            EnsembleConfig with ensemble_id and configuration

        Raises:
            ValueError: If model paths are invalid or strategy is unsupported
        """
        if not model_paths:
            raise ValueError("At least one model path is required")

        if len(model_paths) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid strategies: {', '.join(sorted(self.VALID_STRATEGIES))}"
            )

        # Validate model paths exist
        validated_paths: list[str] = []
        for path in model_paths:
            resolved = Path(path).expanduser().resolve()
            if not resolved.exists():
                raise ValueError(f"Model path does not exist: {path}")
            if not resolved.is_dir():
                raise ValueError(f"Model path is not a directory: {path}")
            validated_paths.append(str(resolved))

        # Validate weights if provided
        if weights is not None:
            if len(weights) != len(model_paths):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(model_paths)})"
                )
            if not all(w >= 0 for w in weights):
                raise ValueError("All weights must be non-negative")
            weight_sum = sum(weights)
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        elif strategy == "weighted":
            # Default to equal weights for weighted strategy
            weights = [1.0 / len(model_paths)] * len(model_paths)

        # Generate ensemble ID
        ensemble_id = f"ensemble-{uuid.uuid4().hex[:8]}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Create config file
        config_path = self._ensembles_dir / f"{ensemble_id}.json"
        config_data = {
            "ensemble_id": ensemble_id,
            "models": validated_paths,
            "routing_strategy": strategy,
            "weights": weights,
            "created_at": created_at,
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        logger.info(
            "Created ensemble %s with %d models using %s strategy",
            ensemble_id,
            len(validated_paths),
            strategy,
        )

        return EnsembleConfig(
            ensemble_id=ensemble_id,
            models=validated_paths,
            routing_strategy=strategy,
            weights=weights,
            created_at=created_at,
            config_path=str(config_path),
        )

    def _load_config(self, ensemble_id: str) -> dict[str, Any]:
        """Load ensemble configuration from disk.

        Args:
            ensemble_id: The ensemble identifier

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If ensemble not found
        """
        config_path = self._ensembles_dir / f"{ensemble_id}.json"
        if not config_path.exists():
            raise ValueError(f"Ensemble not found: {ensemble_id}")

        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid ensemble config: {exc}") from exc

    def list_ensembles(self, limit: int | None = None) -> list[EnsembleConfig]:
        """List available ensemble configurations.

        Args:
            limit: Optional maximum number of ensembles to return.

        Returns:
            List of EnsembleConfig entries.
        """
        max_items = limit if limit is not None else self.DEFAULT_LIST_LIMIT
        configs: list[EnsembleConfig] = []
        for config_path in sorted(self._ensembles_dir.glob(f"*{self.ENSEMBLE_CONFIG_SUFFIX}")):
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid ensemble config: %s", config_path)
                continue

            config = EnsembleConfig(
                ensemble_id=data.get("ensemble_id", config_path.stem),
                models=data.get("models", []),
                routing_strategy=data.get("routing_strategy", "weighted"),
                weights=data.get("weights"),
                created_at=data.get("created_at", ""),
                config_path=str(config_path),
            )
            configs.append(config)
            if len(configs) >= max_items:
                break

        return configs

    def delete(self, ensemble_id: str) -> bool:
        """Delete an ensemble configuration.

        Args:
            ensemble_id: Ensemble identifier.

        Returns:
            True if deleted, False if not found.
        """
        config_path = self._ensembles_dir / f"{ensemble_id}{self.ENSEMBLE_CONFIG_SUFFIX}"
        if not config_path.exists():
            return False
        config_path.unlink()
        return True

    def run(
        self,
        ensemble_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> EnsembleInferenceResult:
        """Execute ensemble inference.

        Args:
            ensemble_id: The ensemble identifier
            prompt: Input prompt
            max_tokens: Maximum tokens to generate per model
            temperature: Sampling temperature

        Returns:
            EnsembleInferenceResult with aggregated response

        Raises:
            ValueError: If ensemble not found or inference fails
        """
        config = self._load_config(ensemble_id)
        models = config["models"]
        strategy = config["routing_strategy"]
        weights = config.get("weights")

        if not models:
            raise ValueError("Ensemble has no models configured")

        start_time = time.time()
        contributions: list[ModelContribution] = []

        # Execute inference on each model
        for i, model_path in enumerate(models):
            model_weight = weights[i] if weights else 1.0 / len(models)
            try:
                result = self._inference_engine.run(
                    model=model_path,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                contributions.append(
                    ModelContribution(
                        model=model_path,
                        response=result.response,
                        weight=model_weight,
                        token_count=result.token_count,
                        duration=result.total_duration,
                    )
                )
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_path, exc)
                # Continue with other models

        if not contributions:
            raise ValueError("All models failed during ensemble inference")

        # Aggregate responses based on strategy
        aggregated_response, aggregation_method = self._aggregate_responses(
            contributions, strategy
        )

        total_duration = time.time() - start_time

        # Build contribution map
        model_contributions = {
            c.model: c.weight for c in contributions
        }

        return EnsembleInferenceResult(
            ensemble_id=ensemble_id,
            prompt=prompt,
            response=aggregated_response,
            model_contributions=model_contributions,
            total_duration=total_duration,
            strategy=strategy,
            models_used=len(contributions),
            aggregation_method=aggregation_method,
        )

    def _aggregate_responses(
        self,
        contributions: list[ModelContribution],
        strategy: str,
    ) -> tuple[str, str]:
        """Aggregate model responses based on strategy.

        Args:
            contributions: List of model contributions
            strategy: Aggregation strategy

        Returns:
            Tuple of (aggregated_response, aggregation_method)
        """
        if strategy == "weighted":
            # Weight-based selection: pick response from highest-weighted model
            # In a real implementation, this would do weighted token-level aggregation
            best = max(contributions, key=lambda c: c.weight)
            return best.response, "weighted_selection"

        elif strategy == "voting":
            # Simple voting: pick most common response (or first if all different)
            responses = [c.response for c in contributions]
            from collections import Counter
            counts = Counter(responses)
            most_common = counts.most_common(1)[0][0]
            return most_common, "majority_vote"

        elif strategy == "cascade":
            # Cascade: use first successful response
            return contributions[0].response, "cascade_first"

        elif strategy == "routing":
            # Routing: in production, would route based on prompt characteristics
            # For now, use weighted selection
            best = max(contributions, key=lambda c: c.weight)
            return best.response, "routing_selection"

        else:
            # Default to first response
            return contributions[0].response, "default"

    def delete(self, ensemble_id: str) -> bool:
        """Delete an ensemble configuration.

        Args:
            ensemble_id: The ensemble identifier

        Returns:
            True if deleted, False if not found
        """
        config_path = self._ensembles_dir / f"{ensemble_id}.json"
        if config_path.exists():
            config_path.unlink()
            logger.info("Deleted ensemble %s", ensemble_id)
            return True
        return False
