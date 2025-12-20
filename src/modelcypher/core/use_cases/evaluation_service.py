from __future__ import annotations

import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import EvaluationResult


@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    metrics: list[str] | None = None
    batch_size: int = 4
    max_samples: int | None = None


@dataclass
class EvalRunResult:
    """Result of running an evaluation."""
    eval_id: str
    model_path: str
    dataset_path: str
    average_loss: float
    perplexity: float
    sample_count: int


class EvaluationService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

    def list_evaluations(self, limit: int = 50) -> dict:
        results = self.store.list_evaluations(limit)
        return {"evaluations": results}

    def get_evaluation(self, eval_id: str) -> EvaluationResult:
        result = self.store.get_evaluation(eval_id)
        if result is None:
            raise RuntimeError(f"Evaluation not found: {eval_id}")
        return result

    def run(
        self,
        model: str,
        dataset: str,
        config: EvalConfig | None = None,
    ) -> EvalRunResult:
        """Execute evaluation on model with dataset.
        
        Args:
            model: Path to model directory.
            dataset: Path to dataset file.
            config: Optional evaluation configuration.
            
        Returns:
            EvalRunResult with eval_id and metrics.
        """
        config = config or EvalConfig()
        eval_id = f"eval-{uuid.uuid4().hex[:8]}"
        
        # In a full implementation, this would run actual evaluation
        # For now, return placeholder metrics
        result = EvalRunResult(
            eval_id=eval_id,
            model_path=model,
            dataset_path=dataset,
            average_loss=0.5,
            perplexity=1.65,
            sample_count=config.max_samples or 100,
        )
        
        return result

    def results(self, eval_id: str) -> dict:
        """Get detailed per-sample results for an evaluation.
        
        Args:
            eval_id: Evaluation ID.
            
        Returns:
            Dictionary with detailed results.
        """
        evaluation = self.get_evaluation(eval_id)
        return {
            "evalId": evaluation.id,
            "modelPath": evaluation.model_path,
            "datasetPath": evaluation.dataset_path,
            "averageLoss": evaluation.average_loss,
            "perplexity": evaluation.perplexity,
            "sampleCount": evaluation.sample_count,
            "sampleResults": evaluation.sample_results,
        }
