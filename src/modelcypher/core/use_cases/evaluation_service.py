from __future__ import annotations

import uuid
from datetime import datetime
from dataclasses import dataclass

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
    adapter_path: str | None = None


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
        adapter: str | None = None,
    ) -> EvalRunResult:
        """Execute evaluation on model with dataset.
        
        Args:
            model: Path to model directory.
            dataset: Path to dataset file.
            config: Optional evaluation configuration.
            adapter: Optional path to LoRA adapter directory.
            
        Returns:
            EvalRunResult with eval_id and metrics.
        """
        config = config or EvalConfig()
        eval_id = f"eval-{uuid.uuid4().hex[:8]}"
        
        # Real MLX Evaluation using mlx_lm
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import numpy as np
            import json
            from mlx_lm import load

            model_path = Path(model)

            # Check if model exists (handle sharded models)
            has_weights = (
                (model_path / "model.safetensors").exists()
                or list(model_path.glob("model-*.safetensors"))
                or (model_path / "config.json").exists() # Some models might only have config if they are remote
            )
            if not has_weights:
                logger.warning(f"No model weights found at {model}, using mock metrics")
                return EvalRunResult(
                    eval_id=eval_id,
                    model_path=model,
                    dataset_path=dataset,
                    average_loss=0.0,
                    perplexity=0.0,
                    sample_count=0,
                )

            # Check dataset exists
            if not Path(dataset).exists():
                raise ValueError(f"Dataset not found: {dataset}")

            # Load model
            logger.info(f"Loading model from {model} (adapter={adapter})")
            llm_model, tokenizer = load(model, adapter_path=adapter)

            # Load and process dataset
            samples = []
            with open(dataset, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get("text", "")
                            if text:
                                samples.append(text)
                        except json.JSONDecodeError:
                            continue

            if config.max_samples:
                samples = samples[: config.max_samples]

            if not samples:
                logger.warning("No valid samples in dataset")
                return EvalRunResult(
                    eval_id=eval_id,
                    model_path=model,
                    dataset_path=dataset,
                    average_loss=0.0,
                    perplexity=0.0,
                    sample_count=0,
                )

            # Compute perplexity over samples
            total_loss = 0.0
            total_tokens = 0

            for text in samples:
                tokens = tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                tokens_mx = mx.array(tokens)
                logits = llm_model(tokens_mx[None, :])
                logits = logits[0, :-1, :]
                targets = tokens_mx[1:]

                log_probs = nn.log_softmax(logits, axis=-1)
                target_log_probs = mx.take_along_axis(
                    log_probs, targets[:, None], axis=-1
                ).squeeze(-1)
                mx.eval(target_log_probs)

                sample_loss = -float(mx.mean(target_log_probs).item())
                total_loss += sample_loss * len(targets)
                total_tokens += len(targets)

            average_loss = total_loss / max(total_tokens, 1)
            perplexity = float(np.exp(average_loss))
            
            sample_count = len(samples)
            logger.info(
                f"Evaluation complete: {sample_count} samples, "
                f"loss={average_loss:.4f}, perplexity={perplexity:.2f}"
            )

            result = EvalRunResult(
                eval_id=eval_id,
                model_path=model,
                dataset_path=dataset,
                average_loss=average_loss,
                perplexity=perplexity,
                sample_count=sample_count,
                adapter_path=adapter,
            )

            # Store result
            self.store.save_evaluation(
                EvaluationResult(
                    id=eval_id,
                    model_path=model,
                    model_name=Path(model).name,
                    dataset_path=dataset,
                    dataset_name=Path(dataset).name,
                    average_loss=average_loss,
                    perplexity=perplexity,
                    sample_count=sample_count,
                    timestamp=datetime.utcnow(),
                    config={"batch_size": config.batch_size},
                    sample_results=[],
                    adapter_path=adapter,
                )
            )
            
            return result
            
        except ImportError:
             logger.error("MLX not installed")
             raise

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
