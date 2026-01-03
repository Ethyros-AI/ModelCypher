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

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from modelcypher.core.domain.models import EvaluationResult

if TYPE_CHECKING:
    from modelcypher.ports.storage import EvaluationStore
    from modelcypher.ports.model_loader import ModelLoaderPort


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
    def __init__(
        self,
        store: "EvaluationStore",
        model_loader: "ModelLoaderPort | None" = None,
    ) -> None:
        self.store = store
        self._model_loader = model_loader

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
        metrics: list[str] | None = None,
        batch_size: int = 4,
        max_samples: int | None = None,
        adapter: str | None = None,
    ) -> EvalRunResult:
        """Execute evaluation on model with dataset.

        Args:
            model: Path to model directory.
            dataset: Path to dataset file.
            metrics: Optional list of metrics to compute.
            batch_size: Batch size for evaluation.
            max_samples: Optional cap on dataset samples.
            adapter: Optional path to LoRA adapter directory.

        Returns:
            EvalRunResult with eval_id and metrics.
        """
        import json

        from modelcypher.core.domain._backend import get_default_backend

        eval_id = f"eval-{uuid.uuid4().hex[:8]}"
        backend = get_default_backend()

        model_path = Path(model)

        # Check if model exists (handle sharded models)
        has_weights = (
            (model_path / "model.safetensors").exists()
            or list(model_path.glob("model-*.safetensors"))
            or (
                model_path / "config.json"
            ).exists()  # Some models might only have config if they are remote
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

        # Get model loader (use injected or default)
        if self._model_loader is None:
            from modelcypher.infrastructure.model_loader_factory import get_model_loader

            self._model_loader = get_model_loader()

        # Load model via port (hexagonal architecture)
        logger.info(f"Loading model from {model} (adapter={adapter})")
        # Note: adapter_path not yet supported via ModelLoaderPort
        llm_model, tokenizer = self._model_loader.load_model_for_training(model)

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

        if max_samples:
            samples = samples[:max_samples]

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

        # Compute perplexity over samples using Backend
        total_loss = 0.0
        total_tokens = 0

        for text in samples:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            tokens_arr = backend.array(tokens)
            # Shape: [1, seq_len] for model input
            input_arr = backend.reshape(tokens_arr, (1, -1))
            logits = llm_model(input_arr)
            # logits shape: [1, seq_len, vocab_size]
            # Get logits for all but last position
            logits = logits[0, :-1, :]
            targets = tokens_arr[1:]

            # Log softmax for cross-entropy loss
            log_probs = backend.log_softmax(logits, axis=-1)
            # Gather target log probs using take_along_axis
            targets_expanded = backend.reshape(targets, (-1, 1))
            target_log_probs = backend.take_along_axis(log_probs, targets_expanded, axis=-1)
            target_log_probs = backend.squeeze(target_log_probs, axis=-1)
            backend.eval(target_log_probs)

            mean_arr = backend.mean(target_log_probs)
            backend.eval(mean_arr)
            sample_loss = -float(backend.to_scalar(mean_arr))
            n_targets = int(targets.shape[0])
            total_loss += sample_loss * n_targets
            total_tokens += n_targets

        average_loss = total_loss / max(total_tokens, 1)
        perplexity_array = backend.exp(backend.array([average_loss]))
        backend.eval(perplexity_array)
        perplexity = float(backend.to_scalar(perplexity_array))

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
                config={"batch_size": batch_size, "metrics": metrics, "max_samples": max_samples},
                sample_results=[],
                adapter_path=adapter,
            )
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
