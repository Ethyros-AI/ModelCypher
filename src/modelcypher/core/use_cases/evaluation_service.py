from __future__ import annotations

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import EvaluationResult


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
