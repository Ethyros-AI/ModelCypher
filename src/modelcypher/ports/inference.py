from __future__ import annotations

from typing import Protocol


class InferenceEngine(Protocol):
    def infer(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict: ...
