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


class HiddenStateEngine(InferenceEngine, Protocol):
    def capture_hidden_states(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        target_layers: set[int] | None = None,
    ) -> dict[int, list[float]]: ...
