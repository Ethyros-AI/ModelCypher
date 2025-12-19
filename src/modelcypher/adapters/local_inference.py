from __future__ import annotations

import time

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.ports.inference import InferenceEngine
from modelcypher.utils.locks import FileLock, FileLockError


class LocalInferenceEngine(InferenceEngine):
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()
        self.lock = FileLock(self.store.paths.base / "training.lock")

    def infer(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        start = time.time()
        try:
            response = self._generate_text(prompt, max_tokens=max_tokens)
            duration = max(time.time() - start, 1e-6)
            token_count = len(response.split())
            return {
                "modelId": model,
                "prompt": prompt,
                "response": response,
                "tokenCount": token_count,
                "tokensPerSecond": token_count / duration,
                "timeToFirstToken": duration / max(token_count, 1),
                "totalDuration": duration,
            }
        finally:
            self.lock.release()

    @staticmethod
    def _generate_text(prompt: str, max_tokens: int) -> str:
        words = prompt.split()
        suffix = "response" if words else "response"
        generated = words + [suffix] * min(max_tokens, 16)
        return " ".join(generated)
