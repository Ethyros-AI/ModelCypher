from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.ports.inference import InferenceEngine
from modelcypher.utils.locks import FileLock, FileLockError

logger = logging.getLogger(__name__)


@dataclass
class BatchInferResult:
    """Result of batched inference."""

    model_id: str
    prompts_file: str
    results: list[dict[str, Any]]
    total_prompts: int
    successful: int
    failed: int
    total_tokens: int
    total_duration: float
    average_tokens_per_second: float


@dataclass
class SuiteInferResult:
    """Result of inference suite execution."""

    model_id: str
    suite_config: str
    test_results: list[dict[str, Any]]
    total_tests: int
    passed: int
    failed: int
    total_duration: float
    summary: dict[str, Any] = field(default_factory=dict)


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

    def run_batch(
        self,
        model: str,
        prompts_file: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> BatchInferResult:
        """Execute batched inference from a prompts file.

        Args:
            model: Model identifier or path
            prompts_file: Path to file containing prompts (one per line or JSONL)
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            BatchInferResult with all inference results

        Raises:
            ValueError: If prompts file doesn't exist or is invalid
        """
        prompts_path = Path(prompts_file).expanduser().resolve()
        if not prompts_path.exists():
            raise ValueError(f"Prompts file does not exist: {prompts_path}")

        # Read prompts from file
        prompts = self._read_prompts(prompts_path)
        if not prompts:
            raise ValueError(f"No prompts found in file: {prompts_path}")

        logger.info("Running batch inference with %d prompts", len(prompts))

        results = []
        successful = 0
        failed = 0
        total_tokens = 0
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            try:
                result = self.infer(model, prompt, max_tokens, temperature, top_p)
                results.append({
                    "index": i,
                    "prompt": prompt[:100],  # Truncate for response
                    "response": result["response"],
                    "tokenCount": result["tokenCount"],
                    "status": "success",
                })
                successful += 1
                total_tokens += result["tokenCount"]
            except Exception as exc:
                results.append({
                    "index": i,
                    "prompt": prompt[:100],
                    "error": str(exc),
                    "status": "failed",
                })
                failed += 1
                logger.warning("Batch inference failed for prompt %d: %s", i, exc)

        total_duration = time.time() - start_time
        avg_tps = total_tokens / max(total_duration, 1e-6)

        return BatchInferResult(
            model_id=model,
            prompts_file=str(prompts_path),
            results=results,
            total_prompts=len(prompts),
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_duration=total_duration,
            average_tokens_per_second=avg_tps,
        )

    def run_suite(
        self,
        model: str,
        suite_config: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> SuiteInferResult:
        """Execute inference suite from a configuration file.

        Args:
            model: Model identifier or path
            suite_config: Path to suite configuration (JSON)
            max_tokens: Default max tokens per response
            temperature: Default sampling temperature

        Returns:
            SuiteInferResult with test results and summary

        Raises:
            ValueError: If suite config doesn't exist or is invalid
        """
        config_path = Path(suite_config).expanduser().resolve()
        if not config_path.exists():
            raise ValueError(f"Suite config does not exist: {config_path}")

        # Load suite configuration
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid suite config JSON: {exc}") from exc

        tests = config.get("tests", [])
        if not tests:
            raise ValueError("Suite config contains no tests")

        logger.info("Running inference suite with %d tests", len(tests))

        test_results = []
        passed = 0
        failed = 0
        start_time = time.time()

        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            prompt = test.get("prompt", "")
            expected = test.get("expected", None)
            test_max_tokens = test.get("max_tokens", max_tokens)
            test_temp = test.get("temperature", temperature)

            try:
                result = self.infer(model, prompt, test_max_tokens, test_temp, 0.95)
                response = result["response"]

                # Check if expected pattern is in response
                test_passed = True
                if expected:
                    if isinstance(expected, str):
                        test_passed = expected.lower() in response.lower()
                    elif isinstance(expected, list):
                        test_passed = any(
                            exp.lower() in response.lower() for exp in expected
                        )

                test_results.append({
                    "name": test_name,
                    "prompt": prompt[:100],
                    "response": response[:200],
                    "expected": expected,
                    "passed": test_passed,
                    "tokenCount": result["tokenCount"],
                    "duration": result["totalDuration"],
                })

                if test_passed:
                    passed += 1
                else:
                    failed += 1

            except Exception as exc:
                test_results.append({
                    "name": test_name,
                    "prompt": prompt[:100],
                    "error": str(exc),
                    "passed": False,
                })
                failed += 1
                logger.warning("Suite test %s failed: %s", test_name, exc)

        total_duration = time.time() - start_time

        return SuiteInferResult(
            model_id=model,
            suite_config=str(config_path),
            test_results=test_results,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary={
                "pass_rate": passed / max(len(tests), 1),
                "average_duration": total_duration / max(len(tests), 1),
                "suite_name": config.get("name", "unnamed"),
            },
        )

    def _read_prompts(self, path: Path) -> list[str]:
        """Read prompts from a file.

        Supports:
        - Plain text (one prompt per line)
        - JSONL with "prompt" field
        """
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        prompts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    if "prompt" in data:
                        prompts.append(data["prompt"])
                    elif "text" in data:
                        prompts.append(data["text"])
                    else:
                        prompts.append(line)
                except json.JSONDecodeError:
                    prompts.append(line)
            else:
                prompts.append(line)

        return prompts

    @staticmethod
    def _generate_text(prompt: str, max_tokens: int) -> str:
        words = prompt.split()
        suffix = "response" if words else "response"
        generated = words + [suffix] * min(max_tokens, 16)
        return " ".join(generated)
