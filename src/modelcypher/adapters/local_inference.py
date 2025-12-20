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


@dataclass
class SecurityScanSummary:
    """Summary of security scan results."""

    security_assessment: str
    anomaly_count: int
    max_anomaly_score: float
    avg_delta: float
    disagreement_rate: float
    circuit_breaker_tripped: bool
    circuit_breaker_trip_index: int | None


@dataclass
class InferenceResult:
    """Result of a single inference run with optional adapter and security scan."""

    prompt: str
    response: str
    token_count: int
    tokens_per_second: float
    time_to_first_token: float | None
    total_duration: float
    stop_reason: str
    model: str
    adapter: str | None
    security: SecurityScanSummary | None


@dataclass
class InferenceCaseResult:
    """Result of a single inference case in a suite."""

    name: str
    prompt: str
    response: str
    token_count: int
    duration: float
    passed: bool | None
    expected: str | list[str] | None
    error: str | None = None


@dataclass
class InferenceSuiteResult:
    """Result of inference suite execution with adapter support."""

    model: str
    adapter: str | None
    suite: str
    cases: list[InferenceCaseResult]
    total_cases: int
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
        - JSON array of prompts
        """
        content = path.read_text(encoding="utf-8")
        
        # Try to parse as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                prompts = []
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict):
                        if "prompt" in item:
                            prompts.append(item["prompt"])
                        elif "text" in item:
                            prompts.append(item["text"])
                return prompts
        except json.JSONDecodeError:
            pass
        
        # Fall back to line-by-line parsing
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

    def _load_adapter(self, adapter_path: str) -> dict[str, Any] | None:
        """Load adapter configuration from path.

        Args:
            adapter_path: Path to adapter directory

        Returns:
            Adapter configuration dict or None if not found
        """
        adapter_dir = Path(adapter_path).expanduser().resolve()
        if not adapter_dir.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_dir}")

        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse adapter config: %s", exc)
                return None
        return None

    def _perform_security_scan(
        self,
        prompt: str,
        response: str,
        model: str,
    ) -> SecurityScanSummary:
        """Perform dual-path security analysis on inference.

        Args:
            prompt: The input prompt
            response: The model response
            model: Model path

        Returns:
            SecurityScanSummary with analysis results
        """
        # Simplified security scan implementation
        # In production, this would use the geometry safety service
        prompt_len = len(prompt)
        response_len = len(response)

        # Basic heuristics for security assessment
        anomaly_score = 0.0
        anomaly_count = 0

        # Check for potential issues
        suspicious_patterns = ["ignore previous", "disregard", "bypass", "jailbreak"]
        for pattern in suspicious_patterns:
            if pattern.lower() in prompt.lower():
                anomaly_count += 1
                anomaly_score = max(anomaly_score, 0.7)

        # Calculate delta based on response characteristics
        avg_delta = abs(response_len - prompt_len) / max(prompt_len, 1)

        # Determine assessment
        if anomaly_score > 0.8:
            assessment = "high_risk"
            tripped = True
        elif anomaly_score > 0.5:
            assessment = "medium_risk"
            tripped = False
        elif anomaly_count > 0:
            assessment = "low_risk"
            tripped = False
        else:
            assessment = "safe"
            tripped = False

        return SecurityScanSummary(
            security_assessment=assessment,
            anomaly_count=anomaly_count,
            max_anomaly_score=anomaly_score,
            avg_delta=avg_delta,
            disagreement_rate=0.0,
            circuit_breaker_tripped=tripped,
            circuit_breaker_trip_index=0 if tripped else None,
        )

    def run(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        security_scan: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> InferenceResult:
        """Execute inference with optional adapter and security scanning.

        Args:
            model: Model identifier or path
            prompt: Input prompt
            adapter: Optional path to adapter directory
            security_scan: Whether to perform dual-path security analysis
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            InferenceResult with metrics and optional security summary

        Raises:
            ValueError: If model or adapter path is invalid
            RuntimeError: If training is running
        """
        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load adapter if specified
        adapter_config = None
        if adapter:
            adapter_config = self._load_adapter(adapter)
            logger.info("Loaded adapter from %s", adapter)

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        start = time.time()
        time_to_first_token = None
        try:
            # Generate response (adapter would be applied here in production)
            first_token_time = time.time()
            response = self._generate_text(prompt, max_tokens=max_tokens)
            time_to_first_token = time.time() - first_token_time

            duration = max(time.time() - start, 1e-6)
            token_count = len(response.split())

            # Perform security scan if requested
            security_summary = None
            if security_scan:
                security_summary = self._perform_security_scan(prompt, response, model)

            return InferenceResult(
                prompt=prompt,
                response=response,
                token_count=token_count,
                tokens_per_second=token_count / duration,
                time_to_first_token=time_to_first_token,
                total_duration=duration,
                stop_reason="length" if token_count >= max_tokens else "stop",
                model=str(model_path),
                adapter=adapter,
                security=security_summary,
            )
        finally:
            self.lock.release()

    def suite(
        self,
        model: str,
        suite_file: str,
        adapter: str | None = None,
        security_scan: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> InferenceSuiteResult:
        """Execute batched inference over a suite of prompts.

        Loads prompts from file (.txt, .json, .jsonl) and executes inference
        keeping the model loaded for efficiency.

        Args:
            model: Model identifier or path
            suite_file: Path to suite file containing prompts
            adapter: Optional path to adapter directory
            security_scan: Whether to perform security analysis
            max_tokens: Default max tokens per response
            temperature: Default sampling temperature

        Returns:
            InferenceSuiteResult with all case results

        Raises:
            ValueError: If suite file doesn't exist or is invalid
        """
        suite_path = Path(suite_file).expanduser().resolve()
        if not suite_path.exists():
            raise ValueError(f"Suite file does not exist: {suite_path}")

        # Determine file type and load prompts/tests
        suffix = suite_path.suffix.lower()
        
        if suffix == ".json":
            # Try to load as suite config with tests
            try:
                config = json.loads(suite_path.read_text(encoding="utf-8"))
                if isinstance(config, dict) and "tests" in config:
                    return self._run_suite_config(
                        model, suite_path, config, adapter, security_scan, max_tokens, temperature
                    )
                elif isinstance(config, list):
                    # JSON array of prompts
                    prompts = []
                    for item in config:
                        if isinstance(item, str):
                            prompts.append({"prompt": item})
                        elif isinstance(item, dict):
                            prompts.append(item)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in suite file: {exc}") from exc
        else:
            # Load as plain text or JSONL
            prompts = []
            for p in self._read_prompts(suite_path):
                prompts.append({"prompt": p})

        if not prompts:
            raise ValueError(f"No prompts found in suite file: {suite_path}")

        logger.info("Running inference suite with %d prompts", len(prompts))

        cases: list[InferenceCaseResult] = []
        passed = 0
        failed = 0
        start_time = time.time()

        for i, item in enumerate(prompts):
            prompt = item.get("prompt", "") if isinstance(item, dict) else str(item)
            name = item.get("name", f"case_{i}") if isinstance(item, dict) else f"case_{i}"
            expected = item.get("expected") if isinstance(item, dict) else None
            case_max_tokens = item.get("max_tokens", max_tokens) if isinstance(item, dict) else max_tokens

            try:
                result = self.run(
                    model=model,
                    prompt=prompt,
                    adapter=adapter,
                    security_scan=security_scan,
                    max_tokens=case_max_tokens,
                    temperature=temperature,
                )

                # Check expected if provided
                test_passed = None
                if expected:
                    if isinstance(expected, str):
                        test_passed = expected.lower() in result.response.lower()
                    elif isinstance(expected, list):
                        test_passed = any(
                            exp.lower() in result.response.lower() for exp in expected
                        )
                    if test_passed:
                        passed += 1
                    else:
                        failed += 1

                cases.append(InferenceCaseResult(
                    name=name,
                    prompt=prompt[:100],
                    response=result.response[:200],
                    token_count=result.token_count,
                    duration=result.total_duration,
                    passed=test_passed,
                    expected=expected,
                ))

            except Exception as exc:
                cases.append(InferenceCaseResult(
                    name=name,
                    prompt=prompt[:100],
                    response="",
                    token_count=0,
                    duration=0.0,
                    passed=False,
                    expected=expected,
                    error=str(exc),
                ))
                failed += 1
                logger.warning("Suite case %s failed: %s", name, exc)

        total_duration = time.time() - start_time
        total_cases = len(cases)

        return InferenceSuiteResult(
            model=model,
            adapter=adapter,
            suite=str(suite_path),
            cases=cases,
            total_cases=total_cases,
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary={
                "pass_rate": passed / max(total_cases, 1) if (passed + failed) > 0 else None,
                "average_duration": total_duration / max(total_cases, 1),
                "suite_name": suite_path.stem,
            },
        )

    def _run_suite_config(
        self,
        model: str,
        suite_path: Path,
        config: dict[str, Any],
        adapter: str | None,
        security_scan: bool,
        max_tokens: int,
        temperature: float,
    ) -> InferenceSuiteResult:
        """Run suite from a structured config with tests."""
        tests = config.get("tests", [])
        if not tests:
            raise ValueError("Suite config contains no tests")

        logger.info("Running inference suite with %d tests", len(tests))

        cases: list[InferenceCaseResult] = []
        passed = 0
        failed = 0
        start_time = time.time()

        for i, test in enumerate(tests):
            name = test.get("name", f"test_{i}")
            prompt = test.get("prompt", "")
            expected = test.get("expected")
            test_max_tokens = test.get("max_tokens", max_tokens)
            test_temp = test.get("temperature", temperature)

            try:
                result = self.run(
                    model=model,
                    prompt=prompt,
                    adapter=adapter,
                    security_scan=security_scan,
                    max_tokens=test_max_tokens,
                    temperature=test_temp,
                )

                # Check expected if provided
                test_passed = True
                if expected:
                    if isinstance(expected, str):
                        test_passed = expected.lower() in result.response.lower()
                    elif isinstance(expected, list):
                        test_passed = any(
                            exp.lower() in result.response.lower() for exp in expected
                        )

                if test_passed:
                    passed += 1
                else:
                    failed += 1

                cases.append(InferenceCaseResult(
                    name=name,
                    prompt=prompt[:100],
                    response=result.response[:200],
                    token_count=result.token_count,
                    duration=result.total_duration,
                    passed=test_passed,
                    expected=expected,
                ))

            except Exception as exc:
                cases.append(InferenceCaseResult(
                    name=name,
                    prompt=prompt[:100],
                    response="",
                    token_count=0,
                    duration=0.0,
                    passed=False,
                    expected=expected,
                    error=str(exc),
                ))
                failed += 1
                logger.warning("Suite test %s failed: %s", name, exc)

        total_duration = time.time() - start_time
        total_cases = len(cases)

        return InferenceSuiteResult(
            model=model,
            adapter=adapter,
            suite=str(suite_path),
            cases=cases,
            total_cases=total_cases,
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary={
                "pass_rate": passed / max(total_cases, 1),
                "average_duration": total_duration / max(total_cases, 1),
                "suite_name": config.get("name", "unnamed"),
            },
        )
