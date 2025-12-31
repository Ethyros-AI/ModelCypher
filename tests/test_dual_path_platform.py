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

"""Tests for dual-path platform selection and configuration.

These tests verify the platform selection logic and dataclass configurations
for the dual-path generator across MLX, CUDA, and JAX backends.
"""

import os
from unittest import mock

import pytest


class TestPlatformDetection:
    """Tests for _is_*_available() helper functions."""

    def test_is_mlx_available_on_darwin(self):
        """MLX availability check on macOS."""
        from modelcypher.core.domain.inference._platform import _is_mlx_available

        # On macOS with MLX installed, should return True
        # This test runs on the current platform
        import platform as plat

        if plat.system() == "Darwin":
            # Clear conftest's MC_DISABLE_MLX to test real detection
            clean_env = {"MC_DISABLE_MLX": ""}
            with mock.patch.dict(os.environ, clean_env, clear=False):
                # MLX should be available on macOS test runners
                assert _is_mlx_available() is True

    def test_is_mlx_disabled_by_env(self):
        """MLX can be disabled via MC_DISABLE_MLX env var."""
        from modelcypher.core.domain.inference._platform import _is_mlx_available

        with mock.patch.dict(os.environ, {"MC_DISABLE_MLX": "1"}):
            assert _is_mlx_available() is False

        with mock.patch.dict(os.environ, {"MC_DISABLE_MLX": "true"}):
            assert _is_mlx_available() is False

        with mock.patch.dict(os.environ, {"MC_DISABLE_MLX": "yes"}):
            assert _is_mlx_available() is False

    def test_is_mlx_available_non_darwin(self):
        """MLX not available on non-Darwin platforms."""
        from modelcypher.core.domain.inference._platform import _is_mlx_available

        with mock.patch("platform.system", return_value="Linux"):
            assert _is_mlx_available() is False

    def test_is_cuda_available_import_error(self):
        """CUDA not available when torch is not installed."""
        # We can't easily mock import failures, but we can verify the function exists
        from modelcypher.core.domain.inference._platform import _is_cuda_available

        # Just verify it returns a boolean
        result = _is_cuda_available()
        assert isinstance(result, bool)

    def test_is_jax_available_import_error(self):
        """JAX availability check."""
        from modelcypher.core.domain.inference._platform import _is_jax_available

        # Just verify it returns a boolean
        result = _is_jax_available()
        assert isinstance(result, bool)


class TestGetInferencePlatform:
    """Tests for get_inference_platform() function."""

    def test_env_override_mc_backend(self):
        """MC_BACKEND env var overrides auto-detection."""
        from modelcypher.core.domain.inference._platform import get_inference_platform

        with mock.patch.dict(os.environ, {"MC_BACKEND": "mlx"}, clear=False):
            assert get_inference_platform() == "mlx"

        with mock.patch.dict(os.environ, {"MC_BACKEND": "cuda"}, clear=False):
            assert get_inference_platform() == "cuda"

        with mock.patch.dict(os.environ, {"MC_BACKEND": "jax"}, clear=False):
            assert get_inference_platform() == "jax"

    def test_env_override_modelcypher_backend(self):
        """MODELCYPHER_BACKEND env var overrides auto-detection."""
        from modelcypher.core.domain.inference._platform import get_inference_platform

        # Remove MC_BACKEND if present
        env = {"MODELCYPHER_BACKEND": "cuda"}
        if "MC_BACKEND" in os.environ:
            env["MC_BACKEND"] = ""
        with mock.patch.dict(os.environ, env, clear=False):
            assert get_inference_platform() == "cuda"

    def test_auto_detect_returns_valid_platform(self):
        """Auto-detection returns one of the known platforms."""
        from modelcypher.core.domain.inference._platform import get_inference_platform

        # Clear override env vars for true auto-detect
        clean_env = {"MC_BACKEND": "", "MODELCYPHER_BACKEND": ""}
        with mock.patch.dict(os.environ, clean_env, clear=False):
            platform = get_inference_platform()
            assert platform in ("mlx", "cuda", "jax", "cpu")

    def test_mlx_priority_on_darwin(self):
        """MLX takes priority on Darwin even if CUDA/JAX available."""
        from modelcypher.core.domain.inference._platform import get_inference_platform

        import platform as plat

        if plat.system() == "Darwin":
            # Clear conftest's MC_DISABLE_MLX to test real detection
            clean_env = {"MC_BACKEND": "", "MODELCYPHER_BACKEND": "", "MC_DISABLE_MLX": ""}
            with mock.patch.dict(os.environ, clean_env, clear=False):
                # On macOS, MLX should be selected
                assert get_inference_platform() == "mlx"


class TestGetDualPathGeneratorClass:
    """Tests for get_dual_path_generator_class() function."""

    def test_returns_class_type(self):
        """Generator class getter returns a type."""
        from modelcypher.core.domain.inference._platform import (
            get_dual_path_generator_class,
        )

        cls = get_dual_path_generator_class()
        assert isinstance(cls, type)

    def test_mlx_returns_dual_path_generator(self):
        """MLX platform returns DualPathGenerator class."""
        with mock.patch.dict(os.environ, {"MC_BACKEND": "mlx"}, clear=False):
            from modelcypher.core.domain.inference._platform import (
                get_dual_path_generator_class,
            )

            cls = get_dual_path_generator_class()
            assert cls.__name__ == "DualPathGenerator"

    def test_cpu_raises_not_implemented(self):
        """CPU platform raises NotImplementedError."""
        from modelcypher.core.domain.inference._platform import (
            get_dual_path_generator_class,
        )

        # Force CPU by mocking all availability checks
        with mock.patch(
            "modelcypher.core.domain.inference._platform._is_mlx_available",
            return_value=False,
        ):
            with mock.patch(
                "modelcypher.core.domain.inference._platform._is_cuda_available",
                return_value=False,
            ):
                with mock.patch(
                    "modelcypher.core.domain.inference._platform._is_jax_available",
                    return_value=False,
                ):
                    with mock.patch.dict(
                        os.environ, {"MC_BACKEND": "", "MODELCYPHER_BACKEND": ""}, clear=False
                    ):
                        with pytest.raises(NotImplementedError, match="cpu"):
                            get_dual_path_generator_class()


class TestGetDualPathConfigClass:
    """Tests for get_dual_path_config_class() function."""

    def test_returns_class_type(self):
        """Config class getter returns a type."""
        from modelcypher.core.domain.inference._platform import get_dual_path_config_class

        cls = get_dual_path_config_class()
        assert isinstance(cls, type)

    def test_mlx_returns_configuration(self):
        """MLX platform returns DualPathGeneratorConfiguration class."""
        with mock.patch.dict(os.environ, {"MC_BACKEND": "mlx"}, clear=False):
            from modelcypher.core.domain.inference._platform import get_dual_path_config_class

            cls = get_dual_path_config_class()
            assert cls.__name__ == "DualPathGeneratorConfiguration"

    def test_cpu_raises_not_implemented(self):
        """CPU platform raises NotImplementedError for config."""
        from modelcypher.core.domain.inference._platform import get_dual_path_config_class

        with mock.patch(
            "modelcypher.core.domain.inference._platform._is_mlx_available",
            return_value=False,
        ):
            with mock.patch(
                "modelcypher.core.domain.inference._platform._is_cuda_available",
                return_value=False,
            ):
                with mock.patch(
                    "modelcypher.core.domain.inference._platform._is_jax_available",
                    return_value=False,
                ):
                    with mock.patch.dict(
                        os.environ, {"MC_BACKEND": "", "MODELCYPHER_BACKEND": ""}, clear=False
                    ):
                        with pytest.raises(NotImplementedError, match="cpu"):
                            get_dual_path_config_class()


class TestGetSecurityScanMetricsClass:
    """Tests for get_security_scan_metrics_class() function."""

    def test_returns_class_type(self):
        """Metrics class getter returns a type."""
        from modelcypher.core.domain.inference._platform import get_security_scan_metrics_class

        cls = get_security_scan_metrics_class()
        assert isinstance(cls, type)

    def test_mlx_returns_security_scan_metrics(self):
        """MLX platform returns SecurityScanMetrics class."""
        with mock.patch.dict(os.environ, {"MC_BACKEND": "mlx"}, clear=False):
            from modelcypher.core.domain.inference._platform import get_security_scan_metrics_class

            cls = get_security_scan_metrics_class()
            assert cls.__name__ == "SecurityScanMetrics"


class TestSecurityScanMetrics:
    """Tests for SecurityScanMetrics dataclass."""

    def test_metrics_creation(self):
        """SecurityScanMetrics can be created with all fields."""
        from modelcypher.core.domain.inference.dual_path_mlx import SecurityScanMetrics

        metrics = SecurityScanMetrics(
            token_count=100,
            time_to_first_token_ms=50.5,
            total_time_ms=1000.0,
            tokens_per_second=100.0,
            circuit_breaker_tripped=False,
            anomaly_alert_count=0,
        )

        assert metrics.token_count == 100
        assert metrics.time_to_first_token_ms == 50.5
        assert metrics.total_time_ms == 1000.0
        assert metrics.tokens_per_second == 100.0
        assert metrics.circuit_breaker_tripped is False
        assert metrics.anomaly_alert_count == 0

    def test_metrics_with_alerts(self):
        """Metrics can track anomaly alerts and circuit breaker."""
        from modelcypher.core.domain.inference.dual_path_mlx import SecurityScanMetrics

        metrics = SecurityScanMetrics(
            token_count=50,
            time_to_first_token_ms=100.0,
            total_time_ms=500.0,
            tokens_per_second=100.0,
            circuit_breaker_tripped=True,
            anomaly_alert_count=3,
        )

        assert metrics.circuit_breaker_tripped is True
        assert metrics.anomaly_alert_count == 3


class TestDualPathGeneratorConfiguration:
    """Tests for DualPathGeneratorConfiguration dataclass."""

    def _make_tracker_config(self):
        """Create a valid tracker configuration for testing."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker

        return EntropyDeltaTracker.Configuration(
            top_k=10,
            anomaly_threshold=0.8,
            consecutive_anomaly_count=3,
            compute_variance=True,
            source="test",
        )

    def test_config_creation_minimal(self):
        """Configuration can be created with minimal required fields."""
        from modelcypher.core.domain.inference.dual_path_mlx import (
            DualPathGeneratorConfiguration,
        )

        tracker_config = self._make_tracker_config()
        config = DualPathGeneratorConfiguration(
            base_model_path="/path/to/model",
            delta_tracker_config=tracker_config,
        )

        assert config.base_model_path == "/path/to/model"
        assert config.adapter_path is None
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.repetition_penalty == 1.0
        assert config.stop_sequences == []
        assert config.halt_on_circuit_breaker is True

    def test_config_creation_full(self):
        """Configuration can be created with all fields."""
        from modelcypher.core.domain.inference.dual_path_mlx import (
            DualPathGeneratorConfiguration,
        )

        tracker_config = self._make_tracker_config()
        config = DualPathGeneratorConfiguration(
            base_model_path="/path/to/model",
            delta_tracker_config=tracker_config,
            adapter_path="/path/to/adapter",
            max_tokens=256,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            stop_sequences=[".", "?", "!"],
            halt_on_circuit_breaker=False,
        )

        assert config.adapter_path == "/path/to/adapter"
        assert config.max_tokens == 256
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.2
        assert config.stop_sequences == [".", "?", "!"]
        assert config.halt_on_circuit_breaker is False

    def test_config_with_zero_temperature(self):
        """Zero temperature (greedy) is valid."""
        from modelcypher.core.domain.inference.dual_path_mlx import (
            DualPathGeneratorConfiguration,
        )

        tracker_config = self._make_tracker_config()
        config = DualPathGeneratorConfiguration(
            base_model_path="/path/to/model",
            delta_tracker_config=tracker_config,
            temperature=0.0,
        )

        assert config.temperature == 0.0


class TestEntropyDeltaTrackerConfiguration:
    """Tests for EntropyDeltaTracker.Configuration."""

    def test_configuration_creation(self):
        """Configuration can be created with all required fields."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker

        config = EntropyDeltaTracker.Configuration(
            top_k=10,
            anomaly_threshold=0.8,
            consecutive_anomaly_count=3,
            compute_variance=True,
            source="test",
        )

        assert config.top_k == 10
        assert config.anomaly_threshold == 0.8
        assert config.consecutive_anomaly_count == 3
        assert config.compute_variance is True
        assert config.source == "test"

    def test_from_baseline_distribution(self):
        """Configuration can be derived from baseline distribution."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker

        # Create sample anomaly scores (simulating baseline)
        samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        config = EntropyDeltaTracker.Configuration.from_baseline_distribution(
            samples,
            alert_percentile=0.90,
            consecutive_count=5,
            top_k=20,
        )

        assert config.top_k == 20
        assert config.consecutive_anomaly_count == 5
        # 90th percentile of [0.1...1.0] is 0.9
        assert config.anomaly_threshold == pytest.approx(0.9, abs=0.1)


class TestEntropyDeltaSample:
    """Tests for EntropyDeltaSample dataclass."""

    def test_sample_creation(self):
        """EntropyDeltaSample can be created with all fields."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaSample

        sample = EntropyDeltaSample(
            token_index=0,
            generated_token=12345,
            base_entropy=2.5,
            base_top_k_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.8,
            adapter_top_k_variance=0.15,
            adapter_top_token=200,
            latency_ms=5.0,
        )

        assert sample.token_index == 0
        assert sample.generated_token == 12345
        assert sample.base_entropy == 2.5
        assert sample.adapter_entropy == 2.8
        assert sample.base_top_token == 100
        assert sample.adapter_top_token == 200

    def test_sample_delta_property(self):
        """EntropyDeltaSample computes delta correctly."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaSample

        sample = EntropyDeltaSample(
            token_index=0,
            generated_token=100,
            base_entropy=3.0,
            base_top_k_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.0,
            adapter_top_k_variance=0.15,
            adapter_top_token=100,
            latency_ms=5.0,
        )

        # delta = base - adapter
        assert sample.delta == pytest.approx(1.0)

    def test_sample_top_token_disagreement(self):
        """EntropyDeltaSample detects top token disagreement."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaSample

        # Same top token
        same = EntropyDeltaSample(
            token_index=0,
            generated_token=100,
            base_entropy=2.0,
            base_top_k_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.0,
            adapter_top_k_variance=0.1,
            adapter_top_token=100,
            latency_ms=5.0,
        )
        assert same.top_token_disagreement is False

        # Different top tokens
        different = EntropyDeltaSample(
            token_index=0,
            generated_token=100,
            base_entropy=2.0,
            base_top_k_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.0,
            adapter_top_k_variance=0.1,
            adapter_top_token=200,
            latency_ms=5.0,
        )
        assert different.top_token_disagreement is True


class TestLogitEntropyCalculator:
    """Tests for LogitEntropyCalculator."""

    def test_calculator_creation(self):
        """LogitEntropyCalculator can be created with top_k."""
        from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator

        calc = LogitEntropyCalculator(top_k=100)
        assert calc.top_k == 100

    def test_compute_returns_entropy_tuple(self):
        """Compute returns a tuple of (entropy, variance)."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator

        backend = get_default_backend()
        calc = LogitEntropyCalculator(top_k=10)

        # Create sample logits
        backend.random_seed(42)
        logits = backend.random_normal((100,))
        entropy, variance = calc.compute(logits)

        assert isinstance(entropy, float)
        assert isinstance(variance, float)
        assert entropy >= 0
        assert variance >= 0

    def test_compute_skip_variance(self):
        """Compute can skip variance calculation."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator

        backend = get_default_backend()
        calc = LogitEntropyCalculator(top_k=10)

        backend.random_seed(42)
        logits = backend.random_normal((100,))
        entropy, variance = calc.compute(logits, skip_variance=True)

        assert isinstance(entropy, float)
        assert variance == 0.0  # Skipped


class TestLogitDivergenceCalculator:
    """Tests for LogitDivergenceCalculator."""

    def test_kl_divergence_same_distribution(self):
        """KL divergence of identical distributions is zero."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitDivergenceCalculator

        backend = get_default_backend()
        calc = LogitDivergenceCalculator()

        backend.random_seed(42)
        logits = backend.random_normal((100,))
        kl = calc.kl_divergence(logits, logits)

        assert kl >= 0
        assert kl < 0.01  # Should be very close to zero

    def test_kl_divergence_different_distributions(self):
        """KL divergence of different distributions is positive."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitDivergenceCalculator

        backend = get_default_backend()
        calc = LogitDivergenceCalculator()

        backend.random_seed(42)
        logits_p = backend.random_normal((100,))
        backend.random_seed(123)
        logits_q = backend.random_normal((100,))
        kl = calc.kl_divergence(logits_p, logits_q)

        assert kl >= 0  # KL divergence is non-negative

    def test_stable_softmax(self):
        """Stable softmax doesn't overflow on large logits."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitDivergenceCalculator

        backend = get_default_backend()
        calc = LogitDivergenceCalculator()

        # Create large logits that could cause overflow without stability
        large_logits = backend.array([1000.0, 1001.0, 1002.0])
        probs = calc.stable_softmax(large_logits)

        # Check probabilities sum to 1
        prob_sum = float(backend.to_numpy(backend.sum(probs)))
        assert prob_sum == pytest.approx(1.0, abs=0.001)
