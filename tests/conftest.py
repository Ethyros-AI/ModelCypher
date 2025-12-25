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

import asyncio
import platform

import numpy as np
import pytest
from hypothesis import settings

from modelcypher.ports.backend import Backend

# =============================================================================
# Backend Availability Detection
# =============================================================================


def _detect_mlx_available() -> bool:
    """Detect if MLX is available (requires Apple Silicon)."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        return False
    try:
        import mlx.core as mx

        # Verify Metal GPU is accessible
        _ = mx.zeros((1,))
        return True
    except (ImportError, RuntimeError):
        return False


def _detect_jax_available() -> bool:
    """Detect if JAX is available with GPU/TPU backend."""
    try:
        import jax

        devices = jax.devices()
        # Check for GPU or TPU (not just CPU)
        has_accelerator = any(d.platform in ("gpu", "tpu") for d in devices)
        return has_accelerator
    except ImportError:
        return False


def _detect_cuda_available() -> bool:
    """Detect if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# Cache availability at import time
HAS_MLX = _detect_mlx_available()
HAS_JAX_GPU = _detect_jax_available()
HAS_CUDA = _detect_cuda_available()


# =============================================================================
# Hypothesis Profiles
# =============================================================================


# Configure hypothesis profiles for fast testing
# Default profile: fast CI testing with minimal examples
settings.register_profile(
    "fast",
    max_examples=10,
    deadline=None,
    suppress_health_check=[],
)

# CI profile: balanced speed and coverage
settings.register_profile(
    "ci",
    max_examples=20,
    deadline=None,
)

# Full profile: thorough testing for release validation
settings.register_profile(
    "full",
    max_examples=100,
    deadline=None,
)

# Load the fast profile by default - override with HYPOTHESIS_PROFILE env var
settings.load_profile("fast")


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async (deferred from pytest-asyncio)")
    # Backend-specific markers are already defined in pyproject.toml
    # but we register them here for completeness
    config.addinivalue_line("markers", "mlx: tests that require MLX (Apple Silicon with Metal GPU)")
    config.addinivalue_line("markers", "jax_gpu: tests that require JAX with GPU/TPU backend")
    config.addinivalue_line("markers", "cuda: tests that require CUDA")
    config.addinivalue_line("markers", "accelerator: tests that require any GPU/accelerator")


@pytest.fixture(autouse=True)
def _clear_cli_composition_cache():
    """Clear CLI composition cache before each test.
    
    The CLI composition module uses @lru_cache on _get_registry() and _get_factory().
    This causes test isolation issues: the first test to use CLI commands caches
    the PortRegistry with whatever MODELCYPHER_HOME was set at that time.
    Later tests that set a different MODELCYPHER_HOME env will still use the
    cached registry, causing "Job not found" errors.
    
    This fixture clears the cache before each test to ensure test isolation.
    """
    yield  # Run the test first
    # Clear caches after each test to ensure next test gets fresh state
    try:
        from modelcypher.cli import composition
        if hasattr(composition._get_registry, 'cache_clear'):
            composition._get_registry.cache_clear()
        if hasattr(composition._get_factory, 'cache_clear'):
            composition._get_factory.cache_clear()
    except ImportError:
        pass  # Module not loaded, nothing to clear


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on hardware availability.

    Tests importing mlx.core at module level will fail to collect on non-Apple machines.
    This hook handles tests that are properly marked with @pytest.mark.mlx etc.
    """
    skip_mlx = pytest.mark.skip(reason="MLX not available (requires Apple Silicon)")
    skip_jax = pytest.mark.skip(reason="JAX GPU/TPU not available")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_accel = pytest.mark.skip(reason="No accelerator available")

    for item in items:
        if "mlx" in item.keywords and not HAS_MLX:
            item.add_marker(skip_mlx)
        if "jax_gpu" in item.keywords and not HAS_JAX_GPU:
            item.add_marker(skip_jax)
        if "cuda" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_cuda)
        if "accelerator" in item.keywords:
            if not (HAS_MLX or HAS_JAX_GPU or HAS_CUDA):
                item.add_marker(skip_accel)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Handle async test functions by running them with asyncio.run()."""
    if asyncio.iscoroutinefunction(pyfuncitem.obj):
        # Get test arguments (fixtures)
        testfunction = pyfuncitem.obj
        funcargs = pyfuncitem.funcargs
        testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
        # Run the async test function synchronously
        asyncio.run(testfunction(**testargs))
        return True  # Indicate we handled the call
    return None  # Let pytest handle sync tests normally


class NumpyBackend(Backend):
    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def reshape(self, array, shape):
        return np.reshape(array, shape)

    def squeeze(self, array, axis=None):
        return np.squeeze(array, axis=axis)

    def transpose(self, array, axes=None):
        return np.transpose(array, axes=axes)

    def matmul(self, lhs, rhs):
        return lhs @ rhs

    def sum(self, array, axis=None, keepdims=False):
        return np.sum(array, axis=axis, keepdims=keepdims)

    def max(self, array, axis=None, keepdims=False):
        return np.max(array, axis=axis, keepdims=keepdims)

    def sqrt(self, array):
        return np.sqrt(array)

    def exp(self, array):
        return np.exp(array)

    def log(self, array):
        return np.log(array)

    def maximum(self, lhs, rhs):
        return np.maximum(lhs, rhs)

    def minimum(self, lhs, rhs):
        return np.minimum(lhs, rhs)

    def abs(self, array):
        return np.abs(array)

    def astype(self, array, dtype):
        return array.astype(dtype)

    def svd(self, array, compute_uv=True):
        if compute_uv:
            u, s, vt = np.linalg.svd(array, full_matrices=False)
            return u, s, vt
        return np.linalg.svd(array, full_matrices=False, compute_uv=False)

    def eval(self, *arrays):
        return None

    def to_numpy(self, array):
        return np.array(array)

    def quantize(self, weight, group_size, bits, mode):
        raise NotImplementedError("Quantization not supported in NumpyBackend")

    def dequantize(self, weight, scales, biases, group_size, bits, mode):
        raise NotImplementedError("Dequantization not supported in NumpyBackend")

    # --- Array Creation (new) ---
    def eye(self, n, m=None, dtype=None):
        return np.eye(n, m, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            return np.arange(start, dtype=dtype)
        return np.arange(start, stop, step, dtype=dtype)

    def diag(self, array, k=0):
        return np.diag(array, k=k)

    def full(self, shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype)

    def ones_like(self, array, dtype=None):
        return np.ones_like(array, dtype=dtype)

    def zeros_like(self, array, dtype=None):
        return np.zeros_like(array, dtype=dtype)

    def linspace(self, start, stop, num, dtype=None):
        return np.linspace(start, stop, num, dtype=dtype)

    # --- Shape Manipulation (new) ---
    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def broadcast_to(self, array, shape):
        return np.broadcast_to(array, shape)

    # --- Reductions (new) ---
    def mean(self, array, axis=None, keepdims=False):
        return np.mean(array, axis=axis, keepdims=keepdims)

    def min(self, array, axis=None, keepdims=False):
        return np.min(array, axis=axis, keepdims=keepdims)

    def argmax(self, array, axis=None):
        return np.argmax(array, axis=axis)

    def argmin(self, array, axis=None):
        return np.argmin(array, axis=axis)

    def var(self, array, axis=None, keepdims=False):
        return np.var(array, axis=axis, keepdims=keepdims)

    def std(self, array, axis=None, keepdims=False):
        return np.std(array, axis=axis, keepdims=keepdims)

    # --- Element-wise Operations (new) ---
    def sign(self, array):
        return np.sign(array)

    def clip(self, array, min_val, max_val):
        return np.clip(array, min_val, max_val)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def softmax(self, array, axis=-1):
        exp_x = np.exp(array - np.max(array, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def cumsum(self, array, axis=None):
        return np.cumsum(array, axis=axis)

    # --- Linear Algebra (new) ---
    def dot(self, a, b):
        return np.dot(a, b)

    def norm(self, array, axis=None, keepdims=False):
        return np.linalg.norm(array, axis=axis, keepdims=keepdims)

    def det(self, array):
        return np.linalg.det(array)

    def eigh(self, array):
        return np.linalg.eigh(array)

    def solve(self, a, b):
        return np.linalg.solve(a, b)

    def qr(self, array):
        return np.linalg.qr(array)

    # --- Indexing ---
    def take(self, array, indices, axis=None):
        return np.take(array, indices, axis=axis)

    # --- Sorting ---
    def sort(self, array, axis=-1):
        return np.sort(array, axis=axis)

    def argsort(self, array, axis=-1):
        return np.argsort(array, axis=axis)

    def argpartition(self, array, kth, axis=-1):
        return np.argpartition(array, kth=kth, axis=axis)

    # --- Random (new) ---
    def random_normal(self, shape, dtype=None):
        arr = np.random.normal(size=shape)
        return arr.astype(dtype) if dtype else arr

    def random_uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        shape = shape or (1,)
        arr = np.random.uniform(low, high, size=shape)
        return arr.astype(dtype) if dtype else arr

    def random_randint(self, low, high, shape=None):
        shape = shape or (1,)
        return np.random.randint(low, high, size=shape)

    def random_seed(self, seed):
        np.random.seed(seed)

    def random_categorical(self, logits, num_samples=1):
        """Sample from categorical distribution defined by logits."""
        logits = np.asarray(logits)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        if probs.ndim == 1:
            return np.random.choice(len(probs), size=num_samples, p=probs)
        else:
            samples = []
            for p in probs:
                samples.append(np.random.choice(len(p), size=num_samples, p=p))
            return np.array(samples)

    # --- Attention Masks ---
    def create_causal_mask(self, seq_len, dtype=None):
        """Create an additive causal attention mask for autoregressive models."""
        # Upper triangular matrix filled with -inf (causal mask)
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        if dtype is not None:
            mask = mask.astype(dtype)
        return mask


__all__ = ["NumpyBackend"]


# --- Pytest Fixtures ---


@pytest.fixture
def numpy_backend() -> NumpyBackend:
    """Provide a NumpyBackend for testing."""
    return NumpyBackend()


@pytest.fixture
def mock_registry(tmp_path):
    """Provide a fully mocked PortRegistry for testing.

    All ports are MagicMock instances, allowing tests to
    configure return values and verify calls.
    """
    from unittest.mock import MagicMock

    registry = MagicMock()

    # Configure mock ports
    registry.model_store = MagicMock()
    registry.dataset_store = MagicMock()
    registry.job_store = MagicMock()
    registry.evaluation_store = MagicMock()
    registry.compare_store = MagicMock()
    registry.manifold_profile_store = MagicMock()
    registry.training_engine = MagicMock()
    registry.inference_engine = MagicMock()
    registry.exporter = MagicMock()
    registry.hub_adapter = MagicMock()
    registry.model_search = MagicMock()
    registry.model_loader = MagicMock()

    # Configure path fields
    registry.base_dir = tmp_path
    registry.logs_dir = tmp_path / "logs"

    return registry


@pytest.fixture
def mock_factory(mock_registry):
    """Provide a ServiceFactory with mocked registry.

    Services created from this factory will use mocked ports.
    """
    from modelcypher.infrastructure.service_factory import ServiceFactory

    return ServiceFactory(mock_registry)


# =============================================================================
# Backend Fixtures
# =============================================================================


@pytest.fixture
def mlx_backend() -> Backend:
    """Provide MLXBackend for testing on Apple Silicon.

    Skips test if MLX is not available.
    """
    if not HAS_MLX:
        pytest.skip("MLX not available (requires Apple Silicon)")
    from modelcypher.backends.mlx_backend import MLXBackend

    return MLXBackend()


@pytest.fixture
def jax_backend() -> Backend:
    """Provide JAXBackend for testing.

    Skips test if JAX GPU/TPU is not available.
    """
    if not HAS_JAX_GPU:
        pytest.skip("JAX GPU/TPU not available")
    from modelcypher.backends.jax_backend import JAXBackend

    return JAXBackend()


@pytest.fixture(params=["numpy", "mlx", "jax"])
def any_backend(request) -> Backend:
    """Parametrized fixture that runs test on all available backends.

    Use this for tests that should verify behavior consistency across backends.
    Tests will be skipped for unavailable backends.
    """
    backend_name = request.param

    if backend_name == "numpy":
        return NumpyBackend()
    elif backend_name == "mlx":
        if not HAS_MLX:
            pytest.skip("MLX not available")
        from modelcypher.backends.mlx_backend import MLXBackend

        return MLXBackend()
    elif backend_name == "jax":
        if not HAS_JAX_GPU:
            pytest.skip("JAX GPU/TPU not available")
        from modelcypher.backends.jax_backend import JAXBackend

        return JAXBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


@pytest.fixture(params=["mlx", "jax"])
def accelerated_backend(request) -> Backend:
    """Parametrized fixture for GPU/accelerator backends only.

    Use this for tests that specifically require hardware acceleration.
    Skips entirely if no accelerators are available.
    """
    backend_name = request.param

    if backend_name == "mlx":
        if not HAS_MLX:
            pytest.skip("MLX not available")
        from modelcypher.backends.mlx_backend import MLXBackend

        return MLXBackend()
    elif backend_name == "jax":
        if not HAS_JAX_GPU:
            pytest.skip("JAX GPU/TPU not available")
        from modelcypher.backends.jax_backend import JAXBackend

        return JAXBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# Export availability flags for use in skipif decorators
__all__ = ["NumpyBackend", "HAS_MLX", "HAS_JAX_GPU", "HAS_CUDA"]
