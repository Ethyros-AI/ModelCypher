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
import os
import platform

import pytest
from hypothesis import settings

from modelcypher.ports.backend import Backend
from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories

# =============================================================================
# Backend Availability Detection
# =============================================================================

_ENV_BACKEND = os.environ.get("MC_BACKEND", "").lower()
if not _ENV_BACKEND:
    _ENV_BACKEND = os.environ.get("MODELCYPHER_BACKEND", "").lower()
_EXPLICIT_MLX = _ENV_BACKEND == "mlx" or os.environ.get("MC_ENABLE_MLX", "").lower() in (
    "1",
    "true",
    "yes",
)


def _detect_mlx_available() -> bool:
    """Detect if MLX is available (requires Apple Silicon)."""
    if os.environ.get("MC_DISABLE_MLX", "").lower() in ("1", "true", "yes"):
        return False
    if _ENV_BACKEND in ("jax", "cuda"):
        return False
    if platform.system() != "Darwin":
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        return False
    from modelcypher.core.domain import _backend as backend_manager

    return backend_manager.probe_mlx_available(explicit=_EXPLICIT_MLX)


def _detect_jax_available() -> bool:
    """Detect if JAX is importable (CPU is acceptable for tests)."""
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def _detect_jax_gpu_available() -> bool:
    """Detect if JAX is available with GPU/TPU backend."""
    try:
        import jax

        devices = jax.devices()
        return any(d.platform in ("gpu", "tpu") for d in devices)
    except Exception:
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
HAS_JAX = _detect_jax_available()
HAS_JAX_GPU = _detect_jax_gpu_available() if HAS_JAX else False
HAS_CUDA = _detect_cuda_available()
HAS_ANY_BACKEND = HAS_MLX or HAS_JAX or HAS_CUDA

if not HAS_MLX:
    os.environ.setdefault("MC_DISABLE_MLX", "1")


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


def pytest_sessionfinish(session, exitstatus):
    """Clean up MLX resources before Python exits to prevent segfaults.

    MLX Metal buffers can cause segfaults (exit code 139) if not properly
    released before Python's garbage collector runs at exit.
    """
    import gc

    # Force garbage collection first to release Python references
    gc.collect()
    gc.collect()

    # Clear MLX cache if available
    if HAS_MLX:
        try:
            import mlx.core as mx

            # Synchronize any pending operations
            mx.eval(mx.zeros(1))

            # Clear the memory cache
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()

            # Final GC pass
            gc.collect()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture(scope="session", autouse=True)
def _register_atlas_defaults():
    """Register default atlas inventories for geometry tests."""
    register_default_atlas_inventories()


@pytest.fixture(scope="session", autouse=True)
def _initialize_default_backend():
    """Initialize the default backend for tests.

    This must happen before any domain code runs. Uses platform detection
    to select the appropriate backend (MLX on macOS, CUDA/JAX elsewhere).
    """
    from modelcypher.backends import initialize_default_backend

    initialize_default_backend()


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
    """Auto-skip tests based on hardware availability and filter non-test items.

    Tests importing mlx.core at module level will fail to collect on non-Apple machines.
    This hook handles tests that are properly marked with @pytest.mark.mlx etc.

    Also filters out functions from source modules that start with test_ but aren't tests.
    """
    skip_mlx = pytest.mark.skip(reason="MLX not available (requires Apple Silicon)")
    skip_jax = pytest.mark.skip(reason="JAX GPU/TPU not available")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_accel = pytest.mark.skip(reason="No accelerator available")

    # Filter out items from src/ directory (e.g., test_hypothesis in prime geometry modules)
    filtered_items = []
    for item in items:
        item_path = str(item.fspath)
        # Only keep items from tests directory
        if "/tests/" in item_path or "\\tests\\" in item_path or item_path.startswith("tests"):
            filtered_items.append(item)
    items[:] = filtered_items

    if not HAS_ANY_BACKEND:
        skip_backend = pytest.mark.skip(
            reason="No backend available. Set MC_BACKEND=mlx/jax/cuda or install JAX for CPU tests."
        )
        for item in items:
            item.add_marker(skip_backend)
        return

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


def pytest_ignore_collect(collection_path, config):
    """Skip MLX-only test modules when MLX is disabled to avoid import-time crashes."""
    if not HAS_ANY_BACKEND:
        return True
    if os.environ.get("MC_DISABLE_MLX", "").lower() not in ("1", "true", "yes"):
        return False
    path_str = str(collection_path)
    if "/tests/" not in path_str and "\\tests\\" not in path_str and not path_str.startswith("tests"):
        return False
    if not path_str.endswith(".py"):
        return False
    try:
        if hasattr(collection_path, "read_text"):
            content = collection_path.read_text(encoding="utf-8")
        else:
            content = collection_path.read()
    except Exception:
        return False
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    content_lower = content.lower()
    mlx_markers = ("import mlx", "from mlx", "mlx.core", "mlx.nn", "mlx_lm")
    return any(marker in content_lower for marker in mlx_markers)


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


# --- Pytest Fixtures ---


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

    Skips test if JAX is not available.
    """
    if not HAS_JAX:
        pytest.skip("JAX not available")
    from modelcypher.backends.jax_backend import JAXBackend

    return JAXBackend()


def _get_available_backends() -> list[str]:
    """Get list of backends worth testing on this machine.

    - MLX: Only on Apple Silicon (always GPU)
    - JAX: Only if GPU/TPU available (skip CPU-only JAX)
    """
    backends = []
    if HAS_MLX:
        backends.append("mlx")
    if HAS_JAX_GPU:
        backends.append("jax")
    # Fallback: if no GPU backends, allow JAX CPU for basic testing
    if not backends and HAS_JAX:
        backends.append("jax")
    return backends


_AVAILABLE_BACKENDS = _get_available_backends()


@pytest.fixture(params=_AVAILABLE_BACKENDS if _AVAILABLE_BACKENDS else ["skip"])
def any_backend(request) -> Backend:
    """Parametrized fixture that runs tests on available GPU backends.

    On Apple Silicon: Uses MLX only (GPU)
    On Linux with GPU: Uses JAX only (GPU)
    Avoids redundant CPU testing when GPU is available.
    """
    backend_name = request.param

    if backend_name == "skip":
        pytest.skip("No backends available")
    elif backend_name == "mlx":
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_name == "jax":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


@pytest.fixture(params=_AVAILABLE_BACKENDS if _AVAILABLE_BACKENDS else ["skip"])
def accelerated_backend(request) -> Backend:
    """Parametrized fixture for GPU/accelerator backends only.

    Uses same backend detection as any_backend - only tests on GPU backends.
    """
    backend_name = request.param

    if backend_name == "skip":
        pytest.skip("No accelerated backends available")
    elif backend_name == "mlx":
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_name == "jax":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# =============================================================================
# Real Model Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def smol_model_path():
    """Provide path to SmolLM-135M model.

    Downloads the model on first use and caches it in tests/fixtures/.models/
    This fixture is session-scoped so the model is only downloaded once.
    """
    from tests.fixtures.models import ensure_model, SMOL_LM_135M

    return ensure_model(SMOL_LM_135M)


@pytest.fixture(scope="session")
def smol_model_weights(smol_model_path):
    """Load SmolLM-135M weights using production loader.

    Returns dict of weight name -> Array.
    """
    from modelcypher.core.domain._backend import get_default_backend
    from tests.fixtures.models import load_model_weights

    backend = get_default_backend()
    return load_model_weights(smol_model_path, backend)


@pytest.fixture(scope="session")
def smol_model_and_tokenizer(smol_model_path):
    """Load SmolLM-135M model and tokenizer.

    Returns tuple of (model, tokenizer).
    """
    from tests.fixtures.models import load_model_and_tokenizer

    return load_model_and_tokenizer(smol_model_path)


@pytest.fixture
def atlas_probes():
    """Provide atlas probe prompts for activation testing.

    Returns list of 100 probe prompt strings from the atlas system.
    """
    from tests.fixtures.models import get_atlas_probes

    return get_atlas_probes(n_samples=100)


# Export availability flags for use in skipif decorators
__all__ = ["HAS_MLX", "HAS_JAX", "HAS_JAX_GPU", "HAS_CUDA"]
