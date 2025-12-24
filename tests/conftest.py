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
import functools
from typing import Callable

import numpy as np
import pytest
from hypothesis import settings, Verbosity

from modelcypher.ports.backend import Backend


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


def pytest_configure(config):
    """Register asyncio marker."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (deferred from pytest-asyncio)"
    )


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
    from pathlib import Path
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
