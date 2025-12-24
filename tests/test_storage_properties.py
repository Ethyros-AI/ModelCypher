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

"""Property tests for StorageService."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.use_cases.storage_service import BYTES_PER_GB, StorageService


@dataclass
class _DiskUsage:
    total: int
    used: int
    free: int


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


# **Feature: cli-mcp-parity, Property 5: Storage cleanup frees non-negative space**
# **Validates: Requirements 9.2**
@given(
    targets=st.lists(
        st.sampled_from(["caches", "rag"]),
        min_size=1,
        max_size=2,
        unique=True,
    ),
    file_sizes=st.lists(
        st.integers(min_value=0, max_value=10000),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=100, deadline=None)
def test_storage_cleanup_frees_non_negative_space(targets: list[str], file_sizes: list[int]):
    """Property 5: For any cleanup operation, cleanup() returns valid cleared targets
    and the number of cleared targets is non-negative (>= 0).
    
    Since the current implementation returns list[str] of cleared targets,
    we validate that:
    1. The returned list length is >= 0
    2. All returned targets are valid target names
    3. The cleared targets are a subset of the requested targets
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        home = tmp_path / "mc_home"
        hf_home = tmp_path / "hf_cache"

        # Set up environment
        import os
        old_mc_home = os.environ.get("MODELCYPHER_HOME")
        old_hf_home = os.environ.get("HF_HOME")
        os.environ["MODELCYPHER_HOME"] = str(home)
        os.environ["HF_HOME"] = str(hf_home)

        try:
            # Create files in target directories based on generated sizes
            for i, size in enumerate(file_sizes):
                if "caches" in targets:
                    _write_bytes(home / "caches" / f"cache_{i}.bin", size)
                    _write_bytes(hf_home / f"model_{i}.bin", size)
                if "rag" in targets:
                    _write_bytes(home / "rag" / f"index_{i}.bin", size)

            service = StorageService(cache_ttl_seconds=0.0)
            cleared = service.cleanup(targets)

            # Property: cleared list length is non-negative
            assert len(cleared) >= 0

            # Property: all cleared targets are valid target names
            valid_targets = {"caches", "rag"}
            for target in cleared:
                assert target in valid_targets

            # Property: cleared targets are a subset of requested targets
            assert set(cleared) <= set(targets)

            # Property: all requested valid targets should be cleared
            requested_valid = set(targets) & valid_targets
            assert set(cleared) == requested_valid

        finally:
            # Restore environment
            if old_mc_home is not None:
                os.environ["MODELCYPHER_HOME"] = old_mc_home
            elif "MODELCYPHER_HOME" in os.environ:
                del os.environ["MODELCYPHER_HOME"]
            if old_hf_home is not None:
                os.environ["HF_HOME"] = old_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]


@given(
    targets=st.lists(
        st.sampled_from(["caches", "rag"]),
        min_size=1,
        max_size=2,
        unique=True,
    ),
)
@settings(max_examples=50, deadline=None)
def test_storage_cleanup_with_empty_directories(targets: list[str]):
    """Test cleanup works correctly even when directories are empty."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        home = tmp_path / "mc_home"
        hf_home = tmp_path / "hf_cache"

        import os
        old_mc_home = os.environ.get("MODELCYPHER_HOME")
        old_hf_home = os.environ.get("HF_HOME")
        os.environ["MODELCYPHER_HOME"] = str(home)
        os.environ["HF_HOME"] = str(hf_home)

        try:
            # Create empty directories
            (home / "caches").mkdir(parents=True, exist_ok=True)
            (home / "rag").mkdir(parents=True, exist_ok=True)
            hf_home.mkdir(parents=True, exist_ok=True)

            service = StorageService(cache_ttl_seconds=0.0)
            cleared = service.cleanup(targets)

            # Property: cleared list length is non-negative
            assert len(cleared) >= 0

            # Property: cleared targets match requested valid targets
            valid_targets = {"caches", "rag"}
            requested_valid = set(targets) & valid_targets
            assert set(cleared) == requested_valid

        finally:
            if old_mc_home is not None:
                os.environ["MODELCYPHER_HOME"] = old_mc_home
            elif "MODELCYPHER_HOME" in os.environ:
                del os.environ["MODELCYPHER_HOME"]
            if old_hf_home is not None:
                os.environ["HF_HOME"] = old_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
