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

"""
Module Import Guard Tests.

This test file ensures that EVERY module in the domain can be imported without errors.
It auto-discovers all Python modules and verifies they load correctly.

This prevents import drift where circular imports or missing dependencies break the module tree.
Run this as part of CI to catch import errors early.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Get the domain directory
DOMAIN_DIR = Path(__file__).parent.parent / "src" / "modelcypher" / "core" / "domain"


def get_all_domain_modules() -> list[str]:
    """Discover all Python modules in the domain directory."""
    modules = []
    for py_file in DOMAIN_DIR.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Convert path to module name
        relative = py_file.relative_to(DOMAIN_DIR.parent.parent.parent)
        module_parts = list(relative.with_suffix("").parts)
        module_name = ".".join(module_parts)
        modules.append(module_name)

    return sorted(modules)


# Get all modules at collection time
ALL_DOMAIN_MODULES = get_all_domain_modules()


class TestModuleImports:
    """Test that all domain modules can be imported."""

    @pytest.mark.parametrize("module_name", ALL_DOMAIN_MODULES)
    def test_module_imports(self, module_name: str):
        """Verify module can be imported without errors.

        This catches:
        - Circular import errors
        - Missing dependencies
        - Syntax errors
        - Type annotation errors
        """
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            pytest.fail(f"Error importing {module_name}: {type(e).__name__}: {e}")


class TestDomainPackageExports:
    """Test that domain package __init__.py exports work correctly."""

    def test_domain_root_import(self):
        """Test that the domain root package can be imported."""
        from modelcypher.core import domain

        assert domain is not None

    def test_geometry_subpackage_import(self):
        """Test that geometry subpackage exports key classes."""
        from modelcypher.core.domain import geometry

        # Verify key exports are accessible
        assert hasattr(geometry, "VectorMath")
        assert hasattr(geometry, "PathNode")
        assert hasattr(geometry, "DoRADecomposition")
        assert hasattr(geometry, "PermutationAligner")

    def test_agents_subpackage_import(self):
        """Test that agents subpackage exports key classes."""
        from modelcypher.core.domain import agents

        assert hasattr(agents, "SemanticPrimeAtlas")
        assert hasattr(agents, "UnifiedAtlasInventory")

    def test_entropy_subpackage_import(self):
        """Test that entropy subpackage exports key classes."""
        from modelcypher.core.domain import entropy

        assert hasattr(entropy, "EntropyTracker")
        assert hasattr(entropy, "ModelStateClassifier")
        assert hasattr(entropy, "EntropyTransition")

    def test_safety_subpackage_import(self):
        """Test that safety subpackage exports key classes."""
        from modelcypher.core.domain import safety

        # Safety package should have been created
        assert safety is not None

    def test_training_subpackage_import(self):
        """Test that training subpackage exports key classes."""
        from modelcypher.core.domain import training

        assert training is not None

    def test_dynamics_subpackage_import(self):
        """Test that dynamics subpackage exports key classes."""
        from modelcypher.core.domain import dynamics

        assert dynamics is not None

    def test_merging_subpackage_import(self):
        """Test that merging subpackage exports key classes."""
        from modelcypher.core.domain import merging

        assert merging is not None

    def test_inference_subpackage_import(self):
        """Test that inference subpackage exports key classes."""
        from modelcypher.core.domain import inference

        assert inference is not None


class TestModuleCount:
    """Test that module count doesn't regress."""

    def test_minimum_module_count(self):
        """Ensure we maintain minimum module count (prevents accidental deletion)."""
        # As of 2024-12, we have 223 domain modules
        assert len(ALL_DOMAIN_MODULES) >= 220, (
            f"Module count dropped to {len(ALL_DOMAIN_MODULES)}! "
            "Modules may have been accidentally deleted."
        )

    def test_modules_are_loaded_in_namespace(self):
        """Verify that importing domain loads most modules."""
        from modelcypher.core import domain  # noqa: F401

        loaded = [m for m in sys.modules.keys() if "modelcypher.core.domain" in m]

        # Should have at least 200 loaded modules (allowing some slack for lazy imports)
        assert len(loaded) >= 200, (
            f"Only {len(loaded)} modules loaded! Check __init__.py exports for missing modules."
        )
