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
CLI Import Guard Tests.

This test file ensures that ALL CLI command modules can be imported without errors.
It auto-discovers all Python modules in cli/commands and verifies they load correctly.

This prevents import drift where broken imports (like HuggingFaceModelLoader)
go undetected until production.

Run this as part of CI to catch import errors early.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Get the CLI commands directory
CLI_COMMANDS_DIR = Path(__file__).parent.parent / "src" / "modelcypher" / "cli" / "commands"


def get_all_cli_modules() -> list[str]:
    """Discover all Python modules in the CLI commands directory."""
    modules = []
    for py_file in CLI_COMMANDS_DIR.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Convert path to module name
        relative = py_file.relative_to(CLI_COMMANDS_DIR.parent.parent.parent)
        module_parts = list(relative.with_suffix("").parts)
        module_name = ".".join(module_parts)
        modules.append(module_name)

    return sorted(modules)


# Get all modules at collection time
ALL_CLI_MODULES = get_all_cli_modules()


class TestCLIModuleImports:
    """Test that all CLI command modules can be imported."""

    @pytest.mark.parametrize("module_name", ALL_CLI_MODULES)
    def test_cli_module_imports(self, module_name: str):
        """Verify CLI module can be imported without errors.

        This catches:
        - Import errors (like HuggingFaceModelLoader not existing)
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


class TestCLIModuleCount:
    """Test that CLI module count doesn't regress."""

    def test_minimum_cli_module_count(self):
        """Ensure we maintain minimum CLI module count (prevents accidental deletion)."""
        # As of 2026-01, we have ~53 CLI command modules
        assert len(ALL_CLI_MODULES) >= 50, (
            f"CLI module count dropped to {len(ALL_CLI_MODULES)}! "
            "Modules may have been accidentally deleted."
        )


class TestCriticalCLICommands:
    """Test that critical CLI command entry points are importable."""

    def test_merge_command_imports(self):
        """Test that merge.py imports correctly - this caught the HuggingFaceModelLoader bug."""
        from modelcypher.cli.commands import merge

        assert hasattr(merge, "app")

    def test_infer_command_imports(self):
        """Test that infer.py imports correctly."""
        from modelcypher.cli.commands import infer

        assert hasattr(infer, "app")

    def test_model_command_imports(self):
        """Test that model.py imports correctly."""
        from modelcypher.cli.commands import model

        assert hasattr(model, "app")

    def test_system_command_imports(self):
        """Test that system.py imports correctly."""
        from modelcypher.cli.commands import system

        assert hasattr(system, "app")

    def test_geometry_commands_import(self):
        """Test that geometry subpackage imports correctly."""
        from modelcypher.cli.commands import geometry

        assert geometry is not None
