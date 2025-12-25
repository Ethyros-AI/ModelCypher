#!/usr/bin/env python3
"""Script to replace numpy with backend in test files."""

import re
import sys
from pathlib import Path


def fix_numpy_in_file(filepath: Path) -> None:
    """Replace numpy usage with backend protocol."""
    content = filepath.read_text()

    # Already has import? Just replace usages
    if "from modelcypher.core.domain._backend import get_default_backend" not in content:
        # Add backend import after existing imports
        content = content.replace(
            "import numpy as np\nimport pytest",
            "import pytest\n\nfrom modelcypher.core.domain._backend import get_default_backend"
        )
        # Remove numpy import
        content = re.sub(r"import numpy as np\n", "", content)

    # Replace np.array( with backend.array(
    # Add backend = get_default_backend() at start of each test method that uses np
    lines = content.split("\n")
    new_lines = []
    in_method = False
    method_has_backend = False
    method_needs_backend = False

    for i, line in enumerate(lines):
        # Detect method start
        if re.match(r"\s+def test_", line):
            in_method = True
            method_has_backend = False
            method_needs_backend = False
            new_lines.append(line)
            continue

        # Check if method already has backend
        if in_method and "backend = get_default_backend()" in line:
            method_has_backend = True

        # Check if line uses np.
        if in_method and re.search(r"\bnp\.", line):
            method_needs_backend = True

        # Add backend line after docstring if needed
        if in_method and method_needs_backend and not method_has_backend:
            if line.strip().startswith('"""') and line.strip().endswith('"""'):
                new_lines.append(line)
                new_lines.append("        backend = get_default_backend()")
                method_has_backend = True
                in_method = False  # Reset for next method
                continue

        # Replace np. usages
        line = re.sub(r"\bnp\.array\(", "backend.array(", line)
        line = re.sub(r"\bnp\.zeros\(", "backend.zeros(", line)
        line = re.sub(r"\bnp\.ones\(", "backend.ones(", line)
        line = re.sub(r"\bnp\.random\.", "backend.random_", line)
        line = re.sub(r"\bnp\.sqrt\(", "backend.sqrt(", line)
        line = re.sub(r"\bnp\.testing\.assert_array_equal\(", "# backend arrays - using plain assert\n        assert backend.allclose(", line)
        line = re.sub(r"\bnp\.testing\.assert_array_almost_equal\(", "assert backend.allclose(", line)

        new_lines.append(line)

    content = "\n".join(new_lines)
    filepath.write_text(content)
    print(f"Fixed {filepath}")


if __name__ == "__main__":
    test_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/test_hessian_estimator.py")
    fix_numpy_in_file(test_file)
