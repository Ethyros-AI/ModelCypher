#!/usr/bin/env python3

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

"""Modernize Python typing to 2025 best practices.

Applies these transformations:
1. List[X] -> list[X]
2. Dict[K, V] -> dict[K, V]
3. Set[X] -> set[X]
4. Tuple[X, Y] -> tuple[X, Y]
5. Optional[X] -> X | None
6. Union[X, Y] -> X | Y
7. Removes deprecated imports from typing

Preserves:
- from __future__ import annotations
- TYPE_CHECKING blocks
- TypeVar, Generic, Protocol, Callable, Any, etc.
"""

import re
import sys
from pathlib import Path


# Typing imports that are still needed (not deprecated)
KEEP_IMPORTS = {
    "TYPE_CHECKING",
    "TypeVar",
    "Generic",
    "Protocol",
    "Callable",
    "Any",
    "Literal",
    "TypedDict",
    "Final",
    "ClassVar",
    "overload",
    "cast",
    "Iterator",
    "Iterable",
    "Generator",
    "Sequence",
    "Mapping",
    "MutableMapping",
    "MutableSequence",
    "Awaitable",
    "Coroutine",
    "AsyncIterator",
    "AsyncIterable",
    "AsyncGenerator",
    "NamedTuple",
    "NoReturn",
    "Never",
    "Self",
    "ParamSpec",
    "TypeAlias",
    "Concatenate",
    "TypeGuard",
    "Required",
    "NotRequired",
    "Unpack",
    "dataclass_transform",
    "runtime_checkable",
}

# Deprecated typing imports (replaced by builtins)
DEPRECATED_IMPORTS = {"List", "Dict", "Set", "Tuple", "FrozenSet", "Type", "Optional", "Union"}


def modernize_file(filepath: Path) -> tuple[bool, list[str]]:
    """Modernize typing in a single file. Returns (changed, changes_made)."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    changes = []

    # 1. Replace List[X] with list[X] (case-sensitive)
    content, n = re.subn(r'\bList\[', 'list[', content)
    if n:
        changes.append(f"List[] -> list[]: {n}")

    # 2. Replace Dict[K, V] with dict[K, V]
    content, n = re.subn(r'\bDict\[', 'dict[', content)
    if n:
        changes.append(f"Dict[] -> dict[]: {n}")

    # 3. Replace Set[X] with set[X]
    content, n = re.subn(r'\bSet\[', 'set[', content)
    if n:
        changes.append(f"Set[] -> set[]: {n}")

    # 4. Replace Tuple[X, Y] with tuple[X, Y]
    content, n = re.subn(r'\bTuple\[', 'tuple[', content)
    if n:
        changes.append(f"Tuple[] -> tuple[]: {n}")

    # 5. Replace FrozenSet[X] with frozenset[X]
    content, n = re.subn(r'\bFrozenSet\[', 'frozenset[', content)
    if n:
        changes.append(f"FrozenSet[] -> frozenset[]: {n}")

    # 6. Replace Type[X] with type[X]
    content, n = re.subn(r'\bType\[', 'type[', content)
    if n:
        changes.append(f"Type[] -> type[]: {n}")

    # 7. Replace Optional[X] with X | None (handles nested brackets)
    def replace_optional_complex(content: str) -> tuple[str, int]:
        """Replace Optional[...] including nested brackets."""
        count = 0
        result = []
        i = 0
        while i < len(content):
            # Look for "Optional["
            if content[i:i+9] == "Optional[":
                # Find matching closing bracket
                start = i + 9
                depth = 1
                j = start
                while j < len(content) and depth > 0:
                    if content[j] == "[":
                        depth += 1
                    elif content[j] == "]":
                        depth -= 1
                    j += 1
                # Extract inner content
                inner = content[start:j-1]
                result.append(f"{inner} | None")
                i = j
                count += 1
            else:
                result.append(content[i])
                i += 1
        return "".join(result), count

    content, n = replace_optional_complex(content)
    if n:
        changes.append(f"Optional[X] -> X | None: {n}")

    # 8. Replace Union[X, Y] with X | Y
    # Simple case: Union[A, B] -> A | B
    def replace_union(match):
        inner = match.group(1)
        # Split by comma, respecting brackets
        parts = []
        depth = 0
        current = ""
        for char in inner:
            if char == "[":
                depth += 1
                current += char
            elif char == "]":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
        return " | ".join(parts)

    # Match Union[...] with simple content (no nested Union)
    content, n = re.subn(r'\bUnion\[([^\[\]]+(?:\[[^\[\]]*\][^\[\]]*)*)\]', replace_union, content)
    if n:
        changes.append(f"Union[X, Y] -> X | Y: {n}")

    # 9. Clean up typing imports
    # Find import lines and remove deprecated ones
    def clean_typing_import(match):
        line = match.group(0)
        imports = match.group(1)

        # Parse imports
        import_list = [i.strip() for i in imports.split(",")]

        # Filter out deprecated ones
        kept = [i for i in import_list if i.split()[0] in KEEP_IMPORTS or i.split()[-1] in KEEP_IMPORTS]

        if not kept:
            return ""  # Remove entire line
        elif len(kept) < len(import_list):
            # Some removed
            return f"from typing import {', '.join(kept)}"
        else:
            return line  # No change

    content, n = re.subn(
        r'^from typing import ([^\n]+)$',
        clean_typing_import,
        content,
        flags=re.MULTILINE
    )
    if n:
        changes.append(f"Cleaned typing imports")

    # Remove triple+ blank lines (left by removed imports) but preserve double blanks
    # PEP 8 requires 2 blank lines between top-level definitions
    content = re.sub(r'\n\n\n\n+', '\n\n\n', content)  # 4+ newlines -> 3 (2 blank lines)

    changed = content != original
    if changed:
        filepath.write_text(content, encoding="utf-8")

    return changed, changes


def main():
    src_dir = Path(__file__).parent.parent / "src"
    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        sys.exit(1)

    total_changed = 0
    all_changes = []

    for py_file in src_dir.rglob("*.py"):
        try:
            changed, changes = modernize_file(py_file)
            if changed:
                total_changed += 1
                rel_path = py_file.relative_to(src_dir.parent)
                all_changes.append((rel_path, changes))
                print(f"✓ {rel_path}")
                for c in changes:
                    print(f"    {c}")
        except Exception as e:
            print(f"✗ {py_file}: {e}")

    print(f"\nModernized {total_changed} files")


if __name__ == "__main__":
    main()
