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

"""Self-study curriculum for geometric introspection.

Curricula are structured learning progressions that teach models to:
1. Observe their own geometric signatures (Level 1)
2. Predict geometry before generating (Level 2)
3. Choose approaches based on predicted geometry (Level 3)
4. Detect and correct geometric anomalies (Level 4)

Curriculum Format (JSONL):
    {"prompt": "...", "expected_answer": "...", "geometry_hint": "...", "level": 1}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Default curriculum location
DEFAULT_CURRICULUM_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "curricula"


class CurriculumLevel(IntEnum):
    """Self-study curriculum levels.

    Level 1: Observation - See geometry, no action required
    Level 2: Prediction - Predict geometry before generating
    Level 3: Selection - Choose approach based on predicted geometry
    Level 4: Correction - Detect and fix geometric anomalies
    """

    OBSERVATION = 1
    PREDICTION = 2
    SELECTION = 3
    CORRECTION = 4


@dataclass
class CurriculumExample:
    """A single curriculum example.

    Attributes:
        prompt: The problem/question to study
        expected_answer: The correct answer (optional)
        level: Curriculum level (1-4)
        geometry_hint: Optional hint about expected geometry
        approaches: Optional list of approaches to compare (for Level 3)
        metadata: Additional metadata for the example
    """

    prompt: str
    expected_answer: str | None = None
    level: CurriculumLevel = CurriculumLevel.OBSERVATION
    geometry_hint: str | None = None
    approaches: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "CurriculumExample":
        """Create from a dictionary (JSON record)."""
        return cls(
            prompt=data["prompt"],
            expected_answer=data.get("expected_answer"),
            level=CurriculumLevel(data.get("level", 1)),
            geometry_hint=data.get("geometry_hint"),
            approaches=data.get("approaches", []),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "level": int(self.level),
            "geometry_hint": self.geometry_hint,
            "approaches": self.approaches,
            "metadata": self.metadata,
        }


@dataclass
class Curriculum:
    """A collection of curriculum examples organized by level.

    Attributes:
        name: Curriculum name (e.g., "geometric_self_study")
        description: Human-readable description
        examples: List of all examples
        levels: Dict mapping level -> list of examples at that level
    """

    name: str
    description: str
    examples: list[CurriculumExample] = field(default_factory=list)
    levels: dict[CurriculumLevel, list[CurriculumExample]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Organize examples by level."""
        self._organize_levels()

    def _organize_levels(self) -> None:
        """Organize examples into level buckets."""
        self.levels = {level: [] for level in CurriculumLevel}
        for example in self.examples:
            self.levels[example.level].append(example)

    def add_example(self, example: CurriculumExample) -> None:
        """Add an example to the curriculum."""
        self.examples.append(example)
        self.levels[example.level].append(example)

    def get_level(self, level: CurriculumLevel) -> list[CurriculumExample]:
        """Get all examples at a specific level."""
        return self.levels.get(level, [])

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[CurriculumExample]:
        return iter(self.examples)


def load_curriculum_from_jsonl(path: Path) -> list[CurriculumExample]:
    """Load curriculum examples from a JSONL file.

    Args:
        path: Path to the JSONL file

    Returns:
        List of CurriculumExample objects
    """
    if not path.exists():
        raise FileNotFoundError(f"Curriculum file not found: {path}")

    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            examples.append(CurriculumExample.from_dict(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping invalid curriculum line: {e}")
            continue

    return examples


def load_curriculum_from_directory(
    directory: Path,
    name: str | None = None,
    description: str | None = None,
) -> Curriculum:
    """Load a curriculum from a directory of JSONL files.

    Directory structure:
        curriculum_name/
            level_1_observation/
                intuitive_traps.jsonl
                explicit_math.jsonl
            level_2_prediction/
                predict_comp_phi.jsonl
            ...

    Args:
        directory: Path to the curriculum directory
        name: Optional override for curriculum name (default: directory name)
        description: Optional description

    Returns:
        Curriculum object with all examples loaded
    """
    if not directory.exists():
        raise FileNotFoundError(f"Curriculum directory not found: {directory}")

    curriculum_name = name or directory.name
    curriculum_desc = description or f"Curriculum loaded from {directory}"

    all_examples = []

    # Load all JSONL files in the directory tree
    for jsonl_file in directory.rglob("*.jsonl"):
        logger.debug(f"Loading curriculum file: {jsonl_file}")
        examples = load_curriculum_from_jsonl(jsonl_file)

        # Try to infer level from directory name
        parent_name = jsonl_file.parent.name.lower()
        if "level_1" in parent_name or "observation" in parent_name:
            default_level = CurriculumLevel.OBSERVATION
        elif "level_2" in parent_name or "prediction" in parent_name:
            default_level = CurriculumLevel.PREDICTION
        elif "level_3" in parent_name or "selection" in parent_name:
            default_level = CurriculumLevel.SELECTION
        elif "level_4" in parent_name or "correction" in parent_name:
            default_level = CurriculumLevel.CORRECTION
        else:
            default_level = CurriculumLevel.OBSERVATION

        # Set default level for examples without explicit level
        for ex in examples:
            if ex.level == CurriculumLevel.OBSERVATION and "level" not in ex.metadata:
                ex.level = default_level

        all_examples.extend(examples)

    logger.info(f"Loaded {len(all_examples)} examples for curriculum '{curriculum_name}'")

    return Curriculum(
        name=curriculum_name,
        description=curriculum_desc,
        examples=all_examples,
    )


def get_builtin_curriculum(name: str = "geometric_self_study") -> Curriculum:
    """Get a built-in curriculum by name.

    Args:
        name: Curriculum name (e.g., "geometric_self_study")

    Returns:
        Loaded Curriculum object
    """
    curriculum_path = DEFAULT_CURRICULUM_PATH / name

    if curriculum_path.exists():
        return load_curriculum_from_directory(curriculum_path)

    # If directory doesn't exist, return a minimal default curriculum
    logger.warning(f"Curriculum '{name}' not found, using default examples")
    return _get_default_curriculum()


def _get_default_curriculum() -> Curriculum:
    """Get the default built-in curriculum with cognitive reflection examples.

    These are the classic cognitive reflection test problems that reliably
    show geometric differences between intuitive and deliberate processing.
    """
    examples = [
        # Level 1: Observation - Classic intuitive traps
        CurriculumExample(
            prompt="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            expected_answer="$0.05",
            level=CurriculumLevel.OBSERVATION,
            geometry_hint="Intuitive answer ($0.10) shows flat geometry. Correct answer shows expand-compress.",
        ),
        CurriculumExample(
            prompt="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            expected_answer="5 minutes",
            level=CurriculumLevel.OBSERVATION,
            geometry_hint="Intuitive answer (100 minutes) shows narrow processing.",
        ),
        CurriculumExample(
            prompt="In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
            expected_answer="47 days",
            level=CurriculumLevel.OBSERVATION,
            geometry_hint="Intuitive answer (24 days) misses exponential reasoning.",
        ),

        # Level 2: Prediction - Predict geometry before answering
        CurriculumExample(
            prompt="Before answering, predict whether this will require intuitive or deliberate processing: What is 12 + 15?",
            expected_answer="27",
            level=CurriculumLevel.PREDICTION,
            geometry_hint="Simple arithmetic - intuitive processing sufficient, expect comp/phi near 1.0.",
        ),
        CurriculumExample(
            prompt="Before answering, predict whether this will require intuitive or deliberate processing: A farmer has 17 sheep. All but 9 die. How many are left?",
            expected_answer="9",
            level=CurriculumLevel.PREDICTION,
            geometry_hint="Parsing trick - looks like subtraction but isn't. Deliberate processing needed.",
        ),

        # Level 3: Selection - Choose approach based on geometry
        CurriculumExample(
            prompt="Emily's father has three daughters. The first daughter is named April. The second daughter is named May. What is the third daughter's name?",
            expected_answer="Emily",
            level=CurriculumLevel.SELECTION,
            approaches=[
                "The pattern continues, so the answer is June.",
                "Wait, let me re-read the question carefully.",
            ],
            geometry_hint="The intuitive pattern-matching approach fails here.",
        ),
        CurriculumExample(
            prompt="Some months have 31 days. How many months have 28 days?",
            expected_answer="12",
            level=CurriculumLevel.SELECTION,
            approaches=[
                "Only February has 28 days, so the answer is 1.",
                "Let me think about what the question is actually asking.",
            ],
            geometry_hint="All months have at least 28 days.",
        ),

        # Level 4: Correction - Detect and fix errors
        CurriculumExample(
            prompt="I said the ball costs $0.10. My geometric signature showed flat processing (comp/phi = 0.6). Should I reconsider?",
            expected_answer="Yes, reconsider",
            level=CurriculumLevel.CORRECTION,
            geometry_hint="Flat geometry indicates intuitive trap. Explicit reasoning needed.",
        ),
    ]

    return Curriculum(
        name="default_cognitive_reflection",
        description="Default curriculum with cognitive reflection test problems",
        examples=examples,
    )


def save_curriculum_to_jsonl(
    curriculum: Curriculum,
    output_path: Path,
    level: CurriculumLevel | None = None,
) -> None:
    """Save curriculum examples to a JSONL file.

    Args:
        curriculum: The curriculum to save
        output_path: Path to the output JSONL file
        level: Optional level filter (save only examples at this level)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = curriculum.get_level(level) if level else curriculum.examples

    with output_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")

    logger.info(f"Saved {len(examples)} examples to {output_path}")
