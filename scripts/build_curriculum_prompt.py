#!/usr/bin/env python3
"""Build a curriculum generation prompt for a frontier model.

Reads a StudentProfile JSON (or generates a blank one) and produces the
structured prompt document that a frontier model uses to generate a curriculum.

Usage:
    poetry run python scripts/build_curriculum_prompt.py \
        --goal "teach basic arithmetic" \
        --domain "logic_math" \
        --output /tmp/prompt.md

    poetry run python scripts/build_curriculum_prompt.py \
        --profile /path/to/student_profile.json \
        --goal "beat 50% on GSM8K" \
        --benchmark "gsm8k" \
        --output /tmp/prompt.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from modelcypher.core.domain.curriculum_protocol.prompt_template import build_prompt
from modelcypher.core.domain.curriculum_protocol.student_profile import (
    GeometricProfile,
    StudentProfile,
)


def _blank_profile() -> StudentProfile:
    """Generate a minimal blank profile for a fresh/unknown model."""
    return StudentProfile(
        model_path="(not specified)",
        model_id="",
        geometric_profile=GeometricProfile(
            architecture="unknown",
            model_family="unknown",
            parameter_count=0,
            hidden_dim=0,
            num_layers=0,
            vocab_size=0,
            context_length=0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a curriculum generation prompt for a frontier model."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Path to student_profile.json. If omitted, a blank profile is used.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Training goal (e.g., 'beat 50%% on GSM8K').",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help="Target domain (e.g., 'logic_math', 'code').",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="",
        help="Target benchmark (e.g., 'gsm8k', 'arc_easy').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    args = parser.parse_args()

    if args.profile is not None:
        with open(args.profile) as f:
            profile = StudentProfile.from_dict(json.load(f))
    else:
        profile = _blank_profile()

    prompt = build_prompt(
        profile=profile,
        goal=args.goal,
        target_domain=args.domain,
        target_benchmark=args.benchmark,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt)
        print(f"Prompt written to {args.output}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
