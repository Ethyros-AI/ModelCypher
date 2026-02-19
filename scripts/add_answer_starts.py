#!/usr/bin/env python3
"""Add answer_start field to JSONL samples for answer-span masked training.

Reads a JSONL file where each line has {"text": "..."} and outputs a new JSONL
with {"text": "...", "answer_start": N} where N is the character offset of the
answer/reasoning portion.

The heuristic: find the last question-like line (ends with ? or contains
"What follows", "What can we conclude", etc.), and set answer_start to the
character after the newline following that line.

Usage:
    python scripts/add_answer_starts.py \
        --input data/training/format_augmented_train.jsonl \
        --output data/training/answer_masked_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Patterns that indicate a question/prompt line (not an answer)
QUESTION_PATTERNS = [
    r"\?$",                        # Ends with ?
    r"^What follows",              # "What follows?"
    r"^What can (we|be)",          # "What can we conclude?"
    r"^What is the conclusion",
    r"^What must be true",
    r"^Determine",
    r"^Consider the following",
    r"^Apply logical reasoning",
    r"^Logical reasoning",
    r"^Analyze this",
    r"^Given:",
    r"^Question:",
]

# Patterns that indicate the START of an answer/reasoning
ANSWER_START_PATTERNS = [
    r"^(Premise|We know|The premise|Since |By modus|The contrapositive|Conclusion|Therefore|It follows|The rule|From the|Using|Applying|Given that|Because|This|If |For|The answer|Yes|No)",
]


def find_answer_start(text: str) -> int:
    """Find the character offset where the answer/reasoning begins.

    Strategy: split text into lines, find the last line that looks like
    a question or prompt, and return the offset of the next line.
    """
    lines = text.split("\n")
    if len(lines) <= 1:
        return 0  # Single line — train on everything

    # Find the last question-like line
    last_question_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                last_question_idx = i
                break

    if last_question_idx < 0:
        # No question found — try to find answer start by looking at first
        # line that matches answer patterns
        offset = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i > 0 and stripped:
                for pattern in ANSWER_START_PATTERNS:
                    if re.match(pattern, stripped, re.IGNORECASE):
                        return offset
            offset += len(line) + 1  # +1 for newline
        return 0  # Fallback: full sequence

    # answer_start = character offset of the line after the last question
    offset = 0
    for i in range(last_question_idx + 1):
        offset += len(lines[i]) + 1  # +1 for the newline character

    return offset


def main():
    parser = argparse.ArgumentParser(description="Add answer_start to JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--verify", action="store_true", help="Print verification stats")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    samples = []
    with input_path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                samples.append(json.loads(stripped))

    results = []
    stats = {"total": 0, "with_answer": 0, "full_sequence": 0}

    for sample in samples:
        text = sample.get("text", "")
        answer_start = find_answer_start(text)
        result = dict(sample)
        result["answer_start"] = answer_start
        results.append(result)

        stats["total"] += 1
        if answer_start > 0:
            stats["with_answer"] += 1
        else:
            stats["full_sequence"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Processed {stats['total']} samples:")
    print(f"  With answer_start > 0: {stats['with_answer']}")
    print(f"  Full sequence (answer_start=0): {stats['full_sequence']}")

    if args.verify:
        print("\nSample verification:")
        for i in [0, 1, len(results) // 2, -1]:
            r = results[i]
            text = r["text"]
            start = r["answer_start"]
            prompt = text[:start].rstrip()
            answer = text[start:]
            print(f"\n--- Sample {i} (answer_start={start}) ---")
            print(f"PROMPT: {prompt[:100]}...")
            print(f"ANSWER: {answer[:100]}...")


if __name__ == "__main__":
    main()
