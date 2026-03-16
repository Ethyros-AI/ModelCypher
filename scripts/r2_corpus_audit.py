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

"""R2 Corpus Audit: Measure answer-style distribution in the quick-aligned training data.

Answers the question: what does the training data teach the model to produce
after "Answer:"? If the corpus overwhelmingly starts GSM8K answers with bare
numbers, the adapter learned "after 'Answer:', produce a number" — and Phase D
showed those numbers are wrong for unseen prompts.

No GPU required. Reads JSONL only.

Usage:
    poetry run python scripts/r2_corpus_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_PATH = Path("data/training/r1_quick_aligned_train.jsonl")
VAL_PATH = Path("data/training/r1_quick_aligned_val.jsonl")
OUTPUT_DIR = Path("results/r2_corpus_audit")


def categorize_first_token(answer: str) -> str:
    """Categorize the first whitespace-delimited token of an answer."""
    token = answer.split()[0] if answer.split() else ""
    if not token:
        return "empty"
    if re.match(r"^-?\d[\d,.\-/]*$", token):
        return "digit"
    if token.lower() in ("yes", "no"):
        return "yes_no"
    if re.match(r"^[A-Da-d]\.?$", token):
        return "letter_label"
    return "content_word"


def parse_records(path: Path) -> list[dict]:
    """Parse JSONL and extract answer portions."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "")
            task = data.get("task", "unknown")

            # Extract answer after "Answer:"
            if "Answer:" in text:
                answer = text.split("Answer:")[-1].strip()
            elif "answer:" in text.lower():
                idx = text.lower().rfind("answer:")
                answer = text[idx + 7:].strip()
            else:
                answer = ""

            first_token = answer.split()[0] if answer.split() else ""
            category = categorize_first_token(answer)
            n_words = len(answer.split())

            records.append({
                "task": task,
                "answer": answer,
                "first_token": first_token,
                "category": category,
                "n_words": n_words,
            })
    return records


def build_report(train_records: list[dict], val_records: list[dict]) -> str:
    lines = [
        "# R2 Corpus Audit: Quick-Aligned Training Data Answer Distribution",
        "",
        f"**Train:** {TRAIN_PATH} ({len(train_records)} records)",
        f"**Val:** {VAL_PATH} ({len(val_records)} records)",
        "",
    ]

    for split_name, records in [("Train", train_records), ("Val", val_records)]:
        lines.extend([f"## {split_name} Split", ""])

        # Overall task distribution
        task_counts = Counter(r["task"] for r in records)
        lines.extend([
            "### Task Distribution",
            "",
            "| Task | Count | % |",
            "|------|------:|--:|",
        ])
        for task, count in task_counts.most_common():
            lines.append(f"| {task} | {count} | {count/len(records)*100:.1f}% |")
        lines.append("")

        # Per-task first-token category
        lines.extend([
            "### First-Token Category by Task",
            "",
            "| Task | digit | yes_no | letter_label | content_word | empty |",
            "|------|------:|-------:|-------------:|-------------:|------:|",
        ])
        tasks = sorted(set(r["task"] for r in records))
        for task in tasks:
            task_recs = [r for r in records if r["task"] == task]
            cats = Counter(r["category"] for r in task_recs)
            n = len(task_recs)
            def pct(cat: str) -> str:
                c = cats.get(cat, 0)
                return f"{c} ({c/n*100:.0f}%)" if c > 0 else "0"
            lines.append(f"| {task} | {pct('digit')} | {pct('yes_no')} | "
                         f"{pct('letter_label')} | {pct('content_word')} | {pct('empty')} |")
        lines.append("")

        # Per-task answer length distribution
        lines.extend([
            "### Answer Length (words) by Task",
            "",
            "| Task | Min | Median | Mean | Max |",
            "|------|----:|-------:|-----:|----:|",
        ])
        for task in tasks:
            lengths = sorted(r["n_words"] for r in records if r["task"] == task)
            if lengths:
                n = len(lengths)
                median = lengths[n // 2]
                mean = sum(lengths) / n
                lines.append(f"| {task} | {lengths[0]} | {median} | {mean:.1f} | {lengths[-1]} |")
        lines.append("")

        # Sample answers per task (first 5)
        lines.extend(["### Sample Answers (first 5 per task)", ""])
        for task in tasks:
            task_recs = [r for r in records if r["task"] == task][:5]
            lines.append(f"**{task}:**")
            for r in task_recs:
                answer_preview = r["answer"][:80] + ("..." if len(r["answer"]) > 80 else "")
                lines.append(f"- [{r['category']}] `{answer_preview}`")
            lines.append("")

        # Most common first tokens per task
        lines.extend(["### Top 10 First Tokens by Task", ""])
        for task in tasks:
            task_recs = [r for r in records if r["task"] == task]
            token_counts = Counter(r["first_token"] for r in task_recs)
            lines.append(f"**{task}:**")
            for token, count in token_counts.most_common(10):
                lines.append(f"- `{token}` ({count}, {count/len(task_recs)*100:.1f}%)")
            lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "If GSM8K answers in the training data are overwhelmingly bare numbers "
        "(category=digit), the adapter learned to produce digits after 'Answer:'. "
        "Phase D showed those digits are often WRONG for unseen benchmark prompts.",
        "",
        "If ARC answers are full content words rather than letter labels, the adapter "
        "learned a different answer format than what the benchmark evaluation might "
        "expect for letter-based checking (though the checker also accepts content).",
        "",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} not found", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {TRAIN_PATH}...")
    train_records = parse_records(TRAIN_PATH)
    print(f"  {len(train_records)} records")

    val_records = []
    if VAL_PATH.exists():
        print(f"Parsing {VAL_PATH}...")
        val_records = parse_records(VAL_PATH)
        print(f"  {len(val_records)} records")

    report = build_report(train_records, val_records)

    # Write JSON
    json_path = OUTPUT_DIR / "corpus_audit.json"
    json_data = {
        "train": {
            "n_records": len(train_records),
            "by_task": {},
        },
        "val": {
            "n_records": len(val_records),
            "by_task": {},
        },
    }
    for split_name, records in [("train", train_records), ("val", val_records)]:
        tasks = sorted(set(r["task"] for r in records))
        for task in tasks:
            task_recs = [r for r in records if r["task"] == task]
            cats = Counter(r["category"] for r in task_recs)
            json_data[split_name]["by_task"][task] = {
                "n": len(task_recs),
                "categories": dict(cats),
                "top_first_tokens": Counter(r["first_token"] for r in task_recs).most_common(20),
            }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    # Write report
    report_path = OUTPUT_DIR / "ANALYSIS.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    print()
    print(report)


if __name__ == "__main__":
    main()
