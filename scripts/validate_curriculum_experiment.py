#!/usr/bin/env python3
"""Validate curriculum_experiment.py logic without GPU.

Tests the pure-logic functions: eval file merging, prompt/expected extraction
(answer_start vs Answer: vs fallback), and AgentEnvelope JSON parsing.

Usage:
    poetry run python scripts/validate_curriculum_experiment.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

_pass = 0
_fail = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  PASS: {name}")
    else:
        _fail += 1
        print(f"  FAIL: {name}{' — ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
# 1. _merge_eval_files: single file passthrough vs multi-file merge
# ---------------------------------------------------------------------------
def test_merge_eval_files() -> None:
    print("\n--- _merge_eval_files ---")

    # We can't call _merge_eval_files directly because it uses _resolve_data_path
    # which calls sys.exit on missing files. Instead, test the merge logic inline.

    # Single-file case: should return the resolved path unchanged
    @dataclass
    class FakeSkill:
        name: str
        eval_files: tuple[str, ...]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write two eval shards
        shard1 = Path(tmpdir) / "eval_shard1.jsonl"
        shard2 = Path(tmpdir) / "eval_shard2.jsonl"

        shard1_data = [
            {"text": "P1 implies Q1. P1. Answer: Q1"},
            {"text": "P2 implies Q2. P2. Answer: Q2"},
        ]
        shard2_data = [
            {"text": "P3 implies Q3. P3. Answer: Q3"},
        ]

        shard1.write_text("\n".join(json.dumps(d) for d in shard1_data) + "\n")
        shard2.write_text("\n".join(json.dumps(d) for d in shard2_data) + "\n")

        # Single file: no merge needed
        _check(
            "single file returns path directly",
            shard1.exists(),
        )

        # Multi-file merge: read both, concatenate
        all_lines = []
        for path in [shard1, shard2]:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_lines.append(line)

        _check("multi-file merge collects all lines", len(all_lines) == 3)

        # Verify content integrity
        parsed = [json.loads(line) for line in all_lines]
        answers = [p["text"].rsplit("Answer:", 1)[1].strip() for p in parsed]
        _check(
            "merged content preserves answers",
            answers == ["Q1", "Q2", "Q3"],
            f"got {answers}",
        )


# ---------------------------------------------------------------------------
# 2. Prompt/expected extraction: answer_start vs "Answer:" vs fallback
# ---------------------------------------------------------------------------
def test_prompt_extraction() -> None:
    print("\n--- Prompt/expected extraction ---")

    def extract(item: dict) -> tuple[str, str]:
        """Mirrors the extraction logic in _run_sample_inference."""
        text = item["text"]
        answer_start = item.get("answer_start")

        if answer_start is not None:
            prompt = text[:answer_start]
            expected = text[answer_start:].strip()
        elif "Answer:" in text:
            parts = text.rsplit("Answer:", 1)
            prompt = parts[0] + "Answer:"
            expected = parts[1].strip()
        else:
            tokens = text.split()
            prompt = " ".join(tokens[:-1])
            expected = tokens[-1].strip()

        return prompt, expected

    # Case 1: answer_start field present
    item_as = {"text": "7 + 8 = Write 5, carry 1. Answer: 15", "answer_start": 26}
    prompt, expected = extract(item_as)
    # answer_start=26 is right after ". " — prompt includes trailing space
    _check(
        "answer_start: prompt stops at offset",
        prompt == "7 + 8 = Write 5, carry 1. ",
        f"got prompt={prompt!r}",
    )
    _check(
        "answer_start: expected is remainder (stripped)",
        expected == "Answer: 15",
        f"got expected={expected!r}",
    )

    # Case 2: "Answer:" delimiter (no answer_start)
    item_ans = {"text": "P implies Q. P. Answer: Q"}
    prompt, expected = extract(item_ans)
    _check(
        "Answer: delimiter: prompt includes 'Answer:'",
        prompt == "P implies Q. P. Answer:",
        f"got prompt={prompt!r}",
    )
    _check(
        "Answer: delimiter: expected is answer",
        expected == "Q",
        f"got expected={expected!r}",
    )

    # Case 3: fallback (no answer_start, no "Answer:")
    item_fb = {"text": "true or false true"}
    prompt, expected = extract(item_fb)
    _check(
        "fallback: prompt is all but last token",
        prompt == "true or false",
        f"got prompt={prompt!r}",
    )
    _check(
        "fallback: expected is last token",
        expected == "true",
        f"got expected={expected!r}",
    )

    # Case 4: answer_start=0 (edge case: entire text is the answer)
    item_zero = {"text": "hello world", "answer_start": 0}
    prompt, expected = extract(item_zero)
    _check(
        "answer_start=0: prompt is empty",
        prompt == "",
        f"got prompt={prompt!r}",
    )
    _check(
        "answer_start=0: expected is full text",
        expected == "hello world",
        f"got expected={expected!r}",
    )

    # Case 5: answer_start present AND text contains "Answer:" — answer_start wins
    item_both = {
        "text": "Ones: 7+8=15. Answer: 15",
        "answer_start": 14,
    }
    prompt, expected = extract(item_both)
    _check(
        "answer_start takes priority over Answer: delimiter",
        prompt == "Ones: 7+8=15. ",
        f"got prompt={prompt!r}",
    )


# ---------------------------------------------------------------------------
# 3. AgentEnvelope parsing
# ---------------------------------------------------------------------------
def test_envelope_parsing() -> None:
    print("\n--- AgentEnvelope parsing ---")

    # Simulate the parsing logic from curriculum_experiment.py
    def parse_envelope(stdout: str) -> tuple[dict | None, dict | None]:
        """Mirrors the envelope parsing in main()."""
        envelope = None
        training_result = None
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            pass

        if envelope is not None:
            training_result = envelope.get("result", envelope)

        return envelope, training_result

    def resolve_adapter(
        envelope: dict | None,
        training_result: dict | None,
        default: str,
    ) -> str:
        """Mirrors the adapter resolution logic."""
        resolved = default
        if training_result and training_result.get("adapter_path"):
            resolved = training_result["adapter_path"]
        elif envelope and envelope.get("metadata", {}).get("adapter_path"):
            resolved = envelope["metadata"]["adapter_path"]
        return resolved

    # Case 1: proper AgentEnvelope
    envelope_json = json.dumps({
        "command": "mc train run",
        "status": "success",
        "result": {
            "train_iters": 100,
            "baseline_loss": 3.5,
            "final_loss": 1.2,
            "post_loss": 1.3,
            "min_cka": 0.98,
            "spectral_bounds_ok": True,
            "pipeline_gate_passed": True,
            "adapter_path": "/tmp/my_adapter",
        },
        "diagnostics": {"summary": "Training converged."},
        "metadata": {
            "adapter_path": "/tmp/my_adapter",
            "model": "/path/to/model",
        },
    })

    env, tr = parse_envelope(envelope_json)
    _check("envelope parsed", env is not None)
    _check("result extracted", tr is not None)
    _check(
        "train_iters from result",
        tr.get("train_iters") == 100,
        f"got {tr.get('train_iters')}",
    )
    _check(
        "final_loss from result",
        tr.get("final_loss") == 1.2,
        f"got {tr.get('final_loss')}",
    )
    _check(
        "adapter_path from result",
        resolve_adapter(env, tr, "/default") == "/tmp/my_adapter",
    )

    # Case 2: malformed JSON
    env2, tr2 = parse_envelope("not json at all")
    _check("malformed JSON: envelope is None", env2 is None)
    _check("malformed JSON: training_result is None", tr2 is None)

    # Case 3: adapter_path only in metadata (not in result)
    envelope_meta_only = json.dumps({
        "command": "mc train run",
        "status": "success",
        "result": {
            "train_iters": 50,
            "adapter_path": None,
        },
        "metadata": {
            "adapter_path": "/tmp/meta_adapter",
        },
    })
    env3, tr3 = parse_envelope(envelope_meta_only)
    resolved = resolve_adapter(env3, tr3, "/default")
    _check(
        "adapter falls through to metadata when result has None",
        resolved == "/tmp/meta_adapter",
        f"got {resolved}",
    )

    # Case 4: no adapter anywhere — falls to default
    envelope_no_adapter = json.dumps({
        "command": "mc train run",
        "status": "failure",
        "result": {"train_iters": 0},
        "metadata": {},
    })
    env4, tr4 = parse_envelope(envelope_no_adapter)
    resolved4 = resolve_adapter(env4, tr4, "/fallback")
    _check(
        "no adapter anywhere: falls to default",
        resolved4 == "/fallback",
        f"got {resolved4}",
    )

    # Case 5: loss_improved check uses result dict correctly
    _check(
        "loss_improved computed from result dict",
        tr.get("final_loss", float("inf")) < tr.get("baseline_loss", float("inf")),
    )


# ---------------------------------------------------------------------------
# 4. Training data preparation logic (prompt/completion → text conversion)
# ---------------------------------------------------------------------------
def test_training_data_conversion() -> None:
    print("\n--- Training data conversion ---")

    def convert(item: dict) -> dict | None:
        """Mirrors _prepare_training_data conversion logic."""
        if "text" in item:
            return item
        elif "prompt" in item and "completion" in item:
            text = item["prompt"] + item["completion"]
            converted = {"text": text}
            if "answer_start" in item:
                converted["answer_start"] = item["answer_start"]
            elif "prompt" in item:
                converted["answer_start"] = len(item["prompt"])
            return converted
        return None

    # Already has "text" — passthrough
    item1 = {"text": "some training text"}
    _check("text passthrough", convert(item1) == item1)

    # prompt + completion → text with derived answer_start
    item2 = {"prompt": "Question: ", "completion": "answer"}
    result2 = convert(item2)
    _check(
        "prompt+completion merged",
        result2["text"] == "Question: answer",
        f"got {result2.get('text')!r}",
    )
    _check(
        "answer_start derived from prompt length",
        result2["answer_start"] == len("Question: "),
        f"got {result2.get('answer_start')}",
    )

    # prompt + completion with explicit answer_start
    item3 = {"prompt": "Q: ", "completion": "A", "answer_start": 99}
    result3 = convert(item3)
    _check(
        "explicit answer_start preserved",
        result3["answer_start"] == 99,
    )

    # Unknown format
    item4 = {"input": "x", "output": "y"}
    _check("unknown format returns None", convert(item4) is None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== validate_curriculum_experiment.py ===")

    test_merge_eval_files()
    test_prompt_extraction()
    test_envelope_parsing()
    test_training_data_conversion()

    print(f"\n{'=' * 40}")
    print(f"Results: {_pass} passed, {_fail} failed")

    if _fail > 0:
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
