#!/usr/bin/env python3
"""Analyze the 6 failing GSM8K problems to understand what training is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

loader = BenchmarkLoader()
gsm_test = loader.load("gsm8k", split="test", limit=30)

# The problems that failed (indices 2, 7, 8, 12, 13, 19 in 0-indexed)
failed_indices = [2, 7, 8, 12, 13, 19]

print("=" * 80)
print("FAILED GSM8K PROBLEMS - FULL TEXT")
print("=" * 80)

for i, sample in enumerate(gsm_test.samples[:20]):
    if i in failed_indices:
        print(f"\n{'='*80}")
        print(f"PROBLEM {i+1} - Expected: {sample.answer}")
        print("=" * 80)
        print(f"Question: {sample.prompt.replace('Answer:', '').strip()}")
        print(f"\nFull Solution:\n{sample.metadata.get('full_answer', 'N/A')}")
