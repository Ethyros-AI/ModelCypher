#!/usr/bin/env python3
"""Check if failing test problems appear in training set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

loader = BenchmarkLoader()

# Load training and test
train = loader.load("gsm8k", split="train", limit=7500)  # Full training set
test = loader.load("gsm8k", split="test", limit=30)

# Failed indices
failed_indices = [0, 2, 8, 12, 13]

# Get failing test questions
failing_questions = [test.samples[i].prompt.replace("Answer:", "").strip()[:100] for i in failed_indices]
failing_answers = [test.samples[i].answer for i in failed_indices]

print("FAILING TEST PROBLEMS:")
for i, (q, a) in enumerate(zip(failing_questions, failing_answers)):
    print(f"\n{i+1}. Answer: {a}")
    print(f"   Q: {q}...")

# Search for similar problems in training
print("\n" + "="*70)
print("SEARCHING TRAINING SET FOR SIMILAR PATTERNS...")
print("="*70)

# Keywords to search for - (name, [keywords_to_match])
patterns = [
    ("Janet/eggs/ducks", ["eggs", "ducks"]),
    ("Josh/flipping/house", ["flipping", "house"]),
    ("John/drives/traffic", ["drives", "traffic"]),
    ("Carlos/lemon/tree", ["lemon", "tree"]),
    ("Melanie/vacuum/sales", ["vacuum", "saleswoman"]),
]

for name, keywords in patterns:
    print(f"\nPattern: {name}")
    count = 0
    for sample in train.samples:
        q = sample.prompt.lower()
        if any(kw.lower() in q for kw in keywords):
            count += 1
            if count <= 3:
                print(f"  Found: {sample.prompt[:80]}...")
                print(f"  Answer: {sample.answer}")
    print(f"  Total found: {count}")
