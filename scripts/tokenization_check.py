#!/usr/bin/env python3
"""Debug: Check tokenization of digit strings."""

from mlx_lm import load
import numpy as np

print("Loading model...")
model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

print("\n=== TOKENIZATION CHECK ===")

# Check how digits tokenize
for digit in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
    tokens = tokenizer.encode(digit)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"'{digit}' encodes to tokens {tokens}: {decoded}")

print("\n=== CHECKING MODEL PREDICTIONS ===")

import mlx.core as mx

for prompt in ["4+1=", "3+1=", "2+1="]:
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].tolist())
    top5 = np.argsort(logits_np)[-5:][::-1]

    print(f"\n'{prompt}'")
    print(f"  Top 5 token IDs: {list(top5)}")
    print(f"  Top 5 decoded: {[tokenizer.decode([t]) for t in top5]}")
    print(f"  Top 5 repr: {[repr(tokenizer.decode([t])) for t in top5]}")

    # What's the token ID for the expected answer?
    expected = str(int(prompt[0]) + int(prompt[2]))  # Extract operands
    expected_tokens = tokenizer.encode(expected)
    print(f"  Expected '{expected}' encodes to: {expected_tokens}")

    # Check if any top token decodes to include the expected digit
    for t in top5:
        decoded = tokenizer.decode([t])
        if expected in decoded:
            print(f"  *** Token {t} ('{decoded}') contains expected '{expected}'")
