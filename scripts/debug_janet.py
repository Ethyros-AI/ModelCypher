#!/usr/bin/env python3
"""Debug Janet's ducks problem - check full output."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_lm import load
import mlx.core as mx

model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
adapter_path = "data/adapters/qwen3_gsm8k_heavy_lora"

print("Loading model...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# Janet's ducks problem
question = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

prompt = f"Question: {question}\n\nAnswer:"
tokens = tokenizer.encode(prompt)
generated = []

print("\nGenerating response...")
for i in range(300):
    logits = model(mx.array([tokens + generated]))
    mx.eval(logits)
    logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()
    next_tok = int(np.argmax(probs))
    generated.append(next_tok)

    decoded = tokenizer.decode(generated)
    if "####" in decoded:
        # Get more for the answer
        for _ in range(15):
            logits = model(mx.array([tokens + generated]))
            mx.eval(logits)
            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()
            next_tok = int(np.argmax(probs))
            generated.append(next_tok)
        break
    if "<|im_end|>" in decoded:
        break

output = tokenizer.decode(generated)
print("\n" + "="*70)
print("FULL OUTPUT:")
print("="*70)
print(output)
print("="*70)

# Show what we're extracting
import re
output_clean = output.replace("<|im_end|>", "")
if "####" in output_clean:
    answer_part = output_clean.split("####")[-1].strip().replace(",", "").replace("$", "")
    print(f"\nAfter ####: '{answer_part}'")
    numbers = re.findall(r'-?\d+', answer_part)
    print(f"Numbers found: {numbers}")
    print(f"Extracted: {numbers[0] if numbers else 'none'}")
