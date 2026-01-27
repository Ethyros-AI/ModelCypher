#!/usr/bin/env python3
"""Debug all 5 failing problems."""

import sys
from pathlib import Path
import numpy as np
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_lm import load
import mlx.core as mx
from modelcypher.core.use_cases.curriculum import BenchmarkLoader

model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
adapter_path = "data/adapters/qwen3_gsm8k_heavy_lora"

print("Loading model...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

loader = BenchmarkLoader()
gsm_test = loader.load("gsm8k", split="test", limit=30)

# Failed indices: 0 (Janet), 2 (Josh), 8 (John), 12 (Carlos), 13 (Melanie)
failed_indices = [0, 2, 8, 12, 13]

for idx in failed_indices:
    sample = gsm_test.samples[idx]
    question = sample.prompt.replace("Answer:", "").strip()
    expected = sample.answer

    print("\n" + "="*70)
    print(f"PROBLEM {idx}: Expected {expected}")
    print("="*70)
    print(f"Q: {question[:100]}...")

    prompt = f"Question: {question}\n\nAnswer:"
    tokens = tokenizer.encode(prompt)
    generated = []

    for _ in range(300):
        logits = model(mx.array([tokens + generated]))
        mx.eval(logits)
        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        next_tok = int(np.argmax(probs))
        generated.append(next_tok)

        decoded = tokenizer.decode(generated)
        if "####" in decoded:
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

    output = tokenizer.decode(generated).replace("<|im_end|>", "").replace("!", "")
    print(f"\nOutput:\n{output[:500]}")

    # Extract
    if "####" in output:
        answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
        numbers = re.findall(r'-?\d+', answer_part)
        predicted = numbers[0] if numbers else ""
    else:
        numbers = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = numbers[-1] if numbers else ""

    status = "OK" if predicted == expected else "XX"
    print(f"\n{status}: Predicted '{predicted}', Expected '{expected}'")

del model
mx.clear_cache()
