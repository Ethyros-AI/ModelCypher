#!/usr/bin/env python3
"""Debug: Why can't models answer 2+2?

Something is clearly wrong. Let's investigate:
1. Different prompt formats
2. Generate more than 1 token
3. Try chat template
4. Look at actual probability distribution
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_debug():
    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("="*80)
    logger.info("DEBUG: Math Prediction Investigation")
    logger.info("="*80)

    # Test with Qwen2.5-Math specifically
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Math-1.5B-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Different prompt formats to try
    prompts = [
        # Our original format
        "2 + 2 equals",
        "The square root of 16 is",

        # More direct
        "2 + 2 =",
        "2+2=",
        "What is 2 + 2?",
        "Calculate: 2 + 2",

        # Question format
        "What is the square root of 16?",
        "sqrt(16) =",

        # More context
        "In mathematics, 2 + 2 equals",
        "Answer: 2 + 2 =",
    ]

    logger.info("\n" + "="*60)
    logger.info("TEST 1: Single token prediction (our method)")
    logger.info("="*60)

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        top_token = int(mx.argmax(logits[0, -1, :]).item())
        top_word = tokenizer.decode([top_token])

        # Get top 3
        import numpy as np
        logits_np = np.array(logits[0, -1, :].tolist())
        top3_idx = np.argsort(logits_np)[-3:][::-1]
        top3 = [tokenizer.decode([int(i)]) for i in top3_idx]

        logger.info(f"\n  '{prompt}'")
        logger.info(f"    Top-1: '{top_word}'")
        logger.info(f"    Top-3: {top3}")

    logger.info("\n" + "="*60)
    logger.info("TEST 2: Full generation (multiple tokens)")
    logger.info("="*60)

    for prompt in prompts[:4]:
        logger.info(f"\n  Prompt: '{prompt}'")

        # Generate more tokens
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=20,
            verbose=False
        )
        logger.info(f"    Generated: '{response}'")

    logger.info("\n" + "="*60)
    logger.info("TEST 3: Check if model has chat template")
    logger.info("="*60)

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        logger.info("  Model HAS a chat template!")

        # Try with chat template
        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info(f"\n  Chat formatted prompt:\n{chat_prompt[:200]}...")

            response = generate(
                model,
                tokenizer,
                prompt=chat_prompt,
                max_tokens=50,
                verbose=False
            )
            logger.info(f"\n  Chat response: '{response}'")
        except Exception as e:
            logger.info(f"  Chat template error: {e}")
    else:
        logger.info("  Model does NOT have a chat template")

    logger.info("\n" + "="*60)
    logger.info("TEST 4: Look at where '4' appears in vocabulary")
    logger.info("="*60)

    # Find all tokens that contain "4"
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
    four_tokens = []
    for i in range(min(vocab_size, 100000)):
        try:
            decoded = tokenizer.decode([i])
            if '4' in decoded and len(decoded.strip()) <= 3:
                four_tokens.append((i, decoded))
        except:
            pass

    logger.info(f"  Tokens containing '4': {four_tokens[:20]}")

    # What's the actual token for "4"?
    four_encoded = tokenizer.encode("4")
    logger.info(f"  '4' encodes to: {four_encoded}")

    four_with_space = tokenizer.encode(" 4")
    logger.info(f"  ' 4' encodes to: {four_with_space}")

    logger.info("\n" + "="*60)
    logger.info("TEST 5: Check probability of '4' vs ' 4' vs 'four'")
    logger.info("="*60)

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)

    import numpy as np
    logits_np = np.array(logits[0, -1, :].tolist())
    probs = np.exp(logits_np) / np.exp(logits_np).sum()

    # Check specific tokens
    check_tokens = {
        "'4'": tokenizer.encode("4")[0] if tokenizer.encode("4") else None,
        "' 4'": tokenizer.encode(" 4")[-1] if tokenizer.encode(" 4") else None,
        "'four'": tokenizer.encode("four")[0] if tokenizer.encode("four") else None,
        "' four'": tokenizer.encode(" four")[-1] if tokenizer.encode(" four") else None,
    }

    logger.info(f"\n  Prompt: '{prompt}'")
    logger.info(f"  Token probabilities:")
    for name, tok_id in check_tokens.items():
        if tok_id is not None:
            prob = probs[tok_id]
            rank = int((logits_np >= logits_np[tok_id]).sum())
            logger.info(f"    {name}: token_id={tok_id}, prob={prob:.6f}, rank={rank}")

    # What's the actual top prediction?
    top_idx = np.argmax(logits_np)
    logger.info(f"\n  Top prediction: token_id={top_idx}, '{tokenizer.decode([int(top_idx)])}', prob={probs[top_idx]:.4f}")

    logger.info("\n" + "="*60)
    logger.info("CONCLUSION")
    logger.info("="*60)


if __name__ == "__main__":
    run_debug()
