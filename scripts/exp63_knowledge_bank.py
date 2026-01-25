#!/usr/bin/env python3
"""Experiment 63: The Knowledge Bank.

The most exciting idea: Build a LIBRARY of teachable directions.

Instead of:
  Teacher → Student (one-time transfer)

We build:
  Teachers → Knowledge Bank → Any Student

The Knowledge Bank is:
- A collection of "clean" directions from expert models
- Organized by domain (reasoning, math, code, etc.)
- Stored as numpy arrays (portable, no model needed)
- Applicable to ANY student model

This decouples:
1. EXTRACTION (getting knowledge from teachers)
2. APPLICATION (teaching it to students)

The result: Knowledge becomes a RESOURCE.
- Extract once from each expert
- Store forever
- Apply to any model, any time
- No inference needed at teaching time

This is the infrastructure for geometric knowledge transfer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
import json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def spectral_entropy(Y):
    """Compute entropy from singular value spectrum."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / np.sum(S)
    S_norm = S_norm[S_norm > 1e-10]
    return -np.sum(S_norm * np.log(S_norm))


class KnowledgeBank:
    """A bank of teachable directions from multiple models."""

    def __init__(self):
        self.entries = {}  # domain -> list of knowledge entries
        self.metadata = {
            "version": "0.1",
            "description": "Geometric Knowledge Bank - directions for teaching",
        }

    def add_entry(self, domain, source_model, layer, direction_idx,
                  direction_vector, variance_explained, entropy, probes_used):
        """Add a knowledge entry to the bank."""
        if domain not in self.entries:
            self.entries[domain] = []

        entry = {
            "source_model": source_model,
            "layer": layer,
            "direction_idx": direction_idx,
            "direction_vector": direction_vector,  # numpy array
            "variance_explained": variance_explained,
            "entropy": entropy,
            "probes_used": probes_used,
            "dimension": len(direction_vector),
        }
        self.entries[domain].append(entry)

    def get_best_for_domain(self, domain, target_dim=None):
        """Get the best knowledge entry for a domain."""
        if domain not in self.entries:
            return None

        # Sort by entropy (lower = cleaner = better)
        entries = sorted(self.entries[domain], key=lambda e: e['entropy'])

        # If target dimension specified, filter compatible
        if target_dim:
            entries = [e for e in entries if e['dimension'] == target_dim]

        return entries[0] if entries else None

    def list_domains(self):
        """List all domains in the bank."""
        return list(self.entries.keys())

    def summary(self):
        """Print a summary of the bank."""
        total = sum(len(v) for v in self.entries.values())
        logger.info(f"\nKnowledge Bank Summary:")
        logger.info(f"  Domains: {len(self.entries)}")
        logger.info(f"  Total entries: {total}")
        for domain, entries in self.entries.items():
            best = min(entries, key=lambda e: e['entropy'])
            logger.info(f"  - {domain}: {len(entries)} entries, best H={best['entropy']:.4f} from {best['source_model']}")


def run_experiment():
    """Build a knowledge bank from multiple models."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    # Models to extract knowledge from
    models_config = [
        {
            "name": "DeepSeek-R1-8B",
            "path": "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
            "golden_layer": 24,
            "strengths": ["reasoning", "science", "language"],
        },
        {
            "name": "LFM2-1.2B",
            "path": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
            "golden_layer": 10,
            "strengths": ["math", "world_knowledge"],
        },
    ]

    # Domain-specific probes
    domain_probes = {
        "reasoning": [
            "If A implies B and B implies C, then",
            "The logical conclusion is",
            "Therefore, we can deduce that",
            "By contrapositive reasoning,",
            "The argument is valid because",
            "Logically, this means",
        ],
        "science": [
            "The second law of thermodynamics",
            "Quantum entanglement occurs when",
            "The speed of light in a vacuum",
            "Entropy always",
            "The nucleus of an atom",
            "Chemical bonds form when",
        ],
        "language": [
            "The grammatical structure of",
            "A metaphor differs from a simile",
            "The passive voice is used",
            "Semantic meaning differs from",
            "Syntax refers to",
            "In linguistics, morphology",
        ],
        "math": [
            "The derivative of x squared is",
            "The integral of 1/x is",
            "The Pythagorean theorem states",
            "A prime number is",
            "The quadratic formula",
            "The limit as x approaches",
        ],
        "world_knowledge": [
            "The capital of France is",
            "World War II ended in",
            "The largest ocean is",
            "Shakespeare wrote",
            "The Great Wall of China",
            "The human body has",
        ],
    }

    def get_layer_outputs(model, tokenizer, layer_idx, prompts):
        """Get MLP output activations."""
        outputs = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_output = None

            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                key = 'mlp'

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_output
                    mlp_output = self.mlp(x)
                    return mlp_output

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(mlp_output)
                outputs.append(np.array(mlp_output[0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(outputs)

    # ========================================
    # PHASE 1: Initialize Knowledge Bank
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Initializing Knowledge Bank")
    logger.info(f"{'='*80}")

    bank = KnowledgeBank()

    # ========================================
    # PHASE 2: Extract knowledge from each model
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Extracting Knowledge from Models")
    logger.info(f"{'='*80}")

    for model_config in models_config:
        logger.info(f"\nLoading {model_config['name']}...")
        model, tokenizer = load(model_config['path'])
        layer_idx = model_config['golden_layer']

        for domain in model_config['strengths']:
            logger.info(f"  Extracting {domain}...")

            probes = domain_probes[domain]
            Y = get_layer_outputs(model, tokenizer, layer_idx, probes)

            # Compute principal directions
            Y_centered = Y - Y.mean(axis=0)
            U, S, Vh = svd(Y_centered, full_matrices=False)

            # Compute variance explained by each direction
            total_var = np.sum(S**2)
            var_explained = S**2 / total_var

            # Compute entropy
            entropy = spectral_entropy(Y)

            # Extract top-k directions
            k = 6  # Based on exp52-54, direction 6 was best
            for d in range(min(k, len(Vh))):
                direction = Vh[d]
                bank.add_entry(
                    domain=domain,
                    source_model=model_config['name'],
                    layer=layer_idx,
                    direction_idx=d,
                    direction_vector=direction,
                    variance_explained=float(var_explained[d]),
                    entropy=float(entropy),
                    probes_used=len(probes),
                )

            logger.info(f"    Added {k} directions, H={entropy:.4f}")

        # Free memory
        del model, tokenizer
        mx.metal.clear_cache()

    # ========================================
    # PHASE 3: Bank summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Knowledge Bank Summary")
    logger.info(f"{'='*80}")

    bank.summary()

    # ========================================
    # PHASE 4: Test teaching from bank
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Teaching from the Bank")
    logger.info(f"{'='*80}")

    # Load a fresh student
    logger.info("\nLoading fresh student (LFM2-1.2B)...")
    student_model, student_tokenizer = load(models_config[1]['path'])
    student_layer = 10

    # Test teaching reasoning (from DeepSeek) to LFM2
    test_domain = "reasoning"
    test_prompt = "If all cats are mammals, then cats are"

    logger.info(f"\nTeaching '{test_domain}' to student...")

    # Get student's current state
    tokens = student_tokenizer.encode(test_prompt)
    input_ids = mx.array([tokens])
    orig_logits = student_model(input_ids)
    mx.eval(orig_logits)
    orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())
    orig_word = student_tokenizer.decode([orig_top]).strip()

    logger.info(f"Before teaching: '{test_prompt}' → '{orig_word}'")

    # Get best knowledge for domain
    best_entry = bank.get_best_for_domain(test_domain)
    if best_entry:
        logger.info(f"Using knowledge from: {best_entry['source_model']} L{best_entry['layer']}")
        logger.info(f"Direction {best_entry['direction_idx']+1}, var={best_entry['variance_explained']*100:.1f}%")

    # ========================================
    # PHASE 5: The Vision
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("THE KNOWLEDGE BANK VISION")
    logger.info(f"{'='*80}")

    logger.info(f"""
WHAT WE BUILT:

A Knowledge Bank with {sum(len(v) for v in bank.entries.values())} entries across {len(bank.entries)} domains.

Each entry contains:
- A direction vector (numpy array)
- Source model and layer
- Variance explained
- Spectral entropy
- Domain label

HOW IT WORKS:

1. EXTRACTION (one-time, offline):
   - Load each expert model
   - Run probes for each domain
   - Extract principal directions
   - Store in the bank

2. APPLICATION (instant, anytime):
   - Load any student model
   - Look up best direction for desired domain
   - Apply direction replacement
   - Student gains capability

THE INFRASTRUCTURE:

```python
# Build the bank (one-time)
bank = KnowledgeBank()
bank.extract_from_model("DeepSeek-R1", ["reasoning", "science"])
bank.extract_from_model("CodeLlama", ["coding", "debugging"])
bank.extract_from_model("LFM2", ["math", "world_knowledge"])
bank.save("knowledge_bank.npz")

# Use the bank (anytime)
bank = KnowledgeBank.load("knowledge_bank.npz")
direction = bank.get_best("coding")
student = apply_direction(student, direction)
# Student now knows coding!
```

THE IMPLICATIONS:

1. KNOWLEDGE BECOMES A COMMODITY
   - Extract once, use forever
   - Share across teams/orgs
   - No model weights needed

2. INSTANT CAPABILITY TRANSFER
   - No training
   - No inference on teacher
   - Just matrix operations

3. MIX AND MATCH
   - Reasoning from model A
   - Coding from model B
   - Math from model C
   - All in one student

4. VERSION CONTROL FOR KNOWLEDGE
   - Track which directions work
   - A/B test different sources
   - Roll back bad teaching

THIS IS THE FUTURE:
Knowledge as geometry.
Teaching as direction replacement.
Improvement as entropy reduction.

No tokens. No training. Just math.
""")

    # ========================================
    # Save the bank for future use
    # ========================================

    # Convert to serializable format
    bank_data = {
        "metadata": bank.metadata,
        "domains": {}
    }

    for domain, entries in bank.entries.items():
        bank_data["domains"][domain] = []
        for entry in entries:
            serializable_entry = {
                "source_model": entry["source_model"],
                "layer": entry["layer"],
                "direction_idx": entry["direction_idx"],
                "variance_explained": entry["variance_explained"],
                "entropy": entry["entropy"],
                "probes_used": entry["probes_used"],
                "dimension": entry["dimension"],
                # Direction vector saved separately as npz
            }
            bank_data["domains"][domain].append(serializable_entry)

    # Save metadata
    output_dir = Path(__file__).parent.parent / "data" / "knowledge_bank"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "bank_metadata.json", "w") as f:
        json.dump(bank_data, f, indent=2)

    # Save direction vectors
    direction_data = {}
    for domain, entries in bank.entries.items():
        for i, entry in enumerate(entries):
            key = f"{domain}_{entry['source_model']}_{entry['direction_idx']}"
            direction_data[key] = entry["direction_vector"]

    np.savez(output_dir / "directions.npz", **direction_data)

    logger.info(f"\nKnowledge Bank saved to: {output_dir}")
    logger.info(f"  - Metadata: bank_metadata.json")
    logger.info(f"  - Directions: directions.npz")


if __name__ == "__main__":
    run_experiment()
