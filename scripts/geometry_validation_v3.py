#!/usr/bin/env python3
"""Geometry Validation V3: Rigorous token-level analysis.

Fixes from V2 based on code review:
- P0: Exact numeric parsing for correctness, proper answer-span tokenization
- P1: Single forward pass for hidden states AND logits (no mismatch)
- P1: No filtering of missing-answer cases - report separately
- P2: No arbitrary thresholds - report raw values
- P3: Seeded random for reproducibility

Key measurement: Hidden state velocity at the FIRST token of the answer.

Usage:
    poetry run python scripts/geometry_validation_v3.py \
        --model /path/to/model \
        --output results/geometry_v3/ \
        --samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TokenMeasurement:
    """Geometry measurement at a single token position."""
    position: int  # Position in full sequence (prompt + generated)
    token_id: int
    token_str: str

    # Per-layer hidden state norms
    layer_norms: list[float]

    # Per-layer velocity (distance from previous token's hidden state)
    layer_velocities: list[float]

    # Per-layer direction change (1 - cosine similarity to previous)
    layer_direction_changes: list[float]


@dataclass
class GenerationResult:
    """Complete generation with per-token geometry."""
    prompt: str
    expected_answer: str
    generated_text: str

    # Correctness (strict numeric parsing)
    extracted_number: float | None  # First number extracted from generation
    is_correct: bool  # extracted_number == expected (within tolerance)

    # Answer token identification
    answer_token_positions: list[int]  # Positions of tokens comprising the answer
    answer_found: bool  # Whether we could identify answer tokens

    # Per-token measurements
    token_measurements: list[TokenMeasurement]

    # Aggregate metrics at answer (first answer token, or None if not found)
    velocity_at_answer: float | None
    direction_change_at_answer: float | None
    layer_velocities_at_answer: list[float] | None

    # Generation metadata
    n_generated_tokens: int
    prompt_length: int


@dataclass
class ExperimentConfig:
    model_path: str
    output_dir: Path
    n_samples: int = 100
    max_tokens: int = 32
    temperature: float = 0.3
    seed: int = 42


class GeometryValidationV3:
    """Rigorous token-level geometry measurement."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0

        # Seed for reproducibility
        random.seed(config.seed)

    def setup(self) -> None:
        from modelcypher.backends import initialize_default_backend

        logger.info(f"Loading model from {self.config.model_path}")
        self.backend = initialize_default_backend()

        model_path = Path(self.config.model_path)
        self.model, self.tokenizer = self.backend.load_model(str(model_path))

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        self.num_layers = len(layers) if layers else 0

        logger.info(f"Model loaded: {self.num_layers} layers")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_number(self, text: str) -> float | None:
        """Extract the first number from text (strict parsing)."""
        # Match integers or decimals, possibly negative
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def _check_correctness(self, extracted: float | None, expected_str: str) -> bool:
        """Check if extracted number matches expected (within floating point tolerance)."""
        if extracted is None:
            return False

        try:
            expected = float(expected_str)
        except ValueError:
            # Expected might not be a number
            return False

        # Integer comparison for arithmetic
        if expected == int(expected):
            return extracted == expected
        else:
            # Float comparison with tolerance
            return abs(extracted - expected) < 1e-6

    def _find_answer_tokens(
        self,
        generated_tokens: list[int],
        extracted_number: float | None,
        prompt_length: int,
    ) -> list[int]:
        """Find token positions that comprise the answer number.

        Returns positions relative to full sequence (prompt + generated).
        """
        if extracted_number is None:
            return []

        # Format the number as the model likely generated it
        if extracted_number == int(extracted_number):
            answer_str = str(int(extracted_number))
        else:
            answer_str = str(extracted_number)

        # Tokenize the answer string to see what tokens to look for
        answer_token_ids = self.tokenizer.encode(answer_str)
        if isinstance(answer_token_ids, list):
            answer_token_ids = answer_token_ids
        else:
            answer_token_ids = self.backend.tolist(answer_token_ids)

        # Search for this sequence in generated tokens
        for start_idx in range(len(generated_tokens) - len(answer_token_ids) + 1):
            match = True
            for j, expected_id in enumerate(answer_token_ids):
                if generated_tokens[start_idx + j] != expected_id:
                    match = False
                    break
            if match:
                # Return positions relative to full sequence
                return [prompt_length + start_idx + j for j in range(len(answer_token_ids))]

        # Fallback: look for first occurrence of answer substring in decoded text
        # and map back to token positions
        generated_text = self.tokenizer.decode(generated_tokens)
        answer_start = generated_text.find(answer_str)

        if answer_start >= 0:
            # Count tokens up to this character position
            char_count = 0
            for i, tok_id in enumerate(generated_tokens):
                tok_str = self.tokenizer.decode([tok_id])
                if char_count >= answer_start:
                    # This token starts at or after the answer
                    # Return positions for as many tokens as needed
                    n_answer_tokens = max(1, len(answer_str) // 2)  # Rough estimate
                    return [prompt_length + i + j for j in range(min(n_answer_tokens, len(generated_tokens) - i))]
                char_count += len(tok_str)

        return []

    def _generate_with_geometry(
        self,
        prompt: str,
        expected_answer: str,
    ) -> GenerationResult:
        """Generate text while capturing hidden states at each token.

        Critical: Capture hidden states and logits in the SAME forward pass.
        """
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector

        b = self.backend
        projector = LayerEntropyProjector(b)

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)

        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        if isinstance(prompt_tokens, list):
            prompt_ids = prompt_tokens
        else:
            prompt_ids = b.tolist(prompt_tokens)

        prompt_length = len(prompt_ids)
        current_ids = b.array([prompt_ids])

        # Storage
        generated_tokens: list[int] = []
        token_measurements: list[TokenMeasurement] = []
        prev_hidden: dict[int, Any] = {}

        for gen_step in range(self.config.max_tokens):
            # Single forward pass that captures BOTH hidden states AND logits
            target_layers = set(range(self.num_layers))

            # Use wrapper to capture hidden states during forward
            captured: dict[int, Any] = {}

            class CaptureWrapper:
                def __init__(wrapper_self, layer: Any, layer_idx: int) -> None:
                    wrapper_self._layer = layer
                    wrapper_self._layer_idx = layer_idx

                def __call__(wrapper_self, *args: Any, **kwargs: Any) -> Any:
                    output = wrapper_self._layer(*args, **kwargs)
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    captured[wrapper_self._layer_idx] = hidden
                    return output

                def __getattr__(wrapper_self, name: str) -> Any:
                    return getattr(wrapper_self._layer, name)

            # Store original layers
            original_layers = list(layers)

            try:
                # Replace layers with wrappers
                for i in target_layers:
                    if 0 <= i < len(layers):
                        layers[i] = CaptureWrapper(original_layers[i], i)

                # Single forward pass - gets both hidden states (via wrappers) and logits
                outputs = base_model(current_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                b.eval(logits)

            finally:
                # Restore original layers
                for i, layer in enumerate(original_layers):
                    layers[i] = layer

            # Extract hidden state at last position for each layer
            last_pos_hidden: dict[int, Any] = {}
            for layer_idx, hidden in captured.items():
                if hidden.ndim == 3:
                    last_pos_hidden[layer_idx] = hidden[0, -1, :]
                else:
                    last_pos_hidden[layer_idx] = hidden[-1, :]
                b.eval(last_pos_hidden[layer_idx])

            # Compute per-layer metrics
            layer_norms = []
            layer_velocities = []
            layer_direction_changes = []

            for layer_idx in range(self.num_layers):
                h = last_pos_hidden[layer_idx]

                # Norm
                norm = float(b.norm(h))
                layer_norms.append(norm)

                # Velocity and direction from previous token
                if layer_idx in prev_hidden and gen_step > 0:
                    prev_h = prev_hidden[layer_idx]
                    diff = h - prev_h
                    velocity = float(b.norm(diff))

                    # Direction change = 1 - cosine similarity
                    dot = float(b.sum(h * prev_h))
                    norm_prod = norm * float(b.norm(prev_h))
                    if norm_prod > 1e-8:
                        cos_sim = dot / norm_prod
                        direction_change = 1.0 - cos_sim
                    else:
                        direction_change = 0.0

                    layer_velocities.append(velocity)
                    layer_direction_changes.append(direction_change)
                else:
                    layer_velocities.append(0.0)
                    layer_direction_changes.append(0.0)

            # Sample next token from logits (same forward pass)
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]

            if self.config.temperature > 0:
                scaled_logits = last_logits / self.config.temperature
                probs = b.softmax(scaled_logits, axis=-1)
                b.eval(probs)
                probs_list = b.tolist(probs)
                next_token = random.choices(range(len(probs_list)), weights=probs_list, k=1)[0]
            else:
                b.eval(last_logits)
                next_token = int(b.argmax(last_logits))

            generated_tokens.append(next_token)

            # Decode token
            try:
                token_str = self.tokenizer.decode([next_token])
            except:
                token_str = f"<{next_token}>"

            # Store measurement
            token_measurements.append(TokenMeasurement(
                position=prompt_length + gen_step,
                token_id=next_token,
                token_str=token_str,
                layer_norms=layer_norms,
                layer_velocities=layer_velocities,
                layer_direction_changes=layer_direction_changes,
            ))

            # Update state
            prev_hidden = last_pos_hidden
            next_token_arr = b.array([[next_token]])
            current_ids = b.concatenate([current_ids, next_token_arr], axis=1)

            # Check EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_token == eos_id:
                break

        # Process results
        generated_text = self.tokenizer.decode(generated_tokens)
        extracted_number = self._extract_number(generated_text)
        is_correct = self._check_correctness(extracted_number, expected_answer)

        # Find answer tokens
        answer_positions = self._find_answer_tokens(generated_tokens, extracted_number, prompt_length)
        answer_found = len(answer_positions) > 0

        # Get metrics at first answer token
        velocity_at_answer = None
        direction_at_answer = None
        layer_velocities_at_answer = None

        if answer_found and answer_positions:
            first_answer_idx = answer_positions[0] - prompt_length  # Convert to index in token_measurements
            if 0 <= first_answer_idx < len(token_measurements):
                tm = token_measurements[first_answer_idx]
                velocity_at_answer = sum(tm.layer_velocities) / max(len(tm.layer_velocities), 1)
                direction_at_answer = sum(tm.layer_direction_changes) / max(len(tm.layer_direction_changes), 1)
                layer_velocities_at_answer = tm.layer_velocities

        return GenerationResult(
            prompt=prompt,
            expected_answer=expected_answer,
            generated_text=generated_text,
            extracted_number=extracted_number,
            is_correct=is_correct,
            answer_token_positions=answer_positions,
            answer_found=answer_found,
            token_measurements=token_measurements,
            velocity_at_answer=velocity_at_answer,
            direction_change_at_answer=direction_at_answer,
            layer_velocities_at_answer=layer_velocities_at_answer,
            n_generated_tokens=len(generated_tokens),
            prompt_length=prompt_length,
        )

    def run(self) -> None:
        """Run experiment."""
        from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

        logger.info("Starting geometry validation V3 (rigorous)")
        self.setup()

        loader = BenchmarkLoader()
        benchmark = loader.load("arithmetic", split="test", limit=self.config.n_samples)

        results: list[GenerationResult] = []

        for i, sample in enumerate(benchmark.samples):
            try:
                result = self._generate_with_geometry(sample.prompt, sample.answer)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed on sample {i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                n_correct = sum(1 for r in results if r.is_correct)
                n_found = sum(1 for r in results if r.answer_found)
                logger.info(f"Progress: {i+1}/{len(benchmark.samples)}, "
                           f"correct: {n_correct}, answer_found: {n_found}")

        self._save_and_analyze(results)

    def _save_and_analyze(self, results: list[GenerationResult]) -> None:
        """Save results and compute statistics."""

        # Save raw results
        output_path = self.config.output_dir / "results.jsonl"
        with open(output_path, "w") as f:
            for r in results:
                record = {
                    "prompt": r.prompt,
                    "expected": r.expected_answer,
                    "generated": r.generated_text,
                    "extracted_number": r.extracted_number,
                    "is_correct": r.is_correct,
                    "answer_found": r.answer_found,
                    "answer_positions": r.answer_token_positions,
                    "velocity_at_answer": r.velocity_at_answer,
                    "direction_at_answer": r.direction_change_at_answer,
                    "layer_velocities_at_answer": r.layer_velocities_at_answer,
                    "n_tokens": r.n_generated_tokens,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {len(results)} results to {output_path}")

        # Analysis
        print("\n" + "=" * 70)
        print("GEOMETRY V3: RIGOROUS ANALYSIS")
        print("=" * 70)

        n_total = len(results)
        n_correct = sum(1 for r in results if r.is_correct)
        n_incorrect = n_total - n_correct
        n_answer_found = sum(1 for r in results if r.answer_found)
        n_answer_not_found = n_total - n_answer_found

        print(f"Total samples: {n_total}")
        print(f"Correct: {n_correct}, Incorrect: {n_incorrect}")
        print(f"Answer token found: {n_answer_found}, Not found: {n_answer_not_found}")
        print()

        # Separate analysis by answer_found status
        print("### SAMPLES WITH ANSWER TOKEN FOUND ###")

        correct_with_answer = [r for r in results if r.is_correct and r.answer_found]
        incorrect_with_answer = [r for r in results if not r.is_correct and r.answer_found]

        print(f"Correct + answer found: {len(correct_with_answer)}")
        print(f"Incorrect + answer found: {len(incorrect_with_answer)}")

        # Velocity analysis (no filtering - include all with answer found)
        c_vel = [r.velocity_at_answer for r in correct_with_answer if r.velocity_at_answer is not None]
        i_vel = [r.velocity_at_answer for r in incorrect_with_answer if r.velocity_at_answer is not None]

        print(f"\nVelocity at answer token:")
        print(f"  Correct samples with velocity: {len(c_vel)}")
        print(f"  Incorrect samples with velocity: {len(i_vel)}")

        if c_vel and i_vel:
            c_mean = sum(c_vel) / len(c_vel)
            i_mean = sum(i_vel) / len(i_vel)

            # Raw difference (no arbitrary interpretation)
            diff = c_mean - i_mean

            # Effect size with CI via bootstrap
            d, ci_low, ci_high = self._bootstrap_effect_size(c_vel, i_vel, n_bootstrap=1000)

            print(f"  Correct mean:   {c_mean:.4f} (n={len(c_vel)})")
            print(f"  Incorrect mean: {i_mean:.4f} (n={len(i_vel)})")
            print(f"  Difference:     {diff:+.4f}")
            print(f"  Effect size d:  {d:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

        # Direction change analysis
        c_dir = [r.direction_change_at_answer for r in correct_with_answer if r.direction_change_at_answer is not None]
        i_dir = [r.direction_change_at_answer for r in incorrect_with_answer if r.direction_change_at_answer is not None]

        print(f"\nDirection change at answer token:")
        if c_dir and i_dir:
            c_mean = sum(c_dir) / len(c_dir)
            i_mean = sum(i_dir) / len(i_dir)
            diff = c_mean - i_mean
            d, ci_low, ci_high = self._bootstrap_effect_size(c_dir, i_dir, n_bootstrap=1000)

            print(f"  Correct mean:   {c_mean:.4f} (n={len(c_dir)})")
            print(f"  Incorrect mean: {i_mean:.4f} (n={len(i_dir)})")
            print(f"  Difference:     {diff:+.4f}")
            print(f"  Effect size d:  {d:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

        # Per-layer analysis
        print(f"\nPer-layer velocity at answer token:")
        for layer_idx in range(self.num_layers):
            c_layer = [r.layer_velocities_at_answer[layer_idx]
                      for r in correct_with_answer
                      if r.layer_velocities_at_answer and layer_idx < len(r.layer_velocities_at_answer)]
            i_layer = [r.layer_velocities_at_answer[layer_idx]
                      for r in incorrect_with_answer
                      if r.layer_velocities_at_answer and layer_idx < len(r.layer_velocities_at_answer)]

            if c_layer and i_layer:
                c_mean = sum(c_layer) / len(c_layer)
                i_mean = sum(i_layer) / len(i_layer)
                d, _, _ = self._bootstrap_effect_size(c_layer, i_layer, n_bootstrap=100)
                print(f"  Layer {layer_idx:2d}: correct={c_mean:.4f}, incorrect={i_mean:.4f}, d={d:+.3f}")

        # Report on samples WITHOUT answer found (transparency)
        print(f"\n### SAMPLES WITHOUT ANSWER TOKEN FOUND ###")
        no_answer_correct = sum(1 for r in results if r.is_correct and not r.answer_found)
        no_answer_incorrect = sum(1 for r in results if not r.is_correct and not r.answer_found)
        print(f"Correct but no answer token: {no_answer_correct}")
        print(f"Incorrect and no answer token: {no_answer_incorrect}")
        print("(These are excluded from velocity analysis - potential bias if asymmetric)")

        print("=" * 70)

    def _bootstrap_effect_size(
        self,
        a: list[float],
        b: list[float],
        n_bootstrap: int = 1000,
    ) -> tuple[float, float, float]:
        """Compute Cohen's d with bootstrap confidence interval."""
        if len(a) < 2 or len(b) < 2:
            return 0.0, 0.0, 0.0

        def cohens_d(x: list[float], y: list[float]) -> float:
            nx, ny = len(x), len(y)
            mx, my = sum(x) / nx, sum(y) / ny
            vx = sum((xi - mx) ** 2 for xi in x) / (nx - 1) if nx > 1 else 0
            vy = sum((yi - my) ** 2 for yi in y) / (ny - 1) if ny > 1 else 0
            pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
            if pooled <= 0:
                return 0.0
            return (mx - my) / (pooled ** 0.5)

        # Point estimate
        d = cohens_d(a, b)

        # Bootstrap
        ds = []
        for _ in range(n_bootstrap):
            a_boot = random.choices(a, k=len(a))
            b_boot = random.choices(b, k=len(b))
            ds.append(cohens_d(a_boot, b_boot))

        ds.sort()
        ci_low = ds[int(0.025 * n_bootstrap)]
        ci_high = ds[int(0.975 * n_bootstrap)]

        return d, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Geometry validation V3")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="results/geometry_v3/", help="Output directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = ExperimentConfig(
        model_path=args.model,
        output_dir=Path(args.output),
        n_samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    experiment = GeometryValidationV3(config)
    experiment.run()


if __name__ == "__main__":
    main()
