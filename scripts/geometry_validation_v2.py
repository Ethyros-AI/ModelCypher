#!/usr/bin/env python3
"""Geometry Validation V2: Token-level analysis at decision points.

This addresses the failures of V1:
1. Measures geometry at ANSWER TOKEN, not averaged over input
2. Uses contrastive pairs (same prompt, correct vs incorrect samples)
3. Tracks trajectory through generation, not just aggregates
4. Targets 1000+ samples for adequate power

Key insight: The difference between correct and incorrect reasoning
should appear WHERE THE MODEL COMMITS TO AN ANSWER, not in how it
processes the question.

Usage:
    poetry run python scripts/geometry_validation_v2.py \
        --model /path/to/model \
        --output results/geometry_v2/ \
        --samples 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TokenGeometry:
    """Geometry at a specific token position."""
    position: int
    token_id: int
    token_str: str

    # Per-layer intrinsic dimension at this position
    layer_ids: list[float]

    # Per-layer spectral entropy at this position
    layer_spectral_entropy: list[float]

    # Hidden state norm at this position (per layer)
    layer_norms: list[float]

    # Velocity from previous token (per layer) - measures "jump size"
    layer_velocity_norms: list[float]

    # Cosine similarity to previous token (per layer) - measures "direction change"
    layer_direction_change: list[float]


@dataclass
class GenerationTrajectory:
    """Full trajectory through generation."""
    prompt: str
    generated_text: str
    full_text: str

    # Per-token geometry
    token_geometries: list[TokenGeometry]

    # Answer-specific metrics (at first answer token)
    answer_token_idx: int
    answer_token_geometry: TokenGeometry | None

    # Trajectory metrics
    total_velocity: float  # Sum of all velocity norms
    mean_velocity: float
    max_velocity: float
    velocity_at_answer: float

    # Did geometry "fork" - sudden direction change at answer?
    direction_change_at_answer: float
    max_direction_change: float

    # Correctness
    expected_answer: str
    is_correct: bool


@dataclass
class ContrastivePair:
    """Same prompt, different outcomes."""
    prompt: str
    expected_answer: str

    correct_trajectory: GenerationTrajectory | None
    incorrect_trajectory: GenerationTrajectory | None

    # Divergence analysis
    divergence_token_idx: int | None  # Where trajectories fork
    divergence_layer: int | None  # Which layer shows divergence first

    # Geometric differences at answer token
    id_diff_at_answer: list[float] | None  # Per-layer ID difference
    velocity_diff_at_answer: list[float] | None
    direction_diff_at_answer: list[float] | None


@dataclass
class ExperimentConfig:
    model_path: str
    output_dir: Path
    target_samples: int = 1000
    samples_per_prompt: int = 1  # Greedy: one deterministic path per prompt
    max_tokens: int = 64
    seed: int = 42


class GeometryValidationV2:
    """Token-level geometry analysis at decision points."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0

    def setup(self) -> None:
        from modelcypher.backends import initialize_default_backend

        logger.info(f"Loading model from {self.config.model_path}")
        self.backend = initialize_default_backend()

        model_path = Path(self.config.model_path)
        self.model, self.tokenizer = self.backend.load_model(str(model_path))

        # Get number of layers
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        self.num_layers = len(layers) if layers else 0

        logger.info(f"Model loaded: {self.num_layers} layers")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _capture_generation_trajectory(
        self,
        prompt: str,
        expected_answer: str,
    ) -> GenerationTrajectory:
        """Generate with full hidden state capture at each token."""
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
        from modelcypher.core.domain.geometry.effective_rank import EffectiveRank

        b = self.backend
        projector = LayerEntropyProjector(b)
        id_estimator = IntrinsicDimension(b)
        rank_estimator = EffectiveRank(b)

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)

        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        if isinstance(prompt_tokens, list):
            input_ids = b.array([prompt_tokens])
        else:
            input_ids = prompt_tokens
            if input_ids.ndim == 1:
                input_ids = b.reshape(input_ids, (1, -1))

        prompt_len = int(input_ids.shape[1])

        # Storage for trajectory
        token_geometries: list[TokenGeometry] = []
        all_hidden_states: list[dict[int, Any]] = []  # Per-token, per-layer hidden states
        generated_tokens: list[int] = []

        # Previous hidden states for velocity computation
        prev_hidden: dict[int, Any] = {}

        # Generate token by token with hidden state capture
        current_ids = input_ids

        for gen_step in range(self.config.max_tokens):
            # Capture hidden states at all layers
            target_layers = set(range(self.num_layers))
            captured = projector._capture_layer_states(
                base_model, layers, current_ids, target_layers
            )

            # Get the hidden state at the LAST position (the one we're predicting from)
            last_pos_hidden: dict[int, Any] = {}
            for layer_idx, hidden in captured.items():
                if hidden.ndim == 3:
                    last_pos_hidden[layer_idx] = hidden[0, -1, :]  # [hidden_dim]
                else:
                    last_pos_hidden[layer_idx] = hidden[-1, :]

            all_hidden_states.append(last_pos_hidden)

            # Compute per-layer metrics at this position
            layer_ids = []
            layer_spectral = []
            layer_norms = []
            layer_velocities = []
            layer_directions = []

            for layer_idx in range(self.num_layers):
                h = last_pos_hidden[layer_idx]
                b.eval(h)

                # Norm
                norm = float(b.norm(h))
                layer_norms.append(norm)

                # Velocity and direction change from previous
                if layer_idx in prev_hidden:
                    prev_h = prev_hidden[layer_idx]
                    diff = h - prev_h
                    velocity = float(b.norm(diff))

                    # Cosine similarity for direction
                    dot = float(b.sum(h * prev_h))
                    norm_prod = norm * float(b.norm(prev_h))
                    if norm_prod > 1e-8:
                        cos_sim = dot / norm_prod
                        direction_change = 1.0 - cos_sim  # 0 = same direction, 2 = opposite
                    else:
                        direction_change = 0.0

                    layer_velocities.append(velocity)
                    layer_directions.append(direction_change)
                else:
                    layer_velocities.append(0.0)
                    layer_directions.append(0.0)

                # ID requires multiple points - use sliding window of recent tokens
                # For single token, use the hidden dim as proxy (log of effective dimensions)
                # This is a simplification - proper ID needs multiple samples
                layer_ids.append(0.0)  # Placeholder - will compute differently

                # Spectral entropy of the hidden state (treat as 1-sample, get variance structure)
                # For single vector, use normalized squared components
                h_sq = h * h
                h_sum = float(b.sum(h_sq))
                if h_sum > 1e-8:
                    p = h_sq / h_sum
                    # Entropy of squared activation distribution
                    log_p = b.log(b.maximum(p, b.full(p.shape, 1e-10)))
                    entropy = -float(b.sum(p * log_p))
                else:
                    entropy = 0.0
                layer_spectral.append(entropy)

            # Get next token via forward pass
            # Note: MLX doesn't need no_grad - it doesn't track gradients by default
            outputs = base_model(current_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # Get logits for last position
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]

            # Greedy decoding — deterministic geometric path
            b.eval(last_logits)
            next_token = int(b.argmax(last_logits))

            generated_tokens.append(next_token)

            # Decode token for storage
            try:
                token_str = self.tokenizer.decode([next_token])
            except:
                token_str = f"<{next_token}>"

            # Store geometry
            token_geometries.append(TokenGeometry(
                position=prompt_len + gen_step,
                token_id=next_token,
                token_str=token_str,
                layer_ids=layer_ids,
                layer_spectral_entropy=layer_spectral,
                layer_norms=layer_norms,
                layer_velocity_norms=layer_velocities,
                layer_direction_change=layer_directions,
            ))

            # Update previous hidden states
            prev_hidden = last_pos_hidden

            # Update input for next step
            next_token_arr = b.array([[next_token]])
            current_ids = b.concatenate([current_ids, next_token_arr], axis=1)

            # Check for EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_token == eos_id:
                break

        # Decode full generation
        generated_text = self.tokenizer.decode(generated_tokens)
        full_text = prompt + generated_text

        # Check correctness
        is_correct = expected_answer.lower() in generated_text.lower()

        # Find answer token (first token that's part of the answer)
        answer_token_idx = -1
        for i, tg in enumerate(token_geometries):
            if expected_answer.lower() in tg.token_str.lower():
                answer_token_idx = i
                break

        # If we can't find exact answer, use first numeric token after prompt
        if answer_token_idx == -1:
            for i, tg in enumerate(token_geometries):
                if any(c.isdigit() for c in tg.token_str):
                    answer_token_idx = i
                    break

        # Compute trajectory-level metrics
        all_velocities = [sum(tg.layer_velocity_norms) / max(len(tg.layer_velocity_norms), 1)
                         for tg in token_geometries]
        all_directions = [sum(tg.layer_direction_change) / max(len(tg.layer_direction_change), 1)
                         for tg in token_geometries]

        total_velocity = sum(all_velocities)
        mean_velocity = total_velocity / max(len(all_velocities), 1)
        max_velocity = max(all_velocities) if all_velocities else 0.0

        velocity_at_answer = all_velocities[answer_token_idx] if 0 <= answer_token_idx < len(all_velocities) else 0.0
        direction_at_answer = all_directions[answer_token_idx] if 0 <= answer_token_idx < len(all_directions) else 0.0
        max_direction = max(all_directions) if all_directions else 0.0

        answer_geometry = token_geometries[answer_token_idx] if 0 <= answer_token_idx < len(token_geometries) else None

        return GenerationTrajectory(
            prompt=prompt,
            generated_text=generated_text,
            full_text=full_text,
            token_geometries=token_geometries,
            answer_token_idx=answer_token_idx,
            answer_token_geometry=answer_geometry,
            total_velocity=total_velocity,
            mean_velocity=mean_velocity,
            max_velocity=max_velocity,
            velocity_at_answer=velocity_at_answer,
            direction_change_at_answer=direction_at_answer,
            max_direction_change=max_direction,
            expected_answer=expected_answer,
            is_correct=is_correct,
        )

    def _create_contrastive_pair(
        self,
        prompt: str,
        expected_answer: str,
    ) -> ContrastivePair | None:
        """Sample multiple times from same prompt to get correct/incorrect pair."""

        correct_traj = None
        incorrect_traj = None

        for attempt in range(self.config.samples_per_prompt):
            try:
                traj = self._capture_generation_trajectory(
                    prompt, expected_answer,
                )

                if traj.is_correct and correct_traj is None:
                    correct_traj = traj
                elif not traj.is_correct and incorrect_traj is None:
                    incorrect_traj = traj

                # Got both, we're done
                if correct_traj is not None and incorrect_traj is not None:
                    break

            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                continue

        # Need at least one trajectory
        if correct_traj is None and incorrect_traj is None:
            return None

        # Compute divergence if we have both
        divergence_idx = None
        divergence_layer = None
        id_diff = None
        velocity_diff = None
        direction_diff = None

        if correct_traj is not None and incorrect_traj is not None:
            # Find where trajectories diverge
            min_len = min(len(correct_traj.token_geometries),
                         len(incorrect_traj.token_geometries))

            for i in range(min_len):
                c_geom = correct_traj.token_geometries[i]
                i_geom = incorrect_traj.token_geometries[i]

                # Check each layer for significant divergence
                for layer_idx in range(self.num_layers):
                    c_dir = c_geom.layer_direction_change[layer_idx]
                    i_dir = i_geom.layer_direction_change[layer_idx]

                    # Significant divergence = direction change difference > 0.1
                    if abs(c_dir - i_dir) > 0.1:
                        divergence_idx = i
                        divergence_layer = layer_idx
                        break

                if divergence_idx is not None:
                    break

            # Compute differences at answer token
            c_ans = correct_traj.answer_token_geometry
            i_ans = incorrect_traj.answer_token_geometry

            if c_ans is not None and i_ans is not None:
                id_diff = [c - i for c, i in zip(c_ans.layer_ids, i_ans.layer_ids)]
                velocity_diff = [c - i for c, i in zip(c_ans.layer_velocity_norms, i_ans.layer_velocity_norms)]
                direction_diff = [c - i for c, i in zip(c_ans.layer_direction_change, i_ans.layer_direction_change)]

        return ContrastivePair(
            prompt=prompt,
            expected_answer=expected_answer,
            correct_trajectory=correct_traj,
            incorrect_trajectory=incorrect_traj,
            divergence_token_idx=divergence_idx,
            divergence_layer=divergence_layer,
            id_diff_at_answer=id_diff,
            velocity_diff_at_answer=velocity_diff,
            direction_diff_at_answer=direction_diff,
        )

    def run(self) -> None:
        """Run the full experiment."""
        from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

        logger.info("Starting geometry validation V2")
        self.setup()

        # Load arithmetic problems (simple, clear answer)
        loader = BenchmarkLoader()

        # Generate enough prompts to hit target samples
        prompts_needed = self.config.target_samples
        benchmark = loader.load("arithmetic", split="test", limit=prompts_needed)

        logger.info(f"Loaded {len(benchmark.samples)} prompts, targeting {self.config.target_samples} contrastive pairs")

        # Collect trajectories
        all_trajectories: list[GenerationTrajectory] = []
        contrastive_pairs: list[ContrastivePair] = []

        for i, sample in enumerate(benchmark.samples):
            if len(all_trajectories) >= self.config.target_samples:
                break

            try:
                pair = self._create_contrastive_pair(sample.prompt, sample.answer)

                if pair is not None:
                    contrastive_pairs.append(pair)

                    if pair.correct_trajectory:
                        all_trajectories.append(pair.correct_trajectory)
                    if pair.incorrect_trajectory:
                        all_trajectories.append(pair.incorrect_trajectory)

            except Exception as e:
                logger.warning(f"Failed on sample {i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                n_correct = sum(1 for t in all_trajectories if t.is_correct)
                n_incorrect = len(all_trajectories) - n_correct
                logger.info(f"Progress: {i+1}/{len(benchmark.samples)}, "
                           f"trajectories: {len(all_trajectories)} "
                           f"({n_correct} correct, {n_incorrect} incorrect)")

        # Save results
        self._save_results(all_trajectories, contrastive_pairs)
        self._analyze_results(all_trajectories, contrastive_pairs)

    def _save_results(
        self,
        trajectories: list[GenerationTrajectory],
        pairs: list[ContrastivePair],
    ) -> None:
        """Save raw results."""
        # Save trajectories
        traj_path = self.config.output_dir / "trajectories.jsonl"
        with open(traj_path, "w") as f:
            for traj in trajectories:
                record = {
                    "prompt": traj.prompt,
                    "generated": traj.generated_text,
                    "expected": traj.expected_answer,
                    "is_correct": traj.is_correct,
                    "answer_token_idx": traj.answer_token_idx,
                    "total_velocity": traj.total_velocity,
                    "mean_velocity": traj.mean_velocity,
                    "max_velocity": traj.max_velocity,
                    "velocity_at_answer": traj.velocity_at_answer,
                    "direction_change_at_answer": traj.direction_change_at_answer,
                    "max_direction_change": traj.max_direction_change,
                    "n_tokens": len(traj.token_geometries),
                }
                if traj.answer_token_geometry:
                    record["answer_layer_norms"] = traj.answer_token_geometry.layer_norms
                    record["answer_layer_velocities"] = traj.answer_token_geometry.layer_velocity_norms
                    record["answer_layer_directions"] = traj.answer_token_geometry.layer_direction_change
                    record["answer_layer_spectral"] = traj.answer_token_geometry.layer_spectral_entropy

                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {len(trajectories)} trajectories to {traj_path}")

        # Save contrastive pairs
        pairs_path = self.config.output_dir / "contrastive_pairs.jsonl"
        with open(pairs_path, "w") as f:
            for pair in pairs:
                record = {
                    "prompt": pair.prompt,
                    "expected": pair.expected_answer,
                    "has_correct": pair.correct_trajectory is not None,
                    "has_incorrect": pair.incorrect_trajectory is not None,
                    "divergence_token_idx": pair.divergence_token_idx,
                    "divergence_layer": pair.divergence_layer,
                    "velocity_diff_at_answer": pair.velocity_diff_at_answer,
                    "direction_diff_at_answer": pair.direction_diff_at_answer,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {len(pairs)} contrastive pairs to {pairs_path}")

    def _analyze_results(
        self,
        trajectories: list[GenerationTrajectory],
        pairs: list[ContrastivePair],
    ) -> None:
        """Analyze for geometric signal at decision points."""

        correct = [t for t in trajectories if t.is_correct]
        incorrect = [t for t in trajectories if not t.is_correct]

        print("\n" + "=" * 70)
        print("GEOMETRY V2: TOKEN-LEVEL ANALYSIS AT DECISION POINTS")
        print("=" * 70)
        print(f"Total trajectories: {len(trajectories)}")
        print(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}")
        print()

        # 1. Velocity at answer token
        print("### VELOCITY AT ANSWER TOKEN ###")
        c_vel = [t.velocity_at_answer for t in correct if t.velocity_at_answer > 0]
        i_vel = [t.velocity_at_answer for t in incorrect if t.velocity_at_answer > 0]

        if c_vel and i_vel:
            c_mean = sum(c_vel) / len(c_vel)
            i_mean = sum(i_vel) / len(i_vel)

            # Effect size
            pooled_std = self._pooled_std(c_vel, i_vel)
            d = (c_mean - i_mean) / pooled_std if pooled_std > 0 else 0

            print(f"Correct mean:   {c_mean:.4f}")
            print(f"Incorrect mean: {i_mean:.4f}")
            print(f"Effect size d:  {d:.3f}")
            print()

        # 2. Direction change at answer token
        print("### DIRECTION CHANGE AT ANSWER TOKEN ###")
        c_dir = [t.direction_change_at_answer for t in correct if t.direction_change_at_answer > 0]
        i_dir = [t.direction_change_at_answer for t in incorrect if t.direction_change_at_answer > 0]

        if c_dir and i_dir:
            c_mean = sum(c_dir) / len(c_dir)
            i_mean = sum(i_dir) / len(i_dir)

            pooled_std = self._pooled_std(c_dir, i_dir)
            d = (c_mean - i_mean) / pooled_std if pooled_std > 0 else 0

            print(f"Correct mean:   {c_mean:.4f}")
            print(f"Incorrect mean: {i_mean:.4f}")
            print(f"Effect size d:  {d:.3f}")
            print()

        # 3. Per-layer analysis at answer token
        print("### PER-LAYER VELOCITY AT ANSWER TOKEN ###")
        for layer_idx in range(self.num_layers):
            c_layer_vel = []
            i_layer_vel = []

            for t in correct:
                if t.answer_token_geometry and layer_idx < len(t.answer_token_geometry.layer_velocity_norms):
                    c_layer_vel.append(t.answer_token_geometry.layer_velocity_norms[layer_idx])

            for t in incorrect:
                if t.answer_token_geometry and layer_idx < len(t.answer_token_geometry.layer_velocity_norms):
                    i_layer_vel.append(t.answer_token_geometry.layer_velocity_norms[layer_idx])

            if c_layer_vel and i_layer_vel:
                c_mean = sum(c_layer_vel) / len(c_layer_vel)
                i_mean = sum(i_layer_vel) / len(i_layer_vel)
                pooled_std = self._pooled_std(c_layer_vel, i_layer_vel)
                d = (c_mean - i_mean) / pooled_std if pooled_std > 0 else 0

                if abs(d) > 0.3:  # Only show layers with signal
                    print(f"Layer {layer_idx:2d}: d={d:+.3f} (correct={c_mean:.4f}, incorrect={i_mean:.4f})")

        print()

        # 4. Contrastive pair analysis
        print("### CONTRASTIVE PAIR ANALYSIS ###")
        complete_pairs = [p for p in pairs if p.correct_trajectory and p.incorrect_trajectory]
        print(f"Complete pairs (both correct and incorrect from same prompt): {len(complete_pairs)}")

        if complete_pairs:
            # Where do trajectories diverge?
            divergence_positions = [p.divergence_token_idx for p in complete_pairs if p.divergence_token_idx is not None]
            divergence_layers = [p.divergence_layer for p in complete_pairs if p.divergence_layer is not None]

            if divergence_positions:
                print(f"Mean divergence position: {sum(divergence_positions)/len(divergence_positions):.1f} tokens after prompt")

            if divergence_layers:
                from collections import Counter
                layer_counts = Counter(divergence_layers)
                print(f"Divergence by layer: {dict(layer_counts)}")

        print("=" * 70)

    def _pooled_std(self, a: list[float], b: list[float]) -> float:
        """Compute pooled standard deviation."""
        if len(a) < 2 or len(b) < 2:
            return 1.0

        a_mean = sum(a) / len(a)
        b_mean = sum(b) / len(b)

        a_var = sum((x - a_mean) ** 2 for x in a) / (len(a) - 1)
        b_var = sum((x - b_mean) ** 2 for x in b) / (len(b) - 1)

        pooled_var = ((len(a) - 1) * a_var + (len(b) - 1) * b_var) / (len(a) + len(b) - 2)
        return pooled_var ** 0.5


def main():
    parser = argparse.ArgumentParser(description="Geometry validation V2 - token-level analysis")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="results/geometry_v2/", help="Output directory")
    parser.add_argument("--samples", type=int, default=100, help="Target number of samples")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per generation")

    args = parser.parse_args()

    config = ExperimentConfig(
        model_path=args.model,
        output_dir=Path(args.output),
        target_samples=args.samples,
        max_tokens=args.max_tokens,
    )

    experiment = GeometryValidationV2(config)
    experiment.run()


if __name__ == "__main__":
    main()
