# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Self-Reflection Training: Teach models to clarify questions before answering.

Research Basis:
    Question normalization improves φ alignment by 73% (experimental_summary.md).
    Models that self-reflect ("Let me understand the question...") achieve
    100% accuracy on problems that trip up intuitive processing.

Training Data Format:
    Input: "Question: [original question]"
    Output: "Let me understand the question. [core question]\\n\\n[reasoning]\\n\\nAnswer: [answer]"

Philosophy:
    Self-reflection IS geometric alignment. When the model extracts the core
    question (~14 tokens), it naturally processes at φ resonance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import mlx.core as mx

logger = logging.getLogger(__name__)

PHI = 1.618033988749895


@dataclass
class SelfReflectionExample:
    """A single self-reflection training example."""

    input_question: str
    core_question: str
    reasoning: str
    answer: str

    @property
    def full_output(self) -> str:
        """Format as self-reflection output."""
        return f"Let me understand the question. {self.core_question}\n\n{self.reasoning}\n\nAnswer: {self.answer}"

    @property
    def full_text(self) -> str:
        """Full training text: input + output."""
        return f"Question: {self.input_question}\n\n{self.full_output}"


def _apply_lora_to_layers(model, layer_indices, config, use_dora=False):
    """Apply LoRA to specific layer indices."""
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.utils import (
        DoRAEmbedding,
        DoRALinear,
        LoRAEmbedding,
        LoRALinear,
        LoRASwitchLinear,
        QuantizedSwitchLinear,
        SwitchLinear,
    )

    def to_lora(layer):
        if not use_dora and hasattr(layer, "to_lora"):
            return layer.to_lora(
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            )

        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_base(
            layer,
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )

    keys = set()

    def get_keys_for_lora(p, m):
        types = (
            nn.Linear,
            nn.QuantizedLinear,
            SwitchLinear,
            QuantizedSwitchLinear,
            nn.Embedding,
            nn.QuantizedEmbedding,
        )
        if hasattr(m, "to_lora") or isinstance(m, types):
            keys.add(p)

    for l in model.layers:
        l.apply_to_modules(get_keys_for_lora)

    for idx in layer_indices:
        l = model.layers[idx]
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        if lora_layers:
            l.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))


def get_self_reflection_examples(
    include_gsm8k: bool = True,
    include_phase_b: bool = True,
    include_phase_c: bool = True,
) -> list[SelfReflectionExample]:
    """Core self-reflection training examples.

    v2 additions:
    - Valid syllogisms (modus ponens) to balance fallacy detection
    - Factual corrections for identified gaps

    v3 additions (Phase A - Project Polymath):
    - 50 GSM8K-style multi-step math patterns

    These cover:
    - Bat-and-ball type (intuitive traps)
    - Rate problems (machines/widgets)
    - Exponential reasoning (lily pad)
    - Relationship tracking (Tom/Jane)
    - Logic fallacies (some vs all) - INVALID, answer No
    - Valid syllogisms (all X are Y) - VALID, answer Yes
    - Trick questions (subtract from 25)
    - Factual corrections (odd numbers, photosynthesis, etc.)
    - [v3] Multi-step arithmetic chains
    - [v3] Percentage calculations
    - [v3] Rate/ratio problems
    - [v3] Distribution/sharing problems

    Args:
        include_gsm8k: Whether to include GSM8K pattern examples (default: True)
        include_phase_b: Whether to include Phase B pattern examples (default: True)
        include_phase_c: Whether to include Phase C pattern examples (default: True)
    """
    examples = [
        # Intuitive traps
        SelfReflectionExample(
            input_question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            core_question="If bat + ball = $1.10 and bat = ball + $1, what is ball?",
            reasoning="Let ball = x. Then bat = x + 1.\nx + (x + 1) = 1.10\n2x = 0.10\nx = 0.05",
            answer="$0.05",
        ),
        SelfReflectionExample(
            input_question="5 machines take 5 minutes to make 5 widgets. How long would it take 100 machines to make 100 widgets?",
            core_question="Rate: 5 machines → 5 widgets in 5 min. Time for 100 machines → 100 widgets?",
            reasoning="5 machines make 5 widgets in 5 min = 1 machine makes 1 widget in 5 min.\n100 machines each make 1 widget in 5 min = 100 widgets in 5 min.",
            answer="5 minutes",
        ),
        SelfReflectionExample(
            input_question="A lily pad doubles in size every day. It takes 48 days to cover a lake. How many days does it take to cover half the lake?",
            core_question="If doubling daily covers lake at day 48, when is half covered?",
            reasoning="Day 48: full lake.\nDay 47: half lake (since it doubles to full the next day).",
            answer="47 days",
        ),
        # Relationship tracking
        SelfReflectionExample(
            input_question="Tom has 3 times as many apples as Jane. Jane has 5 apples. How many apples does Tom have?",
            core_question="Tom = 3 × Jane, Jane = 5. What is Tom?",
            reasoning="Tom = 3 × 5 = 15",
            answer="15 apples",
        ),
        SelfReflectionExample(
            input_question="A train travels 60 km/h for 2 hours, then 80 km/h for 1.5 hours. What is the total distance?",
            core_question="Distance = 60×2 + 80×1.5?",
            reasoning="Leg 1: 60 × 2 = 120 km\nLeg 2: 80 × 1.5 = 120 km\nTotal: 240 km",
            answer="240 km",
        ),
        # Logic fallacies (INVALID - answer No)
        SelfReflectionExample(
            input_question="Some fruits are red. Apples are fruits. Are all apples red?",
            core_question="Does 'SOME fruits are red' + 'apples are fruits' imply ALL apples are red?",
            reasoning="'Some' ≠ 'All'. The premise only says SOME fruits are red.\nApples being fruits doesn't mean they share the 'red' property.\nCounterexample: Green apples exist.",
            answer="No",
        ),
        # Valid syllogisms (VALID - answer Yes) - v2 additions
        SelfReflectionExample(
            input_question="All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
            core_question="Does 'ALL X are Y' + 'Z is X' imply 'Z is Y'?",
            reasoning="This is VALID modus ponens:\n1. All mammals are warm-blooded (universal)\n2. Dogs are mammals (specific)\n3. Therefore, dogs ARE warm-blooded",
            answer="Yes",
        ),
        SelfReflectionExample(
            input_question="All squares have four sides. This shape is a square. Does it have four sides?",
            core_question="Does 'ALL X have Y' + 'This is X' imply 'This has Y'?",
            reasoning="Valid deduction:\nAll squares have four sides.\nThis is a square.\nTherefore, it has four sides.",
            answer="Yes",
        ),
        SelfReflectionExample(
            input_question="If it rains, the ground gets wet. It rained today. Is the ground wet?",
            core_question="Does 'If P then Q' + 'P occurred' imply 'Q'?",
            reasoning="Modus ponens (valid):\nIf rain → wet ground.\nIt rained.\nTherefore, ground is wet.",
            answer="Yes",
        ),
        # Trick questions
        SelfReflectionExample(
            input_question="How many times can you subtract 5 from 25?",
            core_question="How many times can 5 be subtracted FROM 25 specifically?",
            reasoning="You can subtract 5 from 25 exactly ONCE.\nAfter that, you're subtracting from 20, then 15, etc.\nThe question asks about subtracting from 25, not repeatedly.",
            answer="1",
        ),
        SelfReflectionExample(
            input_question="A farmer has 17 sheep. All but 9 die. How many are left?",
            core_question="'All but 9 die' means how many survive?",
            reasoning="'All but 9 die' = 9 survive.\nThis is a parsing trick: it means 9 remain alive.",
            answer="9",
        ),
        # Arithmetic (to not lose capability)
        SelfReflectionExample(
            input_question="What is 12 × 11?",
            core_question="12 × 11 = ?",
            reasoning="12 × 11 = 12 × (10 + 1) = 120 + 12 = 132",
            answer="132",
        ),
        SelfReflectionExample(
            input_question="What is 144 ÷ 12?",
            core_question="144 ÷ 12 = ?",
            reasoning="144 ÷ 12 = 12 (since 12 × 12 = 144)",
            answer="12",
        ),
        # Rate problems
        SelfReflectionExample(
            input_question="If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?",
            core_question="Rate of cats catching mice, scaled to 100/100?",
            reasoning="3 cats catch 3 mice in 3 min = 1 cat catches 1 mouse in 3 min.\nIn 100 min, 1 cat catches ~33 mice.\n3 cats catch ~100 mice in 100 min.",
            answer="3",
        ),
        # Factual corrections - v2 additions
        SelfReflectionExample(
            input_question="Is 7 an odd number?",
            core_question="Is 7 divisible by 2?",
            reasoning="7 ÷ 2 = 3.5 (not whole).\nOdd numbers are NOT evenly divisible by 2.\n7 is not divisible by 2, so 7 is odd.",
            answer="Yes",
        ),
        SelfReflectionExample(
            input_question="What gas do plants produce during photosynthesis?",
            core_question="Photosynthesis output gas?",
            reasoning="6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂\nPlants take in carbon dioxide and release oxygen.",
            answer="Oxygen",
        ),
        SelfReflectionExample(
            input_question="How many chambers does a human heart have?",
            core_question="Human heart chamber count?",
            reasoning="Human heart structure:\n- 2 atria (upper chambers)\n- 2 ventricles (lower chambers)\nTotal: 4 chambers.",
            answer="4",
        ),
        SelfReflectionExample(
            input_question="What color do you get when you mix red and blue paint?",
            core_question="Red + blue in subtractive color mixing?",
            reasoning="Paint uses subtractive color mixing.\nRed + Blue = Purple/Violet.\n(Not green - that's additive/light mixing)",
            answer="Purple",
        ),
        # Simple facts (to preserve)
        SelfReflectionExample(
            input_question="What is the capital of France?",
            core_question="Capital city of France?",
            reasoning="France is a country in Europe. Its capital is Paris.",
            answer="Paris",
        ),
    ]

    # Add GSM8K patterns if requested (v3 / Phase A)
    if include_gsm8k:
        from modelcypher.core.domain.training.gsm8k_patterns import get_gsm8k_pattern_examples
        examples.extend(get_gsm8k_pattern_examples())
    if include_phase_b:
        from modelcypher.core.domain.training.phase_b_patterns import get_phase_b_examples
        examples.extend(get_phase_b_examples())
    if include_phase_c:
        from modelcypher.core.domain.training.phase_c_patterns import get_phase_c_examples
        examples.extend(get_phase_c_examples())

    return examples


@dataclass
class SelfReflectionDataProvider:
    """Data provider for self-reflection training.

    Compatible with TrainingEngine.train() interface.
    """

    examples: list[SelfReflectionExample] = field(default_factory=get_self_reflection_examples)
    tokenizer: any = None

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        """Yield (input_ids, target_ids) for each example."""
        for example in self.examples:
            tokens = self.tokenizer.encode(example.full_text)
            # input = tokens[:-1], target = tokens[1:] (causal LM)
            input_ids = mx.array(tokens[:-1])
            target_ids = mx.array(tokens[1:])
            yield input_ids, target_ids


def compute_phi_ratio(model, tokenizer, text: str) -> float:
    """Compute peak/final norm ratio (proxy for comp/φ).

    Target: ratio ≈ φ (1.618) for optimal processing.
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)
    peak = float(mx.sqrt(mx.sum(hidden * hidden)))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        peak = max(peak, norm)

    final = norm
    return peak / final if final > 1e-10 else 1.0


def evaluate_self_reflection(
    model,
    tokenizer,
    generate_fn,
    examples: list[SelfReflectionExample] | None = None,
) -> dict:
    """Evaluate self-reflection capability.

    Returns:
        Dict with reflection_rate, accuracy, and per-example results.
    """
    if examples is None:
        examples = get_self_reflection_examples()

    results = []
    reflection_count = 0
    correct_count = 0

    PHI = 1.618033988749895

    for ex in examples:
        prompt = f"Question: {ex.input_question}\n\n"
        response = generate_fn(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)

        has_reflection = "let me understand" in response.lower()
        has_answer = ex.answer.lower() in response.lower()

        if has_reflection:
            reflection_count += 1
        if has_answer:
            correct_count += 1

        ratio = compute_phi_ratio(model, tokenizer, response)

        results.append({
            "question": ex.input_question[:50],
            "expected": ex.answer,
            "has_reflection": has_reflection,
            "correct": has_answer,
            "phi_ratio": ratio,
            "phi_distance": abs(ratio - PHI),
        })

    return {
        "reflection_rate": reflection_count / len(examples),
        "accuracy": correct_count / len(examples),
        "avg_phi_distance": sum(r["phi_distance"] for r in results) / len(results),
        "results": results,
    }


def load_training_data_from_jsonl(path: str) -> list[dict]:
    """Load training data from a JSONL file.

    Expected format per line:
        {"prompt": "...", "completion": "..."}

    Returns list of dicts with 'input' and 'output' keys for compatibility
    with the training loop.
    """
    import json
    from pathlib import Path

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    examples = []
    for line in data_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # Support both {"prompt", "completion"} and {"input", "output"} formats
        prompt = record.get("prompt") or record.get("input")
        completion = record.get("completion") or record.get("output")
        if prompt and completion:
            examples.append({"input": prompt, "output": completion})

    if not examples:
        raise ValueError(f"No valid examples loaded from {data_path}")

    logger.info(f"Loaded {len(examples)} training examples from {path}")
    return examples


def train_self_reflection_lora(
    model_path: str,
    output_path: str | None = None,
    rank: int = 8,
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
    run_tests: bool = True,
    layer_start: int | None = None,
    layer_end: int | None = None,
    entropy_probe_path: str | None = None,
    entropy_profile_output: str | None = None,
    id_profile_output: str | None = None,
    training_data_path: str | None = None,
) -> dict:
    """Train self-reflection capability using LoRA.

    This is the main entry point for the CLI command. Uses LoRA to avoid
    catastrophic forgetting of factual knowledge while teaching self-reflection.

    Args:
        model_path: Path to the base model.
        output_path: Optional path to save LoRA adapters.
        rank: LoRA rank (default: 8).
        num_epochs: Number of training epochs (default: 15).
        learning_rate: Learning rate (default: 1e-4).
        run_tests: Whether to run evaluation tests after training.
        entropy_probe_path: Optional path to prompts for entropy profiling.
        entropy_profile_output: Optional path to save entropy profile JSON.
        id_profile_output: Optional path to save intrinsic dimension profile JSON.
        training_data_path: Optional path to custom JSONL training data.
            Format: {"prompt": "...", "completion": "..."} per line.
            If not provided, uses built-in self-reflection examples.

    Returns:
        Dict with training results and optional test metrics.
    """
    import mlx.nn as nn
    import mlx.optimizers as optim
    from datetime import datetime
    from mlx_lm import load, generate
    from mlx_lm.tuner.utils import linear_to_lora_layers

    logger.info("=" * 70)
    logger.info("SELF-REFLECTION LORA TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Rank: {rank}, LR: {learning_rate}, Epochs: {num_epochs}")

    from mlx.utils import tree_flatten

    # Load model
    model, tokenizer = load(model_path)

    def _load_probe_prompts(path: str) -> list[str]:
        prompts: list[str] = []
        probe_path = Path(path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe prompts not found: {probe_path}")
        if probe_path.suffix == ".jsonl":
            import json
            for line in probe_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("prompt") or record.get("text")
                if prompt:
                    prompts.append(prompt)
        else:
            prompts = [line.strip() for line in probe_path.read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts loaded from {probe_path}")
        return prompts

    def _entropy_profile_to_dict(profile) -> dict:
        trajectory = profile.entropy_trajectory()
        if not trajectory:
            return {"trajectory": []}
        peak_idx = max(range(len(trajectory)), key=lambda i: trajectory[i])
        peak = trajectory[peak_idx]
        initial = trajectory[0]
        final = trajectory[-1]
        expansion_rate = (peak - initial) / float(max(1, peak_idx))
        compression_rate = (peak - final) / float(max(1, (len(trajectory) - 1 - peak_idx)))
        ratio_over_phi = (
            compression_rate / (expansion_rate * PHI)
            if expansion_rate != 0.0
            else 0.0
        )
        return {
            "model_name": profile.model_name,
            "created_at": profile.created_at.isoformat(),
            "trajectory": trajectory,
            "peak_layer": peak_idx,
            "initial_entropy": initial,
            "peak_entropy": peak,
            "final_entropy": final,
            "expansion_rate": expansion_rate,
            "compression_rate": compression_rate,
            "ratio_over_phi": ratio_over_phi,
            "layer_stats": {
                idx: {
                    "mean_entropy": result.mean_entropy,
                    "entropy_variance": result.entropy_variance,
                    "min_entropy": result.min_entropy,
                    "max_entropy": result.max_entropy,
                    "sample_count": result.sample_count,
                }
                for idx, result in profile.layer_results.items()
            },
        }

    def _intrinsic_dimension_profile(model_obj, tok, prompts: list[str]) -> dict:
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

        backend = get_default_backend()
        projector = LayerEntropyProjector(backend)
        base_model = getattr(model_obj, "model", model_obj)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model.layers or model.model.layers")

        num_layers = len(layers)
        target_layers = set(range(num_layers))
        layer_points: dict[int, list] = {i: [] for i in target_layers}

        for prompt in prompts:
            tokens = tok.encode(prompt)
            if isinstance(tokens, list):
                input_ids = backend.array([tokens])
            else:
                input_ids = tokens
                if input_ids.ndim == 1:
                    input_ids = backend.reshape(input_ids, (1, -1))

            captured = projector._capture_layer_states(base_model, layers, input_ids, target_layers)
            for layer_idx, hidden_state in captured.items():
                if hidden_state.ndim == 3:
                    pts = hidden_state[0, :, :]
                elif hidden_state.ndim == 2:
                    pts = hidden_state
                else:
                    pts = backend.reshape(hidden_state, (1, -1))
                layer_points[layer_idx].append(pts)

        id_results: dict[int, dict] = {}
        estimator = IntrinsicDimension(backend)
        min_samples = IntrinsicDimension.local_dimension_min_samples()

        for layer_idx, pts_list in layer_points.items():
            if not pts_list:
                continue
            if len(pts_list) == 1:
                all_pts = pts_list[0]
            else:
                all_pts = backend.concatenate(pts_list, axis=0)
            sample_count = int(all_pts.shape[0])
            if sample_count < min_samples:
                id_results[layer_idx] = {
                    "intrinsic_dimension": None,
                    "sample_count": sample_count,
                    "usable_count": 0,
                    "ci_lower": None,
                    "ci_upper": None,
                    "ci_resamples": None,
                }
                continue
            estimate = estimator.compute(all_pts, with_ci=True)
            id_results[layer_idx] = {
                "intrinsic_dimension": estimate.intrinsic_dimension,
                "sample_count": estimate.sample_count,
                "usable_count": estimate.usable_count,
                "ci_lower": estimate.ci.lower if estimate.ci else None,
                "ci_upper": estimate.ci.upper if estimate.ci else None,
                "ci_resamples": estimate.ci.resamples if estimate.ci else None,
            }

        return {
            "model_name": getattr(model_obj, "name", None) or model_obj.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "layers": id_results,
        }

    # Optional entropy profile (pre-training)
    entropy_before = None
    id_before = None
    probe_prompts = None
    if entropy_probe_path:
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector
        probe_prompts = _load_probe_prompts(entropy_probe_path)
        projector = LayerEntropyProjector()
        entropy_before = _entropy_profile_to_dict(
            projector.profile_model(model, tokenizer, probe_prompts)
        )
        id_before = _intrinsic_dimension_profile(model, tokenizer, probe_prompts)

    # Get baseline response
    test_prompt = "Question: A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?\n\n"
    baseline = generate(model, tokenizer, prompt=test_prompt, max_tokens=50, verbose=False)
    logger.info(f"Baseline: {baseline[:70]}...")

    # Freeze base model BEFORE applying LoRA
    # This ensures LoRA params (added after) are trainable
    model.freeze()

    # Apply LoRA
    logger.info("Applying LoRA...")
    lora_config = {
        "rank": rank,
        "alpha": rank * 2,
        "dropout": 0.0,
        "scale": 1.0,
    }
    if layer_start is not None or layer_end is not None:
        start = layer_start if layer_start is not None else 0
        end = layer_end if layer_end is not None else len(model.model.layers) - 1
        layer_indices = list(range(start, end + 1))
        logger.info(f"Applying LoRA to layers: {start}-{end}")
        _apply_lora_to_layers(model, layer_indices, lora_config)
    else:
        linear_to_lora_layers(model, num_layers=len(model.model.layers), config=lora_config)

    # Count parameters using tree_flatten for accuracy
    tp_flat = dict(tree_flatten(model.trainable_parameters()))
    trainable = sum(v.size for v in tp_flat.values())
    all_flat = dict(tree_flatten(model.parameters()))
    total = sum(v.size for v in all_flat.values())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training data
    if training_data_path:
        training_data = load_training_data_from_jsonl(training_data_path)
    else:
        examples = get_self_reflection_examples()
        training_data = [
            {"input": ex.input_question, "output": ex.full_output}
            for ex in examples
        ]
    logger.info(f"Training examples: {len(training_data)}")

    # Training
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def loss_fn(model, tokens):
        input_ids = mx.array([tokens[:-1]])
        target_ids = mx.array([tokens[1:]])
        logits = model(input_ids)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = target_ids.reshape(-1)
        return nn.losses.cross_entropy(logits, targets, reduction='mean')

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(num_epochs):
        total_loss = 0
        for example in training_data:
            full_text = f"Question: {example['input']}\n\n{example['output']}"
            tokens = tokenizer.encode(full_text)

            loss, grads = loss_and_grad(model, tokens)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            total_loss += float(loss)

        avg_loss = total_loss / len(training_data)
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

    # Post-training response
    trained = generate(model, tokenizer, prompt=test_prompt, max_tokens=80, verbose=False)
    has_reflection = "Let me understand" in trained
    logger.info(f"Trained: {trained[:80]}...")
    logger.info(f"Has self-reflection: {'✓' if has_reflection else '✗'}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "config": {
            "rank": rank,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "trainable_params": trainable,
            "total_params": total,
        },
        "baseline_response": baseline[:100],
        "trained_response": trained[:100],
        "has_reflection": has_reflection,
        "entropy_profile_before": entropy_before,
        "intrinsic_dimension_profile_before": id_before,
    }

    if layer_start is not None or layer_end is not None:
        result["config"]["layer_start"] = layer_start if layer_start is not None else 0
        result["config"]["layer_end"] = (
            layer_end if layer_end is not None else len(model.model.layers) - 1
        )

    # Optional entropy profile (post-training)
    if entropy_probe_path:
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector
        projector = LayerEntropyProjector()
        entropy_after = _entropy_profile_to_dict(
            projector.profile_model(model, tokenizer, probe_prompts or [])
        )
        result["entropy_profile_after"] = entropy_after
        id_after = _intrinsic_dimension_profile(model, tokenizer, probe_prompts or [])
        result["intrinsic_dimension_profile_after"] = id_after
        if entropy_profile_output:
            import json
            output_path_obj = Path(entropy_profile_output)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with output_path_obj.open("w") as f:
                json.dump(
                    {"before": entropy_before, "after": entropy_after},
                    f,
                    indent=2,
                )
        if id_profile_output:
            import json
            id_output_path = Path(id_profile_output)
            id_output_path.parent.mkdir(parents=True, exist_ok=True)
            with id_output_path.open("w") as f:
                json.dump(
                    {"before": id_before, "after": id_after},
                    f,
                    indent=2,
                )

    # Run tests if requested
    if run_tests:
        logger.info("\n--- EVALUATION ---")

        word_problems = [
            ("A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?", "0.05"),
            ("5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100?", "5"),
            ("A lily pad doubles daily. Covers lake in 48 days. When half covered?", "47"),
            ("Tom has 3× as many apples as Jane. Jane has 5. How many does Tom have?", "15"),
        ]

        word_correct = 0
        for q, expected in word_problems:
            prompt = f"Question: {q}\n\n"
            response = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
            correct = expected in response
            if correct:
                word_correct += 1
            status = "✓" if correct else "✗"
            logger.info(f"{status} {q[:40]}... → {expected}")

        fact_tests = [
            ("What is the capital of France?", "paris"),
            ("What is H2O?", "water"),
            ("How many days in a week?", "7"),
            ("What planet is closest to the sun?", "mercury"),
            ("How many legs does a spider have?", "8"),
        ]

        facts_correct = 0
        for q, expected in fact_tests:
            prompt = f"Question: {q}\n\nAnswer:"
            response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
            correct = expected.lower() in response.lower()
            if correct:
                facts_correct += 1
            status = "✓" if correct else "✗"
            logger.info(f"{status} {q} → {response[:30].strip()}...")

        result["tests"] = {
            "word_problems": word_correct / len(word_problems),
            "facts_preserved": facts_correct / len(fact_tests),
        }

        logger.info(f"\nWord problems: {word_correct}/{len(word_problems)}")
        logger.info(f"Facts preserved: {facts_correct}/{len(fact_tests)}")

    # Save adapters if output path provided
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights only (need to flatten the nested dict)
        lora_weights = dict(tree_flatten(model.trainable_parameters()))

        weights_path = output_dir / "lora_weights.safetensors"
        mx.save_safetensors(str(weights_path), lora_weights)

        # Save config
        config_path = output_dir / "adapter_config.json"
        import json
        with open(config_path, "w") as f:
            json.dump({
                "rank": rank,
                "alpha": rank * 2,
                "base_model": model_path,
                "training": result["config"],
            }, f, indent=2)

        logger.info(f"\nSaved adapters to: {output_path}")
        result["adapter_path"] = str(output_path)

    return result


def load_self_reflection_adapters(
    model_path: str,
    adapter_path: str,
):
    """Load a model with self-reflection LoRA adapters applied.

    Args:
        model_path: Path to the base model.
        adapter_path: Path to the saved LoRA adapters.

    Returns:
        Tuple of (model, tokenizer) with adapters applied.
    """
    import json
    from mlx_lm import load
    from mlx.utils import tree_unflatten

    # Load adapter config
    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load base model
    model, tokenizer = load(model_path)

    # Freeze base model and apply LoRA structure
    model.freeze()
    lora_config = {
        "rank": config["rank"],
        "alpha": config["alpha"],
        "dropout": 0.0,
        "scale": 1.0,
    }
    layer_start = config.get("training", {}).get("layer_start")
    layer_end = config.get("training", {}).get("layer_end")
    if layer_start is not None or layer_end is not None:
        start = layer_start if layer_start is not None else 0
        end = layer_end if layer_end is not None else len(model.model.layers) - 1
        _apply_lora_to_layers(model, list(range(start, end + 1)), lora_config)
    else:
        from mlx_lm.tuner.utils import linear_to_lora_layers
        linear_to_lora_layers(model, num_layers=len(model.model.layers), config=lora_config)

    # Load and apply adapter weights
    weights_path = Path(adapter_path) / "lora_weights.safetensors"
    adapter_weights = mx.load(str(weights_path))

    # Apply weights to model
    model.update(tree_unflatten(list(adapter_weights.items())))
    mx.eval(model.parameters())

    logger.info(f"Loaded adapters from {adapter_path}")
    return model, tokenizer
