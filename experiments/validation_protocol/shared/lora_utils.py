# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Shared LoRA Training Utilities for Validation Experiments
#
# These utilities enable rapid LoRA training and evaluation for
# validating geometry-based predictions (Fisher Information, Mode Connectivity).

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class QuickTrainResult:
    """Result of a quick LoRA training run."""

    output_path: Path
    final_loss: float
    training_losses: list[float]
    steps_completed: int
    duration_seconds: float
    target_modules: list[str]
    rank: int


@dataclass
class EvalResult:
    """Result of perplexity evaluation."""

    perplexity: float
    loss: float
    tokens_evaluated: int
    samples_evaluated: int


def load_model_and_tokenizer(model_path: Path | str):
    """Load an MLX model and tokenizer.

    Returns:
        (model, tokenizer) tuple
    """
    from modelcypher.adapters.model_loader import load_model_for_training

    model, tokenizer = load_model_for_training(str(model_path))
    return model, tokenizer


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA adapters to specified modules.

    Args:
        model: The base model
        target_modules: List of module names to adapt (e.g., ["q_proj", "v_proj"])
        rank: LoRA rank
        alpha: LoRA alpha (defaults to rank for scale=1.0)
        dropout: LoRA dropout

    Returns:
        Model with LoRA adapters applied
    """
    from modelcypher.core.domain.training.lora_mlx import (
        LoRASettings,
        apply_lora_to_model,
    )

    if alpha is None:
        alpha = float(rank)

    settings = LoRASettings(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    model = apply_lora_to_model(model, settings)
    return model


def freeze_non_lora_params(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters.

    MLX approach: freeze everything, then selectively unfreeze LoRA params.
    Uses named_modules() for proper traversal of the module tree.
    """
    from modelcypher.core.domain.training.lora_mlx import LoRALinear

    # Freeze all parameters first
    model.freeze()

    # Now find all LoRALinear modules and unfreeze their lora_a and lora_b
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"])


def create_data_iterator(
    dataset_path: Path | str,
    tokenizer,
    batch_size: int = 4,
    sequence_length: int = 256,
    max_samples: int | None = None,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Create a data iterator from JSONL dataset.

    Args:
        dataset_path: Path to JSONL file with {"text": ...} format
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        sequence_length: Maximum sequence length
        max_samples: Limit number of samples (None = all)

    Yields:
        (input_ids, target_ids) batches
    """
    # Load all samples
    samples = []
    with open(dataset_path, "r") as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            data = json.loads(line.strip())
            text = data.get("text", "")
            if text:
                samples.append(text)

    if not samples:
        raise ValueError(f"No samples found in {dataset_path}")

    # Tokenize all samples
    all_tokens = []
    for text in samples:
        tokens = tokenizer.encode(text)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids) if hasattr(tokens, 'ids') else list(tokens)
        all_tokens.append(token_ids)

    # Create batches
    def make_batches():
        batch_inputs = []
        batch_targets = []

        for tokens in all_tokens:
            # Truncate or pad to sequence_length
            if len(tokens) > sequence_length + 1:
                tokens = tokens[:sequence_length + 1]
            elif len(tokens) < sequence_length + 1:
                # Pad with tokenizer pad_token_id or 0
                pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0
                tokens = tokens + [pad_id] * (sequence_length + 1 - len(tokens))

            # Input is tokens[:-1], target is tokens[1:]
            batch_inputs.append(tokens[:-1])
            batch_targets.append(tokens[1:])

            if len(batch_inputs) >= batch_size:
                yield (
                    mx.array(batch_inputs, dtype=mx.int32),
                    mx.array(batch_targets, dtype=mx.int32),
                )
                batch_inputs = []
                batch_targets = []

        # Yield remaining samples
        if batch_inputs:
            yield (
                mx.array(batch_inputs, dtype=mx.int32),
                mx.array(batch_targets, dtype=mx.int32),
            )

    return make_batches()


def compute_loss(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    """Compute cross-entropy loss.

    Args:
        model: Language model
        inputs: Input token IDs [batch, seq_len]
        targets: Target token IDs [batch, seq_len]

    Returns:
        Scalar loss value
    """
    # Forward pass
    logits = model(inputs)

    # Cross-entropy loss
    # logits: [batch, seq_len, vocab_size]
    # targets: [batch, seq_len]
    vocab_size = logits.shape[-1]

    # Reshape for cross-entropy
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute cross-entropy
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")

    return loss


def train_lora_quick(
    model_path: Path | str,
    dataset_path: Path | str,
    output_path: Path | str,
    target_modules: list[str],
    rank: int = 8,
    steps: int = 50,
    lr: float = 1e-4,
    batch_size: int = 2,
    sequence_length: int = 256,
    max_samples: int = 100,
) -> QuickTrainResult:
    """Train a LoRA adapter quickly for validation experiments.

    This is a minimal training loop designed for fast iteration
    in validation experiments, not production training.

    Args:
        model_path: Path to base model
        dataset_path: Path to JSONL training data
        output_path: Where to save LoRA weights
        target_modules: Which modules to adapt
        rank: LoRA rank
        steps: Number of training steps
        lr: Learning rate
        batch_size: Batch size
        sequence_length: Max sequence length
        max_samples: Max training samples to use

    Returns:
        QuickTrainResult with training info
    """
    start_time = time.perf_counter()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load_model_and_tokenizer(model_path)

    logger.info("Applying LoRA to modules: %s (rank=%d)", target_modules, rank)
    model = apply_lora(model, target_modules, rank=rank)
    freeze_non_lora_params(model)

    # Create optimizer (only for LoRA params)
    optimizer = optim.AdamW(learning_rate=lr)

    # Create data iterator
    data_iter = list(create_data_iterator(
        dataset_path, tokenizer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_samples=max_samples,
    ))

    if not data_iter:
        raise ValueError("No training data available")

    # Training loop
    losses = []
    step = 0

    def loss_fn(model, inputs, targets):
        return compute_loss(model, inputs, targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    logger.info("Starting training for %d steps", steps)

    while step < steps:
        for inputs, targets in data_iter:
            if step >= steps:
                break

            loss, grads = loss_and_grad(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = float(loss)
            losses.append(loss_val)

            if step % 10 == 0:
                logger.info("  Step %d/%d: loss=%.4f", step, steps, loss_val)

            step += 1

    # Save LoRA weights
    logger.info("Saving LoRA weights to %s", output_path)
    save_lora_weights(model, output_path)

    duration = time.perf_counter() - start_time

    return QuickTrainResult(
        output_path=output_path,
        final_loss=losses[-1] if losses else float('inf'),
        training_losses=losses,
        steps_completed=step,
        duration_seconds=duration,
        target_modules=target_modules,
        rank=rank,
    )


def save_lora_weights(model: nn.Module, output_path: Path) -> None:
    """Save only LoRA adapter weights."""
    from mlx.utils import tree_flatten

    lora_weights = {}
    for name, param in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            lora_weights[name] = param

    if not lora_weights:
        logger.warning("No LoRA weights found to save")
        return

    mx.savez(str(output_path / "lora_weights.npz"), **lora_weights)

    # Save config
    config = {
        "parameter_count": sum(p.size for p in lora_weights.values()),
        "keys": list(lora_weights.keys()),
    }
    with open(output_path / "lora_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved %d LoRA parameters", config["parameter_count"])


def load_lora_weights(
    model: nn.Module,
    lora_path: Path,
    target_modules: list[str] | None = None,
    rank: int = 8,
) -> nn.Module:
    """Load LoRA weights into a model.

    If the model doesn't have LoRA layers, applies them first using the
    target_modules and rank. If not provided, tries to infer from the
    lora_config.json file.
    """
    weights_file = lora_path / "lora_weights.npz"
    if not weights_file.exists():
        raise FileNotFoundError(f"LoRA weights not found at {weights_file}")

    # Check if we need to apply LoRA first
    lora_weights = dict(mx.load(str(weights_file)))

    # Try to load config to get target modules
    config_file = lora_path / "lora_config.json"
    if config_file.exists() and target_modules is None:
        with open(config_file) as f:
            config = json.load(f)
            # Extract module names from the keys
            # Keys look like: model.layers.2.self_attn.q_proj.lora_a
            modules_from_keys = set()
            for key in config.get("keys", []):
                # Extract the projection name (e.g., q_proj)
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if "lora_" in part and i > 0:
                        modules_from_keys.add(parts[i - 1])
            if modules_from_keys:
                target_modules = list(modules_from_keys)

    # Check if model already has LoRA layers
    from modelcypher.core.domain.training.lora_mlx import LoRALinear
    has_lora = any(isinstance(m, LoRALinear) for _, m in model.named_modules())

    if not has_lora and target_modules:
        # Apply LoRA first
        model = apply_lora(model, target_modules, rank=rank)
        logger.info("Applied LoRA to %s before loading weights", target_modules)

    # Use strict=False because LoRA may only be applied to a subset of layers
    # and the saved weights may not cover all model parameters
    model.load_weights(list(lora_weights.items()), strict=False)

    logger.info("Loaded %d LoRA parameters from %s", len(lora_weights), lora_path)
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base weights.

    After merging, the model no longer has LoRA adapters but
    the effect is baked into the base weights.
    """
    from modelcypher.core.domain.training.lora_mlx import LoRALinear

    def merge_module(module):
        if isinstance(module, LoRALinear):
            return module.merge()
        return module

    # Note: This is a simplified version - full implementation would
    # need to traverse the model tree properly
    model.apply(lambda m: merge_module(m) if isinstance(m, LoRALinear) else m)

    return model


def evaluate_perplexity(
    model_path: Path | str,
    dataset_path: Path | str,
    lora_path: Path | str | None = None,
    max_samples: int = 100,
    batch_size: int = 4,
    sequence_length: int = 256,
) -> EvalResult:
    """Evaluate model perplexity on a dataset.

    Args:
        model_path: Path to base model
        dataset_path: Path to evaluation JSONL
        lora_path: Optional path to LoRA weights to apply
        max_samples: Max samples to evaluate
        batch_size: Batch size
        sequence_length: Max sequence length

    Returns:
        EvalResult with perplexity and loss
    """
    import math

    model, tokenizer = load_model_and_tokenizer(model_path)

    if lora_path:
        lora_path = Path(lora_path)
        if lora_path.exists():
            model = load_lora_weights(model, lora_path)

    # Create evaluation data
    data_iter = list(create_data_iterator(
        dataset_path, tokenizer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_samples=max_samples,
    ))

    total_loss = 0.0
    total_tokens = 0
    samples_evaluated = 0

    for inputs, targets in data_iter:
        loss = compute_loss(model, inputs, targets)
        mx.eval(loss)

        batch_tokens = targets.size
        total_loss += float(loss) * batch_tokens
        total_tokens += batch_tokens
        samples_evaluated += inputs.shape[0]

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow

    return EvalResult(
        perplexity=perplexity,
        loss=avg_loss,
        tokens_evaluated=total_tokens,
        samples_evaluated=samples_evaluated,
    )


def collect_layer_activations(
    model_path: Path | str,
    prompts: list[str],
    layer_idx: int,
    lora_path: Path | str | None = None,
) -> mx.array:
    """Collect hidden state activations from a specific layer.

    Args:
        model_path: Path to model
        prompts: List of text prompts
        layer_idx: Which layer to extract from
        lora_path: Optional LoRA weights to apply

    Returns:
        Activations array [n_prompts, hidden_dim]
    """
    model, tokenizer = load_model_and_tokenizer(model_path)

    if lora_path:
        lora_path = Path(lora_path)
        if lora_path.exists():
            model = load_lora_weights(model, lora_path)

    # Get model architecture info
    from modelcypher.ports.model_architecture_factory import get_model_architecture

    config = {}
    if hasattr(model, 'config'):
        if hasattr(model.config, 'to_dict'):
            config = model.config.to_dict()
        elif isinstance(model.config, dict):
            config = model.config

    arch = get_model_architecture(model, config=config)

    activations = []

    for prompt in prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids) if hasattr(tokens, 'ids') else list(tokens)

        input_ids = mx.array([token_ids], dtype=mx.int32)

        # Forward through embedding
        h = arch.embed_module(input_ids)

        # Forward through layers up to target
        for idx, layer in enumerate(arch.layers):
            result = layer(h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result

            if idx == layer_idx:
                break

        # Mean pool over sequence
        pooled = mx.mean(h, axis=(0, 1))
        mx.eval(pooled)
        activations.append(pooled)

    return mx.stack(activations, axis=0)
