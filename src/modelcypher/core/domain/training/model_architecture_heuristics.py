"""Model architecture heuristics for parameter counting and memory estimation.

Provides default architecture configurations based on model size.
"""

from __future__ import annotations


from modelcypher.core.domain.training.checkpoint_models import ModelArchitectureConfig


class ModelArchitectureHeuristics:
    """Heuristics for inferring model architecture from parameter count."""

    @staticmethod
    def config_for_parameter_count(
        parameter_count: int | None,
    ) -> ModelArchitectureConfig:
        """Get architecture config based on parameter count.

        Returns heuristic-based architecture configuration for memory estimation
        when exact architecture is not known.

        Args:
            parameter_count: Number of model parameters.

        Returns:
            Architecture configuration for the given size.
        """
        if parameter_count is None:
            # Default: assume ~7B model
            return ModelArchitectureConfig(
                model_type="simple_transformer",
                vocabulary_size=32000,
                hidden_size=4096,
                num_layers=32,
                num_heads=32,
            )

        if parameter_count >= 6_000_000_000:
            # 7B+ models
            return ModelArchitectureConfig(
                model_type="simple_transformer",
                vocabulary_size=32000,
                hidden_size=4096,
                num_layers=32,
                num_heads=32,
            )
        elif parameter_count >= 2_000_000_000:
            # 3B-6B models
            return ModelArchitectureConfig(
                model_type="simple_transformer",
                vocabulary_size=32000,
                hidden_size=3072,
                num_layers=28,
                num_heads=24,
            )
        else:
            # <3B models
            return ModelArchitectureConfig(
                model_type="simple_transformer",
                vocabulary_size=32000,
                hidden_size=2048,
                num_layers=16,
                num_heads=16,
            )

    @staticmethod
    def estimate_vram_bytes(
        config: ModelArchitectureConfig,
        batch_size: int = 1,
        sequence_length: int = 1024,
        precision_bytes: int = 2,
    ) -> int:
        """Estimate VRAM usage for training.

        Estimates memory requirements based on:
        - Model parameters
        - Optimizer states (8 bytes per param for AdamW)
        - Gradients
        - Activations

        Args:
            config: Model architecture configuration.
            batch_size: Training batch size.
            sequence_length: Maximum sequence length.
            precision_bytes: Bytes per parameter (2 for fp16, 4 for fp32).

        Returns:
            Estimated VRAM in bytes.
        """
        # Estimate parameter count
        vocab_params = config.vocabulary_size * config.hidden_size
        attention_params_per_layer = 4 * config.hidden_size * config.hidden_size
        ffn_params_per_layer = 8 * config.hidden_size * config.hidden_size
        layer_norm_params_per_layer = 4 * config.hidden_size

        params_per_layer = (
            attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer
        )
        total_params = vocab_params + (params_per_layer * config.num_layers)

        # Memory components
        # Model weights
        model_memory = total_params * precision_bytes

        # Optimizer states (AdamW: first/second moments)
        optimizer_memory = total_params * 8

        # Gradients
        gradient_memory = total_params * precision_bytes

        # Activations (rough estimate)
        activation_per_token = config.hidden_size * config.num_layers * precision_bytes
        activation_memory = batch_size * sequence_length * activation_per_token

        # Total with overhead
        total = model_memory + optimizer_memory + gradient_memory + activation_memory
        return int(total * 1.2)  # 20% overhead

    @staticmethod
    def suggest_batch_size(
        config: ModelArchitectureConfig,
        available_vram_bytes: int,
        sequence_length: int = 1024,
        precision_bytes: int = 2,
    ) -> int:
        """Suggest maximum batch size for available VRAM.

        Args:
            config: Model architecture configuration.
            available_vram_bytes: Available VRAM in bytes.
            sequence_length: Maximum sequence length.
            precision_bytes: Bytes per parameter (2 for fp16, 4 for fp32).

        Returns:
            Suggested batch size (minimum 1).
        """
        # Binary search for max batch size
        low, high = 1, 128
        best = 1

        while low <= high:
            mid = (low + high) // 2
            estimated = ModelArchitectureHeuristics.estimate_vram_bytes(
                config, mid, sequence_length, precision_bytes
            )

            if estimated <= available_vram_bytes:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return max(1, best)
