"""
LoRA Adapter Merger: TIES and DARE-TIES Merging Strategies.

Ported 1:1 from the reference Swift implementation.

Implements:
- TIES: "TIES-Merging: Resolving Interference When Merging Models" (Yadav et al. 2023)
- DARE-TIES: DARE sparsity + TIES merging for improved adapter combination

TIES Algorithm:
1. Trim: Keep only top-K% magnitude parameters per adapter
2. Elect: For each position, determine consensus sign direction
3. Merge: Average values with matching signs, zero out conflicts

DARE Enhancement:
- Random dropout of delta weights before TIES (inverted dropout scaling)
- Reduces interference between adapters

NOTE: This module has infrastructure dependencies (mx.load, mx.save_safetensors)
for file I/O that cannot be abstracted via Backend protocol. Math operations
use the Backend protocol.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Infrastructure dependencies (MLX-specific file I/O)
# These cannot be abstracted via Backend protocol
import mlx.core as mx

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.merging.lora_adapter_merger")


class Strategy(str, Enum):
    """Merge strategy for combining LoRA adapters."""
    TIES = "ties"
    DARE_TIES = "dare-ties"


@dataclass
class Config:
    """Configuration for LoRA adapter merging."""
    strategy: Strategy = Strategy.TIES
    # Top-K fraction of parameters to keep during TIES trimming (0-1).
    ties_top_k: float = 0.2
    # Drop rate for DARE (0-1). None = auto-compute from sparsity analysis.
    drop_rate: Optional[float] = None
    # Random seed for deterministic DARE dropout.
    seed: int = 0


@dataclass
class AdapterSparsitySummary:
    """Sparsity analysis for a single adapter."""
    adapter_path: str
    recommended_drop_rate: float
    applied_drop_rate: float
    effective_sparsity: float


@dataclass
class MergeReport:
    """Report of a completed adapter merge."""
    output_directory: str
    adapter_count: int
    strategy: Strategy
    base_model_id: str
    rank: int
    scale: float
    ties_top_k: float
    drop_rate: Optional[float]
    trimmed_fraction: float
    sign_conflict_rate: float
    merged_non_zero_fraction: float
    total_merged_parameters: int
    per_adapter_sparsity: List[AdapterSparsitySummary] = field(default_factory=list)


from modelcypher.core.domain.merging.exceptions import MergeError


# =============================================================================
# Internal Data Structures
# =============================================================================


@dataclass
class LoRAModuleWeights:
    a: Any  # Array type from backend
    b: Any  # Array type from backend


@dataclass
class AdapterPayload:
    directory: Path
    base_model_id: str
    rank: int
    scale: float
    modules: Dict[str, LoRAModuleWeights]
    a_key_by_module: Dict[str, str]
    b_key_by_module: Dict[str, str]
    extra_weights: Dict[str, Any]  # Array type from backend


@dataclass
class MergeMatrixResult:
    merged: Any  # Array type from backend
    conflict_count: int
    merged_non_zero: int
    trimmed_non_zero: int
    trimmed_total: int
    total_elements: int


@dataclass
class TIESMergeResult:
    merged: List[float]
    conflict_count: int
    merged_non_zero: int
    total_elements: int


# =============================================================================
# Seeded Random Generator (for deterministic DARE dropout)
# =============================================================================


class SeededGenerator:
    """Deterministic pseudo-random number generator (SplitMix64)."""
    
    def __init__(self, seed: int):
        self.state = seed if seed != 0 else 0x9E3779B97F4A7C15
    
    def next_uint64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    
    def next_float(self) -> float:
        value = self.next_uint64() >> 40
        return float(value) / float(1 << 24)


# =============================================================================
# LoRA Adapter Merger
# =============================================================================


class LoRAAdapterMerger:
    """
    Merges multiple LoRA adapters using TIES or DARE-TIES strategy.
    
    Example:
        report = LoRAAdapterMerger.merge(
            adapter_directories=[Path("adapter1"), Path("adapter2")],
            output_directory=Path("merged"),
            config=Config(strategy=Strategy.TIES, ties_top_k=0.2)
        )
    """
    
    @staticmethod
    def merge(
        adapter_directories: List[Path],
        output_directory: Path,
        config: Config = Config(),
        backend: "Backend | None" = None,
    ) -> MergeReport:
        """
        Merge multiple LoRA adapters into a single adapter.
        
        Args:
            adapter_directories: Paths to adapter directories (PEFT format).
            output_directory: Where to save the merged adapter.
            config: Merge configuration.
        
        Returns:
            MergeReport with statistics about the merge.
        
        Raises:
            MergeError: If adapters are incompatible or invalid.
        """
        if len(adapter_directories) < 2:
            raise MergeError("At least two adapters are required for merging.")
        
        if not 0.0 <= config.ties_top_k <= 1.0:
            raise MergeError("ties_top_k must be between 0 and 1.")
        
        if config.drop_rate is not None and not 0.0 <= config.drop_rate <= 1.0:
            raise MergeError("drop_rate must be between 0 and 1.")
        
        logger.info(f"Merging {len(adapter_directories)} adapters with strategy {config.strategy.value}")
        
        # Load all adapters
        payloads = [LoRAAdapterMerger._load_adapter(d) for d in adapter_directories]
        first = payloads[0]
        
        # Validate compatibility
        for payload in payloads[1:]:
            if payload.base_model_id != first.base_model_id:
                raise MergeError(f"Adapters target different base models: {first.base_model_id} vs {payload.base_model_id}")
            if payload.rank != first.rank:
                raise MergeError(f"Adapters use different LoRA ranks: {first.rank} vs {payload.rank}")
            if abs(payload.scale - first.scale) > 1e-5:
                raise MergeError(f"Adapters use different scales: {first.scale} vs {payload.scale}")
        
        module_keys = set(first.modules.keys())
        for payload in payloads[1:]:
            if set(payload.modules.keys()) != module_keys:
                raise MergeError("Adapters do not share the same LoRA module set.")
        
        # Resolve drop rates for DARE-TIES
        drop_rates, sparsity_summaries = LoRAAdapterMerger._resolve_drop_rates(payloads, config)
        
        # Merge each module
        merged_weights: Dict[str, mx.array] = {}
        total_elements = 0
        merged_non_zero = 0
        conflict_count = 0
        trimmed_non_zero = 0
        trimmed_total = 0
        
        for module in sorted(module_keys):
            a_matrices = [p.modules[module].a for p in payloads]
            b_matrices = [p.modules[module].b for p in payloads]
            
            a_result = LoRAAdapterMerger._merge_matrices(
                matrices=a_matrices,
                drop_rates=drop_rates,
                config=config,
                seed=config.seed,
                module=module,
                backend=backend,
            )
            b_result = LoRAAdapterMerger._merge_matrices(
                matrices=b_matrices,
                drop_rates=drop_rates,
                config=config,
                seed=config.seed ^ 0x9E3779B97F4A7C15,
                module=module,
                backend=backend,
            )
            
            conflict_count += a_result.conflict_count + b_result.conflict_count
            trimmed_non_zero += a_result.trimmed_non_zero + b_result.trimmed_non_zero
            trimmed_total += a_result.trimmed_total + b_result.trimmed_total
            total_elements += a_result.total_elements + b_result.total_elements
            merged_non_zero += a_result.merged_non_zero + b_result.merged_non_zero
            
            if module in first.a_key_by_module:
                merged_weights[first.a_key_by_module[module]] = a_result.merged
            if module in first.b_key_by_module:
                merged_weights[first.b_key_by_module[module]] = b_result.merged
        
        # Include extra weights from first adapter
        for key, value in first.extra_weights.items():
            merged_weights[key] = value
        
        # Write merged adapter
        LoRAAdapterMerger._write_merged_adapter(
            source_directory=first.directory,
            output_directory=output_directory,
            weights=merged_weights,
            base_model_id=first.base_model_id,
        )
        
        trimmed_fraction = trimmed_non_zero / trimmed_total if trimmed_total > 0 else 0.0
        sign_conflict_rate = conflict_count / total_elements if total_elements > 0 else 0.0
        merged_non_zero_fraction = merged_non_zero / total_elements if total_elements > 0 else 0.0
        
        avg_drop_rate = None
        if config.strategy == Strategy.DARE_TIES and drop_rates:
            avg_drop_rate = sum(drop_rates) / len(drop_rates)
        
        return MergeReport(
            output_directory=str(output_directory),
            adapter_count=len(payloads),
            strategy=config.strategy,
            base_model_id=first.base_model_id,
            rank=first.rank,
            scale=first.scale,
            ties_top_k=config.ties_top_k,
            drop_rate=avg_drop_rate,
            trimmed_fraction=trimmed_fraction,
            sign_conflict_rate=sign_conflict_rate,
            merged_non_zero_fraction=merged_non_zero_fraction,
            total_merged_parameters=total_elements,
            per_adapter_sparsity=sparsity_summaries,
        )
    
    # =========================================================================
    # TIES Core Algorithm
    # =========================================================================
    
    @staticmethod
    def ties_merge(vectors: List[List[float]]) -> TIESMergeResult:
        """
        TIES merge: resolve interference by sign-based consensus.
        
        For each parameter position:
        1. Compute sum to determine consensus sign
        2. Average only values with matching sign
        3. Track conflicts (positions with mixed signs)
        """
        if not vectors:
            return TIESMergeResult(merged=[], conflict_count=0, merged_non_zero=0, total_elements=0)
        
        count = len(vectors[0])
        merged = [0.0] * count
        conflict_count = 0
        merged_non_zero = 0
        
        for idx in range(count):
            total = 0.0
            sign_set = set()
            
            for vector in vectors:
                value = vector[idx]
                if value == 0:
                    continue
                total += value
                sign_set.add(1 if value > 0 else -1)
            
            if len(sign_set) > 1:
                conflict_count += 1
            
            if total > 0:
                sign = 1.0
            elif total < 0:
                sign = -1.0
            else:
                continue  # All zeros or exact cancel
            
            # Average values with matching sign
            matching_sum = 0.0
            matching_count = 0
            for vector in vectors:
                value = vector[idx]
                if value == 0:
                    continue
                if value * sign > 0:
                    matching_sum += value
                    matching_count += 1
            
            if matching_count > 0:
                merged[idx] = matching_sum / matching_count
                merged_non_zero += 1
        
        return TIESMergeResult(
            merged=merged,
            conflict_count=conflict_count,
            merged_non_zero=merged_non_zero,
            total_elements=count,
        )
    
    @staticmethod
    def trim_vector(values: List[float], top_k: float) -> Tuple[List[float], int]:
        """
        Trim vector to top-K% magnitude values.
        
        Args:
            values: Input values.
            top_k: Fraction of values to keep (0-1).
        
        Returns:
            Tuple of (trimmed values, count of kept non-zero values).
        """
        if not values:
            return [], 0
        if top_k <= 0:
            return [0.0] * len(values), 0
        if top_k >= 1:
            kept = sum(1 for v in values if v != 0)
            return values[:], kept
        
        count = len(values)
        k_count = max(1, int(count * top_k))
        magnitudes = [abs(v) for v in values]
        sorted_mags = sorted(magnitudes, reverse=True)
        threshold = sorted_mags[min(k_count - 1, count - 1)]
        
        trimmed = values[:]
        kept = 0
        for i in range(count):
            if magnitudes[i] < threshold:
                trimmed[i] = 0.0
            elif trimmed[i] != 0:
                kept += 1
        
        return trimmed, kept
    
    @staticmethod
    def apply_dare_drop(
        values: List[float],
        drop_rate: float,
        rng: SeededGenerator,
    ) -> List[float]:
        """
        Apply DARE dropout: randomly zero values with rescaling.
        
        Values that survive are scaled by 1/(1-drop_rate) to maintain
        expected magnitude.
        """
        keep_probability = max(0.0, min(1.0, 1.0 - drop_rate))
        if keep_probability <= 0:
            return [0.0] * len(values)
        
        scale = 1.0 / keep_probability
        result = values[:]
        for i in range(len(result)):
            if rng.next_float() > keep_probability:
                result[i] = 0.0
            else:
                result[i] *= scale
        
        return result
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    @staticmethod
    def _load_adapter(directory: Path) -> AdapterPayload:
        """Load PEFT LoRA adapter from directory."""
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"
        
        if not config_path.exists():
            raise MergeError(f"Missing adapter config at {config_path}")
        if not weights_path.exists():
            raise MergeError(f"Missing adapter weights at {weights_path}")
        
        with open(config_path) as f:
            peft_config = json.load(f)
        
        # Extract metadata
        base_model_id = peft_config.get("base_model_name_or_path", "unknown")
        rank = peft_config.get("r", peft_config.get("lora_rank", 16))
        alpha = peft_config.get("lora_alpha", rank)
        scale = alpha / rank if rank > 0 else 1.0
        
        # Load weights
        weights = mx.load(str(weights_path))
        
        # Parse LoRA weights
        modules: Dict[str, LoRAModuleWeights] = {}
        a_by_module: Dict[str, mx.array] = {}
        b_by_module: Dict[str, mx.array] = {}
        a_key_by_module: Dict[str, str] = {}
        b_key_by_module: Dict[str, str] = {}
        extra_weights: Dict[str, mx.array] = {}
        
        for key, value in weights.items():
            # PEFT format: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            if ".lora_A." in key or ".lora_a." in key:
                # Extract module path
                module_path = key.rsplit(".lora_", 1)[0]
                a_by_module[module_path] = value
                a_key_by_module[module_path] = key
            elif ".lora_B." in key or ".lora_b." in key:
                module_path = key.rsplit(".lora_", 1)[0]
                b_by_module[module_path] = value
                b_key_by_module[module_path] = key
            else:
                extra_weights[key] = value
        
        # Pair A and B matrices
        module_keys = set(a_by_module.keys()) & set(b_by_module.keys())
        if not module_keys:
            raise MergeError(f"No LoRA A/B pairs found in {directory}")
        
        for key in module_keys:
            modules[key] = LoRAModuleWeights(a=a_by_module[key], b=b_by_module[key])
        
        return AdapterPayload(
            directory=directory,
            base_model_id=base_model_id,
            rank=rank,
            scale=scale,
            modules=modules,
            a_key_by_module=a_key_by_module,
            b_key_by_module=b_key_by_module,
            extra_weights=extra_weights,
        )
    
    @staticmethod
    def _resolve_drop_rates(
        payloads: List[AdapterPayload],
        config: Config,
    ) -> Tuple[List[float], List[AdapterSparsitySummary]]:
        """Determine drop rates for each adapter (DARE-TIES only)."""
        if config.strategy != Strategy.DARE_TIES:
            return [0.0] * len(payloads), []
        
        drop_rates: List[float] = []
        summaries: List[AdapterSparsitySummary] = []
        
        for payload in payloads:
            # Simple sparsity estimation
            total_params = 0
            zero_params = 0
            for weights in payload.modules.values():
                a_flat = weights.a.reshape(-1).tolist()
                b_flat = weights.b.reshape(-1).tolist()
                total_params += len(a_flat) + len(b_flat)
                zero_params += sum(1 for v in a_flat if abs(v) < 1e-8)
                zero_params += sum(1 for v in b_flat if abs(v) < 1e-8)
            
            effective_sparsity = zero_params / total_params if total_params > 0 else 0.0
            # Recommended drop rate: complement of sparsity (more sparse = lower drop)
            recommended = max(0.0, min(0.9, 0.7 - effective_sparsity * 0.5))
            applied = config.drop_rate if config.drop_rate is not None else recommended
            
            drop_rates.append(applied)
            summaries.append(AdapterSparsitySummary(
                adapter_path=str(payload.directory),
                recommended_drop_rate=recommended,
                applied_drop_rate=applied,
                effective_sparsity=effective_sparsity,
            ))
        
        return drop_rates, summaries
    
    @staticmethod
    def _merge_matrices(
        matrices: List[Any],
        drop_rates: List[float],
        config: Config,
        seed: int,
        module: str,
        backend: "Backend | None" = None,
    ) -> MergeMatrixResult:
        """Merge multiple weight matrices using TIES or DARE-TIES."""
        b = backend or get_default_backend()

        if not matrices:
            raise MergeError(f"No matrices provided for {module}")

        first = matrices[0]
        shape = first.shape
        dtype = first.dtype

        for m in matrices[1:]:
            if m.shape != shape:
                raise MergeError(f"Shape mismatch in module {module}")

        # Flatten to lists (matrices are MLX arrays from mx.load)
        vectors = [b.to_numpy(b.reshape(b.astype(m, "float32"), (-1,))).tolist() for m in matrices]

        # Apply DARE dropout if needed
        if config.strategy == Strategy.DARE_TIES:
            for i in range(len(vectors)):
                drop_rate = drop_rates[i] if i < len(drop_rates) else 0.0
                if drop_rate > 0:
                    stable_seed = LoRAAdapterMerger._stable_seed(seed, module, i)
                    rng = SeededGenerator(stable_seed)
                    vectors[i] = LoRAAdapterMerger.apply_dare_drop(vectors[i], drop_rate, rng)

        # Trim
        trimmed_vectors: List[List[float]] = []
        trimmed_non_zero = 0
        trimmed_total = 0

        for vector in vectors:
            trimmed, kept = LoRAAdapterMerger.trim_vector(vector, config.ties_top_k)
            trimmed_vectors.append(trimmed)
            trimmed_non_zero += kept
            trimmed_total += len(trimmed)

        # TIES merge
        merge_result = LoRAAdapterMerger.ties_merge(trimmed_vectors)

        # Reshape back to original shape using Backend
        merged = b.astype(b.reshape(b.array(merge_result.merged), shape), str(dtype))
        b.eval(merged)

        return MergeMatrixResult(
            merged=merged,
            conflict_count=merge_result.conflict_count,
            merged_non_zero=merge_result.merged_non_zero,
            trimmed_non_zero=trimmed_non_zero,
            trimmed_total=trimmed_total,
            total_elements=merge_result.total_elements,
        )
    
    @staticmethod
    def _stable_seed(base: int, token: str, index: int) -> int:
        """Generate stable seed from base, token, and index (FNV-1a variant)."""
        h = 0xCBF29CE484222325
        for byte in token.encode('utf-8'):
            h ^= byte
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        h ^= base
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        h ^= index
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h
    
    @staticmethod
    def _write_merged_adapter(
        source_directory: Path,
        output_directory: Path,
        weights: Dict[str, mx.array],
        base_model_id: str,
    ) -> None:
        """Write merged adapter to output directory."""
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Copy non-weight files from source
        for item in source_directory.iterdir():
            if item.name in ("adapter_model.safetensors", "adapter_config.json", "lap_manifest.json"):
                continue
            dest = output_directory / item.name
            if not dest.exists():
                if item.is_file():
                    dest.write_bytes(item.read_bytes())
        
        # Save merged weights
        output_weights = output_directory / "adapter_model.safetensors"
        mx.save_safetensors(str(output_weights), weights)
        
        # Update config
        config_path = source_directory / "adapter_config.json"
        with open(config_path) as f:
            config_data = json.load(f)
        
        config_data["base_model_name_or_path"] = base_model_id
        
        output_config = output_directory / "adapter_config.json"
        with open(output_config, 'w') as f:
            json.dump(config_data, f, indent=2)
