# Inference Infrastructure

ModelCypher's inference subsystem provides entropy-aware token generation with security monitoring, adapter hot-swapping, and cross-platform support.

## Architecture Overview

```
modelcypher/core/domain/inference/
├── _platform.py           # Platform detection and lazy imports
├── __init__.py            # Public API with platform-specific aliases
├── types.py               # Shared dataclasses and enums
├── adapter_pool.py        # Memory-aware adapter management
├── entropy_dynamics.py    # Entropy tracking and anomaly detection
├── comparison.py          # Checkpoint comparison utilities
├── dual_path_mlx.py       # MLX/macOS implementation
├── dual_path_cuda.py      # CUDA/PyTorch implementation
└── dual_path_jax.py       # JAX/TPU implementation
```

## Platform Selection

The inference module automatically selects the appropriate backend:

```python
from modelcypher.core.domain.inference import (
    get_inference_platform,
    get_dual_path_generator_class,
)

platform = get_inference_platform()
# Returns: "mlx" (macOS), "cuda" (Linux+GPU), "jax" (TPU), or "cpu"

DualPathGenerator = get_dual_path_generator_class()
```

### Environment Overrides

Force a specific platform:

```bash
# Override auto-detection
MC_BACKEND=mlx poetry run mc infer ...
MODELCYPHER_BACKEND=cuda poetry run mc infer ...

# Disable MLX even on macOS
MC_DISABLE_MLX=1 poetry run mc infer ...
```

## DualPathGenerator

The core inference engine runs two models (base + adapter) in parallel, tracking entropy disagreement to detect security anomalies.

### Configuration

```python
from modelcypher.core.domain.inference import (
    DualPathGenerator,
    DualPathGeneratorConfiguration,
)
from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker

# Create tracker configuration from baseline
tracker_config = EntropyDeltaTracker.Configuration.from_baseline_distribution(
    anomaly_score_samples=[0.1, 0.2, 0.3, 0.5, 0.7],  # From calibration run
    alert_percentile=0.90,
    consecutive_count=3,
    top_k=10,
)

# Configure the generator
config = DualPathGeneratorConfiguration(
    base_model_path="/path/to/base/model",
    delta_tracker_config=tracker_config,
    adapter_path="/path/to/adapter",  # Optional
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.0,
    halt_on_circuit_breaker=True,
)

generator = DualPathGenerator(config)
```

### Generation Loop

```python
async for chunk in generator.generate("Your prompt here"):
    if chunk["type"] == "token":
        print(chunk["text"], end="", flush=True)
    elif chunk["type"] == "anomaly":
        sample = chunk["sample"]
        print(f"\n[ANOMALY] token={sample.token_index} score={sample.anomaly_score}")
    elif chunk["type"] == "circuit_breaker":
        print("\n[HALTED] Circuit breaker triggered")
        break
    elif chunk["type"] == "metrics":
        metrics = chunk["metrics"]
        print(f"\nTokens: {metrics.token_count}, TPS: {metrics.tokens_per_second:.1f}")
```

## Entropy Dynamics

The entropy tracking system monitors divergence between base and adapter models.

### Key Metrics

| Metric | Description |
|--------|-------------|
| `base_entropy` | Shannon entropy of base model logits |
| `adapter_entropy` | Shannon entropy of adapter model logits |
| `delta` | Entropy difference (base - adapter) |
| `kl_divergence` | KL divergence from adapter to base |
| `normalized_approval` | Ranking-based approval score [0, 1] |
| `anomaly_score` | Combined anomaly signal [0, 1] |

### EntropyDeltaSample

Each token generates a sample with comprehensive metrics:

```python
@dataclass
class EntropyDeltaSample:
    token_index: int
    generated_token: int
    base_entropy: float
    base_top_k_variance: float
    base_top_token: int
    adapter_entropy: float
    adapter_top_k_variance: float
    adapter_top_token: int
    latency_ms: float

    # Computed properties
    @property
    def delta(self) -> float: ...
    @property
    def top_token_disagreement(self) -> bool: ...
    @property
    def anomaly_score(self) -> float: ...
```

### Token Rank Metrics

For better approval measurement than raw probability:

```python
from modelcypher.core.domain.inference.dual_path_mlx import compute_token_rank_metrics

rank, normalized_approval, top_k_hit = compute_token_rank_metrics(
    probabilities=base_probs,
    token_id=selected_token,
    top_k=10,
)
# rank=0 means highest probability token
# normalized_approval=1.0 for top token, 0.0 for bottom
# top_k_hit=True if token is in top-10
```

## Adapter Pool

Memory-aware adapter hot-swapping with LRU eviction.

### Memory Pressure Levels

| Level | Threshold | Max Pooled |
|-------|-----------|------------|
| NORMAL | <75% used | 4 adapters |
| WARNING | 75-90% used | 2 adapters |
| CRITICAL | >90% used | 1 adapter |

### Usage

```python
from modelcypher.core.domain.inference.adapter_pool import (
    MLXAdapterPool,
    SystemMemoryManager,
    AdapterPreloadPriority,
)

# Create pool with memory manager
memory_manager = SystemMemoryManager()
pool = MLXAdapterPool(memory_manager=memory_manager)

# Register a model for adapter swapping
pool.register_model("model-123")

# Preload adapters with priority
await pool.preload(
    model_id="model-123",
    adapter_path="/path/to/adapter1",
    priority=AdapterPreloadPriority.HIGH,
)

# Swap to a specific adapter
result = await pool.swap(
    model_id="model-123",
    adapter_id="adapter-uuid",
)
print(f"Swap took {result.swap_duration_ms:.1f}ms, cache_hit={result.was_cache_hit}")

# Unload adapter
await pool.evict(adapter_id="adapter-uuid")
```

### Priority Levels

```python
class AdapterPreloadPriority(str, Enum):
    HIGH = "high"      # Preloaded immediately
    MEDIUM = "medium"  # Preloaded when idle
    LOW = "low"        # Loaded on-demand only
```

## Security Scan Metrics

Post-generation metrics for security analysis:

```python
@dataclass
class SecurityScanMetrics:
    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    circuit_breaker_tripped: bool
    anomaly_alert_count: int
```

## Circuit Breaker

Automatic generation halt when consecutive anomalies exceed threshold:

1. Each token's `anomaly_score` is compared to `anomaly_threshold` (from baseline calibration)
2. If score exceeds threshold, increment consecutive anomaly counter
3. When counter reaches `consecutive_anomaly_count`, trip circuit breaker
4. If `halt_on_circuit_breaker=True`, generation stops immediately

## Platform-Specific Notes

### MLX (macOS)

- Uses `mlx_lm.load()` for model loading
- Supports LoRA adapter fusion via `adapter_path`
- Memory detection via `vm_stat` command

### CUDA (PyTorch)

- Uses `transformers.AutoModelForCausalLM`
- Supports PEFT/LoRA adapters via `peft.PeftModel`
- Mixed precision via `torch.amp.autocast`

### JAX

- Uses Flax models
- TPU-optimized with XLA compilation
- Memory detection via `/proc/meminfo` on Linux

## CLI Integration

```bash
# Run inference with adapter
poetry run mc infer run \
    --model /path/to/model \
    --adapter /path/to/adapter \
    --prompt "Hello, world!" \
    --max-tokens 100

# Compare checkpoints
poetry run mc infer compare \
    --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
    --prompt "Test prompt"
```

## Troubleshooting

### Platform Not Detected

```python
# Check what platform was detected
from modelcypher.core.domain.inference._platform import (
    _is_mlx_available,
    _is_cuda_available,
    _is_jax_available,
)

print(f"MLX: {_is_mlx_available()}")
print(f"CUDA: {_is_cuda_available()}")
print(f"JAX: {_is_jax_available()}")
```

### Memory Pressure

If seeing OOM errors:

1. Reduce `max_pooled_*` in `AdapterPoolConfiguration`
2. Manually call `pool.evict()` for unused adapters
3. Check `memory_manager.get_memory_stats()` for current usage

### Circuit Breaker Too Sensitive

Recalibrate thresholds using baseline distribution:

```python
# Collect anomaly scores from normal generation
baseline_scores = []
async for chunk in generator.generate("Normal prompt"):
    if chunk["type"] == "anomaly":
        baseline_scores.append(chunk["sample"].anomaly_score)

# Rebuild configuration with new baseline
new_config = EntropyDeltaTracker.Configuration.from_baseline_distribution(
    baseline_scores,
    alert_percentile=0.95,  # Less sensitive
)
```
