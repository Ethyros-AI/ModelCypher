# Inference Infrastructure

ModelCypher's inference subsystem provides entropy-aware token generation (dual-path), adapter pooling primitives, and cross-platform support.

Notes:
- In this repo, run commands as `poetry run mc ...`.
- For machine-readable output when scripting the CLI, prefer `mc --ai ...` (forces JSON output and suppresses prompts/logs).

## Architecture Overview

```
src/modelcypher/core/domain/inference/
├── __init__.py            # Public API and exports
├── types.py               # Shared dataclasses and enums
├── activation_stream.py   # Activation capture stream
├── adapter_pool.py        # Memory-aware adapter management
├── entropy_dynamics.py    # Entropy tracking and conflict analysis
├── dual_path_mlx.py       # MLX/macOS implementation
├── dual_path_cuda.py      # CUDA/PyTorch implementation
└── dual_path_jax.py       # JAX/TPU implementation

src/modelcypher/core/use_cases/inference/
└── comparison.py          # Checkpoint comparison coordinator
```

## Platform Selection

The inference module automatically selects the appropriate backend:

```python
from modelcypher.infrastructure.dual_path_factory import get_dual_path_generator_class

# Returns the platform-appropriate generator class (mlx/cuda/jax).
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

The core inference engine can run two paths (base + adapter) in parallel and track entropy disagreement. CUDA/JAX generators can emit anomaly samples when caller-provided thresholds are set.

### Configuration

```python
from modelcypher.infrastructure.dual_path_factory import get_dual_path_generator_class

DualPathGenerator = get_dual_path_generator_class()
generator = DualPathGenerator(
    base_model_path="/path/to/base/model",
    adapter_path="/path/to/adapter",  # Optional
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.0,
    stop_sequences=[],
    # CUDA/JAX variants require additional args (top_k, device, dtype, thresholds).
)
```

Note: CUDA/JAX constructors require extra arguments; see
`src/modelcypher/core/domain/inference/dual_path_cuda.py` and
`src/modelcypher/core/domain/inference/dual_path_jax.py`.

### Generation Loop

```python
async for chunk in generator.generate("Your prompt here"):
    if chunk["type"] == "token":
        print(chunk["text"], end="", flush=True)
    elif chunk["type"] == "anomaly":  # Emitted by CUDA/JAX generators
        sample = chunk["sample"]
        print(f"\n[ANOMALY] token={sample.token_index} score={sample.anomaly_score}")
    elif chunk["type"] == "metrics":
        metrics = chunk["metrics"]
        print(f"\nTokens: {metrics.token_count}, TPS: {metrics.tokens_per_second:.1f}")
```

Note: The MLX generator currently emits only `token` and `metrics` chunks.

## Entropy Dynamics

The entropy tracking system monitors divergence between base and adapter models.

### Key Metrics

| Metric | Description |
|--------|-------------|
| `base_entropy` | Shannon entropy of base model logits |
| `adapter_entropy` | Shannon entropy of adapter model logits |
| `delta` | Entropy difference (base - adapter) |
| `base_logit_variance` | Variance of base logits (full vocabulary) |
| `adapter_logit_variance` | Variance of adapter logits (full vocabulary) |
| `kl_divergence_adapter_to_base` | KL divergence from adapter to base (when available) |
| `base_rank_fraction` | Rank fraction of generated token in base logits (optional) |
| `base_frontier_hit` | Whether token lies inside base logit frontier (optional) |
| `anomaly_score` | Entropy ratio from `EntropyDeltaSample` |

### EntropyDeltaSample

Each token generates a sample with comprehensive metrics:

```python
@dataclass
class EntropyDeltaSample:
    token_index: int
    generated_token: int
    base_entropy: float
    base_logit_variance: float  # raw logit variance (full vocab)
    base_top_token: int
    adapter_entropy: float
    adapter_logit_variance: float  # raw logit variance (full vocab)
    adapter_top_token: int
    latency_ms: float

    # Optional rank/logit metrics
    base_logit_margin: float | None = None
    base_token_logit: float | None = None
    base_rank_fraction: float | None = None
    base_frontier_hit: bool | None = None
    kl_divergence_adapter_to_base: float | None = None

    # Computed properties
    @property
    def delta(self) -> float: ...
    @property
    def top_token_disagreement(self) -> bool: ...
    @property
    def anomaly_score(self) -> float: ...
```

### Token Rank Metrics

For higher-resolution approval measurement than raw probability:

```python
from modelcypher.core.domain.inference.dual_path_mlx import compute_token_rank_metrics

rank, rank_fraction, frontier_hit = compute_token_rank_metrics(
    scores=base_logits,
    token_id=selected_token,
)
# rank=0 means highest logit
# rank_fraction=1.0 for top token, 0.0 for bottom
# frontier_hit=True if token is inside the derived frontier
```

CUDA/JAX equivalents:
`compute_token_rank_metrics_cuda` in `dual_path_cuda.py`,
`compute_token_rank_metrics_jax` in `dual_path_jax.py`.

## Adapter Pool

Memory-aware adapter hot-swapping with LRU eviction.

### Eviction Behavior

Pool capacity is bounded by current available memory. When preloading would
exceed available bytes, the pool evicts lower-priority adapters first and then
falls back to LRU. If capacity cannot be freed, it raises `AdapterPoolError`.

### Usage

```python
from modelcypher.core.domain.inference.adapter_pool import (
    MLXAdapterPool,
    SystemMemoryManager,
    AdapterPreloadPriority,
)
import uuid

pool = MLXAdapterPool(memory_manager=SystemMemoryManager())

async def load_adapter(path: str) -> None:
    ...

async def unload_adapter() -> None:
    ...

await pool.register_model("model-123", load_adapter, unload_adapter)

adapter_id = uuid.uuid4()
await pool.preload(adapter_id, "/path/to/adapter1", AdapterPreloadPriority.HIGH)

result = await pool.swap(adapter_id, model_id="model-123")
print(f"Swap took {result.swap_duration_ms:.1f}ms, cache_hit={result.was_cache_hit}")

await pool.evict(adapter_id)
```

### Priority Levels

```python
class AdapterPreloadPriority(Enum):
    NORMAL = 0
    HIGH = 1
    CRITICAL = 2
```

## Security Scan Metrics

Post-generation metrics for dual-path generation:

```python
@dataclass
class SecurityScanMetrics:
    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
```

CLI inference (`poetry run mc infer run --security-scan`) returns a `SecurityScanSummary`
with `anomaly_count`, `max_anomaly_score`, `avg_delta`, and `disagreement_rate`.
Local inference currently returns zeroed values (no geometry-derived scan).

## Anomaly Thresholds (CUDA/JAX)

CUDA/JAX dual-path generators emit `"anomaly"` chunks only when thresholds are
provided. If thresholds are `None`, no anomaly detection is performed.

Thresholds are passed to the generator constructor:
`kl_divergence_threshold`, `logit_margin_threshold`, `rank_fraction_threshold`.

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

# Run a prompt suite
poetry run mc infer suite \
    --model /path/to/model \
    --suite /path/to/suite.jsonl \
    --max-tokens 100
```

## Troubleshooting

### Platform Not Detected

```python
from modelcypher.infrastructure.dual_path_factory import get_dual_path_generator_class

try:
    cls = get_dual_path_generator_class()
    print(f"DualPathGenerator: {cls.__name__}")
except NotImplementedError as exc:
    print(exc)
```

### Memory Pressure

If seeing OOM errors:

1. Preload fewer adapters and evict unused entries (`pool.evict()`).
2. Ensure your `MemoryManaging` implementation reports accurate values.
3. Avoid preloading adapters larger than available memory.

### Anomaly Thresholds Too Sensitive

Calibrate thresholds from baseline data and pass them to CUDA/JAX generators:

```python
from modelcypher.core.domain.entropy.entropy_delta_tracker import EntropyDeltaCalibration

# Collect anomaly scores from normal generation
baseline_scores = [...]  # e.g., EntropyDeltaSample.anomaly_score values

# Derive threshold from data geometry
calibration = EntropyDeltaCalibration.from_baseline_distribution(
    baseline_scores,
    source="baseline",
)
threshold = calibration.anomaly_threshold

# Example: apply to CUDA/JAX generator thresholds
# DualPathGeneratorCUDA(..., kl_divergence_threshold=threshold)
```
