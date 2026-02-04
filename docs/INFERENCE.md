# Inference Infrastructure

ModelCypher's inference subsystem provides unified generation, entropy-aware monitoring, adapter pooling, and backend-agnostic execution.

Notes:
- In this repo, run commands as `poetry run mc ...`.
- For machine-readable output when scripting the CLI, prefer `mc --ai ...` (forces JSON output and suppresses prompts/logs).

## Architecture Overview

```
src/modelcypher/adapters/inference_engine.py   # Unified inference engine (Backend-driven)
src/modelcypher/core/domain/inference/
├── __init__.py            # Public API and exports
├── types.py               # Shared dataclasses and enums
├── activation_stream.py   # Activation capture stream
├── adapter_pool.py        # Memory-aware adapter management
├── entropy_dynamics.py    # Entropy tracking and conflict analysis
```

## Backend Selection

Inference uses the default backend. Entry points should initialize it before use:

```python
from modelcypher.backends import initialize_default_backend
initialize_default_backend()
```

To force a specific backend, set `MC_BACKEND` or `MODELCYPHER_BACKEND` to a backend key.
Use `mc system probe backends` to list available keys on the current machine.

## InferenceEngine

The unified engine loads models and runs generation through the backend abstraction.

```python
from modelcypher.adapters.inference_engine import InferenceEngine

env = InferenceEngine()
result = env.run(
    model="/path/to/model",
    prompt="Hello, world!",
    adapter=None,
    security_scan=False,
)
print(result.response)
```

### Entropy-Aware Inference

```python
result = env.run_with_entropy(
    model="/path/to/model",
    prompt="Explain geodesics.",
    uncertainty_mode="human_in_loop",
)
print(result.entropy_summary.mean_entropy)
```

## Adapter Pool

Memory-aware adapter hot-swapping with LRU eviction.

```python
from modelcypher.core.domain.inference.adapter_pool import (
    AdapterPool,
    SystemMemoryManager,
    AdapterPreloadPriority,
)
import uuid

pool = AdapterPool(memory_manager=SystemMemoryManager())

async def load_adapter(path: str) -> None:
    ...

async def unload_adapter() -> None:
    ...

await pool.register_model("model-123", load_adapter, unload_adapter)

adapter_id = uuid.uuid4()
await pool.preload(adapter_id, "/path/to/adapter1", AdapterPreloadPriority.HIGH)

result = await pool.swap(adapter_id, model_id="model-123")
print(f"Swap took {result.swap_duration_ms:.1f}ms, cache_hit={result.was_cache_hit}")
```

## Security Scan Metrics

CLI inference (`poetry run mc infer run --security-scan`) returns a `SecurityScanSummary`
with `anomaly_count`, `max_anomaly_score`, `avg_delta`, and `disagreement_rate`.
Local inference currently returns zeroed values until the scan pipeline is wired
through the backend abstraction.

## CLI Integration

```bash
# Run inference with adapter
poetry run mc infer run \
    --model /path/to/model \
    --adapter /path/to/adapter \
    --prompt "Hello, world!"

# Run a prompt suite
poetry run mc infer suite \
    --model /path/to/model \
    --suite /path/to/suite.jsonl
```

## Troubleshooting

### Backend Not Detected

```bash
poetry run mc system status
poetry run mc system probe backends
```

If no backend is available, install the appropriate backend dependencies for your
platform and re-run the probe.
