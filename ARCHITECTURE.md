# ModelCypher Architecture

## Hexagonal Architecture

ModelCypher follows strict hexagonal (ports and adapters) architecture:

```
src/modelcypher/
├── core/
│   ├── domain/        # Pure math + business logic (NO adapter imports)
│   ├── ports/         # Abstract interfaces (ABCs, Protocols)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (filesystem, HF hub)
├── backends/          # Compute backends (MLX, CUDA stub)
├── cli/               # Typer CLI
└── mcp/               # Model Context Protocol server
```

**Dependency Rule**: Dependencies point inward. Domain depends on nothing external; adapters implement ports; CLI/MCP drive the application.

## Backend Abstraction

The `Backend` protocol in `ports/backend.py` abstracts array operations:

- **MLXBackend** (macOS): Production backend using Apple MLX
- **NumpyBackend** (tests): Deterministic testing without GPU
- **CUDABackend** (stub): Future Linux/CUDA support

Domain code uses the Backend protocol, not MLX directly:

```python
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.backend import Array, Backend

def my_geometry_function(backend: Backend, data: Array) -> Array:
    return backend.sqrt(backend.sum(data ** 2))
```

## MLX Constraint: Intentional Exceptions

The file `tests/test_no_mlx_in_domain.py` enforces backend abstraction. However, certain files require direct MLX access for infrastructure that cannot be abstracted:

| File | Reason |
|------|--------|
| `training/lora.py` | `mlx.nn.Module` for neural network layers |
| `training/checkpoints.py` | `mx.save_safetensors`, `mx.load` for I/O |
| `training/engine.py` | Training loop orchestration |
| `inference/dual_path.py` | `mlx_lm` for model loading |
| `merging/lora_adapter_merger.py` | SafeTensors file I/O |

These are tracked in `PENDING_MIGRATION` and represent infrastructure boundaries, not architecture violations. The Backend protocol abstracts mathematical operations; model loading and serialization remain platform-specific.

## Migration Progress

Run `pytest tests/test_no_mlx_in_domain.py -v` to see current status:
- 32+ domain files migrated to Backend protocol
- 8 infrastructure files with intentional MLX dependencies

## Platform Support

- **macOS** (Apple Silicon): Full support via MLX
- **Linux/CUDA**: Stub backend exists; contributions welcome
- **Testing**: NumpyBackend provides deterministic CPU execution
