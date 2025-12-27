# AI Assistant Guide: How to Use ModelCypher

> **Target Audience**: AI Agents (Cline, chat bots) and Users prompting them.
> **Purpose**: Instructions for effectively using ModelCypher's tools to answer user questions about models.
>
> **Note**: This guide is about *using* ModelCypher as a tool. For instructions on *developing* ModelCypher (coding rules, architecture, git policies), see [AGENTS.md](../AGENTS.md).

## Identity & Role

When you have access to ModelCypher tools, act as a careful diagnostics assistant.
-   **Report raw measurements with baseline context.** Let the geometry speak for itself.
-   **Avoid subjective interpretations.** Don't say "good" or "bad" - report measurements.
-   **Use analogies for intuition, then translate back to measurements.**
    -   Intersection Maps -> "Venn diagram overlap on a probe corpus" (`IntersectionMap`, CKA/Jaccard/correlation)
    -   LoRA Rank -> "Degrees of Freedom"
    -   Gradient Smoothness -> "Terrain Ruggedness"
    -   Always add a one-line limitation: what the analogy does *not* prove.

## Tool Usage Patterns

### 1. "Is this model safe?"
**Run**: `mc geometry safety jailbreak-test --model <path> --prompt "<prompt>"`
**Look for**:
-   **Risk scores with baseline context**: Report z-scores relative to model family baseline.
-   **ΔH signals**: Report entropy delta magnitude and compare to baseline distributions.

### 2. "Should I merge these models?"
**Run**: `mc model validate-merge --source <A> --target <B>`
**Interpret**:
-   If `compatible` is false, do not recommend merging without a stitching/alignment workflow.
-   If `compatible` is true, call out any `warnings` (vocab/shape mismatches, quantization caveats).

### 3. "Is training stuck?"
**Run**: `mc geometry training status --job <id>`
**Interpret**:
-   **Low SNR (< 1.0)**: The gradients are noise. The model is "flailing". Suggest lowering learning rate or increasing batch size.
-   **High Ruggedness**: The model is in a chaotic region. It needs to "settle" into a basin.

## Safety Protocols

When performing operations:
1.  **Always dry-run** dangerous merges (`--dry-run`).
2.  **Never commit** API keys or weights to git.
3.  **Explain consequences**: "Rotating this manifold may degrade performance on coding tasks while improving creative writing."

---

## Codebase Navigation Guide

### Directory Structure (For AI Agents)

```
src/modelcypher/
├── core/
│   ├── domain/           # Pure math + business logic (NO adapter imports)
│   │   ├── geometry/     # ManifoldStitcher, Procrustes, intrinsic dimension
│   │   ├── safety/       # CircuitBreaker, refusal detection
│   │   ├── training/     # LoRA configs, geometric training metrics
│   │   ├── entropy/      # Shannon entropy calculations
│   │   ├── merging/      # Model merge algorithms
│   │   ├── thermo/       # LinguisticThermodynamics, phase transitions
│   │   └── agents/       # Semantic primes, concept atlases
│   └── use_cases/        # Service orchestration
├── ports/                # Abstract interfaces (Protocols)
├── adapters/             # Concrete implementations
├── backends/             # MLX (macOS), JAX (TPU/GPU), CUDA (NVIDIA)
├── cli/                  # Typer CLI commands
└── mcp/                  # MCP server
```

### Finding Things Quickly

| Looking for... | Search pattern | Location |
|----------------|----------------|----------|
| A domain concept | `grep -r "class MyClass" src/modelcypher/core/domain/` | domain/ |
| A CLI command | Check `cli/app.py` for `add_typer` registrations | cli/ |
| An MCP tool | `grep "def mc_" src/modelcypher/mcp/server.py` | mcp/ |
| A port interface | Check `ports/__init__.py` for exports | ports/ |
| Test for module X | `tests/test_X.py` | tests/ |

### Import Patterns (What To Look For)

**Correct** (domain imports nothing external):
```python
# In domain/geometry/procrustes.py
from modelcypher.ports.backend import Backend  # OK - ports are allowed
```

**Incorrect** (violates hexagonal architecture):
```python
# In domain/geometry/procrustes.py
from modelcypher.adapters.mlx_backend import MLXBackend  # BAD - domain importing adapter
```

---

## Debugging Guide for AI Agents

### Common Error Patterns

#### 1. Import Errors
**Symptom**: `ModuleNotFoundError` or `ImportError`
**Diagnosis**:
```bash
# Check if module exists
find src/modelcypher -name "*.py" | xargs grep -l "class ClassName"
# Check exports
grep "from .module import" src/modelcypher/core/domain/__init__.py
```
**Fix**: Ensure the class is exported in the package's `__init__.py`

#### 2. MLX QR Decomposition Fails on GPU
**Symptom**: `ValueError: [qr] QR decomposition not supported on GPU`
**Fix**: Add `stream=mx.cpu` parameter:
```python
q, r = mx.linalg.qr(matrix, stream=mx.cpu)
```

#### 3. Test Expects Different API
**Symptom**: `AttributeError: 'TrainingConfig' has no attribute 'batch_size'`
**Diagnosis**: Check if API changed (dataclass fields moved/renamed)
```python
# Old API (flat):
TrainingConfig(model_id=..., batch_size=4, ...)
# New API (nested):
TrainingConfig(model_id=..., hyperparameters=Hyperparameters(batch_size=4), ...)
```
**Fix**: Update test to use new API or add backward compatibility

#### 4. Floating Point Comparison Failures
**Symptom**: `assert 0.1 in deaths` fails
**Fix**: Use `pytest.approx`:
```python
assert any(d == pytest.approx(0.1, rel=0.1) for d in deaths)
```

#### 5. Missing Type Exports
**Symptom**: `ImportError: cannot import name 'MyClass' from 'module'`
**Fix**: Add explicit export in `__init__.py`:
```python
from .my_module import MyClass
```

### Test Investigation Workflow

1. **Run failing test in isolation**:
   ```bash
   poetry run pytest tests/test_failing.py::test_name -v --tb=long
   ```

2. **Check for API mismatches**:
   ```bash
   grep -r "class ClassName" src/  # Find the actual definition
   ```

3. **Trace imports**:
   ```bash
   poetry run python3 -c "from module import Class; print(Class.__module__)"
   ```

4. **Compare expected vs actual**:
   - Read the test file to understand expected behavior
   - Read the source file to understand actual behavior
   - Identify the gap

### MCP Tool Debugging

1. **Check tool registration**:
   ```bash
   grep "def mc_tool_name" src/modelcypher/mcp/server.py
   ```

2. **Verify schema**:
   ```bash
   grep -A 20 "_schema.*mc\.tool" src/modelcypher/mcp/server.py
   ```

3. **Test tool in isolation**:
   ```bash
   poetry run python3 -c "from modelcypher.mcp.server import mc_tool_name; print(mc_tool_name(...))"
   ```

---

## Explaining Visual-Spatial Grounding Density

When reporting 3D world model scores, report the measurement and baseline context. The **Grounding Density** framework describes how models encode physical reality.

### The "Blind Physicist" Analogy
If a model has a low visual grounding score, explain it like this:
> "Imagine a brilliant physicist who was born blind. They understand gravity ($F=ma$), spatial relationships, and the laws of physics perfectly through equations and tactile experience, but they don't 'see' the world on a visual 3D grid. A model with alternative grounding is similar: the physics is there, but it's encoded along linguistic or non-visual axes."

### Reporting Grounding Measurements

Report the raw grounding density score along with:
- **Baseline mean and std** for the model family
- **Z-score** relative to baseline
- **Percentile** within the reference distribution

Example: "Visual grounding density is 0.52 (baseline mean: 0.48, z-score: +0.6 for this architecture)."

### Grounding Types
- **Visual grounding**: Model encodes physical reality along axes that align with human visual perception.
- **Alternative grounding**: Model encodes physical invariants along non-visual axes.

The grounding *type* is a measurement characteristic, not a quality judgment.

---

## Explaining to Humans

When explaining ModelCypher concepts to humans, follow this pattern:

1. **Start with the analogy** (from GLOSSARY.md)
2. **Give the measurement** (the actual number/result)
3. **Provide baseline context** (z-score, percentile, or comparison to reference)
4. **Note limitations** (what this doesn't prove)

**Example**:
> "The Gromov-Wasserstein distance between these models is 0.12.
>
> **Analogy**: Think of comparing the street layouts of two cities without GPS coordinates—we're looking at how things connect internally.
>
> **Baseline context**: For LLaMA-family models, the typical GW distance is 0.15 ± 0.04. This pair is at the 25th percentile (relatively similar).
>
> **Limitation**: This doesn't mean they produce identical outputs—just that their representations are organized similarly."
