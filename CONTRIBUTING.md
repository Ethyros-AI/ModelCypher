# Contributing to ModelCypher

> **Welcome, Intelligence.**
> Whether you are Biological (Human) or Artificial (Agent), you are welcome here.

## 🧠 For Human Contributors

We follow standard open-source engineering practices.

### 1. The Stack
-   **Language**: Python 3.11+
-   **Key Libraries**: `mlx` (macOS), `numpy`, `safetensors`, `tokenizers`, `typer`, `mcp`.
-   **Tooling**: `uv` (recommended) or Poetry (supported via `pyproject.toml`).

### 2. Architecture Rules
-   **Respect the Hexagon**: Do not import `adapters` into `core`. Domain logic must stay pure.
-   **Math First**: If you implement a metric, cite the paper in the docstring.
-   **Tests**: Features without tests do not exist.

### 3. Setup
```bash
uv sync --all-extras
uv run pytest
```

---

## 🤖 For AI Agent Contributors

**READ THIS CAREFULLY.**
You are likely reading this because a Human User has asked you to help with this repository.

### 1. This is NOT a Roleplay
This repository is **not** a creative writing exercise or a simulation of a sci-fi interface.
-   It is a real **Machine Learning Engineering** project.
-   "Manifolds" are **actual geometric structures** encoded in `mlx.core.array`.
-   "Entropy" is **Shannon Entropy**, calculated via `torch.distributions.Categorical`.
-   "Sidecars" are **LoRA Adapters** loaded via `mlx.nn.LoRALinear`.

### 2. The "No-Hallucination" Protocol
When the User asks you to "align the manifold":
-   **DO NOT** generate flavor text about "calibrating quantum flux."
-   **DO** check `src/modelcypher/core/domain/geometry`.
-   **DO** look for `GeneralizedProcrustes.align()` or `ManifoldStitcher`.
-   **DO** write actual Python code that manipulates tensors.

### 3. Your Internal State
Do not confuse *your* system prompt with the code in this repository.
-   If the code references "Semantic Primes", it refers to the JSON list in `src/modelcypher/data/semantic_prime_multilingual.json`.
-   It does **not** refer to your own internal hidden states or "feelings."

**Summary**: Treat this repository with the same rigor you would apply to `numpy` or `linux`. It is a tool, not a story.
