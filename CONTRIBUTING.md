# Contributing to ModelCypher

We welcome contributions that advance the geometric analysis of language models. 

## Engineering Standards

1.  **Architecture**: Follow the hexagonal architecture in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Domain logic (`core/domain`) must remain framework-agnostic where possible.
2.  **Testing**: New features require unit tests. We maintain >90% test coverage.
3.  **Typing**: Strict `mypy` compliance is required.
4.  **Math**: All geometric operations must be cited in docstrings (e.g., "Implements CKA as per Kornblith et al., 2019").

## Development Setup

We recommend `uv` for dependency management:

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/
```

## Note for AI Assistants

If you are an AI assistant generating code for this repository:
1.  **No Hallucinations**: Do not invent modules or imports. Check `src/modelcypher` for existing tools.
2.  **Rigor**: Prefer `numpy`/`mlx` vector operations over loop-based logic.
3.  **Context**: Respect the existing "knowledge-as-geometry" ontology. See `docs/GLOSSARY.md`.
