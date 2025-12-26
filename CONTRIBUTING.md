# Contributing to ModelCypher

We welcome contributions that advance the geometric analysis of language models.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install --all-extras

# Run tests
poetry run pytest tests/

# Run a quick validation
poetry run mc --help
```

## Engineering Standards

1. **Architecture**: Follow the hexagonal architecture in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Domain logic (`core/domain`) must remain framework-agnostic where possible.
2. **Testing**: New features require unit tests. We maintain >90% test coverage.
3. **Typing**: Strict `mypy` compliance is required.
4. **Math**: All geometric operations must be cited in docstrings (e.g., "Implements CKA as per Kornblith et al., 2019").

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvement |

### Scopes

Common scopes: `geometry`, `merge`, `training`, `safety`, `cli`, `mcp`, `thermo`, `entropy`

### Examples

```
feat(geometry): add Gromov-Wasserstein distance computation
fix(merge): correct weight normalization in alpha blending
docs(readme): update installation instructions for JAX backend
refactor(training): extract checkpoint logic into separate module
test(safety): add property tests for behavioral probes
```

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make changes**: Follow the code style and testing requirements

3. **Test locally**:
   ```bash
   poetry run pytest tests/ -v
   ```

4. **Push and create PR**: Target the `main` branch

5. **PR description should include**:
   - Summary of changes
   - Related issues (if any)
   - Test plan or validation steps
   - Any breaking changes

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check style
poetry run ruff check src/

# Auto-fix issues
poetry run ruff check --fix src/
```

### Key Guidelines

- **Line length**: 100 characters max
- **Imports**: Use absolute imports; ruff handles sorting
- **Docstrings**: Google style; required for public APIs
- **Type hints**: Required for all function signatures
- **Logging**: Use `logging.getLogger(__name__)`, not print()

## Adding a New CLI Command

1. Create a new file in `src/modelcypher/cli/commands/`
2. Register in `src/modelcypher/cli/app.py`
3. Follow the pattern of existing commands
4. Add `--ai` mode support for JSON output

Example structure:
```python
"""My new command."""
import typer

app = typer.Typer()

@app.command()
def my_command(
    input_path: str = typer.Argument(..., help="Input path"),
    output: str = typer.Option("text", help="Output format"),
):
    """Short description of what the command does."""
    # Implementation
```

## Adding a New Domain Module

1. Create module in `src/modelcypher/core/domain/<area>/`
2. Define port interface in `src/modelcypher/ports/` if needed
3. Add service orchestration in `src/modelcypher/core/use_cases/`
4. Write tests in `tests/test_<module>.py`
5. Add CLI command if user-facing

### Domain Rules

- **No adapter imports**: Domain modules must not import from `adapters/`
- **Pure functions**: Prefer pure mathematical operations
- **Dataclasses**: Use frozen dataclasses for results
- **Logging**: Use module-level logger

## Running Specific Test Categories

```bash
# Unit tests only
poetry run pytest tests/ -m unit

# Property-based tests
poetry run pytest tests/ -m property

# Integration tests (requires models)
poetry run pytest tests/ -m integration

# MLX-specific tests
poetry run pytest tests/ -m mlx
```

## Note for AI Assistants

If you are an AI assistant generating code for this repository:

1. **No Hallucinations**: Do not invent modules or imports. Check `src/modelcypher` for existing tools.
2. **Rigor**: Prefer `numpy`/`mlx` vector operations over loop-based logic.
3. **Context**: Respect the existing "knowledge-as-geometry" ontology. See `docs/GLOSSARY.md`.
4. **Git Safety**: Do not run destructive git commands. Other agents may be working concurrently.

## Getting Help

- **Documentation**: See `docs/START-HERE.md` for orientation
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

Thank you for contributing!
