# Contributing to ModelCypher

We welcome contributions! As a project built on rigorous high-dimensional geometry and clean architecture, we have high standards for code quality and mathematical correctness.

## Development Setup

1.  **Environment**: We use `uv` for dependency management.
    ```bash
    uv sync
    source .venv/bin/activate
    ```

2.  **Formatting**: We use `ruff` for linting and formatting.
    ```bash
    ruff check .
    ruff format .
    ```

3.  **Testing**:
    ```bash
    pytest src/modelcypher/tests
    ```
    *Note: Some tests require an Apple Silicon GPU.*

## Architecture Rules

-   **Respect the Hexagon**: Do not import `adapters` into the `domain`. If you need external functionality, define a `Port` interface and implement it in `adapters`.
-   **Type Hints**: All code must be fully type-hinted.
-   **Math**: If adding geometric operations, include comments referencing the mathematical basis (e.g., "Procrustes Analysis", "CKA").

## Pull Request Process

1.  Create a feature branch.
2.  Add tests for your new feature.
3.  Ensure `ruff` passes.
4.  Submit a PR with a clear description of the geometric or architectural change.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
