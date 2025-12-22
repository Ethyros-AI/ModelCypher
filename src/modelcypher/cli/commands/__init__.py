"""CLI command modules.

This package contains modular CLI command implementations following hexagonal architecture.
Each submodule defines a Typer app that is registered with the main app.

Structure:
    commands/
    ├── geometry/      # Geometry analysis commands (mc geometry ...)
    │   ├── emotion.py
    │   ├── refinement.py
    │   └── ...
    └── ...
"""
