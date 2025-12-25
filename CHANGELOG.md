# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-24

Initial public release of ModelCypher - a Python framework for measuring and experimenting with the geometry of representations in large language models.

### Added

#### Geometry Engine
- Manifold stitching with Procrustes analysis for aligning model representations
- Intrinsic dimension estimation using MLE and correlation dimension methods
- Topological fingerprints via persistent homology
- CKA (Centered Kernel Alignment) for representation similarity
- Gromov-Wasserstein distance computations
- Activation fingerprinting with dimension-level correlations
- Null space filtering and safety polytope analysis
- Interference prediction for model merging

#### Model Merging
- Unified geometric merge pipeline with service abstraction
- DARE (Drop And REscale) adapter sparsity
- DoRA (Weight-Decomposed Low-Rank Adaptation) support
- Cross-vocabulary merging with comparison-based approach
- Modular merge stages in separate subpackage
- MLX-based weight loading implementation

#### Safety & Monitoring
- Circuit breaker for detecting refusal basins and unstable trajectories
- Behavioral probes for safety auditing
- Entropy differential safety analysis
- Regime state monitoring and intervention triggers

#### Thermodynamics Engine
- Linguistic thermodynamics for activation energy landscapes
- Ridge-cross detection for phase transitions
- Temperature sweep analysis
- Linguistic calorimeter with mathematical invariants

#### Research Domains
- Moral geometry based on Haidt's Moral Foundations Theory
- Temporal topology for time-related representations
- Social geometry with validated research results
- Semantic primes inventory for cross-linguistic anchoring
- UnifiedAtlas concept inventory with multi-domain support

#### Backends
- MLX backend for Apple Silicon (macOS)
- JAX backend for Linux/TPU/GPU environments
- NumPy backend for testing and portability
- Dynamic backend selection based on platform

#### CLI (`mc` / `modelcypher`)
- `mc model probe` - Probe local models for geometric analysis
- `mc geometry training status` - Monitor training geometry
- `mc geometry safety circuit-breaker` - Safety checking
- `mc geometry spatial` - Spatial geometry analysis
- `mc thermo measure/ridge-detect/phase/sweep` - Thermodynamics commands
- `mc adapter blend/ensemble` - Adapter management
- `mc research taxonomy` - Research taxonomy tools
- `mc dataset quality/auto-fix` - Dataset validation

#### MCP Server
- Full Model Context Protocol server (`modelcypher-mcp`)
- 150+ tools for model analysis and manipulation
- IDE integration support (VS Code, Cursor, etc.)

### Technical Highlights

- Hexagonal architecture (Ports and Adapters pattern)
- Strict separation: domain logic has no adapter imports
- Property-based testing with Hypothesis
- 3,030+ passing tests
- Type hints throughout (PEP 561 compliant)

---

## [Unreleased]

### Changed
- Migrated weight loading to MLX-based implementation
- Updated probe count assertions from 321 to 343 probes
- Removed deprecated integration and unit tests
- Replaced vocabulary alignment with comparison-based approach
- Consolidated activation fingerprint definitions

### Removed
- Rotational merger implementation (superseded by unified merge)
- Deprecated audit and verification scripts
