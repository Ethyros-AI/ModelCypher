# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-25

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
- Backend-based weight loading implementation

#### Safety & Monitoring
- Circuit breaker signals for refusal/instability monitoring
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
- Social geometry experiments and probes
- Semantic primes inventory for cross-linguistic anchoring
- UnifiedAtlas concept inventory with multi-domain support

#### Backends
- Apple Silicon backend for macOS
- Linux accelerator backends for GPU/TPU environments
- Dynamic backend selection based on platform
 - No automatic CPU fallback (accelerator backends are required)

#### CLI (`mc` / `modelcypher`)
Key entry points (see `docs/CLI-REFERENCE.md` for the full catalog):
- `mc merge run` - Merge two models via null-space knowledge transplant
- `mc model probe` - Probe local models for architecture/geometry metadata
- `mc geometry ...` - Geometry analysis commands (training, safety, spatial, interference, etc.)
- `mc infer run` / `mc infer suite` - Inference runs with optional security scanning
- `mc thermo ...` - Thermodynamics metrics and calibration
- `mc adapter inspect` / `mc adapter project` / `mc adapter wrap` - Adapter tooling
- `mc research taxonomy` - Research taxonomy tools

### Technical Highlights

- Hexagonal architecture (Ports and Adapters pattern)
- Strict separation: domain logic has no adapter imports
- Property-based testing with Hypothesis
- Thousands of passing tests
- Type hints throughout (PEP 561 compliant)

---

## [Unreleased]

### Added
- 15 new test files for training domain modules covering geometric early stopping, spectral budget, scaled GD, scheduling, loop preservation, gradient smoothness, hessian estimator, checkpoint models/persistence/retention/validation, types, logical shapes, notifications, and benchmark (257 training domain tests total)

### Fixed
- Missing `logger` in `training_notifications.py` — handler exceptions caused `NameError` instead of being caught and logged
- Wrong class reference in `TrainingEventBus.emit_progress()` — referenced undefined `TrainingProgress` instead of `TrainingNotificationProgress`
- Incorrect type annotation on `TrainingEvent.progress` field — was `TrainingProgress`, corrected to `TrainingNotificationProgress`

### Changed
- Migrated weight loading to backend-based implementation
- Updated probe count assertions to match the current UnifiedAtlas inventory
- Removed deprecated integration and unit tests
- Replaced vocabulary alignment with comparison-based approach
- Consolidated activation fingerprint definitions

### Removed
- Rotational merger implementation (superseded by unified merge)
- Deprecated audit and verification scripts
- Dataset validation and quality functionality (focus on core geometry mission)
