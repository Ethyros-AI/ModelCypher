# Requirements Document

## Introduction

ModelCypher is a Python port of TrainingCypher's CLI and MCP tooling with core training, merging, and geometry engines. The goal is 1:1 parity between the Swift TrainingCypher CLI/MCP and the Python ModelCypher CLI/MCP implementations. This spec focuses on completing the remaining CLI commands and MCP tools that are currently PARTIAL or PLANNED status, ensuring developers can use ModelCypher as a drop-in replacement for TrainingCypher's backend functionality.

## Glossary

- **CLI**: Command Line Interface - the `tc` command-line tool for interacting with ModelCypher
- **MCP**: Model Context Protocol - a protocol for AI agents to interact with ModelCypher programmatically
- **MLX**: Apple's machine learning framework for Apple Silicon
- **LoRA**: Low-Rank Adaptation - a parameter-efficient fine-tuning technique
- **JSONL**: JSON Lines format - newline-delimited JSON for datasets
- **Geometry**: High-dimensional analysis tools for training health, alignment drift, and model comparison
- **DARE**: Drop And REscale - a sparsity-based adapter merging technique
- **DoRA**: Weight-Decomposed Low-Rank Adaptation - decomposition into magnitude and direction components
- **Circuit Breaker**: Safety mechanism that monitors training for alignment drift
- **Semantic Primes**: Core conceptual anchors used for model alignment analysis

## Requirements

### Requirement 1

**User Story:** As a developer, I want to validate and preprocess datasets through the CLI, so that I can prepare training data without using the GUI.

#### Acceptance Criteria

1. WHEN a user runs `tc dataset preview` with a dataset path THEN the CLI SHALL display the first N rows with format detection and validation messages
2. WHEN a user runs `tc dataset get-row` with a line number THEN the CLI SHALL return the specified row with validation status
3. WHEN a user runs `tc dataset update-row` with content THEN the CLI SHALL replace the row and return the updated row payload
4. WHEN a user runs `tc dataset add-row` with fields THEN the CLI SHALL append a new row and return the edit payload
5. WHEN a user runs `tc dataset delete-row` with a line number THEN the CLI SHALL remove the row and return confirmation
6. WHEN a user runs `tc dataset convert` with a target format THEN the CLI SHALL convert each row to the specified format

### Requirement 2

**User Story:** As a developer, I want to probe and validate models through the CLI, so that I can verify model integrity and compatibility before training.

#### Acceptance Criteria

1. WHEN a user runs `tc model probe` with a model path THEN the CLI SHALL return architecture details, parameter count, and layer information
2. WHEN a user runs `tc model validate-merge` with source and target models THEN the CLI SHALL return compatibility assessment and warnings
3. WHEN a user runs `tc model analyze-alignment` with two models THEN the CLI SHALL return alignment metrics and drift indicators

### Requirement 3

**User Story:** As a developer, I want to access geometry primes and stitch commands through the CLI, so that I can analyze semantic anchors and perform manifold stitching operations.

#### Acceptance Criteria

1. WHEN a user runs `tc geometry primes list` THEN the CLI SHALL return the inventory of semantic prime anchors
2. WHEN a user runs `tc geometry primes probe` with a model THEN the CLI SHALL return prime activation patterns for the model
3. WHEN a user runs `tc geometry primes compare` with two models THEN the CLI SHALL return prime alignment comparison metrics
4. WHEN a user runs `tc geometry stitch analyze` with checkpoints THEN the CLI SHALL return manifold stitching analysis
5. WHEN a user runs `tc geometry stitch apply` with configuration THEN the CLI SHALL perform the stitching operation

### Requirement 4

**User Story:** As a developer, I want to run evaluations and comparisons through the CLI, so that I can assess model quality programmatically.

#### Acceptance Criteria

1. WHEN a user runs `tc eval run` with model and dataset THEN the CLI SHALL execute evaluation and return metrics
2. WHEN a user runs `tc eval results` with an evaluation ID THEN the CLI SHALL return detailed per-sample results
3. WHEN a user runs `tc compare run` with checkpoints THEN the CLI SHALL execute A/B comparison and return results
4. WHEN a user runs `tc compare checkpoints` with a job ID THEN the CLI SHALL return checkpoint comparison for the job
5. WHEN a user runs `tc compare baseline` with a model THEN the CLI SHALL establish baseline metrics for comparison
6. WHEN a user runs `tc compare score` with comparison ID THEN the CLI SHALL return aggregated comparison scores

### Requirement 5

**User Story:** As a developer, I want to manage adapters through the CLI, so that I can project, wrap, smooth, and inspect LoRA adapters.

#### Acceptance Criteria

1. WHEN a user runs `tc adapter project` with adapter path THEN the CLI SHALL project the adapter to a target space
2. WHEN a user runs `tc adapter wrap-mlx` with adapter path THEN the CLI SHALL wrap the adapter for MLX compatibility
3. WHEN a user runs `tc adapter smooth` with adapter path THEN the CLI SHALL apply smoothing to the adapter weights
4. WHEN a user runs `tc adapter inspect` with adapter path THEN the CLI SHALL return detailed adapter analysis

### Requirement 6

**User Story:** As a developer, I want to access calibration commands through the CLI, so that I can calibrate models for optimal performance.

#### Acceptance Criteria

1. WHEN a user runs `tc calibration run` with model and dataset THEN the CLI SHALL execute calibration and return results
2. WHEN a user runs `tc calibration status` with calibration ID THEN the CLI SHALL return calibration progress and metrics
3. WHEN a user runs `tc calibration apply` with calibration results THEN the CLI SHALL apply calibration to the model

### Requirement 7

**User Story:** As a developer, I want to access thermodynamic analysis through the CLI, so that I can analyze training dynamics using physics-inspired metrics.

#### Acceptance Criteria

1. WHEN a user runs `tc thermo analyze` with a job ID THEN the CLI SHALL return thermodynamic analysis of training
2. WHEN a user runs `tc thermo path` with checkpoints THEN the CLI SHALL return path integration analysis
3. WHEN a user runs `tc thermo entropy` with a job ID THEN the CLI SHALL return entropy metrics over training

### Requirement 8

**User Story:** As a developer, I want to access RAG commands through the CLI, so that I can manage retrieval-augmented generation workflows.

#### Acceptance Criteria

1. WHEN a user runs `tc rag index` with documents THEN the CLI SHALL create a vector index
2. WHEN a user runs `tc rag query` with a query string THEN the CLI SHALL return relevant documents
3. WHEN a user runs `tc rag status` THEN the CLI SHALL return index status and statistics

### Requirement 9

**User Story:** As a developer, I want to access stability suite commands through the CLI, so that I can run stability tests on models.

#### Acceptance Criteria

1. WHEN a user runs `tc stability run` with model THEN the CLI SHALL execute stability suite and return results
2. WHEN a user runs `tc stability report` with suite ID THEN the CLI SHALL return detailed stability report

### Requirement 10

**User Story:** As a developer, I want to access agent evaluation commands through the CLI, so that I can evaluate agent performance.

#### Acceptance Criteria

1. WHEN a user runs `tc agent-eval run` with agent config THEN the CLI SHALL execute agent evaluation
2. WHEN a user runs `tc agent-eval results` with eval ID THEN the CLI SHALL return evaluation results

### Requirement 11

**User Story:** As a developer, I want to access storage management commands through the CLI, so that I can manage disk usage and cleanup.

#### Acceptance Criteria

1. WHEN a user runs `tc storage status` THEN the CLI SHALL return storage usage breakdown
2. WHEN a user runs `tc storage cleanup` with options THEN the CLI SHALL remove old artifacts and return freed space

### Requirement 12

**User Story:** As a developer, I want to access dashboard data through the CLI, so that I can retrieve dashboard metrics for Grafana integration.

#### Acceptance Criteria

1. WHEN a user runs `tc dashboard metrics` THEN the CLI SHALL return current dashboard metrics in Prometheus format
2. WHEN a user runs `tc dashboard export` with format THEN the CLI SHALL export dashboard data

### Requirement 13

**User Story:** As a developer, I want to access help and schema commands through the CLI, so that I can discover available commands and output schemas.

#### Acceptance Criteria

1. WHEN a user runs `tc help ask` with a question THEN the CLI SHALL return contextual help
2. WHEN a user runs `tc completions` with shell type THEN the CLI SHALL output shell completion scripts
3. WHEN a user runs `tc schema` with command name THEN the CLI SHALL return JSON schema for command output

### Requirement 14

**User Story:** As an AI agent, I want complete MCP tool parity with the CLI, so that I can perform all operations programmatically.

#### Acceptance Criteria

1. WHEN an agent calls `tc_model_probe` THEN the MCP server SHALL return model probe results matching CLI output
2. WHEN an agent calls `tc_eval_run` THEN the MCP server SHALL execute evaluation and return results
3. WHEN an agent calls `tc_compare_run` THEN the MCP server SHALL execute comparison and return results
4. WHEN an agent calls `tc_geometry_primes_list` THEN the MCP server SHALL return semantic primes inventory
5. WHEN an agent calls `tc_geometry_stitch_analyze` THEN the MCP server SHALL return stitching analysis

### Requirement 15

**User Story:** As a developer, I want the MCP server to expose all geometry gate detection tools, so that AI agents can analyze model behavior gates.

#### Acceptance Criteria

1. WHEN an agent calls `tc_geometry_path_detect` THEN the MCP server SHALL return detected gates in the model
2. WHEN an agent calls `tc_geometry_path_compare` THEN the MCP server SHALL return gate comparison between models

### Requirement 16

**User Story:** As a developer, I want inference commands to support suite execution, so that I can run batched inference tests.

#### Acceptance Criteria

1. WHEN a user runs `tc infer run` with prompt file THEN the CLI SHALL execute batched inference
2. WHEN a user runs `tc infer suite` with suite config THEN the CLI SHALL execute the inference suite and return results
