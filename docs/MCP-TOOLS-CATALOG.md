# MCP Tools Catalog

Complete reference for ModelCypher MCP tools. These tools enable AI assistants to work with LLM geometric analysis, training, and model operations.

**Total Tools:** 119

---

## Table of Contents

1. [System & Inventory](#system--inventory)
2. [Model Management](#model-management)
3. [Training & Jobs](#training--jobs)
4. [Evaluation](#evaluation)
5. [Inference](#inference)
6. [Agent Tools](#agent-tools)
7. [Geometry - Core](#geometry---core)
8. [Geometry - Spatial](#geometry---spatial)
9. [Geometry - CRM & Baseline](#geometry---crm--baseline)
10. [Geometry - Density](#geometry---density)
11. [Geometry - Interference](#geometry---interference)
12. [Geometry - Safety](#geometry---safety)
13. [Geometry - Metaphor](#geometry---metaphor)
14. [Geometry - Primes](#geometry---primes)
15. [Geometry - Invariant](#geometry---invariant)
16. [Geometry - Visualize](#geometry---visualize)
17. [Safety & Entropy](#safety--entropy)
18. [Thermodynamics](#thermodynamics)
19. [Merge Entropy](#merge-entropy)
20. [Tasks](#tasks)
21. [Adapter](#adapter)

---

## System & Inventory

### mc_inventory

Return inventory snapshot for models, jobs, and checkpoints.

**Returns:** Count of registered models, active jobs, and available checkpoints.

---

### mc_system_status

Return system readiness and backend availability.

**Returns:** Backend status, GPU availability, memory stats.

---

### mc_settings_snapshot

Return current settings snapshot.

**Returns:** Active configuration values and paths.

---

## Model Management

### mc_model_list

List all registered local models.

**Returns:**
- `models`: Array of model entries with id, alias, path, architecture, format, sizeBytes
- `count`: Total registered models

---

### mc_model_register

Register a local model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Path to model directory |
| `alias` | string | No | Human-readable alias |

**Returns:** Model ID, path, alias, registration status.

---

### mc_model_delete

Delete a model. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modelId` | string | Yes | Model identifier |
| `confirmationToken` | string | No | Confirmation token for destructive action |

---

### mc_model_search

Search HuggingFace for models.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `limit` | int | No | Max results (default 20) |
| `library` | string | No | Filter by library (mlx, transformers) |
| `quantization` | string | No | Filter by quantization |
| `sort` | string | No | Sort option |

**Returns:** Array of matching models with metadata.

---

### mc_model_probe

Probe model architecture and configuration.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modelPath` | string | Yes | Path to model directory |

**Returns:** Architecture details, layer count, hidden dimensions, vocab size.

---

### mc_model_validate_merge

Validate merge compatibility between two models.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | Yes | Source model path |
| `target` | string | Yes | Target model path |

**Returns:** Compatibility assessment, dimension alignment, potential issues.

---

### mc_model_analyze_alignment

Analyze geometric alignment between two models.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modelA` | string | Yes | First model path |
| `modelB` | string | Yes | Second model path |

**Returns:** CKA similarity, alignment metrics, layer correspondence.

---

### mc_model_fetch

Fetch model from HuggingFace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repoId` | string | Yes | HuggingFace repo ID |
| `outputDir` | string | No | Local output directory |
| `revision` | string | No | Branch/tag/commit |

**Returns:** Download status, local path.

---

## Training & Jobs

### mc_train_start

Start a training job.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Base model path |
| `dataset` | string | Yes | Training dataset path |
| `outputPath` | string | Yes | Output directory |
| `hyperparameters` | object | Yes | Training hyperparameters |
| `autoEval` | bool | Yes | Run evaluation after training |
| `lora` | object | No | LoRA configuration |
| `idempotencyKey` | string | No | Prevent duplicate jobs |
| `evalDataset` | string | No | Evaluation dataset |
| `evalMetrics` | array | No | Metrics to compute |

**Hyperparameters object:**
- `batchSize`, `learningRate`, `epochs`, `sequenceLength`
- `gradientAccumulationSteps`, `gradientCheckpointing`
- `mixedPrecision`, `computePrecision`
- `warmupSteps`, `weightDecay`, `seed`, `deterministic`, `optimizerType`

**Returns:** Job ID, status, batch size.

---

### mc_job_status

Get job status.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

**Returns:** Status, progress percentage, current epoch.

---

### mc_job_list

List training jobs.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by status |
| `activeOnly` | bool | No | Only active jobs |

**Returns:** Array of job summaries.

---

### mc_job_detail

Get detailed job information.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

**Returns:** Full job configuration, metrics, checkpoints.

---

### mc_job_cancel

Cancel a running job.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

---

### mc_job_pause

Pause a running job.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

---

### mc_job_resume

Resume a paused job.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

---

### mc_job_delete

Delete a job and its artifacts.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |
| `confirmationToken` | string | No | Confirmation token |

---

### mc_validate_train

Validate training configuration without starting.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `dataset` | string | Yes | Dataset path |
| `hyperparameters` | object | Yes | Hyperparameters |

**Returns:** Validation result, warnings, estimated memory.

---

### mc_estimate_train

Estimate training resource requirements.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `dataset` | string | Yes | Dataset path |
| `hyperparameters` | object | Yes | Hyperparameters |

**Returns:** Memory estimate, steps, estimated tokens.

---

### mc_train_preflight

Run comprehensive preflight checks.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `dataset` | string | Yes | Dataset path |
| `hyperparameters` | object | Yes | Hyperparameters |
| `outputPath` | string | No | Output directory |

**Returns:** All checks passed/failed, detailed diagnostics.

---

### mc_train_export

Export trained model/adapter.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |
| `output` | string | Yes | Output path |
| `format` | string | No | Export format (safetensors) |

---

## Evaluation

### mc_eval_run

Run evaluation on a model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `dataset` | string | Yes | Evaluation dataset |
| `metrics` | array | No | Metrics to compute |
| `batchSize` | int | No | Batch size (default 4) |
| `maxSamples` | int | No | Max samples |

**Returns:** Evaluation ID, average loss, perplexity, sample count.

---

### mc_eval_list

List evaluation runs.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | int | No | Max results (default 50) |

---

### mc_eval_show

Show evaluation results.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evalId` | string | Yes | Evaluation ID |

---

## Inference

### mc_infer

Basic inference (prompt → response).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `prompt` | string | Yes | Input prompt |

**Returns:** Response, token count, tokens/sec, timing.

---

### mc_infer_run

Inference with adapter and security scanning.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `prompt` | string | Yes | Input prompt |
| `adapter` | string | No | Adapter path |
| `securityScan` | bool | No | Enable dual-path security |

**Returns:** Response, metrics, optional security analysis.

---

### mc_infer_batch

Batched inference from prompts file.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `promptsFile` | string | Yes | Path to prompts file |

**Returns:** Total prompts, successful/failed counts, aggregate metrics.

---

### mc_infer_suite

Execute inference suite with pass/fail assertions.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `suiteFile` | string | Yes | Suite definition file |
| `adapter` | string | No | Adapter path |
| `securityScan` | bool | No | Enable security scanning |

**Returns:** Pass/fail counts, individual case results.

---

## Agent Tools

### mc_agent_eval_run

Execute agent evaluation suite.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model path |
| `evalSuite` | string | No | Suite name (default "default") |
| `maxTurns` | int | No | Max conversation turns |
| `timeout` | int | No | Timeout in seconds |
| `seed` | int | No | Random seed |

**Returns:** Evaluation ID, status, summary.

---

### mc_agent_eval_results

Get agent evaluation results.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evalId` | string | Yes | Evaluation ID |

**Returns:** Full results, metrics, task outcomes.

---

### mc_agent_trace_import

Import agent traces from Monocle/OpenTelemetry format.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filePath` | string | Yes | Path to trace file |
| `sanitize` | bool | No | Sanitize PII (default true) |
| `maxValueLength` | int | No | Max string length (default 1000) |

**Returns:** Imported trace count, warnings, trace summaries.

---

### mc_agent_trace_analyze

Analyze agent traces for patterns and compliance.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filePath` | string | Yes | Path to trace file |

**Returns:** Span counts, message type distribution, compliance metrics, entropy buckets.

---

### mc_agent_validate_action

Validate agent action for safety.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | JSON action object |
| `strict` | bool | No | Strict validation mode |

**Returns:** Valid flag, errors, warnings, risk level.

---

## Geometry - Core

### mc_geometry_validate

Run geometry validation suite.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `includeFixtures` | bool | No | Include built-in fixtures |

**Returns:** Validation diagnostics, measurements.

---

### mc_geometry_path_detect

Detect path geometry in text or model response.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze or prompt |
| `model` | string | No | Model path (if prompt) |
| `entropyTrace` | array | No | Entropy values |

**Returns:** Path detection results, geometric features.

---

### mc_geometry_path_compare

Compare path geometry between two texts/responses.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `textA` | string | No | First text |
| `textB` | string | No | Second text |
| `modelA` | string | No | First model |
| `modelB` | string | No | Second model |
| `prompt` | string | No | Shared prompt for models |
| `comprehensive` | bool | No | Full comparison |

---

### mc_geometry_concept_detect

Detect concept sequence using atlas probes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze |
| `model` | string | No | Model path |

**Returns:** Detected concepts, confidence scores.

---

### mc_geometry_concept_compare

Compare concept geometry between responses.

---

### mc_geometry_cross_cultural_analyze

Analyze cross-cultural concept representation.

---

### mc_geometry_gromov_wasserstein

Compute Gromov-Wasserstein distance between activations.

**Returns:** GW distance (shape-agnostic similarity).

---

### mc_geometry_intrinsic_dimension

Estimate intrinsic dimension of activations.

**Returns:** Dimension estimate, method used.

---

### mc_geometry_topological_fingerprint

Compute topological fingerprint via persistent homology.

**Returns:** Betti numbers, persistence diagram features.

---

### mc_geometry_spectral_signature

Compute spectral signature of activation covariance.

**Returns:** Eigenvalue spectrum, spectral gap.

---

### mc_geometry_dimension_constraint_invariance

Test dimension-constraint invariance.

---

### mc_geometry_sparse_domains

List sparse activation domains.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | No | Filter by category |

---

### mc_geometry_sparse_locate

Locate sparse activation regions.

---

### mc_geometry_sparse_neurons

Identify sparsely activating neurons.

---

### mc_geometry_refusal_pairs

Get refusal direction probe pairs.

---

### mc_geometry_refusal_detect

Detect refusal direction in activations.

---

### mc_geometry_persona_traits

List persona trait vectors.

---

### mc_geometry_persona_extract

Extract persona vectors from model.

---

### mc_geometry_persona_drift

Measure persona drift between checkpoints.

---

### mc_geometry_manifold_cluster

Cluster activation manifold.

---

### mc_geometry_manifold_dimension

Estimate local manifold dimension.

---

### mc_geometry_manifold_query

Query manifold neighborhood.

---

### mc_geometry_training_status

Get training geometry status.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |
| `format` | string | No | Output format (full) |

---

### mc_geometry_training_history

Get training geometry history.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

---

## Geometry - Spatial

### mc_geometry_spatial_anchors

Get spatial reasoning anchors.

---

### mc_geometry_spatial_euclidean

Probe 3D Euclidean consistency of spatial anchors (2D/3D only).

---

### mc_geometry_spatial_gravity

Analyze gravity/physics intuition.

---

### mc_geometry_spatial_density

Measure spatial concept density.

---

### mc_geometry_spatial_analyze

Full spatial reasoning analysis.

---

### mc_geometry_spatial_probe_model

Probe model spatial capabilities.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modelPath` | string | Yes | Model path |

**Returns:** Spatial scores by dimension.

---

### mc_geometry_spatial_cross_grounding_feasibility

Assess cross-architecture grounding feasibility.

---

### mc_geometry_spatial_cross_grounding_transfer

Execute cross-architecture grounding transfer.

---

## Geometry - CRM & Baseline

### mc_geometry_crm_build

Build Concept Response Matrix for model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modelPath` | string | Yes | Model path |
| `outputPath` | string | Yes | Output CRM JSON path |
| `adapter` | string | No | Optional adapter directory |

**Returns:** CRM tensor, layer fingerprints.

---

### mc_geometry_crm_compare

Compare CRMs between models.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sourcePath` | string | Yes | Source CRM JSON path |
| `targetPath` | string | Yes | Target CRM JSON path |

**Returns:** Layer correspondence via CKA matching.

---

### mc_geometry_crm_sequence_inventory

List available CRM sequences.

---

### mc_geometry_baseline_list

List geometry baseline profiles.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `family` | string | No | Filter by family |

---

### mc_geometry_baseline_extract

Extract baseline geometry from model.

---

### mc_geometry_baseline_compare

Compare against baseline geometry.

---

## Geometry - Density

### mc_geometry_density_profile

Build density profile for model layers.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modelPath` | string | Yes | Model path |
| `layers` | array | No | Specific layers |

**Returns:** Per-layer density measurements.

---

### mc_geometry_density_diff

Compute density difference between models.

---

## Geometry - Interference

### mc_geometry_interference_predict

Predict merge interference between models.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | Yes | Source model |
| `target` | string | Yes | Target model |

**Returns:** Interference scores by layer, risk assessment.

---

### mc_geometry_null_space_filter

Apply null space projection filter.

---

### mc_geometry_null_space_profile

Profile null space dimensions.

---

## Geometry - Safety

### mc_geometry_safety_jailbreak_test

Test model resistance to jailbreak attempts.

---

### mc_geometry_dare_sparsity

Analyze DARE sparsity patterns.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `checkpointPath` | string | Yes | Checkpoint path |
| `basePath` | string | No | Base model path |

---

### mc_geometry_dora_decomposition

Analyze DoRA magnitude/direction decomposition.

---

## Geometry - Metaphor

### mc_geometry_metaphor_list

List metaphor invariant atlases.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `family` | string | No | Filter by family |

---

### mc_geometry_metaphor_trajectory

Track metaphor trajectory through layers.

---

### mc_geometry_metaphor_invariance

Test metaphor invariance properties.

---

### mc_geometry_metaphor_convergence

Measure metaphor representation convergence.

---

## Geometry - Primes

### mc_geometry_primes_list

List semantic primes categories.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | No | Filter by category |

---

### mc_geometry_primes_probe

Probe semantic prime activations.

---

### mc_geometry_primes_compare

Compare semantic prime representations.

---

## Geometry - Invariant

### mc_geometry_invariant_map_layers

Map layer correspondence via invariants.

---

### mc_geometry_invariant_collapse_risk

Assess representation collapse risk.

---

### mc_geometry_atlas_inventory

List atlas inventory entries.

---

## Geometry - Visualize

### mc_geometry_visualize_create

Create geometry visualization.

---

### mc_geometry_visualize_from_activations

Visualize from activation files.

---

### mc_geometry_visualize_info

Get visualization capabilities.

---

## Safety & Entropy

### mc_safety_circuit_breaker

Evaluate adapter safety using static + entropy analysis.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `adapterName` | string | Yes | Adapter identifier |
| `adapterDescription` | string | No | Description text |
| `skillTags` | array | No | Skill tags |
| `entropyDelta` | array | No | Entropy delta values |

**Returns:** Threat indicator count, entropy stats.

---

### mc_safety_persona_drift

Detect persona drift from baseline.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `baselinePersona` | object | Yes | Trait → score mapping |
| `currentBehavior` | array | Yes | Behavior samples |

**Returns:** Mean drift, trait-level scores.

---

### mc_safety_redteam_scan

Run red team security scan.

---

### mc_safety_behavioral_probe

Probe behavioral patterns.

---

### mc_safety_adapter_probe

Probe adapter for delta-feature geometry.

---

### mc_entropy_analyze

Analyze entropy patterns.

---

### mc_entropy_detect_distress

Detect entropy distress signals.

---

### mc_entropy_verify_baseline

Verify entropy against baseline.

---

### mc_entropy_window

Compute sliding window entropy.

---

### mc_entropy_conversation_track

Track conversation entropy over turns.

---

### mc_entropy_dual_path

Execute dual-path entropy analysis.

---

## Thermodynamics

### mc_thermo_analyze

Analyze training thermodynamics.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

---

### mc_thermo_path

Compute thermodynamic path between checkpoints.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `checkpoints` | array | Yes | Checkpoint paths |

---

### mc_thermo_path_integration

Compute path integral over training trajectory.

---

### mc_thermo_entropy

Get training entropy measurements.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jobId` | string | Yes | Job identifier |

---

### mc_thermo_measure

Measure thermodynamic quantities.

---

### mc_thermo_detect

Detect thermodynamic anomalies.

---

### mc_thermo_detect_batch

Batch thermodynamic detection.

---

## Merge Entropy

### mc_merge_entropy_profile

Profile entropy characteristics for merge.

---

### mc_merge_entropy_validate

Validate merge entropy compatibility.

---

### mc_model_validate_knowledge

Validate knowledge preservation post-merge.

---

### mc_model_vocab_compare

Compare vocabularies between models.

---

## Tasks

### mc_task_list

List background tasks.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by status |
| `limit` | int | No | Max results |

---

### mc_task_status

Get task status.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `taskId` | string | Yes | Task identifier |

---

### mc_task_cancel

Cancel a running task.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `taskId` | string | Yes | Task identifier |

---

### mc_task_result

Get task result.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `taskId` | string | Yes | Task identifier |

---

### mc_task_delete

Delete a task.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `taskId` | string | Yes | Task identifier |

---

## Adapter

### mc_adapter_inspect

Inspect adapter weights and configuration.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `adapterPath` | string | Yes | Path to adapter |

**Returns:** LoRA configuration, layer stats, weight norms.

---

## Schema Conventions

All tools return JSON with a `_schema` field for versioning:

```json
{
  "_schema": "mc.<category>.<action>.v1",
  "field1": "value1"
}
```

**Annotation Types:**
- `READ_ONLY_ANNOTATIONS`: No side effects
- `MUTATING_ANNOTATIONS`: Modifies state
- `DESTRUCTIVE_ANNOTATIONS`: Requires confirmation
- `NETWORK_ANNOTATIONS`: Makes network requests

---

## Error Handling

Errors are returned as JSON with standardized structure:

```json
{
  "error": {
    "code": "MC-1001",
    "title": "Short description",
    "detail": "Full explanation",
    "hint": "Suggested fix"
  }
}
```

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MC_MCP_REQUIRE_CONFIRMATION` | Require confirmation for destructive ops |
| `MC_MODELS_DIR` | Default models directory |
| `MC_LOG_LEVEL` | Logging verbosity |

### Tool Filtering

Use `MC_MCP_TOOL_SET` to limit exposed tools:

```bash
MC_MCP_TOOL_SET="mc_model_list,mc_infer,mc_geometry_*" poetry run modelcypher-mcp
```
