# Design Document: CLI/MCP Parity

## Overview

This design document outlines the implementation of remaining CLI commands and MCP tools to achieve 1:1 parity between TrainingCypher (Swift) and ModelCypher (Python). The implementation follows ModelCypher's existing hexagonal architecture with clear separation between CLI presentation, MCP server, use case services, and domain logic.

**Existing Implementation Status:**
- ✅ Dataset editing (preview, get-row, update-row, add-row, delete-row, convert) - DONE
- ✅ Geometry adapter (DARE sparsity, DoRA decomposition) - DONE  
- ✅ Storage (status, cleanup) - DONE
- ✅ Geometry validation, training status/history, safety circuit-breaker - DONE
- ✅ Geometry path detect/compare - DONE
- ⚠️ Evaluation (list/show done, run/results missing)
- ⚠️ Compare (list/show done, run/checkpoints/baseline/score missing)
- ❌ Model probe/validate-merge/analyze-alignment - NOT STARTED
- ❌ Geometry primes (list/probe/compare) - NOT STARTED
- ❌ Geometry stitch (analyze/apply) - NOT STARTED
- ❌ Adapter (project/wrap-mlx/smooth/inspect) - NOT STARTED
- ❌ Calibration - NOT STARTED
- ❌ Thermo - NOT STARTED
- ❌ RAG - NOT STARTED
- ❌ Stability - NOT STARTED
- ❌ Agent-eval - NOT STARTED
- ❌ Dashboard - NOT STARTED
- ❌ Help/completions/schema - NOT STARTED
- ❌ Inference suite (run/suite) - NOT STARTED

The design prioritizes:
1. **API Compatibility** - Identical command signatures and output schemas
2. **Testability** - Property-based testing for correctness guarantees
3. **Modularity** - Each command group maps to a dedicated service
4. **Consistency** - MCP tools mirror CLI commands exactly
5. **Reuse** - Leverage existing services and domain logic

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ dataset  │ │  model   │ │ geometry │ │  eval    │  ...      │
│  │ commands │ │ commands │ │ commands │ │ commands │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
┌───────┼────────────┼────────────┼────────────┼──────────────────┐
│       │            │            │            │   MCP Layer      │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐           │
│  │tc_dataset│ │tc_model  │ │tc_geom   │ │tc_eval   │  ...      │
│  │  tools   │ │  tools   │ │  tools   │ │  tools   │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
┌───────┴────────────┴────────────┴────────────┴──────────────────┐
│                      Use Case Services                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ DatasetEditor   │ │ ModelProbe      │ │ GeometryPrimes  │    │
│  │ Service         │ │ Service         │ │ Service         │    │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘    │
│           │                   │                   │              │
│  ┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐    │
│  │ EvaluationSuite │ │ AdapterService  │ │ ThermoService   │    │
│  │ Service         │ │                 │ │                 │    │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘    │
└───────────┼──────────────────┼──────────────────┼───────────────┘
            │                  │                  │
┌───────────┴──────────────────┴──────────────────┴───────────────┐
│                        Domain Layer                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │ Dataset     │ │ Model       │ │ Geometry    │ │ Adapter    │ │
│  │ Validation  │ │ Analysis    │ │ Primes      │ │ Operations │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Model Probe Service (NEW)

**Location:** `src/modelcypher/core/use_cases/model_probe_service.py`

```python
@dataclass
class ModelProbeResult:
    architecture: str
    parameter_count: int
    layers: list[LayerInfo]
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    quantization: str | None

@dataclass
class MergeValidationResult:
    compatible: bool
    warnings: list[str]
    architecture_match: bool
    vocab_match: bool
    dimension_match: bool

@dataclass
class AlignmentAnalysisResult:
    drift_magnitude: float
    layer_drifts: list[LayerDrift]
    assessment: str
    interpretation: str

class ModelProbeService:
    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details."""
        
    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models."""
        
    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models."""
```

### 2. Geometry Primes Service (NEW)

**Location:** `src/modelcypher/core/use_cases/geometry_primes_service.py`

Leverages existing domain logic in `core/domain/` for semantic primes.

```python
@dataclass
class SemanticPrime:
    id: str
    name: str
    category: str
    embedding: list[float] | None

@dataclass
class PrimeActivation:
    prime_id: str
    activation_strength: float
    layer_activations: dict[str, float]

@dataclass
class PrimeComparisonResult:
    alignment_score: float
    divergent_primes: list[str]
    convergent_primes: list[str]
    interpretation: str

class GeometryPrimesService:
    def list_primes(self) -> list[SemanticPrime]:
        """List all semantic prime anchors."""
        
    def probe(self, model_path: str) -> list[PrimeActivation]:
        """Probe model for prime activation patterns."""
        
    def compare(self, model_a: str, model_b: str) -> PrimeComparisonResult:
        """Compare prime alignment between two models."""
```

### 3. Geometry Stitch Service (NEW)

**Location:** `src/modelcypher/core/use_cases/geometry_stitch_service.py`

Leverages existing `manifold_stitcher.py` and `affine_stitching_layer.py` domain logic.

```python
@dataclass
class StitchAnalysisResult:
    manifold_distance: float
    stitching_points: list[StitchPoint]
    recommended_config: dict
    interpretation: str

@dataclass
class StitchApplyResult:
    output_path: str
    stitched_layers: int
    quality_score: float

class GeometryStitchService:
    def analyze(self, checkpoints: list[str]) -> StitchAnalysisResult:
        """Analyze manifold stitching between checkpoints."""
        
    def apply(self, config: StitchConfig) -> StitchApplyResult:
        """Apply stitching operation."""
```

### 4. Evaluation Service (EXTEND existing)

**Location:** `src/modelcypher/core/use_cases/evaluation_service.py`

Add `run` and `results` methods to existing service.

```python
class EvaluationService:
    # Existing: list_evaluations, get_evaluation
    
    def run(self, model: str, dataset: str, config: EvalConfig) -> EvalRunResult:
        """Execute evaluation and return metrics."""
        
    def results(self, eval_id: str) -> EvalDetailedResults:
        """Get detailed per-sample results."""
```

### 5. Compare Service (EXTEND existing)

**Location:** `src/modelcypher/core/use_cases/compare_service.py`

Add `run`, `checkpoints`, `baseline`, `score` methods to existing service.

```python
class CompareService:
    # Existing: list_sessions, get_session
    
    def run(self, checkpoints: list[str], prompt: str, config: CompareConfig) -> CompareRunResult:
        """Execute A/B comparison."""
        
    def checkpoints(self, job_id: str) -> CheckpointComparisonResult:
        """Compare checkpoints for a job."""
        
    def baseline(self, model: str) -> BaselineResult:
        """Establish baseline metrics."""
        
    def score(self, comparison_id: str) -> CompareScoreResult:
        """Get aggregated comparison scores."""
```

### 6. Adapter Service (NEW)

**Location:** `src/modelcypher/core/use_cases/adapter_service.py`

Extends existing `GeometryAdapterService` with additional operations.

```python
@dataclass
class AdapterInspectResult:
    rank: int
    alpha: float
    target_modules: list[str]
    sparsity: float
    parameter_count: int
    layer_analysis: list[LayerAdapterInfo]

class AdapterService:
    def project(self, adapter_path: str, target_space: str) -> ProjectResult:
        """Project adapter to target space."""
        
    def wrap_mlx(self, adapter_path: str, output_path: str) -> WrapResult:
        """Wrap adapter for MLX compatibility."""
        
    def smooth(self, adapter_path: str, strength: float) -> SmoothResult:
        """Apply smoothing to adapter weights."""
        
    def inspect(self, adapter_path: str) -> AdapterInspectResult:
        """Return detailed adapter analysis."""
```

### 7. Calibration Service (NEW)

**Location:** `src/modelcypher/core/use_cases/calibration_service.py`

```python
class CalibrationService:
    def run(self, model: str, dataset: str) -> CalibrationRunResult:
        """Execute calibration."""
        
    def status(self, calibration_id: str) -> CalibrationStatus:
        """Get calibration progress."""
        
    def apply(self, calibration_id: str, model: str) -> CalibrationApplyResult:
        """Apply calibration to model."""
```

### 8. Thermo Service (NEW)

**Location:** `src/modelcypher/core/use_cases/thermo_service.py`

Leverages existing `thermo_path_integration.py` domain logic.

```python
class ThermoService:
    def analyze(self, job_id: str) -> ThermoAnalysisResult:
        """Thermodynamic analysis of training."""
        
    def path(self, checkpoints: list[str]) -> ThermoPathResult:
        """Path integration analysis."""
        
    def entropy(self, job_id: str) -> ThermoEntropyResult:
        """Entropy metrics over training."""
```

### 9. RAG Service (NEW)

**Location:** `src/modelcypher/core/use_cases/rag_service.py`

```python
class RAGService:
    def index(self, documents: list[str], output_path: str) -> RAGIndexResult:
        """Create vector index from documents."""
        
    def query(self, query: str, top_k: int = 5) -> RAGQueryResult:
        """Query the index for relevant documents."""
        
    def status(self) -> RAGStatusResult:
        """Get index status and statistics."""
```

### 10. Stability Service (NEW)

**Location:** `src/modelcypher/core/use_cases/stability_service.py`

```python
class StabilityService:
    def run(self, model: str, config: StabilityConfig) -> StabilityRunResult:
        """Execute stability suite."""
        
    def report(self, suite_id: str) -> StabilityReport:
        """Get detailed stability report."""
```

### 11. Agent Eval Service (NEW)

**Location:** `src/modelcypher/core/use_cases/agent_eval_service.py`

```python
class AgentEvalService:
    def run(self, config: AgentEvalConfig) -> AgentEvalRunResult:
        """Execute agent evaluation."""
        
    def results(self, eval_id: str) -> AgentEvalResults:
        """Get evaluation results."""
```

### 12. Dashboard Service (NEW)

**Location:** `src/modelcypher/core/use_cases/dashboard_service.py`

```python
class DashboardService:
    def metrics(self) -> str:
        """Return metrics in Prometheus format."""
        
    def export(self, format: str) -> DashboardExportResult:
        """Export dashboard data."""
```

### 13. Help Service (NEW)

**Location:** `src/modelcypher/core/use_cases/help_service.py`

```python
class HelpService:
    def ask(self, question: str) -> HelpResponse:
        """Return contextual help for a question."""
        
    def completions(self, shell: str) -> str:
        """Generate shell completion script."""
        
    def schema(self, command: str) -> dict:
        """Return JSON schema for command output."""
```

### 14. Inference Suite (EXTEND existing LocalInferenceEngine)

**Location:** `src/modelcypher/adapters/local_inference.py`

```python
class LocalInferenceEngine:
    # Existing: infer
    
    def run_batch(self, model: str, prompts_file: str, config: InferConfig) -> BatchInferResult:
        """Execute batched inference from prompt file."""
        
    def run_suite(self, model: str, suite_config: str) -> SuiteInferResult:
        """Execute inference suite."""
```

## Data Models

### Model Probe Models

```python
@dataclass
class LayerInfo:
    name: str
    type: str
    parameters: int
    shape: list[int]

@dataclass
class LayerDrift:
    layer_name: str
    drift_magnitude: float
    direction: str
```

### Existing Data Models (Already Implemented)

The following models already exist in `core/domain/` and `core/use_cases/`:
- `DatasetRowSnapshot` - Dataset row with format detection
- `DatasetEditResult` - Result of dataset edit operations
- `DatasetPreviewResult` - Preview of dataset rows
- `DARESparsityAnalyzer.SparsityAnalysis` - DARE sparsity analysis
- `DoRADecomposition.DecompositionResult` - DoRA decomposition
- `StorageUsage`, `StorageSnapshot` - Storage metrics
- `EvaluationResult`, `CompareSession` - Evaluation and comparison results

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Note: Dataset editing properties (1-5 from requirements) are already covered by existing tests. The following properties focus on NEW functionality.

### Property 1: Model probe returns required fields
*For any* valid model path, `probe(path)` SHALL return a result containing architecture, parameter_count, and layers fields with non-null values.
**Validates: Requirements 2.1**

### Property 2: Model merge validation is symmetric for compatibility
*For any* two models A and B, `validate_merge(A, B).compatible` SHALL equal `validate_merge(B, A).compatible`.
**Validates: Requirements 2.2**

### Property 3: Alignment analysis returns bounded drift
*For any* two models A and B, `analyze_alignment(A, B).drift_magnitude` SHALL be in range [0.0, 1.0].
**Validates: Requirements 2.3**

### Property 4: Geometry primes list is non-empty
*For any* call to `list_primes()`, the result SHALL contain at least one semantic prime with valid id and name.
**Validates: Requirements 3.1**

### Property 5: Prime probe returns activations for all primes
*For any* valid model and prime inventory, `probe(model)` SHALL return activations for each prime in the inventory.
**Validates: Requirements 3.2**

### Property 6: Prime comparison is symmetric for alignment score
*For any* two models A and B, `compare(A, B).alignment_score` SHALL equal `compare(B, A).alignment_score`.
**Validates: Requirements 3.3**

### Property 7: Evaluation run-results round trip
*For any* model, dataset, and evaluation config, after `run(model, dataset, config)` returns eval_id, calling `get_evaluation(eval_id)` SHALL return results for that evaluation.
**Validates: Requirements 4.1, 4.2**

### Property 8: Comparison run-score round trip
*For any* set of checkpoints and comparison config, after `run(checkpoints, config)` returns comparison_id, calling `get_session(comparison_id)` SHALL return the session for that comparison.
**Validates: Requirements 4.3, 4.6**

### Property 9: Adapter inspect returns valid analysis
*For any* valid adapter path, `inspect(path)` SHALL return a result with rank > 0, alpha > 0, and non-empty target_modules.
**Validates: Requirements 5.4**

### Property 10: Adapter smoothing reduces variance
*For any* valid adapter and smoothing strength > 0, `smooth(adapter, strength)` SHALL produce an adapter with weight variance <= original variance.
**Validates: Requirements 5.3**

### Property 11: MCP output matches CLI output
*For any* command C with arguments A, the MCP tool `tc_C(A)` SHALL return output with the same schema and equivalent values as CLI command `tc C A --output json`.
**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**

### Property 12: Schema command returns valid JSON schema
*For any* valid command name, `schema(command)` SHALL return a valid JSON Schema document that validates the command's output.
**Validates: Requirements 13.3**

### Property 13: Thermo analysis returns bounded metrics
*For any* valid job_id with training history, `analyze(job_id)` SHALL return metrics with entropy values in range [0.0, ∞) and temperature values > 0.
**Validates: Requirements 7.1**

### Property 14: RAG index-query round trip
*For any* set of documents and query string, after `index(documents)` completes, `query(q)` SHALL return results that are subsets of the indexed documents.
**Validates: Requirements 8.1, 8.2**

## Error Handling

All services follow a consistent error handling pattern:

```python
class ServiceError(Exception):
    """Base class for service errors."""
    def __init__(self, code: str, title: str, detail: str, hint: str | None = None):
        self.code = code
        self.title = title
        self.detail = detail
        self.hint = hint

class NotFoundError(ServiceError):
    """Resource not found."""

class ValidationError(ServiceError):
    """Input validation failed."""

class OperationError(ServiceError):
    """Operation failed."""
```

Error codes follow the pattern `MC-XXXX` where:
- `MC-1XXX`: Validation errors
- `MC-2XXX`: Not found errors
- `MC-3XXX`: Operation errors
- `MC-4XXX`: Resource errors
- `MC-5XXX`: System errors

## Testing Strategy

### Dual Testing Approach

The implementation uses both unit tests and property-based tests:

1. **Unit Tests** - Verify specific examples and edge cases
2. **Property-Based Tests** - Verify universal properties across all inputs

### Property-Based Testing Framework

**Framework:** `hypothesis` (Python's standard PBT library)

**Configuration:**
- Minimum 100 iterations per property
- Explicit seed for reproducibility
- Shrinking enabled for minimal counterexamples

### Test Organization

```
tests/
├── unit/
│   ├── test_dataset_editor.py
│   ├── test_model_probe.py
│   ├── test_geometry_primes.py
│   └── ...
├── property/
│   ├── test_dataset_properties.py
│   ├── test_model_properties.py
│   ├── test_mcp_cli_parity.py
│   └── ...
└── fixtures/
    ├── sample_datasets/
    ├── sample_models/
    └── sample_adapters/
```

### Property Test Annotations

Each property-based test MUST include a comment referencing the correctness property:

```python
# **Feature: cli-mcp-parity, Property 1: Dataset row operations preserve file integrity**
# **Validates: Requirements 1.3, 1.4, 1.5**
@given(dataset=valid_jsonl_datasets(), operations=lists(dataset_operations()))
@settings(max_examples=100)
def test_dataset_operations_preserve_integrity(dataset, operations):
    ...
```

### Test Generators

Custom Hypothesis strategies for domain objects:

```python
@composite
def valid_jsonl_datasets(draw) -> str:
    """Generate valid JSONL dataset files."""
    
@composite
def dataset_operations(draw) -> DatasetOperation:
    """Generate valid dataset edit operations."""
    
@composite
def valid_model_paths(draw) -> str:
    """Generate paths to valid model directories."""
```
