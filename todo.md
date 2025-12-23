# ModelCypher Todo List

## Active Tasks

*None currently*

---

## Backlog

### Benchmark Automation (Deferred from Audit)
**Priority**: Low | **Effort**: High

Implement `mc eval validate-merge` CLI command that orchestrates external benchmark suites.

**Implementation Path**:
1. Create `src/modelcypher/adapters/benchmark_adapter.py` with interfaces for:
   - lm-evaluation-harness (MMLU, GSM8K, HellaSwag)
   - HarmBench (safety/red teaming)
   - MergeBench (merge quality)
2. Add `mc eval validate-merge` CLI command
3. Output consolidated JSON report with scores and recommendations

**Why Deferred**: External suites have complex dependencies; core functionality works independently.

---

### CUDA Training Stubs
**Priority**: Low | **Effort**: Medium

Implement CUDA versions of training modules for Linux/NVIDIA support.

**Files to implement**:
- [ ] `training/engine_cuda.py` - PyTorch training loop
- [ ] `training/checkpoints_cuda.py` - torch.save/load
- [ ] `training/evaluation_cuda.py` - PyTorch evaluation
- [ ] `training/lora_cuda.py` - PEFT LoRA integration
- [ ] `training/loss_landscape_cuda.py` - torch.autograd

**Infrastructure ready**: `_platform.py` auto-detects and routes to correct implementation.

---

## Known Issues

### vocabulary_analyzer import error
**File**: `src/modelcypher/core/domain/vocabulary/vocabulary_analyzer.py`
- Test `test_module_imports[modelcypher.core.domain.vocabulary.vocabulary_analyzer]` fails
- Pre-existing issue with circular or missing import

---

## Completed (Reference)

### December 23, 2025
- [x] MLX Backend Abstraction (51 files migrated)
- [x] CABE-4 for emotion_concept_atlas.py
- [x] EntropyMergeValidator MCP & CLI integration
- [x] Platform selection infrastructure
- [x] Guard tests for architecture compliance
- [x] Audit report cleanup and summary
- [x] Fix EvaluationExecutionEngine mock entropy (added entropy_fn, scoring_fn callbacks)
- [x] Add tests for EvaluationExecutionEngine (14 new tests)
