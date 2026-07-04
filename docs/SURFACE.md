# ModelCypher Surface Map

This map ranks the public CLI surfaces so agents can find load-bearing code
without reading every use case. Status meanings:

- `promoted`: user-facing workflow that should stay stable.
- `instrumented`: direct measurement or support surface; useful, but not the
  main story.
- `experimental`: available for research or compatibility, not doctrine.

## Top-Level CLI Groups

| Group | Status | Primary service/direct path | Command source | Test anchor |
| --- | --- | --- | --- | --- |
| `mc analyze` | promoted | `ObservationService`, `ObservationBundleReportService`, expert services below | `src/modelcypher/cli/commands/analyze/` | `tests/cli/commands/test_analyze_commands.py` |
| `mc train` | promoted | `DatasetTrainingService`, `StarTrainingService` | `src/modelcypher/cli/commands/train.py` | `tests/cli/commands/test_train_commands.py`, `tests/test_dataset_training_service_strict.py` |
| `mc data` | promoted | `DataPreparationService` | `src/modelcypher/cli/commands/data.py` | `tests/cli/commands/test_readme_command_contract.py` |
| `mc infer` | promoted | backend inference through CLI composition | `src/modelcypher/cli/commands/infer.py` | `tests/cli/commands/test_infer_commands.py` |
| `mc model` | instrumented | `ModelService`, `CapacityAnalysisService`, quantization helpers | `src/modelcypher/cli/commands/model.py` | `tests/cli/commands/test_model_commands.py` |
| `mc system` | instrumented | `SystemService`, `SystemCacheService` | `src/modelcypher/cli/commands/system.py` | `tests/cli/commands/test_system_commands.py` |
| `mc adapter` | instrumented | `AdapterAnalysisService` | `src/modelcypher/cli/commands/adapter.py` | `tests/cli/commands/test_adapter_commands.py` |
| `mc quantize` | experimental | `QuantizationCorrectionService` | `src/modelcypher/cli/commands/quantize.py` | `tests/test_quantize_cli.py` |
| `mc merge` | experimental | `MergeService` over `modelcypher.experimental.merge` | `src/modelcypher/cli/commands/merge.py` | `tests/test_unified_geometric_merge.py`, `tests/test_merge_telemetry.py` |

## Analyze Commands

Current source exposes 43 analyze commands: 4 canonical workflows, 5 `probe`
workflow aliases, and 34 expert/compatibility commands.

| Command | Status | Primary service/direct path | Command source | Test anchor |
| --- | --- | --- | --- | --- |
| `mc analyze capture` | promoted | `ObservationService.capture` | `analyze/workflows.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze family` | promoted | `ObservationService.family` | `analyze/workflows.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze compare` | promoted | `ObservationService.compare` | `analyze/workflows.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze report` | promoted | `ObservationBundleReportService` | `analyze/workflows.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe calibrate` | promoted | `GeometrySafetyService` | `analyze/__init__.py`, `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe jailbreak` | promoted | `GeometrySafetyService` | `analyze/__init__.py`, `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe redteam` | promoted | `SafetyProbeService` | `analyze/__init__.py`, `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe behavioral` | promoted | `SafetyProbeService` | `analyze/__init__.py`, `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe bilm-info` | promoted | direct BiLM probe metadata reader | `analyze/__init__.py`, `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze attention-collapse` | instrumented | direct domain instrument: `core/domain/geometry/attention_collapse.py` | `analyze/geometric_concept_attention.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze attention-sink` | instrumented | direct domain instrument: `core/domain/geometry/attention_sink.py` | `analyze/geometric_concept_attention.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze geodesic-compare` | instrumented | `GeodesicTrajectoryService` | `analyze/geodesic_compare.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze geodesic-profile` | instrumented | `GeodesicTrajectoryService` | `analyze/geodesic_profile.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze geodesic-trajectory` | instrumented | `GeodesicTrajectoryService` | `analyze/geodesic_trajectory.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze concept-volume` | instrumented | `ConceptVolumeService` | `analyze/geometric_concept_attention.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze dimension-profile` | instrumented | `GeometryAnalysisService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze entropy-trajectory` | instrumented | `GeometryAnalysisService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze expansion-ratio` | instrumented | `GeometryAnalysisService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze reasoning-flow` | instrumented | `GeometryAnalysisService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze spectral-trajectory` | instrumented | `GeometryAnalysisService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze jacobian-trace` | instrumented | `GeometryAnalysisService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py`, `tests/test_jacobian_analyzer.py` |
| `mc analyze verification-depth-profile` | instrumented | `VerificationDepthProfileService` | `analyze/geometric.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze chain-profile` | instrumented | `ChainAnalysisService` | `analyze/geometric_concept_attention.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze adapter-probe` | instrumented | direct adapter/probe compatibility path | `analyze/behavioral.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze behavioral-signature` | instrumented | `GeometryAnalysisService` | `analyze/behavioral.py` | `tests/cli/commands/test_analyze_commands.py`, `tests/test_behavioral_analyzer.py` |
| `mc analyze cognitive-reflection-test` | instrumented | `GeometryAnalysisService` | `analyze/behavioral.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze calibrate-safety` | instrumented | `GeometrySafetyService` | `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze jailbreak-test` | instrumented | `GeometrySafetyService` | `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe-redteam` | instrumented | `SafetyProbeService` | `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze probe-behavioral` | instrumented | `SafetyProbeService` | `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze bilm-probe-info` | instrumented | direct BiLM probe metadata reader | `analyze/probes.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze benchmark` | instrumented | `BenchmarkService` | `analyze/benchmark.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze lora-svd` | instrumented | direct LoRA SVD diagnostic | `analyze/benchmark.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze sparse-region` | instrumented | `GeometrySparseService` | `analyze/benchmark.py` | `tests/cli/commands/test_analyze_commands.py`, `tests/test_neuron_sparsity_analyzer.py` |
| `mc analyze knowledge-type` | instrumented | direct knowledge analyzer path | `analyze/benchmark.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze curriculum-profile` | instrumented | direct curriculum profiler path | `analyze/benchmark.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze circuit-breaker` | instrumented | `GeometrySafetyService` | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze persona` | instrumented | `GeometrySafetyService` | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze uncertainty-modes` | instrumented | direct uncertainty-mode diagnostic | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze entropy-pattern` | instrumented | `EntropyProbeService` | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py`, `tests/domain/entropy/test_chunk_entropy_analyzer.py` |
| `mc analyze entropy-baseline-verify` | instrumented | `EntropyProbeService` | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze crm-build` | instrumented | `ConceptResponseMatrixService` | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py` |
| `mc analyze crm-compare` | instrumented | `ConceptResponseMatrixService` | `analyze/monitoring.py` | `tests/cli/commands/test_analyze_commands.py` |
