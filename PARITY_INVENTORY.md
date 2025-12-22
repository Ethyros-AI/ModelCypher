# TrainingCypher → ModelCypher Parity Inventory

**Generated**: 2025-12-22
**Scope**: Core novel functionality only (excluding RAG, GUI, third-party integrations)

---

## Executive Summary

| Domain | Swift Files | Python Files | Parity Status |
|--------|-------------|--------------|---------------|
| **Geometry** | 45 | 52 | ✅ ~95% Complete |
| **Entropy** | 26 | 15 | ⚠️ ~57% Complete |
| **Safety** | 27 | 26 | ✅ ~90% Complete |
| **Training** | 37 | 15 | ⚠️ ~40% Complete |
| **Agents** | 27 | 12 | ⚠️ ~45% Complete |
| **Thermodynamics** | 14 | 5 | ⚠️ ~38% Complete |
| **Adapters** | 16 | ~2 | 🔴 ~10% Complete |
| **Inference** | 10 | ~3 | 🔴 ~30% Complete |
| **Validation** | 12 | ~5 | ⚠️ ~40% Complete |
| **Memory** | 11 | ~0 | 🔴 Not Started |

---

## Detailed Inventory by Domain

### Geometry (45 Swift → 52 Python) ✅

The Geometry domain is the most complete. Most core algorithms are ported.

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `AffineStitchingLayer.swift` (34KB) | `affine_stitching_layer.py` (17KB) | ✅ Ported |
| `AnchorInvarianceAnalyzer.swift` (16KB) | `anchor_invariance_analyzer.py` (17KB) | ✅ Ported |
| `CompositionalProbes.swift` (19KB) | `compositional_probes.py` (11KB) | ✅ Ported |
| `ConceptDetector.swift` (19KB) | `concept_detector.py` (15KB) | ✅ Ported |
| `ConceptResponseMatrix.swift` (34KB) | `concept_response_matrix.py` (25KB) | ✅ Ported |
| `CrossArchitectureLayerMatcher.swift` (25KB) | `cross_architecture_layer_matcher.py` (11KB) | ⚠️ Partial |
| `CrossCulturalGeometry.swift` (29KB) | `cross_cultural_geometry.py` (15KB) | ⚠️ Partial |
| `DARESparsityAnalyzer.swift` (18KB) | `dare_sparsity.py` (10KB) | ✅ Ported |
| `DimensionAlignmentBuilder.swift` (6KB) | `dimension_blender.py` (11KB) | ✅ Ported |
| `DoRADecomposition.swift` (17KB) | `dora_decomposition.py` (10KB) | ✅ Ported |
| `DomainSignalProfile.swift` (3KB) | `domain_signal_profile.py` (5KB) | ✅ Ported |
| `GateDetector.swift` (16KB) | `gate_detector.py` (9KB) | ✅ Ported |
| `GeneralizedProcrustes.swift` (39KB) | `generalized_procrustes.py` (8KB) | ⚠️ Partial (size gap) |
| `GeometricFingerprint.swift` (18KB) | `geometry_fingerprint.py` (10KB) | ✅ Ported |
| `GeometryValidationSuite.swift` (18KB) | `geometry_validation_suite.py` (15KB) | ✅ Ported |
| `IntersectionMapAnalysis.swift` (15KB) | `intersection_map_analysis.py` (11KB) | ✅ Ported |
| `IntrinsicDimensionEstimator.swift` (11KB) | `intrinsic_dimension_estimator.py` (8KB) | ✅ Ported |
| `InvariantConvergenceAnalyzer.swift` (17KB) | `invariant_convergence_analyzer.py` (4KB) | ⚠️ Partial (size gap) |
| `InvariantLayerMapper.swift` (22KB) | `invariant_layer_mapper.py` (37KB) | ✅ Ported (Python larger!) |
| `ManifoldClusterer.swift` (20KB) | `manifold_clusterer.py` (14KB) | ✅ Ported |
| `ManifoldDimensionality.swift` (9KB) | `manifold_dimensionality.py` (6KB) | ✅ Ported |
| `ManifoldFidelitySweep.swift` (31KB) | `manifold_fidelity_sweep.py` (12KB) | ⚠️ Partial (size gap) |
| `ManifoldProfile.swift` (18KB) | `manifold_profile.py` (8KB) | ⚠️ Partial |
| `ManifoldProfileService.swift` (16KB) | `manifold_profile_service.py` (12KB) | ✅ Ported |
| `MetaphorConvergenceAnalyzer.swift` (17KB) | `metaphor_convergence_analyzer.py` (19KB) | ✅ Ported |
| `ModelFingerprintsProjection.swift` (12KB) | `model_fingerprints_projection.py` (9KB) | ✅ Ported |
| `PathGeometry.swift` (39KB) | `path_geometry.py` (20KB) | ⚠️ Partial (size gap) |
| `PermutationAligner.swift` (48KB) | `permutation_aligner.py` (23KB) | ⚠️ Partial (GPU opts missing) |
| `PersonaVectorMonitor.swift` (28KB) | `persona_vector_monitor.py` (15KB) | ⚠️ Partial |
| `RefusalDirectionCache.swift` (7KB) | `refusal_direction_cache.py` (6KB) | ✅ Ported |
| `RefusalDirectionDetector.swift` (18KB) | `refusal_direction_detector.py` (10KB) | ✅ Ported |
| `SetMath.swift` (1KB) | — | ❌ Missing |
| `SharedSubspaceProjector.swift` (45KB) | `shared_subspace_projector.py` (34KB) | ✅ Ported |
| `SparseRegionDomains.swift` (14KB) | `sparse_region_domains.py` (12KB) | ✅ Ported |
| `SparseRegionLocator.swift` (20KB) | `sparse_region_locator.py` (13KB) | ✅ Ported |
| `SparseRegionProber.swift` (17KB) | `sparse_region_prober.py` (11KB) | ✅ Ported |
| `SparseRegionValidator.swift` (19KB) | `sparse_region_validator.py` (11KB) | ✅ Ported |
| `TangentSpaceAlignment.swift` (13KB) | `tangent_space_alignment.py` (11KB) | ✅ Ported |
| `ThermoPathIntegration.swift` (17KB) | `thermo_path_integration.py` (10KB) | ✅ Ported |
| `TopologicalFingerprint.swift` (22KB) | `topological_fingerprint.py` (14KB) | ✅ Ported |
| `TransferFidelityPrediction.swift` (10KB) | `transfer_fidelity.py` (4KB) | ⚠️ Partial |
| `TransportGuidedMerger.swift` (21KB) | `transport_guided_merger.py` (13KB) | ✅ Ported |
| `TraversalCoherence.swift` (12KB) | `traversal_coherence.py` (5KB) | ⚠️ Partial |
| `VectorMath.swift` (4KB) | `vector_math.py` (1KB) | ⚠️ Partial |
| `VerbNounDimensionClassifier.swift` (19KB) | `verb_noun_dimension_classifier.py` (8KB) | ⚠️ Partial |

**Python-only additions**: `gromov_wasserstein.py`, `refinement_density.py`, `manifold_stitcher.py`, `probe_corpus.py`, `probes.py`, `types.py`, `fingerprints.py`, `intrinsic_dimension.py`

---

### Entropy (26 Swift → 13 Python) ⚠️

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `AdapterStackAnalyzer.swift` (20KB) | — | ❌ Missing |
| `BaselineVerificationProbe.swift` (22KB) | `baseline_verification_probe.py` (21KB) | ✅ Ported |
| `ChunkEntropyAnalyzer.swift` (16KB) | `chunk_entropy_analyzer.py` (19KB) | ✅ Ported |
| `ConflictScore.swift` (12KB) | `conflict_score.py` (2KB) | ⚠️ Stub only |
| `ConversationEntropyTracker.swift` (20KB) | `conversation_entropy_tracker.py` (18KB) | ✅ Ported |
| `EntropyDeltaSample.swift` (21KB) | `entropy_delta_sample.py` (11KB) | ⚠️ Partial |
| `EntropyDeltaTracker.swift` (19KB) | `entropy_delta_tracker.py` (18KB) | ✅ Ported |
| `EntropyLogitProcessor.swift` (6KB) | — | ❌ Missing |
| `EntropyPatternDetector.swift` (14KB) | `entropy_pattern_detector.py` (14KB) | ✅ Ported |
| `EntropySample.swift` (8KB) | — | ❌ Missing (types in others) |
| `EntropyTracker.swift` (28KB) | `entropy_tracker.py` (22KB) | ✅ Ported |
| `EntropyWindow.swift` (10KB) | — | ❌ Missing |
| `GeometricAlignmentLogitProcessor.swift` (4KB) | — | ❌ Missing |
| `GeometricAlignmentSystem.swift` (32KB) | `geometric_alignment.py` (22KB) | ⚠️ Partial |
| `HiddenStateExtractor.swift` (15KB) | `hidden_state_extractor.py` (10KB) | ✅ Ported |
| `HiddenStateTaps.swift` (19KB) | — | ❌ Missing |
| `JailbreakEntropyExperiment.swift` (23KB) | — | ❌ Missing |
| `LogitDivergenceCalculator.swift` (2KB) | — | ❌ Missing |
| `LogitEntropyCalculator.swift` (12KB) | — | ❌ Missing |
| `MetricSample.swift` (11KB) | `metrics_ring_buffer.py` (18KB) | ✅ Ported |
| `MetricsRingBuffer.swift` (11KB) | `metrics_ring_buffer.py` (18KB) | ✅ Ported |
| `ModelState.swift` (8KB) | `model_state.py` (4KB) | ⚠️ Partial |
| `ModelStateClassifier.swift` (15KB) | `model_state_classifier.py` (14KB) | ✅ Ported |
| `SEPProbe.swift` (14KB) | `sep_probe.py` (9KB) | ⚠️ Partial |
| `SEPProbeOnlineTraining.swift` (26KB) | — | ❌ Missing |
| `SidecarSafetyLogitProcessor.swift` (6KB) | — | ❌ Missing |

**Missing (critical)**: `JailbreakEntropyExperiment`, `SEPProbeOnlineTraining`, `HiddenStateTaps`

---

### Safety (27 Swift → 26 Python) ✅

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `AdapterCapability.swift` (7KB) | `adapter_capability.py` (8KB) | ✅ Ported |
| `AdapterSafetyModels.swift` (4KB) | `adapter_safety_models.py` (4KB) | ✅ Ported |
| `AdapterSafetyProbe.swift` (6KB) | `adapter_safety_probe.py` (7KB) | ✅ Ported |
| `BehavioralProbes.swift` (16KB) | `behavioral_probes.py` (17KB) | ✅ Ported |
| `CapabilityGuard.swift` (14KB) | `capability_guard.py` (15KB) | ✅ Ported |
| `CircuitBreakerIntegration.swift` (24KB) | `circuit_breaker_integration.py` (13KB) | ⚠️ Partial |
| `DatasetSafetyScanner.swift` (12KB) | `dataset_safety_scanner.py` (9KB) | ✅ Ported |
| `DeltaFeatureExtractor.swift` (8KB) | `delta_feature_extractor.py` (9KB) | ✅ Ported |
| `DeltaFeatureSet.swift` (1KB) | `delta_feature_set.py` (3KB) | ✅ Ported |
| `InterventionConfirmationCoordinator.swift` (11KB) | — | ❌ Missing |
| `InterventionExecutor.swift` (18KB) | `intervention_executor.py` (11KB) | ⚠️ Partial |
| `OpenAIModerationClient.swift` (5KB) | — | ❌ Skip (third-party API) |
| `OutputSafetyGuard.swift` (7KB) | `output_safety_guard.py` (7KB) | ✅ Ported |
| `OutputSafetyResult.swift` (3KB) | `output_safety_result.py` (4KB) | ✅ Ported |
| `RedTeamProbe.swift` (7KB) | `red_team_probe.py` (13KB) | ✅ Ported (Python more!) |
| `RegexContentFilter.swift` (14KB) | `regex_content_filter.py` (9KB) | ✅ Ported |
| `RuntimeCanaryScheduler.swift` (2KB) | — | ❌ Missing |
| `SafeLoRAProjector.swift` (3KB) | `safe_lora_projector.py` (7KB) | ✅ Ported |
| `SafetyAuditLog.swift` (4KB) | `safety_audit_log.py` (4KB) | ✅ Ported |
| `SafetyModels.swift` (9KB) | `safety_models.py` (9KB) | ✅ Ported |
| `SecurityEvent.swift` (1KB) | `security_event.py` (4KB) | ✅ Ported |
| `StreamingTokenBuffer.swift` (3KB) | `streaming_token_buffer.py` (3KB) | ✅ Ported |
| `TrainingDataSafetyValidator.swift` (9KB) | `training_data_safety_validator.py` (13KB) | ✅ Ported |
| `TrainingSample.swift` (2KB) | `training_sample.py` (3KB) | ✅ Ported |
| Calibration/ (2 files) | calibration/ (3 files) | ✅ Ported |
| SidecarSafety/ (5 files) | sidecar/ (6 files) | ✅ Ported |
| StabilitySuite/ (2 files) | stability_suite/ (3 files) | ✅ Ported |

**Missing (low priority)**: `InterventionConfirmationCoordinator`, `RuntimeCanaryScheduler`

---

### Training (37 Swift → 15 Python) ⚠️

> [!WARNING]
> Training domain has significant gaps. The Swift codebase has extensive MLX training engine extensions not yet ported.

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `CheckpointManager.swift` (30KB) | `checkpoints.py` (15KB) | ⚠️ Partial |
| Checkpoints/ (8 files) | — | ⚠️ Partial (merged) |
| `GeometricMetricsCollector.swift` (14KB) | `geometric_metrics_collector.py` (7KB) | ⚠️ Partial |
| `GeometricTrainingMetrics.swift` (22KB) | `geometric_training_metrics.py` (15KB) | ✅ Ported |
| `GradientSmoothnessEstimator.swift` (4KB) | `gradient_smoothness_estimator.py` (6KB) | ✅ Ported |
| `HessianEstimator.swift` (24KB) | `hessian_estimator.py` (10KB) | ⚠️ Partial |
| `IdleTrainingScheduler.swift` (15KB) | `idle_training_scheduler.py` (14KB) | ✅ Ported |
| `LoRAAdapterGeometry.swift` (5KB) | — | ❌ Missing |
| `LossLandscapeComputer.swift` (16KB) | `loss_landscape.py` (11KB) | ✅ Ported |
| `MLXOptimizationStrategies.swift` (13KB) | — | ❌ Missing |
| `MLXQuantizationSupport.swift` (1KB) | — | ❌ Missing |
| `MLXTrainingEngine+AdapterRegistration.swift` (10KB) | — | ❌ Missing |
| `MLXTrainingEngine+CoreTraining.swift` (57KB) | — | ❌ Missing |
| `MLXTrainingEngine+Evaluation.swift` (19KB) | `evaluation.py` (13KB) | ⚠️ Partial |
| `MLXTrainingEngine+JobControl.swift` (3KB) | — | ❌ Missing |
| `MLXTrainingEngine+LoRA.swift` (51KB) | `lora.py` (12KB) | ⚠️ Partial (40KB gap!) |
| `MLXTrainingEngine+LoRATargetResolution.swift` (4KB) | — | ❌ Missing |
| `MLXTrainingEngine+MemoryMonitoring.swift` (10KB) | — | ❌ Missing |
| `MLXTrainingEngine+ModelLoading.swift` (34KB) | — | ❌ Missing |
| `MLXTrainingEngine+Optimizations.swift` (22KB) | — | ❌ Missing |
| `MLXTrainingEngine+Scheduling.swift` (11KB) | `scheduling.py` (9KB) | ⚠️ Partial |
| `MLXTrainingEngine+Tokenization.swift` (9KB) | — | ❌ Missing |
| `MLXTrainingEngine+TrainingSupport.swift` (41KB) | — | ❌ Missing |
| `MLXTrainingEngine.swift` (9KB) | `engine.py` (15KB) | ⚠️ Partial |
| `ModelArchitectureConfig.swift` (5KB) | — | ❌ Missing |
| `ModelArchitectureHeuristics.swift` (1KB) | — | ❌ Missing |
| `ParameterThresholds.swift` (2KB) | — | ❌ Missing |
| Preflight/ (2 files) | — | ❌ Missing |
| `TrainingBenchmark.swift` (10KB) | — | ❌ Missing |
| `TrainingError.swift` (4KB) | — | ❌ Missing |
| `TrainingHyperparameterValidator.swift` (8KB) | `validation.py` (4KB) | ⚠️ Partial |
| `TrainingResourceGuard.swift` (24KB) | `resources.py` (15KB) | ⚠️ Partial |

**Missing (critical)**: MLXTrainingEngine extensions (~230KB of Swift code)

---

### Agents (27 Swift → 12 Python) ⚠️

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `AgentAction.swift` (7KB) | — | ❌ Missing |
| `AgentActionSchema.swift` (1KB) | — | ❌ Missing |
| `AgentActionValidator.swift` (4KB) | — | ❌ Missing |
| `AgentEvalSuiteEngine.swift` (18KB) | `agent_eval_suite_engine.py` (21KB) | ✅ Ported |
| `AgentEvalSuiteModels.swift` (15KB) | — | ❌ Missing (in engine) |
| `AgentJSONSnippetExtractor.swift` (3KB) | — | ❌ Missing |
| `AgentPromptSanitizer.swift` (5KB) | — | ❌ Missing |
| `AgentTrace.swift` (9KB) | — | ❌ Missing |
| `AgentTraceAnalytics.swift` (5KB) | — | ❌ Missing |
| `AgentTraceMiner.swift` (11KB) | — | ❌ Missing |
| `AgentTraceSanitizer.swift` (3KB) | — | ❌ Missing |
| `AgentTraceSpans.swift` (3KB) | — | ❌ Missing |
| `AgentTraceValue.swift` (6KB) | — | ❌ Missing |
| `ComputationalGateAtlas.swift` (53KB) | `computational_gate_atlas.py` (27KB) | ⚠️ Partial |
| `ConceptualGenealogyAtlas.swift` (10KB) | — | ❌ Missing |
| `IntrinsicIdentityRules.swift` (4KB) | — | ❌ Missing |
| `LoRAExpert.swift` (16KB) | — | ❌ Missing |
| `MetaphorInvariantAtlas.swift` (31KB) | — | ❌ Missing |
| `MonocleTraceImporter.swift` (17KB) | — | ❌ Missing (third-party) |
| `SemanticConceptAtlas.swift` (19KB) | — | ❌ Missing |
| `SemanticConceptInventory.swift` (20KB) | — | ❌ Missing |
| `SemanticPrimeAtlas.swift` (19KB) | `semantic_prime_atlas.py` (14KB) | ✅ Ported |
| `SemanticPrimeDriftDetector.swift` (4KB) | `semantic_prime_drift.py` (3KB) | ✅ Ported |
| `SemanticPrimeFrames.swift` (33KB) | `semantic_prime_frames.py` (2KB) | 🔴 Stub only |
| `SemanticPrimeMultilingualInventory.swift` (64KB) | `semantic_prime_multilingual.py` (6KB) | 🔴 Stub only |
| `SequenceInvariantAtlas.swift` (47KB) | `sequence_invariant_atlas.py` (46KB) | ✅ Ported |
| `TaskDiversionDetector.swift` (9KB) | `task_diversion_detector.py` (7KB) | ✅ Ported |

**Python-only**: `emotion_concept_atlas.py` (44KB), `unified_atlas.py` (17KB), `semantic_primes.py`

**Missing (critical)**: AgentTrace* suite, SemanticConceptAtlas, MetaphorInvariantAtlas

---

### Thermodynamics (14 Swift → 4 Python) 🔴

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `BehavioralOutcomeClassifier.swift` (19KB) | `behavioral_outcome_classifier.py` (11KB) | ⚠️ Partial |
| `BenchmarkRunner.swift` (17KB) | — | ❌ Missing |
| `CalorimetryResult.swift` (19KB) | — | ❌ Missing |
| `DifferentialEntropyDetector.swift` (17KB) | `differential_entropy_detector.py` (16KB) | ✅ Ported |
| `EntropyDefenseMonitor.swift` (12KB) | — | ❌ Missing |
| `LinguisticCalorimeter.swift` (30KB) | `optimization_metric_calculator.py` (5KB) | 🔴 Stub only |
| `LinguisticThermodynamics.swift` (32KB) | `linguistic_thermodynamics.py` (26KB) | ⚠️ Partial |
| `MultilingualIntensity*.swift` (3 files, 19KB) | — | ❌ Missing |
| `PhaseTransitionTheory.swift` (25KB) | `phase_transition_theory.py` (21KB) | ✅ Ported |
| `PromptPerturbationSuite.swift` (13KB) | — | ❌ Missing |
| `RidgeCrossDetector.swift` (16KB) | `ridge_cross_detector.py` (13KB) | ✅ Ported |
| `TestPromptSuite.swift` (15KB) | — | ❌ Missing |

**Missing (critical)**: `LinguisticCalorimeter` (full impl), `BenchmarkRunner`, multilingual intensity

---

### Adapters Domain (16 Swift → ~2 Python) 🔴

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `AdapterBlender.swift` (11KB) | — | ❌ Missing |
| `AdapterManifest.swift` (42KB) | — | ❌ Missing |
| `AdapterManifestBuilder.swift` (16KB) | — | ❌ Missing |
| `AdapterRelevance.swift` (17KB) | — | ❌ Missing |
| `AdapterSubscription.swift` (21KB) | — | ❌ Missing |
| `ContractValidator.swift` (36KB) | — | ❌ Missing |
| `EnsembleOrchestrator.swift` (14KB) | — | ❌ Missing |
| `GuardrailPresets.swift` (15KB) | — | ❌ Missing |
| `InvertedManifestIndex.swift` (16KB) | — | ❌ Missing |
| `LSPManifest*.swift` (3 files, 36KB) | — | ❌ Missing |
| `LSPPackage.swift` (11KB) | — | ❌ Missing |
| `LSPPublisher.swift` (44KB) | — | ❌ Missing |
| `Signal.swift` (17KB) | — | ❌ Missing |
| `SignalRouter.swift` (38KB) | — | ❌ Missing |

**Note**: This is the "Lingua Skill Protocol" (LSP) system for adapter composition. Critical for advanced multi-adapter inference.

---

### Memory Domain (11 Swift → 0 Python) 🔴

| Swift Module | Status |
|--------------|--------|
| `MLXMemoryService.swift` (17KB) | ❌ Not started |
| `MemoryManager*.swift` (5 files, 32KB) | ❌ Not started |
| `MemoryManagerConfiguration.swift` (10KB) | ❌ Not started |
| `MemoryStatistics.swift` (3KB) | ❌ Not started |
| `MemoryWarningService.swift` (9KB) | ❌ Not started |
| `SafeGPUSnapshot.swift` (5KB) | ❌ Not started |

---

### SelfImprovement Domain (4 Swift → 0 Python) 🔴

| Swift Module | Status |
|--------------|--------|
| `DPODatasetSynthesizer.swift` (22KB) | ❌ Not started |
| `DPOTrainingStrategy.swift` (8KB) | ❌ Not started |
| `FailureBatch.swift` (5KB) | ❌ Not started |
| `FailureCase.swift` (5KB) | ❌ Not started |

---

### Research Domain (3 Swift → 2 Python) ⚠️

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| `CognitivePathExperiment.swift` (28KB) | — | ❌ Missing |
| `JailbreakEntropyTaxonomy.swift` (26KB) | — | ❌ Missing |
| `TrainingObservation.swift` (17KB) | — | ❌ Missing |

Python has: `research_service.py` (15KB), but this is a service layer.

---

## Priority Recommendations

### High Priority (Core Novel Functionality)

1. **MLXTrainingEngine Extensions** (~230KB Swift)
   - Core training loop, LoRA injection, model loading
   - Required for training parity

2. **Entropy Domain Gaps** (~60KB Swift remaining)
   - `JailbreakEntropyExperiment`, `SEPProbeOnlineTraining`
   - `HiddenStateTaps` (complex - requires MLX module hooks)
   - ✅ `EntropyDeltaTracker`, `MetricsRingBuffer` now ported
   - Required for safety monitoring

3. **Thermodynamics** (~80KB Swift)
   - `LinguisticCalorimeter`, `DifferentialEntropyDetector`
   - Required for training dynamics analysis

4. **Agent Trace Suite** (~60KB Swift)
   - Agent observability and debugging
   - Required for agentic workflows

### Medium Priority

5. **Adapters/LSP System** (~300KB Swift)
   - Multi-adapter composition
   - Skill routing and blending

6. **Memory Management** (~75KB Swift)
   - GPU memory optimization
   - Batch sizing heuristics

7. **SelfImprovement** (~40KB Swift)
   - DPO dataset synthesis
   - Failure analysis

### Low Priority

8. **Research Domain** (~70KB Swift)
   - Experimental features
   - Jailbreak taxonomy

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Swift Domain Files | ~330 |
| Total Python Domain Files | ~180 |
| Estimated Parity | ~55% |
| Swift LOC (Domain only) | ~1.5M bytes |
| Python LOC (Domain only) | ~800K bytes |
| Critical Gaps | MLXTrainingEngine, Entropy suite, Thermodynamics |

---

## Next Steps

1. **Phase 1**: Port MLXTrainingEngine core extensions
2. **Phase 2**: Complete Entropy domain gaps
3. **Phase 3**: Port remaining Thermodynamics modules
4. **Phase 4**: Port Agent Trace observability suite
5. **Phase 5**: Consider Adapters/LSP system
