# ModelCypher Parity Tracker

**Last Updated**: 2025-12-23

This document tracks parity between TrainingCypher (Swift) and ModelCypher (Python).
Status values: DONE, PARTIAL, PLANNED.

Focus: High-dimensional geometry, core math, CLI/MCP parity. RAG tooling is de-prioritized.

---

## Executive Summary

| Domain | Swift Files | Python Files | Status |
|--------|-------------|--------------|--------|
| **Geometry** | 45 | 59 | ✅ ~95% Complete |
| **Entropy** | 26 | 18 | ✅ ~85% Complete |
| **Safety** | 27 | 23 | ✅ ~95% Complete |
| **Training** | 37 | 24 | ✅ Core Complete |
| **Agents** | 27 | 24 | ✅ ~90% Complete |
| **Thermodynamics** | 14 | 4 | ✅ Core Complete |
| **Validation** | 12 | 9 | ✅ ~90% Complete |
| **Dataset** | 10 | 9 | ✅ ~95% Complete |
| **Adapters** | 16 | 4 | ⚪ Out of Scope |
| **Inference** | 10 | ~3 | ⚪ Out of Scope |
| **Memory** | 11 | ~0 | ⚪ Out of Scope |

**Note**: Domains marked "Out of Scope" are platform-specific (macOS/MLX) or experimental features not required for ModelCypher's core functionality.

---

## Phase 2: Core Domain Parity - DONE

Comprehensive port of TrainingCypher core domain modules to Python (Dec 2025).
All modules syntax-verified, import-tested, and passing 1020 unit tests.

### Safety Domain (25 files, ~2,540 LOC) - DONE
- Safety models, security events, training samples
- Adapter safety types (status, tier, triggers, scorecard)
- Output safety (guards, streaming buffer, audit log)
- Dataset safety scanner with two-layer moderation
- Capability guard and audit logging
- Adapter safety probes (delta features, L2 norms, sparsity)
- Safe LoRA projector
- Calibration (geometric alignment, semantic prime baseline)
- Sidecar safety (decisions, configuration, session control, policy)
- Stability suite (models, prompt battery)

### Entropy Domain (14 files, ~3,350 LOC) - DONE
- Logit entropy/divergence calculators
- Sliding window entropy tracker with circuit breaker
- Metrics ring buffer with event markers
- Entropy delta tracker (dual-path adapter security)
- Adapter stack analyzer
- Chunk entropy analyzer (RAG injection detection)
- Conversation entropy tracker (oscillation/manipulation detection)
- Model state classifier (2D entropy+variance)
- SEP probe online training
- Geometric alignment (Sentinel/Oscillator/Monitor/Director)

### Agents Domain (12 files, ~3,000 LOC) - DONE
- Agent trace (spans, status, summary, source, store)
- Agent trace sanitizer and value types
- Agent trace analytics (compliance, entropy buckets)
- Agent action validator
- Agent prompt sanitizer (injection detection)
- Agent JSON extractor
- LoRA expert routing (registry, activation)
- Monocle/OpenTelemetry trace importer
- Intrinsic identity rules
- Metaphor invariant atlas

### Training Domain (11 files, ~2,200 LOC) - DONE
- Checkpoint validation, recovery, retention, persistence
- Model architecture heuristics (VRAM, batch size)
- Parameter thresholds
- Preflight checks
- Architecture config loader
- Training benchmark and comparison
- Training notifications
- Resource guard

### Validation Domain (6 files, ~2,300 LOC) - DONE
- Dataset validation models and stats
- Dataset format analyzer (text/chat/instruction/completion/tools)
- Dataset validator (quick and full modes with caching)
- Intrinsic identity linter
- Dataset text extractor
- Dataset file enumerator (JSONL/CSV/TSV/JSON with gzip)

### Dataset Domain (8 files, ~1,700 LOC) - DONE
- Chat message types
- Chat template library (24 templates: Llama3, Qwen, Gemma, Mistral, Phi, Cohere, DeepSeek, Granite, etc.)
- Document chunker (hierarchical with boundary preservation)
- Dataset slicer (head/chunk modes)
- Streaming shuffler (memory-bounded reservoir sampling)
- JSONL parser with normalization
- Dataset export formatter (format conversion)
- Token counter service with LRU cache

---

## CLI Command Groups

| Command Group | Status | Commands |
|---------------|--------|----------|
| inventory | PARTIAL | outputs differ; missing system/version alignment |
| explain | PARTIAL | stub output only |
| train | PARTIAL | start/preflight/status/pause/resume/cancel/logs/export; no real MLX training |
| job | PARTIAL | list/show/attach/delete; missing filters and loss stats |
| checkpoint | PARTIAL | list/delete/export (in train.py); export formats stubbed |
| model | PARTIAL | list/register/delete/fetch/search/probe/validate-merge/analyze-alignment/merge/geometric-merge/unified-merge |
| dataset | DONE | validate/preprocess/list/delete/pack-asif/quality/auto-fix/preview/get-row/update-row/add-row/delete-row/convert/format-analyze/chunk/template |
| doc | PARTIAL | convert/validate; preflight missing |
| system | PARTIAL | status/probe; readiness details need parity |
| eval | PARTIAL | list/show/run |
| compare | PARTIAL | list/show/run/checkpoints/baseline/score |
| validate | PARTIAL | train/dataset |
| estimate | PARTIAL | train |
| infer | PARTIAL | mlx-lm inference + run/suite/batch; security scan heuristic |
| geometry | DONE | 13 submodules: crm/emotion/adapter/invariant/manifold/metrics/path/persona/refinement/refusal/safety/sparse/stitch/training/transport |
| entropy | DONE | analyze/detect-distress/verify-baseline/window/conversation-track/dual-path |
| safety | DONE | adapter-probe/dataset-scan/lint-identity |
| agent | DONE | trace-import/trace-analyze/validate-action |
| agent-eval | PARTIAL | run/results |
| adapter | DONE | inspect/project/wrap-mlx/smooth/merge |
| calibration | PARTIAL | run/status/apply (in adapter.py) |
| thermo | DONE | analyze/path/entropy/measure/detect/detect-batch/ridge-detect/phase/sweep |
| storage | PARTIAL | filesystem + manifold profile store |
| research | DONE | taxonomy run/cluster/report |
| sparse-region | DONE | domains/locator/prober/validator (in geometry/sparse) |
| rag | OPTIONAL | de-prioritized |
| stability | PLANNED | |
| dashboard | PLANNED | |

---

## MCP Tools

**Total: 136 tools implemented** (organized by category)

### Core Tools
| Tool | Status |
|------|--------|
| mc_inventory | PARTIAL |
| mc_settings_snapshot | DONE |
| mc_system_status | PARTIAL |
| mc_doc_convert | DONE |
| mc_help_ask | DONE |
| mc_schema | DONE |

### Training & Jobs
| Tool | Status |
|------|--------|
| mc_train_start | PARTIAL |
| mc_train_preflight | DONE |
| mc_train_export | DONE |
| mc_job_status/list/detail/cancel/pause/resume/delete | PARTIAL |
| mc_validate_train | PARTIAL |
| mc_estimate_train | PARTIAL |
| mc_checkpoint_list/export/delete | PARTIAL |

### Model Management
| Tool | Status |
|------|--------|
| mc_model_list/fetch/search/probe/register/delete | DONE |
| mc_model_validate_merge/analyze_alignment | DONE |
| mc_model_merge | PARTIAL |

### Inference
| Tool | Status |
|------|--------|
| mc_infer | PARTIAL |
| mc_infer_run/batch/suite | PARTIAL |

### Geometry (40 tools)
| Tool | Status |
|------|--------|
| mc_geometry_validate | DONE |
| mc_geometry_path_detect/compare | DONE |
| mc_geometry_training_status/history | DONE |
| mc_geometry_primes_list/probe/compare | DONE |
| mc_geometry_crm_build/compare/sequence_inventory | DONE |
| mc_geometry_stitch_analyze/apply/train | PARTIAL |
| mc_geometry_dare_sparsity | DONE |
| mc_geometry_dora_decomposition | DONE |
| mc_geometry_gromov_wasserstein | DONE |
| mc_geometry_intrinsic_dimension | DONE |
| mc_geometry_topological_fingerprint | DONE |
| mc_geometry_sparse_domains/locate | DONE |
| mc_geometry_refusal_pairs/detect | DONE |
| mc_geometry_persona_traits/extract/drift | DONE |
| mc_geometry_manifold_cluster/dimension/query | DONE |
| mc_geometry_transport_merge/synthesize | DONE |
| mc_geometry_invariant_map_layers/collapse_risk | DONE |
| mc_geometry_atlas_inventory | DONE |
| mc_geometry_refinement_analyze | DONE |
| mc_geometry_domain_profile | DONE |
| mc_geometry_safety_jailbreak_test | DONE |

### Safety & Entropy
| Tool | Status |
|------|--------|
| mc_safety_circuit_breaker | DONE |
| mc_safety_persona_drift | DONE |
| mc_safety_redteam_scan | DONE |
| mc_safety_behavioral_probe | DONE |
| mc_safety_adapter_probe | DONE |
| mc_safety_dataset_scan | DONE |
| mc_safety_lint_identity | DONE |
| mc_entropy_analyze | DONE |
| mc_entropy_detect_distress | DONE |
| mc_entropy_verify_baseline | DONE |
| mc_entropy_window | DONE |
| mc_entropy_conversation_track | DONE |
| mc_entropy_dual_path | DONE |

### Dataset
| Tool | Status |
|------|--------|
| mc_dataset_validate | PARTIAL |
| mc_dataset_get_row/update_row/add_row/delete_row | DONE |
| mc_dataset_convert | DONE |
| mc_dataset_list/delete/preprocess | DONE |
| mc_dataset_format_analyze | DONE |
| mc_dataset_chunk | DONE |
| mc_dataset_template | DONE |

### Agent
| Tool | Status |
|------|--------|
| mc_agent_eval_run/results | PARTIAL |
| mc_agent_trace_import | DONE |
| mc_agent_trace_analyze | DONE |
| mc_agent_validate_action | DONE |

### Thermodynamics
| Tool | Status |
|------|--------|
| mc_thermo_measure | DONE |
| mc_thermo_detect/detect_batch | DONE |
| mc_thermo_analyze/path/entropy | DONE |

### Adapters & Ensembles
| Tool | Status |
|------|--------|
| mc_adapter_inspect/merge | DONE |
| mc_ensemble_create/run/list/delete | DONE |

### Merge Validation (NEW)
| Tool | Status |
|------|--------|
| mc_merge_validate | DONE |
| mc_merge_perplexity | DONE |
| mc_merge_coherence | DONE |
| mc_merge_probe | DONE |
| mc_merge_diagnose | DONE |

### Evaluation (NEW)
| Tool | Status |
|------|--------|
| mc_eval_run/list/show | PARTIAL |

### Calibration (NEW)
| Tool | Status |
|------|--------|
| mc_calibration_run/status/apply | PARTIAL |

### Stability (NEW)
| Tool | Status |
|------|--------|
| mc_stability_run/report | PLANNED |

### Dashboard (NEW)
| Tool | Status |
|------|--------|
| mc_dashboard_metrics/export | PLANNED |

### Research (NEW)
| Tool | Status |
|------|--------|
| mc_research_sparse_region | DONE |
| mc_research_afm | DONE |

### Storage
| Tool | Status |
|------|--------|
| mc_storage_usage | DONE |
| mc_storage_cleanup | DONE |

### RAG (Optional)
| Tool | Status |
|------|--------|
| mc_rag_build/query/list/delete | PARTIAL |

---

## Detailed Domain Inventory

### Geometry (45 Swift → 59 Python) ✅ ~95%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| AffineStitchingLayer.swift | affine_stitching_layer.py | ✅ |
| AnchorInvarianceAnalyzer.swift | anchor_invariance_analyzer.py | ✅ |
| CompositionalProbes.swift | compositional_probes.py | ✅ |
| ConceptDetector.swift | concept_detector.py | ✅ |
| ConceptResponseMatrix.swift | concept_response_matrix.py | ✅ |
| CrossArchitectureLayerMatcher.swift | cross_architecture_layer_matcher.py | ✅ |
| DARESparsityAnalyzer.swift | dare_sparsity.py | ✅ |
| DoRADecomposition.swift | dora_decomposition.py | ✅ |
| GateDetector.swift | gate_detector.py | ✅ |
| GeneralizedProcrustes.swift | generalized_procrustes.py | ✅ |
| GeometricFingerprint.swift | geometry_fingerprint.py | ✅ |
| GeometryValidationSuite.swift | geometry_validation_suite.py | ✅ |
| IntrinsicDimensionEstimator.swift | intrinsic_dimension_estimator.py | ✅ |
| InvariantLayerMapper.swift | invariant_layer_mapper.py | ✅ |
| ManifoldClusterer.swift | manifold_clusterer.py | ✅ |
| ManifoldDimensionality.swift | manifold_dimensionality.py | ✅ |
| ManifoldProfileService.swift | manifold_profile_service.py | ✅ |
| PermutationAligner.swift | permutation_aligner.py | ✅ |
| RefusalDirectionDetector.swift | refusal_direction_detector.py | ✅ |
| SharedSubspaceProjector.swift | shared_subspace_projector.py | ✅ |
| SparseRegion*.swift | sparse_region_*.py | ✅ |
| TopologicalFingerprint.swift | topological_fingerprint.py | ✅ |
| TransportGuidedMerger.swift | transport_guided_merger.py | ✅ |

**Python-only**: gromov_wasserstein.py, refinement_density.py, manifold_stitcher.py

### Entropy (26 Swift → 18 Python) ✅ ~85%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| AdapterStackAnalyzer.swift | adapter_stack_analyzer.py | ✅ Phase 2 |
| BaselineVerificationProbe.swift | baseline_verification_probe.py | ✅ |
| ChunkEntropyAnalyzer.swift | chunk_entropy_analyzer.py | ✅ |
| ConflictScore.swift | conflict_score.py | ✅ Phase 2 |
| ConversationEntropyTracker.swift | conversation_entropy_tracker.py | ✅ |
| EntropyDeltaSample.swift | entropy_delta_sample.py | ✅ Phase 2 |
| EntropyDeltaTracker.swift | entropy_delta_tracker.py | ✅ |
| EntropyPatternDetector.swift | entropy_pattern_detector.py | ✅ |
| EntropyTracker.swift | entropy_tracker.py | ✅ |
| EntropyWindow.swift | entropy_window.py | ✅ Phase 2 |
| HiddenStateExtractor.swift | hidden_state_extractor.py | ✅ |
| LogitEntropyCalculator.swift | logit_entropy_calculator.py | ✅ Phase 2 |
| LogitDivergenceCalculator.swift | logit_divergence_calculator.py | ✅ Phase 2 |
| MetricsRingBuffer.swift | metrics_ring_buffer.py | ✅ |
| ModelStateClassifier.swift | model_state_classifier.py | ✅ |
| SEPProbe.swift | sep_probe.py | ✅ |
| SEPProbeOnlineTraining.swift | sep_probe_online_training.py | ✅ Phase 2 |

**Remaining**: JailbreakEntropyExperiment, HiddenStateTaps (complex MLX hooks)

### Safety (27 Swift → 23 Python) ✅ ~95%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| AdapterCapability.swift | adapter_capability.py | ✅ |
| AdapterSafetyModels.swift | adapter_safety_models.py | ✅ |
| AdapterSafetyProbe.swift | adapter_safety_probe.py | ✅ |
| BehavioralProbes.swift | behavioral_probes.py | ✅ |
| CapabilityGuard.swift | capability_guard.py | ✅ |
| DatasetSafetyScanner.swift | dataset_safety_scanner.py | ✅ |
| DeltaFeatureExtractor.swift | delta_feature_extractor.py | ✅ |
| DeltaFeatureSet.swift | delta_feature_set.py | ✅ |
| OutputSafetyGuard.swift | output_safety_guard.py | ✅ |
| RedTeamProbe.swift | red_team_probe.py | ✅ |
| RegexContentFilter.swift | regex_content_filter.py | ✅ |
| SafeLoRAProjector.swift | safe_lora_projector.py | ✅ |
| SafetyAuditLog.swift | safety_audit_log.py | ✅ |
| SafetyModels.swift | safety_models.py | ✅ |
| SecurityEvent.swift | security_event.py | ✅ |
| StreamingTokenBuffer.swift | streaming_token_buffer.py | ✅ |
| TrainingDataSafetyValidator.swift | training_data_safety_validator.py | ✅ |
| Calibration/ | calibration/ | ✅ |
| SidecarSafety/ | sidecar/ | ✅ |
| StabilitySuite/ | stability_suite/ | ✅ |

**Remaining**: InterventionConfirmationCoordinator, RuntimeCanaryScheduler (low priority)

### Agents (27 Swift → 24 Python) ✅ ~90%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| AgentAction.swift | agent_action.py | ✅ Phase 2 |
| AgentActionValidator.swift | agent_action_validator.py | ✅ Phase 2 |
| AgentEvalSuiteEngine.swift | agent_eval_suite_engine.py | ✅ |
| AgentJSONSnippetExtractor.swift | agent_json_extractor.py | ✅ Phase 2 |
| AgentPromptSanitizer.swift | agent_prompt_sanitizer.py | ✅ Phase 2 |
| AgentTrace.swift | agent_trace.py | ✅ |
| AgentTraceAnalytics.swift | agent_trace_analytics.py | ✅ Phase 2 |
| AgentTraceSanitizer.swift | agent_trace_sanitizer.py | ✅ Phase 2 |
| AgentTraceValue.swift | agent_trace_value.py | ✅ Phase 2 |
| ComputationalGateAtlas.swift | computational_gate_atlas.py | ✅ |
| IntrinsicIdentityRules.swift | intrinsic_identity_rules.py | ✅ Phase 2 |
| LoRAExpert.swift | lora_expert.py | ✅ Phase 2 |
| MetaphorInvariantAtlas.swift | metaphor_invariant_atlas.py | ✅ Phase 2 |
| MonocleTraceImporter.swift | monocle_trace_importer.py | ✅ Phase 2 |
| SemanticPrimeAtlas.swift | semantic_prime_atlas.py | ✅ |
| SequenceInvariantAtlas.swift | sequence_invariant_atlas.py | ✅ |
| TaskDiversionDetector.swift | task_diversion_detector.py | ✅ |

**Python-only**: emotion_concept_atlas.py, unified_atlas.py

### Training (37 Swift → 24 Python) ✅ Core Complete

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| CheckpointManager.swift | checkpoints.py | ✅ |
| Checkpoints/ | checkpoint_*.py | ✅ Phase 2 |
| GeometricMetricsCollector.swift | geometric_metrics_collector.py | ✅ |
| GeometricTrainingMetrics.swift | geometric_training_metrics.py | ✅ |
| GradientSmoothnessEstimator.swift | gradient_smoothness_estimator.py | ✅ |
| HessianEstimator.swift | hessian_estimator.py | ✅ |
| IdleTrainingScheduler.swift | idle_training_scheduler.py | ✅ |
| LossLandscapeComputer.swift | loss_landscape.py | ✅ |
| ModelArchitectureHeuristics.swift | model_architecture_heuristics.py | ✅ Phase 2 |
| ParameterThresholds.swift | parameter_thresholds.py | ✅ Phase 2 |
| Preflight/ | preflight_check.py | ✅ Phase 2 |
| TrainingBenchmark.swift | training_benchmark.py | ✅ Phase 2 |
| TrainingResourceGuard.swift | resource_guard.py | ✅ Phase 2 |

**Remaining**: MLXTrainingEngine extensions (~230KB Swift) - core training loop, LoRA injection

### Validation (12 Swift → 9 Python) ✅ ~90%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| DatasetFormatAnalyzer.swift | dataset_format_analyzer.py | ✅ Phase 2 |
| DatasetValidationModels.swift | dataset_validation_models.py | ✅ Phase 2 |
| DatasetValidator.swift | dataset_validator.py | ✅ Phase 2 |
| IntrinsicIdentityLinter.swift | intrinsic_identity_linter.py | ✅ Phase 2 |
| DatasetTextExtractor.swift | dataset_text_extractor.py | ✅ Phase 2 |
| DatasetFileEnumerator.swift | dataset_file_enumerator.py | ✅ Phase 2 |

### Dataset (10 Swift → 9 Python) ✅ ~95%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| ChatMessage.swift | chat_message.py | ✅ Phase 2 |
| ChatTemplateLibrary.swift | chat_template_library.py | ✅ Phase 2 |
| DocumentChunker.swift | document_chunker.py | ✅ Phase 2 |
| DatasetSlicer.swift | dataset_slicer.py | ✅ Phase 2 |
| StreamingShuffler.swift | streaming_shuffler.py | ✅ Phase 2 |
| JSONLParser.swift | jsonl_parser.py | ✅ Phase 2 |
| DatasetExportFormatter.swift | dataset_export_formatter.py | ✅ Phase 2 |
| TokenCounterService.swift | token_counter_service.py | ✅ Phase 2 |

### Thermodynamics (14 Swift → 4 Python) ✅ Core Complete

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| BehavioralOutcomeClassifier.swift | behavioral_outcome_classifier.py | ✅ |
| DifferentialEntropyDetector.swift | differential_entropy_detector.py | ✅ |
| LinguisticThermodynamics.swift | linguistic_thermodynamics.py | ✅ |
| PhaseTransitionTheory.swift | phase_transition_theory.py | ✅ |
| RidgeCrossDetector.swift | ridge_cross_detector.py | ✅ |
| PromptPerturbationSuite.swift | prompt_perturbation_suite.py | ✅ |

**Remaining**: LinguisticCalorimeter (full), BenchmarkRunner, multilingual intensity

### Adapters (16 Swift → ~4 Python) 🔴 ~25%

| Swift Module | Python Equivalent | Status |
|--------------|-------------------|--------|
| LoRAAdapterMerger.swift | lora_adapter_merger.py | ✅ |
| UnifiedManifoldMerger.swift | unified_manifold_merger.py | ⚠️ Core only |
| RotationalModelMerger.swift | rotational_merger.py | ✅ |
| AdapterBlender.swift | adapter_blender.py | ✅ Phase 2 |
| EnsembleOrchestrator.swift | ensemble_orchestrator.py | ✅ Phase 2 |

**Remaining**: LSP system (Lingua Skill Protocol) for multi-adapter composition (~300KB)

### Memory (11 Swift → 0 Python) 🔴

Not started. Includes MLXMemoryService, MemoryManager, SafeGPUSnapshot.

---

## Scope and Status

### Completed ✅
1. **Core Domain Parity (Phase 2)** - Safety, Entropy, Agents, Training, Validation, Dataset

### Out of Scope (Not Planned for Porting)
The following domains exist in TrainingCypher (Swift) but are intentionally **not planned** for porting to ModelCypher:
- **Memory Management** - MLXMemoryService, MemoryManager, SafeGPUSnapshot (platform-specific)
- **RAG Domain** - De-prioritized, optional functionality
- **Adapters/LSP System** - Multi-adapter composition (~300KB Swift)
- **Research Domain** - Experimental features
- **SelfImprovement** - DPO synthesis
- **MLXTrainingEngine extensions** - Core training loop deeply tied to macOS/MLX

These are platform-specific features or experimental research modules that are not required for ModelCypher's core functionality.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Swift Domain Files | ~330 |
| Total Python Domain Files | 174 |
| Functional Parity | ~75% (core domains complete) |
| Phase 2 Files Added | 76 |
| Phase 2 LOC Added | ~15,090 |
| Test Count | 1717 passing |
| Test Files | 125 |
| Import Coverage | 98% (174/178 modules load) |
