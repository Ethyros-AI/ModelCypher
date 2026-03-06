# Product-Maintenance Drift Audit

Date: 2026-03-06

Scope:
- `src/`
- `scripts/`
- `tests/`
- `docs/`

Audit rule:
- No backward compatibility
- No permissive unknown defaults
- No user-facing bypasses in canonical runtime paths
- No convenience wrappers that duplicate a canonical operator surface
- Exact alternates are allowed only when they preserve operator semantics

Scanner:
- `poetry run python scripts/report_doctrine_audit.py`

Current scanner inventory after cleanup:
- `P1 legacy_alias_or_deprecated = 29`
- `P2 heuristic_or_product_language = 297`
- `P2 override_or_bypass = 208`
- Root split:
  - `docs`: `P1=20`, `P2=170`
  - `scripts`: `P1=6`, `P2=85`
  - `src`: `P2=170`
  - `tests`: `P1=3`, `P2=80`

Runtime result:
- `src/` is clean for scanner-defined `P0/P1`
- Unresolved `P0/P1` in canonical runtime paths: none

## Resolved Findings

| Severity | Category | Path / line | Violation | Decision | Replacement rule |
| --- | --- | --- | --- | --- | --- |
| P1 | Convenience wrapper | `src/modelcypher/adapters/model_loader.py` (wrappers deleted; canonical class begins at line 36) | Loader aliases duplicated the canonical loader surface and kept multiple call paths alive. | Delete | `ModelLoader` is the only loader surface. Callers instantiate `ModelLoader(backend)` directly. |
| P1 | Convenience wrapper | `src/modelcypher/cli/composition.py:92-111` and `src/modelcypher/cli/commands/safety/geodesic_trajectory.py:92-98` | `get_model_loader()` in the CLI composition root kept a second loader access path alive. | Delete | Composition keeps `get_backend()`; loader callers construct `ModelLoader(get_backend())`. |
| P1 | Import shim | `src/modelcypher/core/domain/geometry/orthogonal_probe_generator.py` | Pure re-export shim existed only to preserve old imports. | Delete | Import directly from `null_space`, `probe_generator`, `trajectory_analysis`, or `trajectory_projection`. |
| P0 | Unknown provenance treated as mergeable | `src/modelcypher/experimental/merge/lora_adapter_merger.py:172-197` | Merge used adapter metadata without strict literal validation. Unknown values could be treated as capability-carrying. | Refactor | Missing provenance raises `MergeError`; `capability_transfer` must be literal `'true'` or `'false'`; `'false'` is rejected. |
| P0 | Unknown activation schema accepted | `src/modelcypher/core/domain/profile.py:838-855` | `load_activations()` accepted old tensor-key formats instead of enforcing the canonical schema. | Refactor | Unknown tensor keys now raise `ValueError`. Profile activations must use canonical names. |
| P0 | Missing measurement treated as safe | `src/modelcypher/experimental/merge/models.py:188-212` | `get_transfer_safety()` returned `1.0` when no boundary radii were measured and treated missing layer radii as maximally safe. | Refactor | Missing `boundary_radii` raises `ValueError`; missing layer radius raises `KeyError`; zero maximum radius returns `0.0`. |
| P0 | Undefined local geometry defaulted to unit scale | `src/modelcypher/core/domain/geometry/geodesic_deviation.py:443-463` | `_get_local_edge_length()` returned `1.0` when the graph carried no local geometric evidence. | Refactor | Undefined local edge length now raises `ValueError`. |
| P1 | Payload alias preserved old output contract | `src/modelcypher/core/use_cases/adapter_divergence_profile_service.py:144-147` | `layer_agreement_rate` duplicated `dominant_adapter_rate` only to preserve an older payload shape. | Delete | `dominant_adapter_rate` is the only exported field. |
| P1 | Old path label preserved | `src/modelcypher/experimental/merge/stages/transplant_stage.py:338-345` | `legacy_profile_path` encoded an old path contract and old naming. | Refactor | The path is now `profile_path`; labels use `profile` vs `trajectory`. |
| P1 | Compatibility signature branch | `src/modelcypher/core/domain/lora_memory_store.py:814-817` | `try/except TypeError` preserved an old `derive_optimizer_geometry_config()` call signature. | Delete | Only the current optimizer derivation signature is accepted. |
| P2 | Canonical train CLI exposed manual LR | `src/modelcypher/cli/commands/train.py:80-181` | `--lr` kept a user-facing hyperparameter escape hatch in the one training path. | Delete | `mc train run` exposes instrumentation only; LR is always derived. |
| P2 | Canonical training service accepted manual safety bypasses | `src/modelcypher/core/use_cases/dataset_training_service.py:398-1400` | `lr_override`, `constraint_state_override`, `scale_bound_override`, and `research_allow_quantization_frontier_invalid` reopened non-derived branches inside the canonical training path. | Delete | Training always derives LR, scale bounds, and quantization-frontier gating from measurement. |
| P2 | MASS math layer preserved LR override bypass | `src/modelcypher/core/domain/training/mass_step_size.py:42-189` and `src/modelcypher/backends/_mlx_training_adapter_train_mixin.py:45-706` | The pure math and adapter loop both allowed explicit LR override to bypass Weyl and validation-derived bounds. | Delete | MASS now has one derivation path: spectral ceiling, sqrt(N) correction, and validation backoff. |
| P2 | Adapter injection preserved global scale-bound override | `src/modelcypher/backends/_mlx_training_adapter_core_mixin.py:569-621` | `scale_bound_override` bypassed per-layer spectral safety. | Delete | NB-LoRA scale bound is always `(sigma_k / 2) * (1 - sqrt(eps))`. |
| P2 | Merge injection-layer environment override | `src/modelcypher/experimental/merge/pipeline.py:860-882` and `src/modelcypher/experimental/merge/stages/probe_from_profile.py:255-275` | `MC_INJECTION_LAYER` let callers force a merge location outside measured alignment. | Delete | Injection layer comes only from measured profile alignment or measured transmission geometry. |
| P2 | Embedding transplant environment overrides | `src/modelcypher/experimental/merge/stages/transplant_embeddings.py:60-114` | `MC_SKIP_EMBEDDING_TRANSPLANT` and `MC_FORCE_EMBEDDING_TRANSPLANT` turned a hard geometry boundary into a manual escape hatch. | Delete | Unsafe cross-vocab embedding transplant stays blocked; no environment override path remains. |

## Exact Alternates Kept

These survived the audit because they are not backward compatibility. They are exact or backend-equivalent execution alternates.

| Path / line | Alternate | Why it stays |
| --- | --- | --- |
| `src/modelcypher/experimental/merge/stages/probe_inference.py:17-21` | Parallel vs sequential probe inference | Same operator semantics; only memory scheduling changes. |
| `src/modelcypher/experimental/merge/stages/transplant_stage.py:892-898` | Pre-stacked activation matrix vs iterable of rows | Same activation tensor after stacking; one path is storage layout, not a different contract. |
| `src/modelcypher/backends/mlx_backend.py:330-333` | `self.mx.clear_cache()` vs older backend location | Backend-equivalent cache release path; semantic operator is identical. |
| `src/modelcypher/core/domain/lora_memory_store.py:467-560` | Spectral regularization helpers | Kept as explicit non-canonical experimental operators, not as compatibility shims. |

## Unresolved P0/P1

Canonical runtime (`src/`):
- None

Non-runtime archival references:
- `docs/LFM2-350M-WORK-SUMMARY.md:98,101,286,287`
- `docs/research/COMPRESSION-RESEARCH-SYNTHESIS.md:314`
- `docs/research/ENTROPY-CURVATURE-GQA-FALSIFIER-PROTOCOL.md:133,135,224`
- `docs/research/FIRST_PRINCIPLES_REVIEW_LEDGER.md:20-21`
- `docs/research/MATH-FOUNDATIONS.md:365`
- `docs/research/geometry_only_hard_mode_experiment_matrix.md:33,38,65,67,69,71,79,163`
- `docs/research/quantization_frontier_bedrock_review_2026_03_05.md:280`
- `scripts/analyze_real_adapters.py:261`
- `scripts/entropy_curvature_operator_split.py:1055`
- `scripts/experiment_adaptive_lr.py:383`
- `scripts/validate_gqa_falsifier_artifacts.py:185`
- `scripts/validate_isometry_real.py:30`
- `scripts/weight_geometry_falsification.py:502`
- `tests/cli/commands/test_train_commands.py:49,88`
- `tests/integration/test_cka_pipeline.py:99`

Adjudication:
- These are historical references, archive paths, or tests that discuss removed legacy behavior.
- They do not create live permissive behavior in canonical runtime code.
- They should be cleaned in a documentation pass if we want the entire repo text surface to stop mentioning the old vocabulary.

## Next-Wave P2 Debt

These are the highest-value remaining doctrine violations because they preserve manual bypasses or non-derived control surfaces in live runtime code.

| Path / line | Problem | Decision direction |
| --- | --- | --- |
| `src/modelcypher/core/use_cases/lora_memory_service.py:326` | Rank override language still treats manual rank injection as supported. | Remove once the memory-store path is aligned to the same single derived rank rule as main training. |
| `src/modelcypher/core/domain/training/checkpoint_models.py:149` | Memory-estimation metadata still describes optional override fields. | Replace with measured-memory inputs or remove the optional override contract. |
| `src/modelcypher/core/domain/training/geometric_training_metrics.py:153-155` | A monitoring-only heuristic score remains explicitly heuristic. | Re-derive or demote out of doctrine-critical code paths. |

## Guardrails Added

Files:
- `scripts/report_doctrine_audit.py`
- `tests/repo/test_doctrine_audit.py`
- `tests/domain/training/test_mission_alignment_training.py`
- `tests/test_lora_adapter_merger.py`
- `tests/experimental/test_merge_models.py`

What they enforce:
- No deleted loader aliases or deleted probe shim imports reappear in `src/`
- No backward-compatibility wording reappears in runtime source
- Missing or malformed adapter provenance fails closed
- Transfer-safety queries require measured boundary radii

## Verification

Commands run:
- `poetry run pytest tests/repo/test_doctrine_audit.py tests/domain/training/test_mission_alignment_training.py tests/test_lora_adapter_merger.py tests/domain/test_profile.py tests/test_adapter_divergence_profile_service.py tests/domain/geometry/test_geodesic_deviation.py tests/experimental/test_merge_models.py tests/adapters/test_model_loader_iter_weights.py tests/test_gram_aligner_integration.py -q`
- `poetry run pytest tests/integration/test_entropy_workflow.py -q`
- `poetry run pytest tests/cli/commands/test_train_commands.py tests/test_dataset_training_service_strict.py tests/test_mlx_training_adapter_strict.py tests/domain/training/test_mass_step_size.py tests/domain/training/test_mission_alignment_training.py tests/repo/test_doctrine_audit.py -q`
- `poetry run pytest tests/test_probe_from_profile.py tests/test_merge_pipeline_behavior_jacobian.py tests/experimental/test_merge_models.py -q`

Result:
- Focused doctrine suites passed after cleanup.
