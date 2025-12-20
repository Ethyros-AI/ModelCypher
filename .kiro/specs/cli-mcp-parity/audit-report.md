# CLI/MCP Parity Audit Report

## Summary

This audit compares all CLI commands with their corresponding MCP tools to identify gaps in coverage.

**Total CLI Commands:** 102
**Total MCP Tools:** 66
**Coverage:** 64.7%

## CLI Commands with MCP Coverage ✓

| CLI Command | MCP Tool | Status |
|-------------|----------|--------|
| `mc inventory` | `mc_inventory` | ✓ |
| `mc train start` | `mc_train_start` | ✓ |
| `mc train status` | `mc_job_status` | ✓ |
| `mc train pause` | `mc_job_pause` | ✓ |
| `mc train resume` | `mc_job_resume` | ✓ |
| `mc train cancel` | `mc_job_cancel` | ✓ |
| `mc job list` | `mc_job_list` | ✓ |
| `mc job show` | `mc_job_detail` | ✓ |
| `mc checkpoint export` | `mc_checkpoint_export` | ✓ |
| `mc model list` | `mc_model_list` | ✓ |
| `mc model fetch` | `mc_model_fetch` | ✓ |
| `mc model search` | `mc_model_search` | ✓ |
| `mc model probe` | `mc_model_probe` | ✓ |
| `mc model validate-merge` | `mc_model_validate_merge` | ✓ |
| `mc model analyze-alignment` | `mc_model_analyze_alignment` | ✓ |
| `mc system status` | `mc_system_status` | ✓ |
| `mc dataset validate` | `mc_dataset_validate` | ✓ |
| `mc dataset convert` | `mc_dataset_convert` | ✓ |
| `mc dataset get-row` | `mc_dataset_get_row` | ✓ |
| `mc dataset update-row` | `mc_dataset_update_row` | ✓ |
| `mc dataset add-row` | `mc_dataset_add_row` | ✓ |
| `mc dataset delete-row` | `mc_dataset_delete_row` | ✓ |
| `mc validate train` | `mc_validate_train` | ✓ |
| `mc estimate train` | `mc_estimate_train` | ✓ |
| `mc geometry validate` | `mc_geometry_validate` | ✓ |
| `mc geometry training status` | `mc_geometry_training_status` | ✓ |
| `mc geometry training history` | `mc_geometry_training_history` | ✓ |
| `mc geometry safety circuit-breaker` | `mc_safety_circuit_breaker` | ✓ |
| `mc geometry safety persona` | `mc_safety_persona_drift` | ✓ |
| `mc geometry safety jailbreak-test` | `mc_geometry_safety_jailbreak_test` | ✓ |
| `mc geometry adapter sparsity` | `mc_geometry_dare_sparsity` | ✓ |
| `mc geometry adapter decomposition` | `mc_geometry_dora_decomposition` | ✓ |
| `mc geometry primes list` | `mc_geometry_primes_list` | ✓ |
| `mc geometry primes probe` | `mc_geometry_primes_probe` | ✓ |
| `mc geometry primes compare` | `mc_geometry_primes_compare` | ✓ |
| `mc geometry stitch analyze` | `mc_geometry_stitch_analyze` | ✓ |
| `mc geometry stitch apply` | `mc_geometry_stitch_apply` | ✓ |
| `mc infer` | `mc_infer` | ✓ |
| `mc infer run` | `mc_infer_run` | ✓ |
| `mc infer suite` | `mc_infer_suite` | ✓ |
| `mc adapter merge` | `mc_adapter_merge` | ✓ |
| `mc thermo measure` | `mc_thermo_measure` | ✓ |
| `mc thermo detect` | `mc_thermo_detect` | ✓ |
| `mc thermo detect-batch` | `mc_thermo_detect_batch` | ✓ |
| `mc calibration run` | `mc_calibration_run` | ✓ |
| `mc calibration status` | `mc_calibration_status` | ✓ |
| `mc calibration apply` | `mc_calibration_apply` | ✓ |
| `mc rag index` | `mc_rag_index` | ✓ |
| `mc rag query` | `mc_rag_query` | ✓ |
| `mc rag status` | `mc_rag_status` | ✓ |
| `mc stability run` | `mc_stability_run` | ✓ |
| `mc stability report` | `mc_stability_report` | ✓ |
| `mc agent-eval run` | `mc_agent_eval_run` | ✓ |
| `mc agent-eval results` | `mc_agent_eval_results` | ✓ |
| `mc dashboard metrics` | `mc_dashboard_metrics` | ✓ |
| `mc dashboard export` | `mc_dashboard_export` | ✓ |
| `mc help ask` | `mc_help_ask` | ✓ |
| `mc schema` | `mc_schema` | ✓ |
| `mc storage status` | `mc_storage_status` | ✓ |
| `mc storage cleanup` | `mc_storage_cleanup` | ✓ |
| `mc ensemble create` | `mc_ensemble_create` | ✓ |
| `mc ensemble run` | `mc_ensemble_run` | ✓ |
| `mc research sparse-region` | `mc_research_sparse_region` | ✓ |
| `mc research afm` | `mc_research_afm` | ✓ |

## CLI Commands Missing MCP Coverage ✗

| CLI Command | Expected MCP Tool | Priority | Notes |
|-------------|-------------------|----------|-------|
| `mc explain` | `mc_explain` | Low | Utility command |
| `mc train preflight` | `mc_train_preflight` | Medium | Pre-flight validation |
| `mc train export` | `mc_train_export` | Medium | Export trained model |
| `mc train logs` | `mc_train_logs` | Low | Log streaming |
| `mc job attach` | `mc_job_attach` | Low | Log streaming |
| `mc job delete` | `mc_job_delete` | Medium | Job cleanup |
| `mc checkpoint list` | `mc_checkpoint_list` | Medium | List checkpoints |
| `mc checkpoint delete` | `mc_checkpoint_delete` | Medium | Checkpoint cleanup |
| `mc model register` | `mc_model_register` | Medium | Register local model |
| `mc model merge` | `mc_model_merge` | High | Model merging |
| `mc model delete` | `mc_model_delete` | Medium | Model cleanup |
| `mc system probe` | `mc_system_probe` | Low | System diagnostics |
| `mc dataset preprocess` | `mc_dataset_preprocess` | Medium | Dataset preprocessing |
| `mc dataset preview` | `mc_dataset_preview` | Low | Dataset preview |
| `mc dataset list` | `mc_dataset_list` | Medium | List datasets |
| `mc dataset delete` | `mc_dataset_delete` | Medium | Dataset cleanup |
| `mc dataset pack-asif` | `mc_dataset_pack_asif` | Low | ASIF packaging |
| `mc eval list` | `mc_eval_list` | Medium | List evaluations |
| `mc eval show` | `mc_eval_show` | Medium | Show evaluation |
| `mc eval run` | `mc_eval_run` | High | Run evaluation |
| `mc compare list` | `mc_compare_list` | Low | List comparisons |
| `mc compare show` | `mc_compare_show` | Low | Show comparison |
| `mc doc convert` | `mc_doc_convert` | Low | Document conversion |
| `mc doc validate` | `mc_doc_validate` | Low | Document validation |
| `mc validate dataset` | `mc_validate_dataset` | Low | Alias for dataset validate |
| `mc geometry path detect` | `mc_geometry_path_detect` | Medium | Path detection |
| `mc geometry path compare` | `mc_geometry_path_compare` | Medium | Path comparison |
| `mc geometry training levels` | `mc_geometry_training_levels` | Low | List instrumentation levels |
| `mc adapter inspect` | `mc_adapter_inspect` | Medium | Adapter inspection |
| `mc adapter project` | `mc_adapter_project` | Low | Adapter projection |
| `mc adapter wrap-mlx` | `mc_adapter_wrap_mlx` | Low | MLX wrapping |
| `mc adapter smooth` | `mc_adapter_smooth` | Low | Adapter smoothing |
| `mc thermo analyze` | `mc_thermo_analyze` | Medium | Thermo analysis |
| `mc thermo path` | `mc_thermo_path` | Medium | Thermo path integration |
| `mc thermo entropy` | `mc_thermo_entropy` | Medium | Entropy metrics |
| `mc completions` | N/A | N/A | Shell-specific, not applicable for MCP |
| `mc ensemble list` | `mc_ensemble_list` | Medium | List ensembles |
| `mc ensemble delete` | `mc_ensemble_delete` | Medium | Delete ensemble |

## MCP-Only Tools (No CLI Equivalent)

| MCP Tool | Notes |
|----------|-------|
| `mc_settings_snapshot` | Settings snapshot for MCP context |
| `mc_infer_batch` | Batch inference (CLI uses suite) |

## Tool Naming Convention Verification

All MCP tools follow the `mc_*` naming convention. ✓

### Naming Pattern Analysis:
- Underscores replace hyphens: `jailbreak-test` → `mc_geometry_safety_jailbreak_test` ✓
- Nested commands use underscores: `geometry safety` → `mc_safety_*` or `mc_geometry_safety_*` ✓
- Consistent prefix: All tools start with `mc_` ✓

## Recommendations

### High Priority (Core Functionality)
1. Add `mc_model_merge` - Model merging is a core feature
2. Add `mc_eval_run` - Evaluation is essential for model assessment

### Medium Priority (Useful for Agents)
3. Add `mc_checkpoint_list` - Checkpoint management
4. Add `mc_checkpoint_delete` - Checkpoint cleanup
5. Add `mc_job_delete` - Job cleanup
6. Add `mc_model_register` - Local model registration
7. Add `mc_model_delete` - Model cleanup
8. Add `mc_dataset_list` - Dataset discovery
9. Add `mc_dataset_delete` - Dataset cleanup
10. Add `mc_adapter_inspect` - Adapter analysis
11. Add `mc_thermo_analyze` - Thermo analysis
12. Add `mc_thermo_path` - Path integration
13. Add `mc_thermo_entropy` - Entropy metrics
14. Add `mc_ensemble_list` - Ensemble discovery
15. Add `mc_ensemble_delete` - Ensemble cleanup
16. Add `mc_geometry_path_detect` - Path detection
17. Add `mc_geometry_path_compare` - Path comparison
18. Add `mc_eval_list` - Evaluation listing
19. Add `mc_eval_show` - Evaluation details
20. Add `mc_train_preflight` - Pre-flight validation
21. Add `mc_train_export` - Export trained model
22. Add `mc_dataset_preprocess` - Dataset preprocessing

### Low Priority (Utility/Specialized)
- `mc_explain` - Command explanation
- `mc_train_logs` - Log streaming (not ideal for MCP)
- `mc_job_attach` - Log streaming (not ideal for MCP)
- `mc_system_probe` - System diagnostics
- `mc_dataset_preview` - Dataset preview
- `mc_dataset_pack_asif` - ASIF packaging
- `mc_compare_list/show` - Comparison utilities
- `mc_doc_convert/validate` - Document utilities
- `mc_geometry_training_levels` - Instrumentation levels
- `mc_adapter_project/wrap_mlx/smooth` - Specialized adapter ops

## Conclusion

The CLI/MCP parity is at approximately 64.7% coverage. The core training, inference, geometry, and safety features have good MCP coverage. The main gaps are in:

1. **Resource management** (delete/cleanup operations)
2. **Listing operations** (list checkpoints, datasets, ensembles, evaluations)
3. **Thermo analysis** (analyze, path, entropy)
4. **Geometry path** (detect, compare)
5. **Model operations** (merge, register)
6. **Evaluation** (run, list, show)

All existing MCP tools follow the `mc_*` naming convention correctly.
