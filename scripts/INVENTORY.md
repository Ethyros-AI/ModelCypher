# Scripts Inventory

**Total:** 284 Python scripts
**Lines:** ~105,000
**Generated:** 2026-01-29

This inventory documents all scripts before archiving to `/Volumes/CodeCypher/archive/modelcypher-scripts/`.

---

## Research Arc Summary

| Phase | Scripts | Topic | Key Finding |
|-------|---------|-------|-------------|
| 1 | exp9-exp33 | RMT compression | Marchenko-Pastur filtering works; gate layers at 85% SVD energy |
| 2 | exp38-exp44 | Golden layer | Layer at ~67% depth (golden ratio) is optimal across architectures |
| 3 | exp45-exp55 | Cross-arch merge | CKA=0.9255 via F=pinv(src)@tgt; single direction achieves 91.7% |
| 4 | exp56-exp65 | Failure analysis | Pattern interference, entropy-gated teaching, knowledge banks |
| 5 | exp66-exp75 | Self-improvement | Models can observe own manifold; geometric self-play works |
| 6 | exp76-exp82 | Scaling/Teachers | 70% ceiling; teacher bridging breaks through barriers |
| 7 | exp83-exp87 | Limits | Generation-based eval shows 20pp gap over single-token |

---

## Numbered Experiments (exp9-exp87)

### Phase 1: RMT Compression (exp9-exp33)

| File | Description |
|------|-------------|
| exp9_rmt_compression.py | RMT-based rank detection comparing naive pinv vs. signal/noise-separated compression |
| exp10_active_subspace_compression.py | Active subspace projection for compression in variance-defined low-dimensional spaces |
| exp11_gw_layer_prediction.py | Gromov-Wasserstein distance as predictor for safe layer combinations |
| exp12_geodesic_vs_euclidean_rank.py | Comparison of geodesic vs. Euclidean rank revealing manifold structure for compression |
| exp13_ranking_vs_mse.py | Ranking-preserving optimization vs. MSE loss for improved post-compression accuracy |
| exp14_entropy_coverage.py | Calibration entropy as predictor for held-out generalization using energy-based analysis |
| exp15_gate_layer_detection.py | Auto-detection of uncompressible gate layers via top-1 singular value energy threshold |
| exp18_contiguous_analysis.py | Contiguous vs. non-contiguous layer combinations testing error propagation and isolation |
| exp19_max_lossless_compression.py | Combining safe zones to achieve maximum lossless compression from identified safe layers |
| exp20_mega_skip.py | Single linear transform to skip 27 layers replacing entire transmission zone |
| exp21_attention_only.py | Attention-rich model with linearized MLPs preserving context understanding |
| exp22_spread_compression.py | Spread vs. sequential layer compression testing error compounding in non-adjacent layers |
| exp23_error_growth_rate.py | Error propagation analysis fitting exponential model with Euler's number scaling |
| exp24_margin_dynamics.py | Exponential margin decay tracking through compression with e² critical threshold |
| exp25_perturbation_spectrum.py | Spectral structure analysis of compression error with random matrix theory |
| exp26_entropy_dynamics.py | Entropy preservation under compression detecting rotation vs. noise/diffusion errors |
| exp27_metric_preservation.py | Pairwise distance preservation across compression detecting topological distortion |
| exp28_distortion_preserving.py | Compression preserving MLP distortion patterns using metric tensor Procrustes |
| exp29_scaled_compression.py | Scaling compressed matrix by mean expansion ratio to preserve distortion scale |
| exp30_downstream_preserving.py | High-variance direction preservation in MLP output for downstream layer requirements |
| exp31_minimal_subspace.py | Finding minimal essential subspace dimensionality where accuracy peaks and entropy minimizes |
| exp32_lowrank_multilayer.py | Low-rank multi-layer compression testing error compounding prevention across layers |
| exp33_adaptive_rank.py | Per-layer adaptive rank selection to remove noise while preventing error accumulation |

### Phase 2: Golden Layer & Unified Compression (exp38-exp44)

| File | Description |
|------|-------------|
| exp38_unified_compression.py | Combines low-rank projection, reverse chain order, spread pattern, and entropy monitoring for maximum compression |
| exp39_entropy_optimal_compression.py | Identifies layers achieving negative entropy change for true compression-based knowledge transfer |
| exp40_100_percent_layers.py | Systematically tests all layers with varying k values to find all achieving 100% accuracy |
| exp41_golden_layer_geometry.py | Analyzes activation covariance to identify geometric signatures of "golden" layers enabling perfect compression |
| exp42_cross_arch_golden.py | Tests whether every architecture has a golden layer at ~67% depth (golden ratio hypothesis) |
| exp43_combination_failure.py | Investigates why compressing two high-accuracy layers causes degradation despite individual success |
| exp44_attention_compression.py | Extends compression to attention layer projections (Q, K, V) testing if attention is more compressible |

### Phase 3: Cross-Architecture Merge (exp45-exp55)

| File | Description |
|------|-------------|
| exp45_compressed_transplant.py | Tests whether compressed source layers transfer better by removing noise while preserving behavior |
| exp46_cross_arch_merge.py | Transplants DeepSeek-R1's MLP behavior into LFM2 using activation alignment at aligned depths |
| exp46b_stabilized_merge.py | Improves cross-arch merge with ridge regression and low-rank projection for numerical stability |
| exp46c_normalized_merge.py | Fixes numerical overflow by normalizing activations to unit variance before computing transplant |
| exp46d_correct_svd.py | Corrects SVD dimension handling for proper projection when samples differ from features |
| exp47_curriculum_transplant.py | Tests pedagogically-designed curriculum with diverse examples and progressive difficulty |
| exp48_minimal_curriculum.py | Discovers minimum training examples needed to achieve 80%+ accuracy in cross-arch teaching |
| exp49_multi_layer_teaching.py | Applies progressive teaching to multiple layers with recalibration after each transplant |
| exp50_optimal_curriculum.py | Selects most effective training samples by maximizing coverage of essential k=6 dimensions |
| exp51_directional_teaching.py | Decomposes layers into k essential directions and teaches each topic independently |
| exp52_essential_direction.py | Investigates why single direction (direction 6) achieves 91.7%, outperforming full layer teaching |
| exp53_dual_essentials.py | Explores combining two essential directions (6 and 8) that each achieve 91.7% |
| exp54_optimal_replacement.py | Improves accuracy by replacing only target's specific directions with source directions |
| exp55_stubborn_failure.py | Investigates single remaining failure case in 91.7% accurate directional transplant |

### Phase 4: Failure Analysis & Teaching (exp56-exp65)

| File | Description |
|------|-------------|
| exp56_entropy_reduction.py | Cross-architecture entropy reduction by transplanting knowledge from larger to smaller models |
| exp57_selective_denoising.py | Selective entropy-based teaching that only applies transfer when it reduces uncertainty |
| exp58_iterative_distillation.py | Iterative entropy-gated distillation extracting knowledge until no further reduction possible |
| exp59_manifold_self_teaching.py | Pure geometric knowledge transfer via spectral entropy measurement without token manipulation |
| exp60_iterative_self_teaching.py | Iterative self-teaching using spectral entropy to find and transfer optimal layer-pair directions |
| exp61_capability_teaching.py | Capability-focused teaching transferring domain-specific knowledge without compression |
| exp62_reciprocal_teaching.py | Bidirectional knowledge exchange where models with complementary strengths teach each other |
| exp63_knowledge_bank.py | Reusable library of extracted "clean" directions from expert models as portable numpy arrays |
| exp64_lfm2_upgrade.py | Systematic upgrade of LFM2-1.2B by identifying weaknesses and applying targeted teaching |
| exp65_robust_teaching.py | Robust cross-architecture teaching using SVD-based dimensionality reduction for dimension mismatches |

### Phase 5: Self-Improvement (exp66-exp75)

| File | Description |
|------|-------------|
| exp66_same_arch_teaching.py | Same-architecture teaching where LFM2.5-1.2B-Instruct upgrades LFM2-1.2B without projection |
| exp67_multi_layer_teaching.py | Multi-layer teaching with numerical stability applying improvements one layer at a time |
| exp68_targeted_teaching.py | Domain-specific teaching focusing on weak domains like math with targeted probes |
| exp69_logit_surgery.py | Direct manipulation of embedding space and lm_head weights to steer logit probabilities |
| exp70_correctness_geometry.py | Discovery that correct answers have invariant geometric signatures (high kurtosis, low entropy) |
| exp71_geometry_guided_teaching.py | Teaching guided by geometric objectives to increase kurtosis and decrease spectral entropy |
| exp72_geometric_self_play.py | Self-improvement without teacher by randomly perturbing directions and keeping geometry improvements |
| exp73_self_direction_play.py | Self-play using model's own principal directions by boosting SVD components |
| exp74_perpetual_improvement.py | Continuous recursive self-improvement loop exploring geometry-improving modifications until convergence |
| exp75_cascaded_improvement.py | Layer-by-layer cascaded improvement with downstream recalibration to stack improvements |

### Phase 6: Scaling & Teachers (exp76-exp82)

| File | Description |
|------|-------------|
| exp76_forward_flow_improvement.py | Optimize layers sequentially respecting forward information flow without backward recalibration |
| exp77_exhaustive_layer_optimization.py | Exhaustively search all directions and boost factors for optimal geometry at each layer |
| exp78_fast_exhaustive_optimization.py | Fast exhaustive optimization using geometry-only first pass and coarse-to-fine grid search |
| exp79_multistart_basin_exploration.py | Test if different starting perturbations converge to same 70% ceiling or different basins |
| exp80_teacher_bridge_injection.py | Inject teacher model directions to break through student's 70% ceiling |
| exp80b_teacher_bridge_from_70.py | Self-improve to 70% first, then inject teacher directions to exceed either alone |
| exp81_targeted_direction_transplant.py | Extract and transplant only specific directions from teacher encoding failing cases |
| exp82_multilayer_teacher_bridge.py | Find which teacher layer(s) best encode failing cases and inject at corresponding student layers |

### Phase 7: Limits & Generation (exp83-exp87)

| File | Description |
|------|-------------|
| exp83_find_better_teacher.py | Evaluate all available models to find teacher exceeding student's 70% ceiling |
| exp84_better_teacher_bridge.py | Inject directions from Qwen2.5-Coder (80% teacher) to break 70% barrier |
| exp85_math_case_analysis.py | Analyze why both student and teacher fail on same math cases (architectural constraints) |
| exp86_proper_evaluation.py | Evaluate accuracy using generation-based metrics instead of single-token prediction |
| exp87_generation_based_self_improvement.py | Self-improve using generation accuracy as metric to break single-token ceiling |

### Unnumbered Experiments (exp_*)

| File | Description |
|------|-------------|
| exp_adversarial_trajectories.py | Test if adversarial inputs show pathological dimensional trajectories |
| exp_cross_domain_geodesic.py | Test if geodesic structure is domain-independent across math and science |
| exp_dec_geodesic.py | Compute DEC geodesics and test if Hodge decomposition separates correct/incorrect |
| exp_difficulty_vs_expansion.py | Test if harder problems require more dimensional expansion and later peak layers |
| exp_dimensional_curve.py | Track intrinsic dimension through layers to understand expansion-compression curve |
| exp_dimensional_modes.py | Analyze two dimensional processing modes: immediate recognition vs expansion-then-compression |
| exp_dimensional_trajectory_vis.py | Visualize dimensional trajectories through all layers |
| exp_entropy_correct_vs_incorrect.py | Compare entropy trajectories for correct vs incorrect answers |
| exp_entropy_trajectory_full.py | Measure entropy at every layer to capture full expand-compress trajectory |
| exp_explicit_math_unlock.py | Test if making implicit math explicit unlocks expansion capability |
| exp_failure_cartography.py | Map what triggers constrained encoding via entropy trajectories and structural features |
| exp_geometry_metric_tensor.py | Derive metric tensor coefficients from eigenspectrum geometry without heuristics |
| exp_temporal_spatial_duality.py | Test if optimal learning follows geodesics in joint temporal-spatial space |

---

## Training Scripts (train_*)

| File | Description |
|------|-------------|
| train_and_save_self_reflection.py | Trains self-reflection and extracts core questions first, then saves weights |
| train_automatic_self_reflection.py | Learns question normalization through behavioral training (73% φ improvement) |
| train_combined_mastery.py | Combines full GSM8K with explicit reasoning for six failing patterns |
| train_complete_patterns.py | Covers all 11 logical shapes discovered through error analysis |
| train_cot_preserve_geometry.py | Trains chain-of-thought while preserving natural geometry |
| train_distilled_logic.py | Distills fundamental logical shapes into 10 perfect examples per pattern |
| train_early_layer_expansion.py | Trains early-layer adapter (0-10) for implicit math recognition |
| train_final_mastery.py | Trains 13 patterns with reinforced weak areas and common regression points |
| train_for_phi.py | Trains for comp/φ = 1.0 geometry balancing expansion and compression |
| train_geometric_alignment.py | Trains to recognize problem complexity and adjust geometric trajectory |
| train_geometric_gap.py | Geometric induction training with synthetic examples for six failing patterns |
| train_geometry_driven.py | Derives all hyperparameters from loss landscape geometry |
| train_gsm8k_cot.py | Trains explicit chain-of-thought format for autonomous reasoning |
| train_gsm8k_full.py | Uses real GSM8K data with full chain-of-thought solutions |
| train_gsm8k_full_heavy.py | Trains on all 7473 GSM8K samples with extended iterations |
| train_gsm8k_heavy.py | Trains on full GSM8K prioritizing real data over synthetic |
| train_gsm8k_mastery.py | Reaches 70%+ via more data, heavy arithmetic preservation (60%), longer training |
| train_gsm8k_qwen3.py | Trains multi-step math reasoning on Qwen3-8B with cumulative training |
| train_gsm8k_targeted.py | Targets six failure patterns with specialized training data |
| train_gsm8k_v2.py | Balanced curriculum with 50% arithmetic preservation |
| train_logical_analogies.py | Teaches logical patterns through analogies to arithmetic |
| train_math_correlation.py | Correlates symbolic arithmetic with counting by reinforcing invariants |
| train_math_qwen3.py | Trains math capability on Qwen3-8B using text continuation format |
| train_math_qwen3_v2.py | Fixes remaining issues from v1 using cumulative training |
| train_multistep_qwen3.py | Bridges arithmetic-to-GSM8K gap with 2-3 step intermediate problems |
| train_reflection_lora.py | Adds self-reflection via LoRA without destroying base knowledge |
| train_reflection_lora_v2.py | Uses mlx-lm built-in LoRA for correct gradient flow |
| train_self_awareness.py | Trains geometric uncertainty rewarding comp/φ ≈ 1.0 on correct answers |
| train_self_improvement.py | Model self-teaching through verified self-play data |
| train_self_improvement_v2.py | Extends self-improvement with proper chat format for instruct models |
| train_surgical_reasoning.py | Targets five reasoning pattern failures with explicit chain-of-thought |
| train_unified_expansion_adapter.py | Combines early-layer math recognition with GSM8K solving |
| train_universal_reasoning_adapter.py | Extends math adapter to logical, causal, and comparison patterns |
| training_true_gaps.py | Targeted gap filling training language-to-equation parsing |

---

## Evaluation & Benchmark Scripts

| File | Description |
|------|-------------|
| eval_geometric_alignment.py | Compares base vs trained adapter on accuracy by difficulty and geometry |
| eval_gsm8k_cot.py | Evaluates GSM8K with Chain-of-Thought prompting |
| evaluate_alignment_quality.py | Tests if geometric self-alignment improves coherence, logic, fact-checking |
| evaluate_gsm8k_unified.py | Evaluates GSM8K accuracy with unified adapter on Qwen3-8B |
| evaluate_mastery_fixed.py | Evaluates mastery across difficulty tiers with corrected number extraction |
| evaluate_math_complete.py | Evaluates Qwen3-8B math by generating multiple tokens |
| evaluate_multi_benchmark.py | Tests geometric adaptation on GSM8K, ARC-Challenge, HellaSwag, BoolQ |
| evaluate_universal_adapter.py | Compares all adapters on multiple benchmarks |
| benchmark_aligned_model.py | Tests if SVD alignment produces real capability improvement |
| benchmark_baseline.py | Establishes LFM2-350M baseline across reasoning categories |
| benchmark_lfm2_350m.py | Measures LFM2-350M accuracy and geometric signatures by category |
| benchmark_lfm2_350m_hard_math.py | Identifies where LFM2-350M math breaks down |
| benchmark_with_reflection.py | Trains self-reflection then benchmarks performance |
| fixed_evaluation.py | Evaluates on symbolic arithmetic with correct tokenization |

---

## Geometric Analysis Scripts

| File | Description |
|------|-------------|
| geometric_bridge_signature.py | Identifies geometric signatures of disconnected vs working capabilities |
| geometric_experiments.py | Runs rigorous mathematical experiments on geometry constants |
| geometric_invariant_discovery.py | Distinguishes geometric signatures of facts from speculation |
| geometric_knowledge_discovery.py | Framework finding counterfactual sensitivity as key knowledge marker |
| geometric_lora_test.py | Constructs geometrically-structured LoRA adapters |
| geometric_self_awareness.py | Enables models to monitor own geometric uncertainty via comp/φ drift |
| geometric_self_detection.py | Auto-detects disconnected capabilities via condition number threshold |
| geometric_self_play_loop.py | Guides model exploration of own manifold using geometric signatures |
| geometric_signature_comparison.py | Compares SVD patterns between correct and incorrect answers |
| geometric_training.py | Derives training parameters directly from geometric measurements |
| geometry_derived_training.py | Trains where all hyperparameters derive from Gram matrix geometry |
| geodesic_performance_benchmarks.py | Measures computational overhead of geodesic vs Euclidean metrics |
| gsm8k_geometric_analysis.py | Diagnoses if GSM8K failures are disconnected capabilities or true gaps |
| gsm8k_geometric_training.py | Geometry-derived training using Fisher Information and null-space projection |
| gsm8k_surgical_alignment.py | Surgically aligns GSM8K with all parameters from geometry |
| gsm8k_entropy_training.py | Minimizes manifold entropy for GSM8K toward fundamental constants |

---

## Analysis & Debug Scripts

| File | Description |
|------|-------------|
| activation_geometry_analysis.py | Analyzes SVD-based activation geometry to correlate with accuracy |
| analyze_failures.py | Analyzes 6 failing GSM8K problems to understand required patterns |
| analyze_remaining_failures.py | Investigates remaining GSM8K failures with entropy trajectories |
| broken_structure_analysis.py | Finds structural misalignment explaining off-by-one arithmetic errors |
| correct_vs_incorrect.py | Analyzes geometric differences between correct and incorrect computations |
| debug_failures.py | Debugs all 5 failing GSM8K problems with full output analysis |
| debug_janet.py | Debugs Janet's ducks problem with detailed output tracking |
| debug_math_prediction.py | Investigates why models fail at simple arithmetic |
| debug_trajectory_norms.py | Analyzes activation norm trajectories through layers |
| deep_geometric_analysis.py | Full geometric toolkit: dimension, curvature, CKA, attention geometry |
| representation_analysis.py | Analyzes collapsed symbolic vs structured counting representations |
| why_4plus1_works.py | Investigates what makes "4+1=" special (frequency, tokenization, geometry) |
| why_math_failed.py | Analyzes why gradient-guided modification improved language but not math |

---

## Self-Improvement & Autonomous Scripts

| File | Description |
|------|-------------|
| autonomous_benchmark_learning.py | Autonomous self-improvement using benchmarks as curricula |
| autonomous_loop.py | Complete autonomous self-improvement detecting deficiencies |
| autonomous_manifold_completion.py | Indefinite loop to complete model manifold using generation-based eval |
| closed_loop_self_improvement.py | Full autonomous loop integrating gap detection, classification, bridging |
| complete_self_awareness.py | Geometry-based self-awareness using comp/φ ratios |
| counterfactual_self_play.py | Uses counterfactual sensitivity to identify gaps and complete manifold |
| complexity_self_play.py | Calibrates complexity-dimension law using alignment error |
| geometric_self_play_loop.py | Guides exploration using only geometric signatures |
| real_self_improvement.py | Tests if model can identify gaps and specify self-improvement |
| run_autonomous_completion.py | Self-play without supervision modifying weights with fundamental constants |
| run_self_alignment.py | Geometric self-alignment to reduce entropy using fundamental constants |
| safe_self_play_training.py | Generate verified training data for word problem improvement |
| seed_explorer.py | Evolve random matrices toward geometric fitness without training data |
| self_directed_learning.py | Full learning autonomy identifying gaps and learning without forgetting |

---

## Alignment & Bridge Scripts

| File | Description |
|------|-------------|
| additive_alignment_test.py | Adds geometric structure to small SVs while preserving large ones |
| affine_bridge_loss_test.py | Compares coordinate MSE vs relational CKA-based loss |
| aggressive_alignment_test.py | Tests stronger geometric perturbations and multi-scale alignment |
| alignment_driven_training.py | Trains by minimizing misalignment between counting and symbolic |
| automatic_bridge_computation.py | Computes effective priming bridges from geometry via Procrustes |
| bridge_application_methods.py | Tests logit steering, token probability, auto-prompt generation |
| fundamental_alignment.py | Gradient-guided modification to fix broken arithmetic |
| quick_alignment_test.py | Run limited geometric alignment rounds and evaluate |
| relational_alignment.py | Compute transform from broken to correct relational structure |
| relational_bridge_test.py | Tests if relational primes work better than individual concept primes |
| soft_alignment_test.py | Compare hard vs soft SVD alignment on benchmark |
| targeted_alignment_test.py | Identify and strengthen weak SVD dimensions |
| word_problem_bridge.py | Test language-to-math bridges using metaphors and analogies |

---

## Priming & Intervention Scripts

| File | Description |
|------|-------------|
| activation_steering.py | Tests inference-time activation nudging toward geometric constants |
| activation_steering_test.py | Tests modifying hidden states during inference |
| counting_context_priming.py | Tests if counting sequence priming improves symbolic arithmetic |
| minimal_intervention_test.py | Binary search for smallest scale modification for improvement |
| minimal_prime_test.py | Find smallest effective intervention for arithmetic |
| nsm_prime_test.py | Tests Natural Semantic Metalanguage primitives vs verbose primes |
| operation_specific_primes.py | Investigates if each operation needs its own semantic prime |
| period_prime_test.py | Tests generalization of period-before-problem priming |
| prime_mechanism_analysis.py | Analyzes what actually happens during priming |
| priming_comprehensive_test.py | Tests generalization of semantic priming beyond +1 problems |
| priming_limits_test.py | Extends priming to multiplication, division, word problems |
| say_primitive_test.py | Explores Wierzbicka's "say" primitive for arithmetic |
| subtraction_primitive.py | Identifies semantic primitive that unlocks subtraction |

---

## Compression & Transform Scripts

| File | Description |
|------|-------------|
| fix_transform.py | Train transform to work with raw prompts without priming |
| fix_transform_v2.py | Train on continued text sequences for natural context |
| fix_word_problems.py | Teach models to parse and solve word problems |
| full_manifold_compression.py | Test manifold compression across all 455 arithmetic facts |
| manifold_compression.py | Test if compressing math space coherence aligns individual facts |

---

## Gradient & Modification Scripts

| File | Description |
|------|-------------|
| arch_test_gradient.py | Tests if gradient-guided modification works across architectures |
| gradient_guided_merge.py | Apply gradient-guided principles to model merging |
| gradient_guided_modification.py | Find safe weight modification directions orthogonal to preservation |
| gradient_plus_geometric.py | Combine gradient guidance with geometric structure analysis |
| iterative_orthogonal.py | Test if multiple iterations of orthogonal modification compound |

---

## Dimensional & Curvature Scripts

| File | Description |
|------|-------------|
| complexity_dimension_correlation.py | Tests correlation between statement complexity and intrinsic dimension |
| curvature_analysis.py | Analyzes manifold curvature to correlate with capabilities |
| deep_invariant_discovery.py | Multi-metric confidence signature using variance, rank, attractors, entropy |
| dimensional_hierarchy.py | Tests hypothesis of dimensional hierarchy from facts to meta-cognition |
| fractional_dimension_probe.py | Investigates fractional dimensional spaces in model merging |
| scale_curvature_analysis.py | Aggregate activations across statements to measure manifold curvature |
| validate_dimensional_geodesic.py | Validate dimensional geodesic theory: dim = (e/π)×complexity + (π/e) |

---

## Miscellaneous Scripts

| File | Description |
|------|-------------|
| all_category_optimization.py | Find direction improving weak categories while preserving strong |
| arithmetic_tables_check.py | Validates foundational arithmetic facts |
| baseline_scan_qwen3.py | Comprehensive capability scan for Qwen3-8B |
| capability_chain_algebra.py | Uses verified arithmetic as oracle to learn algebraic simplifications |
| check_training_overlap.py | Checks if failing test problems appear in training set |
| cka_linear_vs_geodesic.py | Compares linear CKA vs geodesic CKA for structure capture |
| compare_positive_geometry_adapter.py | Compares positive Grassmann signatures before/after adapter |
| concept_correlation.py | Tests if math concepts exist in non-arithmetic forms |
| consistency_geodesic_test.py | Compares Euclidean cosine vs geodesic distance for consistency |
| consistency_geometry_study.py | Studies correlation between consistency and geometric constant alignment |
| constant_ablation.py | Ablates constant families (π/e, φ, √2) to determine quality drivers |
| counterfactual_sensitivity.py | Measures representation changes when facts are violated |
| cross_architecture_transfer.py | Tests if learned capabilities transfer across architectures |
| crystallization_test.py | Nudges near-constant SV ratios to exact constants |
| derive_mlp_scale.py | Derives correct MLP scale correction from geometry |
| differentiable_phi_loss.py | Trains toward golden ratio compression using differentiable loss |
| discover_primitives.py | Discovers model's semantic primitives via single-word activation |
| dormant_activation_test.py | Creates constant ratios in unused SV regions |
| euclidean_vs_geodesic_distances.py | Quantify differences between Euclidean and geodesic distances |
| gap_detection_calibration.py | Tests if consistency metrics predict accuracy before seeing answers |
| generate_math_equivalence_data.py | Generate training data showing arithmetic equals counting |
| gram_aligner_cka_test.py | Compare linear CKA vs geodesic CKA for alignment issues |
| inference_sharpening.py | Sharpen logit predictions using temperature from activation sharpness |
| internal_structure_comparison.py | Analyze relational structure differences between math and non-math |
| iteration_tracking_test.py | Track at which iteration each capability improves or degrades |
| logit_sharpness.py | Measure geometric sharpness and derive effective temperature |
| map_invariant_structure.py | Discover where fundamental constants appear in model geometry |
| measure_reflection_geometry.py | Verify self-reflection training improves φ-alignment |
| multi_category_improvement.py | Test if combined improvement gradients work across categories |
| multi_topic_learning.py | Test cumulative improvement without learning interference |
| orthogonal_subspace_analysis.py | Characterize safe subspace orthogonal to preservation gradients |
| parallel_pathway_test.py | Add geometric delta as new pathway without modifying originals |
| phi_alignment_training.py | Train for comp/φ = 1.0 using chain-of-thought with geometric loss |
| post_alignment_learning.py | Test if gradient-guided learning improves harder math after alignment |
| profile_model_geometry.py | Measure dimensional trajectory and compression/φ ratio for any model |
| question_normalization.py | Test if core question extraction ensures φ resonance point |
| relational_structure_comparison.py | Compare Gram matrices between broken and working models |
| research_integration.py | Pipeline for converting web content into QA training pairs |
| residual_connection_test.py | Test if geometric structure belongs in residual blending ratios |
| run_contrastive_learning.py | Train through contrasting coherent vs incoherent statements |
| run_geometric_learning.py | Direct weight space optimization for SVD alignment |
| run_iterative_learning.py | Full loop of thinking, surgical alignment, and repetition |
| run_progressive_learning.py | Iterative thinking with weight locking building on previous gains |
| run_surgical_alignment.py | Align only SVD ratios proximal to fundamental constants |
| run_surgical_only.py | Ablation test removing thinking phase |
| run_thinking_loop.py | Test if iterative self-questioning increases constant signatures |
| scale_test_gradient.py | Test if gradient-guided modification works on larger models |
| semantic_direction_discovery.py | Find semantic directions separating categories |
| semantic_invariance_test.py | Test if factual knowledge exhibits semantic invariance |
| sharpness_training.py | Train for increased confidence on answer tokens |
| single_topic_learning.py | Combine gap detection, research, and modification for weak categories |
| successor_analysis.py | Analyze what model computes for "+1" operations |
| targeted_sharpness_training.py | Increase sharpness specifically on correct tokens |
| test_early_layer_expansion.py | Test if teaching early layers increases activation expansion |
| test_manifold_entropy.py | Validate manifold entropy measurements and SVD constant matches |
| test_self_awareness_on_failures.py | Validate geometric prediction of failures using comp/φ deviation |
| tokenization_check.py | Debug digit string tokenization and prediction behavior |
| true_gap_detection.py | Distinguish disconnected capabilities from truly missing ones |
| verification_oracle.py | Use verified arithmetic to check word problem parses |
| verify_intermediate_geometry.py | Check φ alignment at self-reflection intermediate steps |

---

## Archive Mapping

When archived to `/Volumes/CodeCypher/archive/modelcypher-scripts/`:

| Archive Folder | Scripts |
|----------------|---------|
| `compression/` | exp9-exp33, manifold_compression, full_manifold_compression |
| `golden_layer/` | exp38-exp44 |
| `cross_arch/` | exp45-exp55 |
| `failure_analysis/` | exp56-exp65, analyze_*, debug_*, why_* |
| `self_improvement/` | exp66-exp75, autonomous_*, self_*, run_autonomous_*, run_self_* |
| `scaling/` | exp76-exp82 |
| `pathological/` | exp83-exp87, exp_* |
| `techniques/` | counterfactual_*, geometric_*, train_distilled_logic |
| `training/` | train_* |
| `evaluation/` | eval_*, evaluate_*, benchmark_*, fixed_evaluation |
| `utilities/` | All remaining scripts |
