# Open Mathematical Questions

**Updated:** 2026-03-08

## What This File Is For

This file is now the active derivation backlog.

A question belongs here only if all three are true:

1. it blocks promotion of a mission, roadmap, or vision claim
2. it still needs a causal operator, equation, measurement operator, or falsifier
3. it is still open today

If a topic is mainly about prioritization, benchmarking, repo cleanup, or
artifact hygiene, it belongs in `docs/RESEARCH-ROADMAP.md`, not here.

Historical solved and refuted material has not been discarded. It now belongs in
dedicated research notes and the git history instead of living in one giant
mixed-status file.

## Active Questions

### Q1. What operator predicts behavioral failure when structural safety still passes?

**Why this is open**

The canonical training path can pass structural checks and still fail behavioral
preservation.

**Why it matters**

This is the main blocker on the claim that training can be easier for people
because the geometry removes guessing.

**Current evidence**

- `results/pipeline_validation/verdict.json`: 350M structural pass `5/5`,
  inference pass `3/5`, composite pass `3/5`
- `results/pipeline_validation/REPORT.md`: failures show high
  `cka_blindness_ratio`, low `null_access_min_behavioral_preserved_fraction`,
  and margin-sign flips despite acceptable structural measurements

**What is missing**

- a causal operator from adapter perturbation to behavioral degradation
- a commensurable measurement linking null-space access, CKA blindness, and
  answer degradation

**Next falsifier**

Pre-register a layer-local intervention that predicts failure before online eval
degrades. If the predicted layer does not move the degradation metric, the
candidate operator is wrong.

### Q2. When is a global MASS ceiling sufficient, and when is per-layer control required?

**Why this is open**

MASS is the active training controller, but the scale law is still incomplete.

**Why it matters**

Without this, the one-command training claim cannot honestly generalize across
architectures and scales.

**Current evidence**

- `docs/MISSION.md`: MASS is the canonical controller
- `results/g5_8b_validation_multiseed/multiseed_gates.json`: mechanical gates
  are partly healthy at 8B, but full closure is not
- `results/pipeline_validation/verdict.json`: 350M still shows unresolved
  behavioral failure cases

**What is missing**

- an architecture-conditioned and scale-conditioned law saying when the global
  ceiling is enough
- a falsifiable boundary between "global controller is valid" and
  "per-layer controller is required"

**Next falsifier**

Run matched global-vs-per-layer MASS experiments on the same model, dataset, and
preservation suite. If per-layer control does not change the failure regime, the
missing term is elsewhere.

### Q3. What is the quantization frontier law?

**Why this is open**

We have promising quantized correction results but not the law that predicts
when the quantized model can preserve geometry and behavior.

**Why it matters**

Quantized-first is central to the vision. A few good corrective runs are not
enough.

**Current evidence**

- `docs/research/quantization_frontier_precheck_v1_implementation_2026_03_05.md`
- `results/quantization_frontier/`
- `results/closedform_sequential_correction/20260227T173057Z/closedform_correction.json`

**What is missing**

- the architecture-conditioned equation linking crossing severity to achievable
  CKA floor and behavioral preservation
- a commensurable operator across bit-depths and architectures

**Next falsifier**

Run paired FP-to-quantized sweeps across multiple bit-depths and model families.
If the same crossing statistic does not order the achieved CKA floor and
degeneration outcomes, the frontier statistic is incomplete.

### Q4. Is unused-subspace residual energy causally anti-degeneration?

**Why this is open**

Corrective and Tikhonov-style quantization work improved degeneration, but the
causal mechanism is not yet identified.

**Why it matters**

This question decides whether we actually understand preservation under
correction or are just benefiting from a lucky regularization effect.

**Current evidence**

- `results/closedform_sequential_correction/`
- `results/stacked_corrective_recovery/` (archived: `/Volumes/CodeCypher/archive/results-refuted/`)
- `results/corrective_lora_training/` (archived: `/Volumes/CodeCypher/archive/results-refuted/`)

**What is missing**

- the operator linking unused-subspace residuals to repetition suppression
- an intervention that separates "less damage" from "better implicit
  regularization"

**Next falsifier**

Use covariance-matched re-noise controls and direct interventions on
`E_unused`. If degeneration does not track the manipulated residual energy, the
anti-degeneration story is wrong.

### Q5. What is the architecture-conditioned law from entropy or logit state to curvature and ID?

**Why this is open**

We have a strong middle-chain research thread, but the cross-family law is still
underspecified.

**Why it matters**

This blocks promotion of the full causal chain behind highway phases and
cross-architecture geometry claims.

**Current evidence**

- `results/entropy_curvature_operator_split/`
- `results/f5_sign_law_analysis_6models/cross_model_summary.json`
- `docs/research/SOTA-AUDIT-2026-03.md`

**What is missing**

- explicit architecture terms for same-GQA divergences such as Qwen3 vs Qwen3.5
- the correct split between attention entropy, logit entropy, norm coupling, and
  sublayer effects

**Next falsifier**

Fit one family-conditioned law and require it to predict sign and direction
before reruns. If the law cannot predict the held-out family, the mechanism is
still underspecified.

### Q6. What local geometric operator actually drives TwoNN ID changes?

**Why this is open**

Several attractive explanations have already failed.

**Why it matters**

Without this, phase-language remains descriptive instead of mechanistic.

**Current evidence**

- `results/covariance_rank_id/` (archived): covariance-rank injection is not the mechanism
- `results/tangent_subspace_id_mechanism/`: repaired atlas-backed reruns are the
  active path; historical 2026-03-07 evidence is exploratory only
- `docs/research/tangent_subspace_id_mechanism_2026_03_08.md`: repaired rerun
  checkpoint with LFM2/Qwen completed and Llama still blocked in local tangent
  alignment at `N=324`

**What is missing**

- a local operator commensurable with geodesic TwoNN
- a falsifier that survives beyond one neighborhood-size choice

**Next falsifier**

Rerun the tangent-space mechanism with patched operators and higher probe count.
If the signal disappears under a commensurable geodesic operator, tangent
misalignment is not the explanation.

### Q7. What is the correct DPI-compatible accessible-information observable for deterministic residual networks?

**Why this is open**

The old mutual-information depth-decay story is dead, but the replacement
observable is not fully closed.

**Why it matters**

Information claims will keep reappearing unless the replacement observable is
explicit and enforced.

**Current evidence**

- `docs/research/linear_accessible_information_derivation.md`
- `results/dpi_analysis/` (archived: `/Volumes/CodeCypher/archive/results-refuted/`)
- `results/information_bridge/` (archived: `/Volumes/CodeCypher/archive/results-refuted/`)
- `results/information_bridge_linear_cka/`

**What is missing**

- a DPI-compatible observable that is valid for deterministic residual chains
- debiased or sampling-aware CKA integration in the canonical geometry path

**Next falsifier**

If the debiased observable changes the cross-model conclusion, then the current
operator is still too biased to promote.

### Q8. What measurements are sufficient to certify portable cross-architecture adapters?

**Why this is open**

We have merge machinery and portability claims, but not yet a complete
certificate for transfer and stacking.

**Why it matters**

This is the mathematical bottleneck between current merge work and the identity
layer described in `docs/VISION.md`.

**Current evidence**

- `src/modelcypher/experimental/merge/`
- `src/modelcypher/experimental/self_improve/lora_stacker.py`
- `results/geometry_sota/analysis_summary.json`
- `results/sota_audit_2026_03/scorecard.md`

**What is missing**

- a commensurable preservation certificate across model families
- mandatory comparison against standard merge baselines
- a clear line between "alignment on probes" and "portable behavior"

**Next falsifier**

If a MergeBench-style comparison shows that null-space transfer does not beat or
match standard merge baselines on preserved behavior, then portability is still
an internal hypothesis, not a user-facing advantage.

### Q9. What consolidation operator adds new user structure without forgetting old structure?

**Why this is open**

The nightly consolidation part of the vision depends on a specific update law,
not on the existence of continual-learning code.

**Why it matters**

Without this operator, the "identity layer" remains narrative rather than
stable mechanism.

**Current evidence**

- `src/modelcypher/experimental/continual/`
- `src/modelcypher/experimental/use_cases/consolidation_service.py`
- `results/continual_learning/`

**What is missing**

- the exact additive or projected update operator
- a preservation metric that is strong enough to certify non-forgetting
- an intervention showing why the operator works better than replay or standard
  continual-learning baselines

**Next falsifier**

Run before/after preservation tests under a fixed update operator and compare to
standard replay-style baselines. If preservation is not measurably better, the
consolidation story is still open.

## Questions That Are No Longer Open

These do not belong in the active blocker list anymore:

- Layer Jacobians are not rank-1 in trained transformers; the rank-collapse
  claim was refuted
- Highway location is not governed by one universal heuristic
- The old Lipschitz step-size derivation failed and was replaced by MASS
- Shannon-style mutual-information decay through deterministic residual depth was
  refuted
- `beta_1` as a direct predictor of reasoning success was refuted

If a solved or refuted question still matters, it should live in a dedicated
note with artifact references, not remain in this file as pseudo-open work.

## Historical Detail Lives Here Now

- `docs/research/SOTA-AUDIT-2026-03.md`
- `docs/research/field_map_external_methods.md`
- `docs/research/linear_accessible_information_derivation.md`
- `docs/research/lr_derivation_analysis.md`
- `docs/research/covariance_rank_id_phase2_review_2026_03_05.md`
- `docs/research/TANGENT-SUBSPACE-ID-FALSIFIER-PROTOCOL.md`
- `docs/research/tangent_subspace_id_mechanism_2026_03_07.md`
- `docs/research/quantization_frontier_precheck_v1_implementation_2026_03_05.md`
- `docs/research/cross-architecture-geometry.md`

## Inclusion Rule Going Forward

When a question is solved, refuted, or clearly downgraded to an operational
roadmap item:

1. remove it from this file
2. add the dedicated note or artifact pointer
3. keep only the active blocker list here

This file should stay short enough that a researcher can read it in one pass and
know exactly which mathematical questions still block promotion.
