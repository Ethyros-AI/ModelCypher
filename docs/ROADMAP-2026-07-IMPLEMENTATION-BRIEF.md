# ModelCypher Implementation Brief — July 2026

**Status:** Authoritative work order, supersedes TODO.md's open items. Produced by a
five-dimension frontier-model audit (origin/history, mathematical soundness, empirical
evidence, engineering, state-of-the-art) commissioned 2026-07-04. Companion to
[RESEARCH-ROADMAP.md](RESEARCH-ROADMAP.md) (which remains the research closure ladder —
R1–R6 and its operating rules stay binding).

**For the implementing agent:** work the workstreams in order (WS0 → WS4). Each item
has file paths and an acceptance criterion. Do not add new research surfaces; this
brief is convergence work. Respect AGENTS.md (hexagonal boundaries, derive-or-label
constants, no mixed-model narratives). Where this brief says a claim must be
downgraded, downgrade it — the audit evidence is summarized inline; do not re-litigate
it by softening language.

---

## The strategic reframe (read first)

The audit's central conclusion: **ModelCypher's defensible asset is the instrument,
not the optimizer.**

- The "replaces all 15 hyperparameters" thesis is (a) contradicted by the shipped
  default path (`cli/commands/train.py` → calibrated AdamW lr=2e-4, cosine, betas
  0.9/0.999 — knobs #1/#3/#7 are industry values), (b) evidenced *against* by the
  project's own retained R1 result (canonical geometric path won 0/7 lm-eval tasks and
  fell below the untrained baseline on several, while plain PiSSA on the same surface
  gained), and (c) eroding as external work ships the same idea with the benchmark
  closure ModelCypher lacks (Schedule-Free, Prodigy, Muon/SF-NorMuon spectral
  optimizers).
- **`mc analyze` is the genuinely differentiated surface.** An MLX-native, CLI-first,
  reproducible-bundle observability tool for local models on Apple Silicon has no
  direct competitor — TransformerLens and nnsight are PyTorch research libraries, and
  the MLX ecosystem ships inference tooling with no interpretability layer. The
  measurement quantities (CKA, intrinsic dimension, curvature, entropy trajectories)
  are exactly what the 2025–26 representational-geometry literature is converging on.
- **The epistemic discipline is a real, rare asset.** Pre-registered falsifiers honored
  through negative outcomes, a refuted-work archive, an internal claim registry that
  downgrades the project's own claims. Most solo AI projects claim; this one falsifies.
  Do not sand this off — surface it (WS0.6, WS4.3).

The through-line of every workstream: **make the public story match the evidence
ledger, harden the one surface that has no competitor, and either close the training
benchmark honestly or reframe the thesis as a research program.** Do not let the
15-knob narrative — which the default runtime contradicts — remain the headline.

Where the math is genuinely sound, it is named so below and must be preserved: Weyl's
inequality for the LoRA scale bound (correctly proven and applied; the strongest
validated result, shipped in `mc analyze lora-svd`), the exact `project_to_null_space`
linear algebra, the standard-error stopping statistics, and the measurement-side
estimators (Shannon effective rank, TwoNN/Levina-Bickel ID, CKA, randomized SVD per
Halko 2011). This brief attacks the overclaimed math, not the honest core.

---

## WS0 — Truth and positioning (cheap, do first)

**WS0.1 — Auto-generated 15-knob status matrix.**
Replace the README "What Gets Derived" table with a generated status matrix with
values {derived+shipped-default | derived+research-mode-only | formula-exists-unwired
| dead-code | removed}. Write `scripts/generate_knob_matrix.py` introspecting
`cli/commands/train.py` defaults, `core/domain/training/`, and the import graph.
Current truth per audit: LR/momentum/schedule = calibrated AdamW on the default path;
dropout/init-as-documented/residual-scaling = unwired or dead; epsilon/weight-decay =
formulas ignored by the shipped optimizer; rank/target-modules/batch-size/stopping
(+ clipping/warmup on the Cayley research path) = genuinely operative.
*Acceptance:* README table row for LR reads "default: calibrated AdamW 2e-4 cosine;
MASS on research modes"; a CI check fails when the table and code defaults diverge.

**WS0.2 — Reposition README.md.**
Headline = measurement workbench (mc analyze, observation bundles, MLX-native). Move
the 15-hyperparameter program into `docs/research/` labeled as a research program
with per-row evidence state. Rewrite the thesis sentence from "replaces all 15" to
"derives all 15" (derivation is real; superiority is not).
*Acceptance:* no claim in README.md that the README's own Evidence Snapshot
contradicts.

**WS0.3 — Fix the public GitHub metadata.**
The Ethyros-AI/ModelCypher About text and web-visible copy still advertise pre-pivot
theater ("Relational Manifold Projection", "universal basis of 439 probes",
"Geometric LoRAs without gradient training"). Replace with the measurement-workbench
positioning. *Acceptance:* fresh clone and repo page tell the same story.
*(Owner action: requires GitHub access.)*

**WS0.4 — Rosetta stone citation audit.**
`docs/research/geometric_hyperparameter_rosetta_stone.md`: for each of the 15 rows,
relabel as **derived** (only where the cited theorem actually yields the formula),
**adopted** (mainstream method used as-is: row 7 Schedule-Free/Defazio, row 8
McCandlish gradient-noise scale), or **convention** (IEEE-754 sqrt(eps) rows 2/9/13).
Propagate the R2 finding into the σ_k row: the repo's own REPORT.md states the
structural σ_k budget is "diagnostic-only — does not predict behavioral damage";
the rosetta stone still says [VALIDATED]. *Acceptance:* every remaining "derived"
label survives a citation check.

**WS0.5 — Single refutation ledger.**
Merge `papers/NEGATIVE-RESULTS.md`, `docs/research/OPEN-MATHEMATICAL-QUESTIONS-
REFUTATIONS.md`, and an index of `/Volumes/CodeCypher/archive/modelcypher-scripts/
refuted/` (15 scripts) into one `docs/research/REFUTATION-LEDGER.md` — columns:
hypothesis, date registered, date killed, kill mechanism, artifact paths. Recover the
K-FAC negative result from git (`git show ff5be1be:scripts/kfac_validation.py`;
removal commit 49060514, 2026-02-25) and write its one-page refutation note — the
clearest technique-abandonment in the history currently has no recorded reason.
*Acceptance:* one authoritative refutation count; K-FAC entry exists.

**WS0.6 — docs/HISTORY.md.**
Record the arc so it doesn't require 2,402-commit archaeology: Dec 19 TrainingCypher
port → Dec–Jan SmolLM-360M multi-donor compression era (284 scripts, 70% ceiling) →
Jan 29 archival reset → Feb 7–23 hyperparameter-derivation thesis → Feb 25–Mar 3
refutation wave (K-FAC same-day reversal; 5/5 external-theory refutations; Information
Bridge 5/8) → Mar 26 measurement-workbench pivot → Apr R1 no-go → pause. Link the
archival commits. *Acceptance:* a newcomer can reconstruct the epistemic arc from one
document.

**WS0.7 — Stale bookkeeping regeneration.**
README test count (says 6,809; actual collection 7,733 — script it, don't hand-edit);
CHANGELOG entries for 2026-03-02→05-02 from git log (BiLM margin training, atlas
family report, Cayley init, Qwen3.5 layer paths, PiSSA casting); TODO.md refresh —
including the G5 line, which still says "training run in progress" for a run that
died 2026-03-01 with a 0-byte train_result.json.tmp. Bump version past 1.0.0.
*Acceptance:* no self-reported number in README/TODO/CHANGELOG contradicts an
artifact.

---

## WS1 — Engineering foundation (enables everything else)

**WS1.1 — CI (highest priority in the entire brief).**
`.github/workflows/ci.yml`: (a) macos-14 arm64: `poetry install` +
`poetry run pytest -m "not real_model and not slow" -q`; (b) ubuntu with
`MC_DISABLE_MLX=1` exercising the JAX-CPU fallback (conftest.py supports this);
(c) ruff check; (d) mypy job — add `[tool.mypy]` to pyproject.toml, permissive with
per-module strictness. There is currently **no .github directory in a 2,402-commit
repo**; every green test run in history happened on one laptop.
*Acceptance:* green on a no-op PR; red on an intentionally broken domain test.

**WS1.2 — Dependency prune.**
Remove from mandatory deps (zero imports verified anywhere in src/scripts/tests):
`gwpy`, `gwosc`, `geoopt`, `openai-whisper` (drags torch into every install),
`tenacity`. Demote `matplotlib`, `plotly` → `viz` extra; `lm-eval` → `eval` extra
(used only by 5 files in scripts/). Add `tests/test_dependency_usage.py` asserting
every mandatory dep is imported somewhere under src/ (mirror the AST pattern in
`tests/test_hexagonal_boundaries.py`). Regenerate poetry.lock.
*Acceptance:* full suite passes on the pruned lock; dependency test is green and
would fail if a dep lost its last import.

**WS1.3 — Rescue gitignored knowledge.**
`results/nblora_vs_standard/REPORT.md` — the canonical R2 handoff CLAUDE.md line 157
points at — is gitignored and exists only on the owner's machine. Move research
REPORT/AUDIT markdowns (not data artifacts) into tracked `docs/research/reports/`,
updating references. *Acceptance:* a fresh `git clone` contains the R2 handoff.

**WS1.4 — Dead code deletion.**
`src/modelcypher/experimental/thermo.py` (shadowed by thermo/ package — import
resolves to the package; verify no direct file refs), empty
`src/modelcypher/cli/commands/safety/`, merge `util/math_utils.py` into `utils/`
(single consumer: `experimental/lora_isometry.py`), `core/domain/training/
residual_scaling.py` (zero importers), `spectral_normalized_lora_init` (unwired;
shipped init is PiSSA), `compute_geometric_dropout` wiring decision (wire it or
delete it), `core/domain/training/scaled_gd.py` if unused.
*Acceptance:* every surviving public function is reachable from `mc train run`,
`mc analyze`, `mc merge`, or an explicitly labeled research script (grep-verified).

**WS1.5 — Resolve the experimental→production leak.**
`cli/composition.py:479` backs the shipped `mc merge` with
`experimental.merge.merger.UnifiedGeometricMerger`, and 4 core/use_cases files import
from experimental/ — violating CLAUDE.md's own promotion policy. Either promote
experimental/merge/ into core or mark `mc merge` experimental in help/docs. Then
extend `tests/test_hexagonal_boundaries.py` to forbid `modelcypher.experimental`
imports from cli/ and core/use_cases (seed allowlist with the current 5 offenders;
burn it down). *Acceptance:* boundary test enforces the policy.

**WS1.6 — mlx-lm compat unblocking.**
Raise `mlx-lm ^0.30.7` to allow 0.31.x; run the compat-sensitive suites
(`tests/test_qwen35_training_compat.py`, `tests/test_mlx_training_adapter_strict.py`,
consumers of `backends/_mlx_qwen35_compat.py`). Add a weekly allowed-to-fail CI job
against latest mlx-lm — this is the fastest-rotting seam in the codebase.
*Acceptance:* suite green on 0.31.x; scheduled job exists.

**WS1.7 — Hygiene batch.**
(a) Wheel packaging: pyproject ships only `data/*.json`; `data/domain_taxonomy.yaml`
is silently excluded and `core/domain/domains.py` falls back to a hardcoded taxonomy
for pip users — add the yaml include + a wheel smoke test. (b) Unify agent docs:
AGENTS.md becomes single source (fix its directory diagram — ports/ is top-level;
add infrastructure/, experimental/, utils/), fold CLAUDE.md's operational content in
or into a tracked OPERATIONS.md, make CLAUDE.md the symlink .gitignore claims, and
rewrite GEMINI.md to reference AGENTS.md — its "discard all industry heuristics"
framing miscalibrates fresh agents. (c) `docs/SURFACE.md`: rank the 9 CLI groups and
43 analyze subcommands {promoted | instrumented | experimental}, each pointing at its
use_case service and test file, so an agent can find load-bearing code among 68
use_case services without reading 251k lines. (d) Split the modules over the repo's
own 20k-token budget: `core/use_cases/dataset_training_service.py` (2,892 lines),
`backends/_mlx_training_adapter_train_mixin.py` (2,837), `cli/commands/analyze/
geometric.py` (2,449, split by subcommand cluster). *Acceptance:*
`scripts/report_token_budget.py --threshold 20000` reports no src/ file over budget;
wheel smoke test asserts the yaml ships.

---

## WS2 — Mathematical correctness (the poke-holes fixes)

These are the specific errors a hostile referee would use to discredit the honest
core. Fix them so the good math (Weyl bound, null-space projection, SE stopping,
measurement estimators) is not tarred by the overclaimed math sitting next to it.

**WS2.1 — Unify MP noise-variance estimation and fix the wrong moment identity.**
Three files disagree: `marchenko_pastur.py`/`tikhonov_correction.py` use
σ²=trace(C)/D (mean eigenvalue *including* signal spikes, which inflates the noise
edge), while `rmt_signal_separation.py` uses median/lower-bulk estimators. Create one
spike-robust estimator module (iteratively remove top-k or use median-of-bulk with the
d−n exact-zero eigenvalues excluded), used by all three. Fix the wrong comment in
`rmt_signal_separation.py:197` ("E[λ]=σ²(1+γ)" — the MP mean is σ²). Add synthetic
spiked-covariance tests asserting signal-rank recovery for γ<1 and γ>1, **including
the N≪D probe regime** where the current code misclassifies pure signal as noise
(repro: 32 unit-signal eigenvalues among 256 dims → noise_edge 1.83 > 1.0, all signal
lost). *Acceptance:* the probe-regime test passes; one estimator, one answer.

**WS2.2 — Effective-sample-size correction before any MP application to activations.**
Token activations are autocorrelated, not i.i.d., so MP's aspect ratio γ=D/N is wrong
(effective N ≪ token count). Estimate token autocorrelation length τ and use
N_eff = N/τ in `rmt_signal_separation.py` and `tikhonov_correction.py`. Document the
i.i.d. violation and its measured magnitude on one real capture.
*Acceptance:* the aspect ratio uses N_eff; a note quantifies the correction on real data.

**WS2.3 — Fix the geodesic null-space filter.**
`core/domain/geometry/geodesic_null_space.py:_compute_basis` computes geodesic
distances, a Fréchet mean, and log-map tangent vectors (lines ~502–543) and then
**never uses the tangent vectors** — the applied operation (line ~388) is
`delta * keep_weights_row`, per-coordinate diagonal scaling in the standard basis,
which cannot represent the actual (rotated) signal subspace and silently depends on
the coordinate basis. Either (a) delete the unused geodesic computation (wasted GPU
feeding one diagnostic scalar), and (b) replace diagonal keep_weights with a proper
projection onto the noise eigenvector subspace `P_noise = V_noise @ V_noise.T` (the
eigendecomposition already exists in `compute_rmt_null_space_weights`). Add a unit
test that jointly rotates activations and delta and asserts filter output is invariant.
Run the existing merge falsifier comparing diagonal-scaling vs subspace projection on
one real merge pair. *Acceptance:* rotation-invariance test passes; the honest
`project_to_null_space` (lines ~839–999) remains the reference.

**WS2.4 — Make eta_weyl per-layer and space-consistent.**
`mass_step_size.py` compares a global flattened L2 norm of the update (across all
adapter parameters, in Cayley free-parameter space) against a per-layer weight
singular value σ_k — three category jumps (global vs per-layer, vector norm vs
spectral norm, free-parameter vs effective-delta space). Bound per-layer
‖ΔW_l‖₂ of the *effective* delta against that layer's σ_k(W_l); measure the actual
per-step spectral displacement of `get_effective_delta()` and log its ratio to the
current global-norm proxy. *Acceptance:* the Weyl label names the quantity it bounds.

**WS2.5 — Falsify or fix the √N Brownian step-budget model.**
`compute_conformal_margin_rate` justifies √N displacement growth from one data point
(964 steps → 29.3% of budget); gradient-descent steps are correlated, so real growth
can be ~N. Write a script reading saved controller step traces across all archived
runs and fit the displacement-vs-steps exponent; if it's closer to 1 than 0.5,
replace `apply_sqrt_n_epoch_correction`. Also log per-run which of {eta_sps, eta_weyl,
eta_ceiling, eta_margin} binds each step — the audit's numerical check (σ values from
the repo's own docs) shows eta_ceiling binds while SPS and Weyl never do, i.e. MASS is
a static derived LR plus safety monitors, not an adaptive controller. If so, state
that in docs. *Acceptance:* the √N law is checked against real telemetry, not assumed;
docs describe MASS as it actually behaves.

**WS2.6 — Replace f\*=0 in SPS for cross-entropy.**
`mass_step_size.py:819` uses f\*=0 for CE loss; the Polyak step needs the true optimum,
and CE has a nonzero irreducible floor, so eta_sps systematically overestimates. Use a
measured floor (running val-loss minimum or token-entropy lower bound).
*Acceptance:* SPS uses a nonzero CE floor; the MSE-distillation path keeps its
RMT-derived f\*.

**WS2.7 — Flat-data null control for curvature; rename "angular curvature."**
`riemannian_core_curvature.py` estimates sectional curvature from
(graph_geodesic − chord)/chord, but kNN-graph shortest paths overestimate geodesics
with a positive density- and k-dependent bias (Bernstein 2000 conditions unchecked),
so the "defect" is positive on flat data too — the sign is not identified. Compute the
defect on matched-size uniform-hyperplane samples at the same (n,d,k) and report
curvature sign only when the measured defect exceeds the flat-data distribution.
Rename the entropy-chain "angular curvature" to "layer rotation angle" (a well-defined
measurement mislabeled) in `mc analyze` output and docs. Replace confidence=1/(1+std)
with a bootstrap CI or remove it. *Acceptance:* curvature sign reported only against a
null; no metric named "curvature" that is a rotation angle.

**WS2.8 — Purge or honestly relabel doctrine-violating constants.**
`core/domain/training/hyperparameter_validation.py` ships bare human defaults under a
"bounds are DERIVED, not heuristics" banner: batch ∈ [1,8], SEQUENCE_MIN=128,
SEQUENCE_MAX=4096, GRAD_ACCUM_MAX=16, LR_MAX=1/√eps≈2896 (GD stability depends on
curvature, not dtype), LR_MIN=eps. Either derive from measured memory/architecture or
mark "engineering limit, underived" and drop the banner. Same for
`NBLoRAConfig init_scale=0.5*scale_bound` (bare 0.5) and the (1−√eps)≈0.99965 "margin"
that does nothing. Replace IEEE-derived thresholds on *sampled behavioral* quantities
in `geometric_early_stopping.py` (entropy-drift, margin-collapse) with bootstrap
variance of the baseline — sampling noise exceeds float32 rounding by orders of
magnitude — and fix the Higham citation (matmul error ~n·eps·‖A‖‖B‖, not √eps).
*Acceptance:* no constant claims derivation it doesn't have; the "derive every number"
doctrine is honored or the label is dropped.

**WS2.9 — Fix the dimensional error in compute_geometric_epsilon (claim #2).**
`geometric_optimizer.py:compute_geometric_epsilon` returns
max(σ_k², √eps·σ_max²) in weight-singular-value² units, but Adam's ε lives in
gradient-second-moment units (added to √v). Either derive ε from the measured v_t
spectrum and ablate {1e-8, √eps_f32, derived} on the 350M ship-path benchmark and wire
the winner, or drop claim #2 from the table. The shipped optimizer ignores it anyway.
*Acceptance:* claim #2 is either wired with units that match or removed.

**WS2.10 — Resolve the σ_k vs σ_max/2 scale-bound fork; fix the Cayley docstring.**
`cayley_lora.py:732` ships scale_bound=(σ_max/2)·(1−√eps) with a comment that it
"removes the redundant preservation constraint," abandoning the Weyl preservation
argument that `lora_spectral_theory.md` and the rosetta stone still teach as σ_k.
Either restore σ_k (accept the capacity cost) or promote the change through
`FIRST_PRINCIPLES_REVIEW_PROTOCOL.md` (mechanism/equation/falsifier) and update both
docs. Separately, fix the `cayley_transform` docstring: I+Z is invertible because the
symmetric part (YᵀY) is PSD ⇒ Re λ ≥ 0, **not** because "eigenvalues of Z are
imaginary" (false once Z includes the PSD term) — a hostile reviewer uses exactly this
kind of error to discredit the correct surrounding math.
*Acceptance:* one scale bound, one story, across code and docs; docstring is correct.

---

## WS3 — Close the credibility benchmark and fix evidence discipline

The training thesis is currently evidenced *against*, not merely unproven. Either close
it honestly on the claim the code can actually support, or reframe it as research.

**WS3.1 — Pre-register and run the ship-path closure benchmark.**
Charter at `results/r1_ship_closure/R1-SHIP-CHARTER.md`, three arms: (a) recipe
baseline standard LoRA (fixed public recipe, e.g. TorchTune lr 3e-4, r=8, alpha=16,
cosine+warmup, q+v); (b) geometry-derived *config* driving standard LoRA (the CLAUDE.md
"Ship Path": derived rank/alpha/targets/stopping + PiSSA + B_crit + certificate
stopping); (c) canonical geometric_lora. Models LFM2-350M and Qwen3.5-0.8B; seeds
{42,123,456}; full lm-eval-harness 7-task limit=None plus commensurable val CE.
Pre-register success: arm (b) or (c) beats (a) outside the tie band 2·√(se₁²+se₂²) on
≥2 tasks with no task loss outside the band, across pooled seeds. **Critically, add
optimizer-class hyperparameter-free baselines** — Prodigy, Schedule-Free AdamW, one
Muon-class optimizer at matched budget — because the claim is "no LR, no schedule," and
a reviewer's first objection is that the R1 controls are all LoRA variants, no
optimizer-class baselines. *Acceptance:* a completed multi-seed table with a
promote/no-promote verdict appended to the ledger. If arm (b) wins, that is the ship
story. If all lose, WS0.2's thesis rewrite from "replaces" to "derives" stands as
final.

**WS3.2 — Fix the benchmark quick suite before any credibility run.**
`core/use_cases/benchmark_service.py` uses n=10 per task with ceiling-saturated tasks
(the "GSM8K 70%, ARC 100%, BoolQ 100%" headline is 7/10, 10/10, 10/10 — an improvement
was arithmetically impossible on two of them). Raise per-task samples to ≥200
(stratified); route pre/post comparison through the non-ceiling eval set
(`results/g5_8b_validation/non_ceiling_eval_set_8b.json`) so a 100%-pre task can never
be an improvement target; retain per-item outputs for McNemar-style paired testing.
*Acceptance:* no ceiling-saturated task is selectable as an improvement target; paired
tests available.

**WS3.3 — Rerun G5 Qwen3-8B to completion or update its status.**
The 2026-03-01 run died after batch-size derivation with a 0-byte
train_result.json.tmp; TODO.md still says "in progress" four months later. Rerun with
gradient accumulation, 3 seeds, pre AND post benchmark on the fixed non-ceiling set,
under a completion contract: not reportable until non-tmp train_result.json plus
post-benchmark JSON exist. Update TODO.md to real status either way.
*(Operational: 8B is production-confidence only per CLAUDE.md; check GPU processes
first; do not run during other GPU work.)* *Acceptance:* G5 line reflects a completed
run or an explicit PARKED status.

**WS3.4 — Reconstruct and retain the MP-Tikhonov vs binary projector A/B.**
The binary projector was permanently deleted on the strength of an A/B ("won all 5
metrics," CHANGELOG: preserved fraction +35%, degeneration 0.088 vs 0.759) whose
result artifact was never retained and cannot be verified from any file. Recover the
binary projector from git, run `scripts/merge_ab_test.py` on ≥2 model pairs, write
5-metric results to `results/merge_projector_ab/REPORT.md`. If binary matches or beats
Tikhonov on a majority, the registry falsifier CR-MRG-001 fires and the removal
reopens. *Acceptance:* the CHANGELOG numbers are backed by a retained artifact or
corrected.

**WS3.5 — Behaviorally calibrate the CKA preservation gate.**
`scripts/g5_8b_validation.py:514` requires min_cka ≥ 1−√eps_f32 ≈ 0.99966 — a
machine-precision threshold any learning adapter fails, so the gate is near-unpassable
and (per the R2 finding that structural budgets don't predict behavioral damage) is
measuring the wrong thing. Regress online-eval degradation against min-CKA across the
retained 350M pipeline_validation trials and G5 seeds; pick the CKA level below which
significant degradation (existing CI machinery) becomes likely; document the
derivation; re-evaluate retained gate summaries under the new threshold.
*Acceptance:* the gate threshold is derived from measured CKA↔damage data, not dtype.

**WS3.6 — Two-tier validation tags + a seed/retention policy.**
Split `[VALIDATED]` into `[VALIDATED-ENG]` (code/memory/mechanics; 1 run OK) and
`[VALIDATED-EFF]` (benchmark efficacy; ≥3 seeds, pooled effect outside 2·SE). Apply
retroactively (data-rank ceiling and gradient accumulation → ENG). Add a retention
rule: raw per-seed gates.json / train_result.json / benchmark JSON may not be deleted
until the aggregate verdict is computed and committed — the seed43 deletion that left
a 1-seed "multiseed" family must not recur. *Acceptance:* every efficacy claim carries
a seed count; retention rule documented in AGENTS.md.

**WS3.7 — Close the R2 mechanism (needed for any preservation claim).**
Execute the pre-registered next falsifier in `results/nblora_vs_standard/
NEXT-FALSIFIER.md` for inference-representation collapse (train CKA 0.95 vs inference
CKA min 0.01). R3/R5 gates are meaningless until an operator predicts behavioral damage
before online degradation. Note: the R2 collapse (healthy train-space CKA, collapsing
inference-manifold CKA — format memorization destroying inference geometry) is
plausibly publishable on its own as a cautionary measurement study (see WS4.3).
*Acceptance:* a pre-registered operator that predicts failure before online
degradation, survives intervention, and explains the retained 350M failure cases.

**WS3.8 — Sync the external drive to current state.**
`/Volumes/CodeCypher/CATALOG.json` (dated 2026-01-03) describes only the superseded
SmolLM-360M compression project and indexes none of the ~118 experiments/ directories;
`experiments/README.md` still advertises the downgraded "88.5% cross-model invariance"
claim. Regenerate the catalog with per-experiment status (active/refuted/superseded)
linked to the claim registry; either create `archive/results-refuted/` as referenced
in briefs or fix all references to the actual `archive/modelcypher-scripts/refuted/`.
*Acceptance:* drive metadata matches the repo's current evidence state.

---

## WS4 — The differentiated wedge (product direction)

This is where the defensible value is. Prioritize after WS0–WS1; WS4.1 and WS4.3 are
the highest-leverage moves in the whole brief for a public release someday.

**WS4.1 — Ship `mc advise`: the measurement-to-config product.**
A command that measures a model (spectral decay, tail_dims, effective rank per layer)
and emits a ready-to-use LoRA config (rank, alpha, target modules, early-stop
criterion) for standard stacks — PEFT/Axolotl YAML and mlx-lm LoRA config output. This
decouples the *defensible* derivation claim (config) from the *unclosed* one
(optimizer). No one ships an auto-config product; PiSSA/EVA/AdaLoRA remain
methods-in-papers. *Acceptance:* `mc advise` output drops into axolotl/mlx-lm
unmodified, and an A/B on one model shows the derived config ≥ r=16/alpha=16 defaults.

**WS4.2 — Build a replication anchor for the workbench.**
Use `mc analyze` to reproduce two published results on local models: (a) the layerwise
intrinsic-dimension expansion→compression profile with explicit TwoNN vs MLE estimator
comparison, citing the Ansuini lineage and 2025–26 ID papers — the "semantic highway"
result (15.8→1.8→9.6) is a rediscovery of this known phenomenon and must cite/compare,
not name-and-claim; (b) the curvature↔next-token-entropy coupling (arXiv 2604.23985).
Store as a canonical `results/` family with REPORT.md. *Acceptance:* metrics match
published qualitative profiles; discrepancies documented. This is the fastest path to
"the instrument reads true."

**WS4.3 — Write and post the workbench preprint.**
Merge `papers/paper-4-modelcypher-toolkit.md` and `paper-5-semantic-highway.md` into
one arXiv submission: "an MLX-native measurement workbench for local-model
representation geometry, validated by replication," with the R2 train-CKA-vs-inference-
CKA collapse as the flagship case study. Remove doctrine language ("probability does
not cause events," softmax-is-observer-side) and self-assigned PROVEN badges; add a
related-work section mapping every ModelCypher term to field-standard vocabulary
(semantic highway → ID expansion-compression; entropy trajectory → next-token entropy
dynamics). Target a mech-interp or efficient-ML workshop. *Acceptance:* a submission
draft with standard-estimator baselines and no unbaselined claim.

**WS4.4 — Add an interop export path.**
`mc analyze export` writing bundle tensors (layer activations, per-layer metrics) as
.safetensors/.npz plus a starter notebook reproducing the same analysis in PyTorch, so
TransformerLens/nnsight users can verify ModelCypher numbers in their own stack.
*Acceptance:* a round-trip demo notebook in examples/ runs on one HF model.

**WS4.5 — Relicense the measurement surface for adoption.**
Current AGPL-3.0 suppresses academic and industrial evaluation of tooling. If research
adoption is a goal, dual-license or relicense the `mc analyze` surface to Apache-2.0 /
MIT. *(Owner decision.)* *Acceptance:* the observability CLI carries a permissive
license.

**WS4.6 — Gate the safety and merging surfaces.**
Before any public claim from paper-2, run the entropy-signal probes head-to-head
against a linear-probe baseline (HiddenDetect-style) on a shared jailbreak set — show
an advantage or relabel the CLI verbs (calibrate-safety, jailbreak-test, probe-redteam)
experimental in `docs/CLI-REFERENCE.md`. Hold the line on merging (no new algorithms
until R1–R4 close, per the roadmap); when R5 activates, evaluate on MergeBench against
TIES, Task Arithmetic, WUDI-Merging, and model soups rather than a custom harness. The
distinctive merging artifact is the machine-readable *preservation certificate*
(verdict.json with declared gates) — scope that as a measurement product independent of
which merge algorithm wins. *Acceptance:* no unbaselined safety/merge claim ships as
fact.

**WS4.7 — De-orbit the cross-domain forays.**
`plasma/` (5-shot MAST tokamak work, torch-based, unintegrated, untested, no baseline
vs published disruption predictors) plus the drive's geometric_cryptanalysis /
sha256_manifold / modelcypher-astronomy forays lend the flagship repo apparent breadth
while being unvalidated single-run work. Move `plasma/` to its own repo or
`/Volumes/CodeCypher/archive/`; delete plasma tasks from TODO.md. If the "1s disruption
lead time" claim is ever made externally, it needs a DisruptionBench baseline first.
*Acceptance:* the main repo contains only validated or clearly-labeled-research
surfaces; torch leaves the main install.

---

## Sequencing summary

- **First (days, cheap, high-trust):** WS0 entirely, WS1.1 (CI), WS1.3 (rescue
  gitignored REPORT), WS1.2 (dep prune).
- **Second (correctness, protects credibility):** WS2 entirely — do WS2.1/2.3/2.10
  first (the errors most visible to a reviewer), then the rest.
- **Third (the credibility question):** WS3.1–WS3.3 (close or reframe the benchmark),
  then WS3.4–WS3.8 (evidence hygiene).
- **Fourth (the future):** WS4.1 (`mc advise`) and WS4.2–WS4.3 (replication + preprint)
  are the moves that make this citable and releasable.

**One-line verdict for the owner:** the instrument is real and nearly unique; the
optimizer thesis is not yet earned and may not be. Reframe around the instrument, fix
the math a referee would attack, close or park the benchmark honestly — and the work
becomes something the world could actually appreciate.
