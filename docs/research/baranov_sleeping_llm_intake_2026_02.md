# Baranov Sleeping-LLM Intake (2026-02)

Date: 2026-02-26  
Owner: ModelCypher research intake  
Default stance: no wholesale external code import; treat external work as hypotheses to re-derive under `docs/MISSION.md`.

## 1. Immutable Source Registry

### 1.1 Paper Artifacts (local + DOI + Zenodo)

| Paper | Local PDF | SHA256 | DOI | DOI URL | Zenodo record | Zenodo publication date | Zenodo files |
|---|---|---|---|---|---|---|---|
| 1 | `/Users/jasonkempf/Downloads/1-Sleep-Wake-Consolidation.pdf` | `ad13c39f45d732f6df0b2dd9aef892c2066a0e88837a99a72db3b4e8c7c28e0b` | `10.5281/zenodo.18778760` | `https://doi.org/10.5281/zenodo.18778760` | `https://zenodo.org/record/18778760` | `2026-02-01` | `1-Sleep-Wake-Consolidation.pdf` |
| 2 | `/Users/jasonkempf/Downloads/2-Alignment-Tax.pdf` | `3f383c0bb871e39ea3093636c85c286db6503c60d688fa82efab86f10a7266cc` | `10.5281/zenodo.18778762` | `https://doi.org/10.5281/zenodo.18778762` | `https://zenodo.org/record/18778762` | `2026-02-08` | `2-Alignment-Tax.pdf` |
| 3 | `/Users/jasonkempf/Downloads/3-Dual-System-Memory-Consolidation.pdf` | `46546e57de103b61f36971c47c7fbab47008fc5f50bdb2b1d1d951cf39c2ee69` | `10.5281/zenodo.18778764` | `https://doi.org/10.5281/zenodo.18778764` | `https://zenodo.org/record/18778764` | `2026-02-12` | `3-Dual-System-Memory-Consolidation.pdf` |
| 4 | `/Users/jasonkempf/Downloads/4-Sleeping-LLM.pdf` | `0a7264797a006d462558d16869e2bd0a7ebb216f6b30e043307dcbd76cf01477` | `10.5281/zenodo.18778766` | `https://doi.org/10.5281/zenodo.18778766` | `https://zenodo.org/record/18778766` | `2026-02-18` | `4-Sleeping-LLM.pdf` |
| 5 | `/Users/jasonkempf/Downloads/5-Sleep-Wake-Memory-Convergence.pdf` | `abcf52d54a2a81f6a031d7410dbe82808ac9037478ff5eaf47228664d79419df` | `10.5281/zenodo.18778768` | `https://doi.org/10.5281/zenodo.18778768` | `https://zenodo.org/record/18778768` | `2026-02-25` | `5-Sleep-Wake-Memory-Convergence.pdf` |
| 6 | `/Users/jasonkempf/Downloads/6-Per-Fact-Graduated-Consolidation.pdf` | `ab70b1c090ba08bac255bdb9a1c3bc0324b32660e203c2dd2084acc23494bb66` | `10.5281/zenodo.18779159` | `https://doi.org/10.5281/zenodo.18779159` | `https://zenodo.org/record/18779159` | `2026-02-25` | `6-Per-Fact-Graduated-Consolidation.pdf` |

### 1.2 External Code Artifact

| Field | Value |
|---|---|
| Repository URL | `https://github.com/vbario/sleeping-llm` |
| Audited clone path | `/tmp/sleeping-llm-zj8L6L` |
| Commit SHA | `111aa740eff1be5994d9f685e012bf9a465f9ed7` |
| Commit timestamp | `2026-02-25T21:35:41-05:00` |
| Commit subject | `Merge branch 'report'` |

## 2. Provenance Confidence

- Identity binding for papers 1-6 is strong: local PDFs, Zenodo records, and DOI endpoints agree on title and numbering.
- Reproducibility provenance is limited: Zenodo records currently contain single PDF artifacts only (no full code/config/result bundles attached to records).
- Citation confidence for provenance metadata: high.
- Citation confidence for headline performance claims: medium pending independent replication.

## 3. Repository Credibility Audit (sleeping-llm)

| Audit dimension | Observation | Risk for adoption |
|---|---|---|
| License hygiene | No repository `LICENSE` file present in audited commit; `README.md` claims MIT. | Legal ambiguity for code reuse; keep no-import stance. |
| CI presence | No `.github/workflows` found. | No automated regression guardrails. |
| Test surface | `tests/` has 3 files, 56 test functions. Direct file execution passes in audited environment. | Coverage is narrow relative to claim breadth. |
| Dependency pinning | `requirements*.txt` use lower-bound pins (`>=`), no lockfile discipline in external repo. | Re-runs may drift over time. |
| Release/version hygiene | `VERSIONS.md` is branch-history text, no clear semver/tagged release process. | Weak provenance of experiment-to-release mapping. |
| Artifact traceability | Many scripts/configs/results present under `experiments/`, `results/`, `notes/`. | Better than paper-only, but still uneven and not DOI-bundled. |
| Seed/control rigor | Papers explicitly state single-run limitations in multiple versions. Some scripts expose `--seed`. | Most headline claims remain [EMPIRICAL], not [VALIDATED]. |

## 4. Reproducibility Scorecard (per paper headline)

Scoring rule: `script + config + raw result + seed/control definition` => max 4.

| Paper | Headline area | Executable script | Config artifact | Raw result artifact | Seed/control definition | Score | Intake read |
|---|---|---|---|---|---|---|---|
| 1 | 3B LoRA sleep-wake memory formation on Mac | Partial (`src.main` flow; no dedicated paper script) | Yes (`config.yaml`) | Partial (narrative + limited files) | Partial (single-run; no sweep protocol) | 2/4 | Mechanism demo, weak replication package |
| 2 | Alignment tax inverse scaling (3B/8B/70B) | Partial (pipeline exists; no one-command repro script) | Yes (`experiments/configs/*.yaml`) | Yes (`experiments/results/*`) | Partial (single-run; limited 70B sweep) | 3/4 | Re-runnable but under-controlled |
| 3 | Dual-system MEMIT + LoRA | Yes (`experiments/test_*`, lifecycle scripts) | Yes | Yes | Partial (single-run emphasized) | 3/4 | Good scriptability, weak statistical rigor |
| 4 | Two-phase SWS+REM | Yes (`experiments/test_consolidation.py`, `test_rem_ppl.py`) | Yes | Yes (`experiments/results/test_rem_ppl*.json`) | Partial (single-run emphasized) | 3/4 | Replicable pipeline, limited controls |
| 5 | Wake threshold + sleep convergence | Yes (`memit_capacity_test.py`, `v7_convergence_test.py`) | Yes | Yes (`v7_*`, convergence JSONs) | Partial (single-run emphasized) | 3/4 | Strongest script/result linkage |
| 6 | Per-fact graduated consolidation | Yes (`sweep_consolidation_capacity.py`) | Yes | Yes (`sweep_consolidation_*.json`) | Partial (3-cycle horizon, no external baselines) | 3/4 | Useful experimental seed, not validated |

## 5. Claim Matrix (atomic, normalized, adoption-gated)

Evidence labels follow `docs/EVIDENCE-TAXONOMY.md`.

| ID | Atomic claim | Type | Evidence label | Missing controls / blockers | ModelCypher mapping | Decision |
|---|---|---|---|---|---|---|
| C1 | Sleep-wake LoRA cycle can persist some conversational facts across restart (3B setup). | empirical result | [EMPIRICAL] | Single-run, synthetic fact source, narrow eval. | `core/domain/lora_memory_store.py`, `core/use_cases/lora_memory_service.py` | defer |
| C2 | A narrow LR window around `1e-4` is required for stability in paper-1 setup. | engineering heuristic + empirical result | [EMPIRICAL] | Hyperparameter sweep is ad hoc; no geometric derivation. | training derivation stack under `core/domain/training/*` | reject (as rule), defer (as hypothesis) |
| C3 | Spaced replay across sleep cycles improves recall. | mechanism + empirical result | [EMPIRICAL] | No multi-seed confidence interval; curation confounders. | replay-like behavior via event queues in memory modules | defer |
| C4 | LoRA-based memory consolidation shows inverse scaling in instruction models (3B>8B>70B recall). | empirical result | [EMPIRICAL] | Single family, 4-bit quantization confound, sparse 70B sweep. | benchmark and validation use-cases (`benchmark_service.py`) | adopt (as replication target) |
| C5 | Alignment prior (RLHF/SFT stack) causally suppresses learned recall at large scale. | speculative interpretation | [CONJECTURAL] | Causal mechanism not isolated experimentally. | requires mode-split evaluator + causal probes | defer |
| C6 | 70B can converge train loss while exhibiting zero recall in targeted probes. | empirical result | [EMPIRICAL] | Needs independent replication with controls. | validation pipeline + recall-mode split | adopt (failure signature) |
| C7 | Dual MEMIT+LoRA provides instant wake recall + post-sleep persistence. | empirical result | [EMPIRICAL] | Single-run and synthetic facts only. | experimental continual stack + LoRA memory path | defer |
| C8 | Cross-edit null-space constraints preserve prior edits (near-1 retention claims). | mechanism + empirical result | [EMPIRICAL] | Broader model/scale falsification missing. | `core/domain/geometry/null_space_tracker.py`, `experimental/continual/knowledge_encoder.py` | adopt (experimental) |
| C9 | Sleep pressure from fixed weighted signals (`edits/time/ppl`) is a valid scheduler. | engineering heuristic | [CONJECTURAL] | Fixed weights and thresholds are heuristic. | background consolidation and policy layers | reject (until re-derived) |
| C10 | Covariance-regularized MEMIT can keep PPL drift near zero while editing. | empirical result | [EMPIRICAL] | Metric scope narrow; identity-text proxy risks mismatch. | safety/validation metrics + preservation suite | defer |
| C11 | SWS+REM staging reduces PPL damage versus SWS-only in some regimes. | empirical result | [EMPIRICAL] | Non-monotonic scaling unresolved; single-run. | experimental consolidation services | defer |
| C12 | Fixed gates (`tau_ppl`, `tau_recall`, fixed stage rules) safely govern consolidation. | engineering heuristic | [CONJECTURAL] | Constants not geometry-derived. | consolidation control layer | reject |
| C13 | Wake capacity has a sharp tipping threshold (8B: 13->14 facts). | empirical result | [EMPIRICAL] | Needs multi-seed and model-family replication. | continual interference experiments + null-space metrics | adopt (capacity-track hypothesis) |
| C14 | Sleep refresh cycles can recover from severe degradation to high recall. | empirical result | [EMPIRICAL] | Convergence guarantees not formally validated. | `experimental/use_cases/consolidation_service.py` + tracker metrics | defer |
| C15 | Pruning can induce a death-spiral regression mode. | empirical result | [EMPIRICAL] | Root-cause isolation incomplete. | regression test target for consolidation pipelines | adopt (negative control / regression guard) |
| C16 | Per-fact gating outperforms per-edit all-or-nothing gating for consolidation progression. | empirical result | [EMPIRICAL] | Needs broader workload mix. | missing experimental interfaces (fact-level stage state) | adopt (experimental interface) |
| C17 | Fixed graduated schedule `1.0 -> 0.5 -> 0.1 -> 0.0` is generally correct. | engineering heuristic | [CONJECTURAL] | Static schedule is not derived from dtype/spectrum/baselines. | consolidation stage progression | reject (fixed constants) |
| C18 | Cumulative fusing progressively erodes alignment tax and enables transfer. | mechanism + empirical result | [EMPIRICAL] | Mechanistic evidence incomplete; long-horizon unknown. | experimental consolidation-only pathway | defer |
| C19 | Effective lifetime capacity is unbounded under per-fact graduated consolidation. | speculative interpretation | [CONJECTURAL] | Long-horizon and saturation limits not tested. | capacity metrics + long-run protocol | reject (current evidence) |

## 6. Relevance Map to ModelCypher

### 6.1 Paper 1 (LoRA sleep-wake consolidation)

What already exists:
- Two-tier LoRA memory store and merge path:
  - `src/modelcypher/core/domain/lora_memory_store.py`
  - `src/modelcypher/core/use_cases/lora_memory_service.py`
- Experimental consolidation path:
  - `src/modelcypher/experimental/use_cases/consolidation_service.py`
  - `src/modelcypher/experimental/use_cases/entropy_learning_bridge.py`

What is missing for direct replication:
- Paper-style curation/replay pipeline for conversational QA extraction with reproducible dataset manifests.
- Explicit restart-persistence protocol artifacts tied to controlled fact sets.

What conflicts with `MISSION.md`:
- Fixed LR windows and fixed validation thresholds from external work.
- Any keyword-score heuristics for curation without geometric/statistical derivation.

### 6.2 Paper 2 (Alignment tax)

What already exists:
- General benchmarking and evaluation service:
  - `src/modelcypher/core/use_cases/benchmark_service.py`
- Chat formatting support:
  - `src/modelcypher/core/domain/chat_template.py`

What is missing:
- Dedicated recall-mode evaluator that scores both `raw_completion` and `chat_template` from the same fact set.
- Pre-registered alignment-tax replication harness with CKA drift and preserved-fraction metrics.

What conflicts with `MISSION.md`:
- Interpreting fixed percentage degradations as universal thresholds.
- Accepting LoRA success based on single score gates without geometry-derived acceptance criteria.

### 6.3 Papers 3-6 (MEMIT, staged consolidation, capacity convergence)

What already exists:
- Null-space projection and behavioral-preservation metrics:
  - `src/modelcypher/core/domain/geometry/null_space_tracker.py`
  - `src/modelcypher/experimental/continual/knowledge_encoder.py`
- Experimental manifold-completion and consolidation scaffolding:
  - `src/modelcypher/experimental/continual/manifold_completion.py`
  - `src/modelcypher/experimental/use_cases/consolidation_service.py`

What is missing:
- Direct MEMIT-equation experimental module (Woodbury-style update path with traceable matrices).
- Fact-level data model for staged progression:
  - `FactTriple`
  - `EditState`
  - `ConsolidationStage`
- Per-fact advancement/retreat service with rollback-safe state transitions.

What conflicts with `MISSION.md`:
- Static stage schedules and hardcoded rollback percentages.
- Arbitrary iteration floors/caps not tied to geometry ceilings.

## 7. Heuristic Removal and Re-Derivation Spec

External heuristic forms observed in source materials are blocked unless replaced by one of:
- dtype precision (`eps`, `sqrt(eps)`),
- spectral/SVD structure,
- baseline distribution statistics from measured controls.

| External heuristic pattern | External example | ModelCypher-compliant replacement | Status |
|---|---|---|---|
| Fixed degradation threshold | `degraded_threshold = 0.5` | Threshold from baseline degradation distribution (quantile or z-score gate recorded per run). | required rewrite |
| Fixed PPL rollback percentage | `max_ppl_increase = 0.10` / `0.15` | Accept/reject based on measured baseline drift envelope on held-out controls, with confidence interval. | required rewrite |
| Static stage schedule | `1.0, 0.5, 0.1, 0.0` | Stage weight from measured transfer strength and preserved-fraction trajectory; no hardcoded ladder. | required rewrite |
| Arbitrary per-fact iteration budget | `iters_per_fact = 10` | Stop by convergence from `sqrt(eps)` and spectral exhaustion criteria. | required rewrite |
| Hard cap refresh count | `max_refresh_per_cycle = 10` | Bound from available null rank / trajectory rank and measured update survival. | required rewrite |
| Fixed sleep-pressure weights | e.g. `0.6/0.3/0.1` | Data-derived weighting from observed predictive power on degradation onset. | blocked until derived |

If no principled derivation is available, feature remains `[CONJECTURAL]` and is not promoted.

## 8. Adoption Gate Summary

Decision totals from this intake:
- `adopt`: 6 claims (as experimental hypotheses / interfaces only)
- `defer`: 9 claims (requires independent replication and controls)
- `reject`: 4 claims (heuristic/static or over-claimed)

Policy for this intake phase:
- No production CLI changes.
- No external code import.
- External citations are allowed only with explicit evidence label and scope qualifier.
- Promotion requires passing the replication protocol in `docs/research/baranov_replication_protocol_2026_02.md`.

## 9. Execution Status

### Patchset 1: Experimental Interfaces & Scaffolding (2026-02-26)

**Status: Implemented.**

Implemented under `src/modelcypher/experimental/baranov/`:

| Component | File | Status |
|-----------|------|--------|
| `FactTriple`, `EditStatus`, `EditState`, `ConsolidationStage` | `models.py` | Done |
| `RecallEvaluator` protocol, `compute_recall_aggregate` | `recall_evaluator.py` | Done (protocol only, no concrete evaluator) |
| `FactConsolidationTracker` (per-fact stage machine) | `consolidation_tracker.py` | Done |
| `EditApplicator` protocol stub | `edit_applicator.py` | Done (interface only) |
| `ReplicationManifest` + `validate_manifest` | `manifest.py` | Done |
| Artifact writers (JSON/CSV/markdown) | `artifact_writer.py` | Done |
| Clopper-Pearson CI (promoted to public API) | `core/domain/statistics.py` | Done |

Tests: `tests/experimental/baranov/` (unit, integration, numeric literal audit).

### Still Pending

- Concrete `RecallEvaluator` implementation (patchset 2).
- Concrete `EditApplicator` implementation -- Woodbury/MEMIT path (patchset 2).
- Track runner scripts (`scripts/baranov_track_{a,b,c}.py`).
- Actual model runs with artifact collection.
