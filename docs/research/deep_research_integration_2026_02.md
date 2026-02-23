# Deep Research Integration (2026-02) `[EMPIRICAL]`

## Purpose

This document is the canonical integration record for six externally generated
deep-research reports supplied on 2026-02-22. It captures provenance,
deduplication, normalized claim handling, evidence labels, and file-level
mapping into the ModelCypher code/docs surface.

Scope constraints for this pass:
- Documentation and docstring/comment updates only
- No runtime behavior changes
- No schema/interface changes
- No raw report copies committed to this repository

---

## Source Registry (Provenance + Dedupe)

| Report | Absolute source path | SHA-256 | Status |
|---|---|---|---|
| 4 | `/Users/jasonkempf/Downloads/deep-research-report (4).md` | `0deac52d399a61e29bbef275d8f615c05a59b904f80406e030d9997fcddaec19` | Unique |
| 5 | `/Users/jasonkempf/Downloads/deep-research-report (5).md` | `3d8016da6eacc70653b30ec694bb38737919be33ff477ef49fb5a98c3e03aef9` | Unique |
| 6 | `/Users/jasonkempf/Downloads/deep-research-report (6).md` | `d2b58706d758d94152dac85a12179848d178dd88b61b0e9e43a5a7f0763a45f0` | Unique canonical for 6/8 |
| 7 | `/Users/jasonkempf/Downloads/deep-research-report (7).md` | `1a5c675ddccc630b71e350c87bd5276970a86e7906ff94b93980c3df8b7b1c34` | Unique |
| 8 | `/Users/jasonkempf/Downloads/deep-research-report (8).md` | `d2b58706d758d94152dac85a12179848d178dd88b61b0e9e43a5a7f0763a45f0` | Duplicate of report 6 |
| 9 | `/Users/jasonkempf/Downloads/deep-research-report (9).md` | `68a24b6ec4b1ecceb92cbad79c038e5d88afff325b061b2178ae9cd43907e03a` | Unique |

Deduplication rule for this pass:
- Reports 6 and 8 are byte-identical and are treated as one evidence source.

---

## Normalized Reference Anchor Set

Imported reports contained tool-specific citation artifacts. Incorporated claims
were normalized to repository-usable anchors:

- `docs/EVIDENCE-TAXONOMY.md`
- `docs/research/lr_derivation_analysis.md`
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- `docs/research/architecture_geometry_theory.md`
- `docs/research/mhc_null_space_connection.md`
- `docs/research/topological_fingerprints.md`
- `docs/research/field_map_external_methods.md`
- `docs/references/BIBLIOGRAPHY.md`

External-paper identifiers retained where already present in repository docs:
- `arXiv:2512.24880` (mHC)
- `arXiv:2305.18290` (DPO)
- `arXiv:1905.00414` (CKA)

---

## Claim Normalization Matrix

| Imported claim | Evidence label | ModelCypher status | Mapped file/module | Decision |
|---|---|---|---|---|
| Public frontier labs show a geometric-mechanism turn | `[CONJECTURAL]` | Contextual landscape, not direct mechanism proof in this repo | `docs/research/field_map_external_methods.md` | Adopt (context only) |
| ~~`beta_1` / `delta_beta_1` can index reasoning reliability~~ | `[DISPROVEN]` | Falsification protocol (6 tests, n=50, LFM2-350M) failed 3/6: metric robustness, held-out replication, subsample stability. See `results/beta1_falsification/full/LFM2-350M/FALSIFICATION_REPORT.md` | `docs/research/topological_fingerprints.md` | Disproven |
| Inference-time exact PH is expensive; graph-cycle proxy is practical | `[EMPIRICAL]` | Computationally aligned with current topology implementation limits | `docs/research/topological_fingerprints.md` | Adopt |
| Global HVP-Lipschitz LR is brittle for nonsmooth stochastic training | `[VALIDATED]` | Already observed in repo ablations; MASS already implemented | `docs/research/lr_derivation_analysis.md`, `src/modelcypher/backends/mlx_training_adapter.py` | Adopt (consistency cleanup) |
| Retraction-based Armijo + measured step conditions outperform fixed-L logic | `[EMPIRICAL]` | Already partly implemented in training adapter | `src/modelcypher/backends/mlx_training_adapter.py` (comments/docs only this pass) | Adopt (documentation) |
| CE on long reasoning traces often optimizes format over invariants | `[VALIDATED]` (repo scope) | Matches existing internal findings and roadmap language | `docs/RESEARCH-ROADMAP.md`, `docs/research/lr_derivation_analysis.md` | Adopt |
| LIMO / RLVR suggest outcome-aligned objectives + entropy are key | `[CONJECTURAL]` to `[EMPIRICAL]` by sub-claim | Directionally aligned with current outcome training stack | `src/modelcypher/core/domain/training/regime_selection.py`, `docs/research/lr_derivation_analysis.md` | Adopt with bounded claim strength |
| Architecture knobs (depth/width/head/GQA/RoPE) shape geometric regimes | `[CONJECTURAL]` | Existing theory doc has partial coverage; add caveats on confounds | `docs/research/architecture_geometry_theory.md` | Adopt |
| C3 abstention should be evaluated by selective-risk protocol, not confidence only | `[CONJECTURAL]` | Existing protocol had high-level framing; needs concrete benchmark schema | `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md` | Adopt |
| mHC + null-space can be unified as intersection of invariance sets | `[CONJECTURAL]` | Existing synthesis present; claim-strength bounds needed | `docs/research/mhc_null_space_connection.md` | Adopt (bounded language) |
| DPO vs REINFORCE under spectral budget is an engineering tradeoff | `[CONJECTURAL]` | Discussion-level only; no production switch in this pass | `docs/RESEARCH-ROADMAP.md`, `docs/research/field_map_external_methods.md` | Defer (research thread) |

---

## Per-Report Integration Notes

## Report 4: Public Geometric-Turn Signals

Classification:
- External context and field-signaling only
- Not sufficient for mechanism validation claims

Integration:
- Added as evidence-labeled context in `docs/research/field_map_external_methods.md`
- Explicitly marked as non-mechanistic and non-decisive for internal theorem status

## Report 5: `beta_1` / `delta_beta_1` Reasoning Signatures [DISPROVEN]

Classification:
- ~~Mechanistically plausible, requires robustness controls~~
- **Disproven (2026-02-22).** Robustness protocol executed; claim failed 3/6 tests.

Integration:
- Added deployment split (offline exact PH vs online graph proxy) to
  `docs/research/topological_fingerprints.md`
- Added robustness protocol requirements: distance sensitivity, subsample
  stability, null-shuffle controls, layer-window calibration, and proxy gating
- **2026-02-22:** Full falsification protocol executed (`scripts/beta1_falsification.py`,
  n=50, LFM2-350M). F1 (metric robustness) FAIL, F3 (held-out replication) FAIL,
  F5 (subsample stability) FAIL. Claim moved to `[DISPROVEN]` across all docs.

## Reports 6/8: Nonsmooth Stochastic Manifold Step-Size Theory

Classification:
- Strongly aligned with existing internal failure analysis

Integration:
- Used to reconcile stale `eta = 1/L` active wording across docs/docstrings
- Canonical history remains `docs/research/lr_derivation_analysis.md`
- No algorithm changes introduced in this pass

## Report 7: CE Trace Memorization vs Objective Alignment

Classification:
- Consistent with existing ModelCypher findings on CE trace failure mode

Integration:
- Cross-linked in roadmap/training documentation as objective-level caution
- Retained bounded language between validated repo findings and broader claims

## Report 9: Architecture Predictors, C1/C2/C3, mHC-null-space, DPO/REINFORCE

Classification:
- Mixed: strong synthesis value, heterogeneous evidence quality

Integration:
- Added predictor caveats and family-confound guidance in
  `docs/research/architecture_geometry_theory.md`
- Added concrete C3 selective-risk evaluation protocol details in
  `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- Refined mHC/null-space framing in
  `docs/research/mhc_null_space_connection.md` with bounded claim language

---

## Adoption Rules Applied

- Claims without in-repo reproduction or theorem proof remain `[CONJECTURAL]`
- Historical/disproven LR derivation language is kept only as historical context
- Tool-specific citation artifacts are excluded from repository text
- No hidden thresholds introduced; all thresholds remain data-derived, epsilon-derived,
  or explicitly marked as research defaults

---

## File-Level Mapping (Adopted)

Documentation updates linked to this integration pass:
- `docs/TRAINING-GUIDE.md`
- `docs/CURRENT-STATE.md`
- `docs/CLI-REFERENCE.md`
- `docs/MISSION.md`
- `docs/research/1p2b_training_configuration.md`
- `docs/research/geometric_hyperparameter_rosetta_stone.md`
- `docs/research/topological_fingerprints.md`
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- `docs/research/architecture_geometry_theory.md`
- `docs/research/mhc_null_space_connection.md`
- `docs/research/field_map_external_methods.md`
- `docs/RESEARCH-ROADMAP.md`
- `src/modelcypher/core/domain/training/geometric_optimizer.py` (docstring only)
- `src/modelcypher/backends/mlx_training_adapter.py` (docstring/comments only)

---

## Out of Scope (Explicitly Deferred)

- Runtime retraining policy changes (for example, replacing REINFORCE with DPO)
- New CLI surface for report-specific diagnostics
- New benchmark datasets or artifact generation in this pass
- Bulk bibliography ingestion of every external citation from source reports

