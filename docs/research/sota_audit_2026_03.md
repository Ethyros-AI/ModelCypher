# SOTA Audit (2024-2026): ModelCypher vs Industry/Public Signals

**Date:** 2026-03-03
**Scope:** Section 7, Section 9, mission-closure open items, LoRA/PEFT theory, and model-merging position.
**Method:** Firecrawl search/scrape/extract only; retained sources tiered T0-T4 and crosswalked to internal claim records.

## Executive Summary

ModelCypher is at the edge in three places:
1. **Information-theoretic measurement validity** for transformer-layer analysis (sigma commensurability, degeneration diagnostics, DPI-violation mechanism accounting).
2. **Residual-stream non-Markov diagnostics** linking DPI-violation magnitude to bypass strength across three model families.
3. **Integrated geometric training controller** breadth (single-stack derivation of multiple controls vs fragmented external methods).

ModelCypher should adapt external work in three places:
1. **CKA reliability corrections** (debiased/sparse-sampling-aware estimators).
2. **LoRA theorem-conditioned failure diagnostics** from 2025 proof-level results.
3. **Benchmark methodology for mission closure** (contamination-aware rolling eval and leakage quantification).

Frontier work remains open in four places:
1. **Signed DPI-violation direction law** (current model explains magnitude but not sign).
2. **Quantization frontier equation** (Weyl crossing severity to CKA floor, architecture terms explicit).
3. **Unused-subspace residual causal mechanism** for degeneration reduction.
4. **Entropy-curvature mechanism split** (logit entropy vs attention entropy, architecture-conditioned sublayer effects).

## Comparative Verdicts

- `CUTTING_EDGE`: 8 claims
- `ADAPT_OTHERS`: 5 claims
- `PUSH_FURTHER`: 6 claims
- `DEPRIORITIZE`: 2 claims

Detailed per-claim decisions are in:
- `results/sota_audit_2026_03/claim_crosswalk.json`
- `results/sota_audit_2026_03/scorecard.md`

## Where ModelCypher Is Clearly Ahead

### 1) Measurement validity + sigma commensurability
Internal status:
- Shared-sigma non-degenerate interval calibration is validated across LFM2-350M, LFM2-700M, Qwen3.5-0.8B.
- Gap-heuristic sigma pathologies were corrected via feasibility-interval calibration.

External gap:
- No retained 2024-2026 source provided a complete all-layer non-degeneracy interval method with explicit empty-interval failure semantics.

### 2) DPI violation mechanism with residual bypass telemetry
Internal status:
- P6 (`DPI holds at fixed sigma`) is refuted 3/3.
- `rho(|Delta_l|, ||F_l||/||h_l||)` is strong and significant across all three models.

External gap:
- Retained residual-flow papers discuss bottlenecks/routing, but none provided this estimator-specific cross-family DPI-violation quantification.

### 3) Integrated geometric controller scope
Internal status:
- Mission stack integrates derived controls beyond isolated tactics (rank/step/stopping/geometry checks).

External gap:
- External literature is strong on individual components but remains fragmented at system level.

## Where External Work Is Stronger and Should Be Adapted

### 1) CKA bias corrections
Evidence:
- [Correcting Biased CKA](https://arxiv.org/abs/2405.01012)
- [Sparse-sampled CKA estimator](https://arxiv.org/abs/2502.15104)

Action:
- Integrate debiased/sampling-aware CKA in [`src/modelcypher/core/domain/geometry/cka.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/core/domain/geometry/cka.py).

### 2) LoRA convergence/failure diagnostics
Evidence:
- [LoRA convergence theorem (ICML 2025)](https://proceedings.mlr.press/v267/kim25n.html)

Action:
- Add regime-aware diagnostics in [`src/modelcypher/core/domain/training/geometric_lora.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/core/domain/training/geometric_lora.py).

### 3) Mission-closure eval methodology
Evidence:
- [LiveBench](https://arxiv.org/abs/2406.19314)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [Kernel Divergence Score](https://arxiv.org/abs/2502.00678)
- [Benchmark saturation study](https://arxiv.org/abs/2602.16763)

Action:
- Upgrade non-ceiling 8B closure protocol in [`scripts/g5_build_non_ceiling_eval_set.py`](/Users/jasonkempf/ModelCypher/scripts/g5_build_non_ceiling_eval_set.py).

## Where To Push Frontier

### 1) Signed DPI direction mechanism
- Current mechanism explains `|Delta|`, not `sign(Delta)`.
- Needed: derivation + intervention evidence bundle.

### 2) Quantization frontier law
- Current status: architecture-dependent CKA floor and crossing severity observed.
- Needed: explicit causal equation tested across 4/8-bit and architecture terms.

### 3) Unused-subspace residual causality
- Current status: empirical and promising but not causal-closed.
- Needed: intervention protocol and falsifier-driven conclusion.

### 4) Entropy operator and architecture dependence
- Current status: **[EMPIRICAL]** on 4 models (LFM2-350M/700M, Qwen3.5-0.8B, Qwen2.5-3B). H_logit is the primary operator (r=0.867 on Qwen2.5-3B vs H_attn r=-0.062). F1 PASS 4/4, F3 PASS. F5 CONSISTENT_SIGN (threshold DERIVED: Fisher-SE MDE + Bretherton autocorrelation correction). 2/4 models resolvable (LFM2-350M, Qwen3.5-0.8B), both negative sign. Mechanism prediction 4/4 (Qwen3.5-0.8B coverage raised to 100% via identity-core decomposition on non-full-attention layers). Cross-scale validated within LFM2 family.
- F5 depth confound identified: raw sign inconsistency explained as depth confound. After depth control with derived detection floor, sign is consistently negative among resolvable models (cross-family: hybrid + standard transformer).
- Promotion to [VALIDATED] was premature (reverted 2026-03-04): only 2/4 models are currently resolvable despite derived threshold closure.
- Open: add models for stronger cross-family evidence; resolve Qwen2.5-3B high autocorrelation (ρ₁=0.905, n_eff=4); derive architecture term for component-sign split.
- Claim record: `CR-EC-001` (`[EMPIRICAL]`, `PUSH_FURTHER`).

## Threads To Retire

- “Rényi MI must decay with layer distance.”
- “ID-defined highway labels imply bypass-dominance behavior.”
- Any deterministic-layer Shannon-MI depth narrative without commensurability proof.

## Contradiction Handling

All contradictions were tagged to one of:
- `measurement_invalidity`
- `mechanism_underspecification`
- `direct_refutation`

No contradictory row was promoted to a stronger claim state without satisfying the first-principles protocol.

## High-Impact Manual Verification

Mission-impacting rows were manually verified from scraped source text (not extraction summaries only), specifically for:
- 8B evaluation methodology
- contamination controls
- benchmark saturation evidence
- quantization-method comparators

See `manual_source_verified=true` and `source_verification_mode="scrape_text_manual"` in `claim_crosswalk.json`.

## Data Contracts Used in This Audit

```json
{
  "SourceCard": {
    "source_id": "string",
    "canonical_url": "string",
    "external_tier": "T0|T1|T2|T3|T4",
    "track_tags": ["string"],
    "key_claim": "string"
  },
  "ClaimRecord": {
    "claim_id": "string",
    "statement": "string",
    "current_status": "taxonomy label",
    "operator": "string",
    "architecture_terms": "string",
    "scale_terms": "string",
    "falsifier": "string",
    "latest_results_path": "path"
  },
  "ClaimCrosswalkRecord": {
    "claim_id": "string",
    "best_external_tier": "T0|T1|T2|T3|T4",
    "classification": "CUTTING_EDGE|ADAPT_OTHERS|PUSH_FURTHER|DEPRIORITIZE",
    "contradiction_tag": "measurement_invalidity|mechanism_underspecification|direct_refutation|null"
  },
  "ActionItem": {
    "action_id": "string",
    "claim_ids": ["string"],
    "bucket": "adapt_now|validate_now|push_frontier|stop_doing",
    "target_path": "string|null"
  }
}
```

## Primary Artifacts

- `results/sota_audit_2026_03/source_registry.json`
- `results/sota_audit_2026_03/source_cards.json`
- `results/sota_audit_2026_03/internal_claim_registry.json`
- `results/sota_audit_2026_03/claim_crosswalk.json`
- `results/sota_audit_2026_03/scorecard.md`
- `results/sota_audit_2026_03/action_map.md`
