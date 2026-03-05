# First-Principles Review Protocol

**Status:** Canonical claim-audit contract
**Effective date:** 2026-03-03
**Applies to:** `docs/`, `AGENTS.md`, `MISSION.md`, `VISION.md`, experiment reports, and promotion decisions

---

## 1. Purpose

This protocol blocks correlation-first narratives from becoming doctrine.

A claim is promotable only if it is derived from causal geometry and survives
falsification under explicit architecture and scale terms.

If a result is "confirmed on one model, refuted on another," that is not a
conclusion. It is either:
1. mechanism underspecification, or
2. measurement invalidity.

---

## 2. Required Claim Form (No Exceptions)

Every promoted claim must be written in this form before experiments run:

```text
observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)
```

Required fields per claim:

1. **Causal operator**: deterministic map that produces the effect.
2. **Equation/theorem**: formal derivation with assumptions.
3. **Architecture term**: explicit dependence on operator type/routing/layer family.
4. **Scale term**: explicit dependence on width/depth/parameter count or derived dimensional terms.
5. **Precision state**: explicit numeric representation and quantization operator parameters.
6. **Measurement operator**: statistic/kernel and domain of validity.
7. **Commensurability proof**: proof measurements are comparable across compared objects.
8. **Directional prediction**: sign/magnitude expectation before seeing data.
9. **Falsifier**: exact observation that invalidates the claim.

Missing any field means the claim is **exploratory only**.

---

## 3. Commensurability Requirements

A measurement is valid for cross-object comparison only if:

1. It is invariant to nuisance transformations that differ across compared objects
   (or those transformations are explicitly controlled).
2. Its calibration parameters are shared, derived from a joint principle, or proven
   to induce equivalent resolution.
3. Its sensitivity floor and saturation regime are characterized.
4. Pairwise comparisons do not mix incompatible scales without a proof of equivalence.

If these conditions are not proven, cross-model conclusions are invalid.

---

## 4. Mixed-Outcome Classification Rule

Given a claim tested on multiple models:

1. **Same sign, same regime, falsifier not triggered** -> candidate for validation.
2. **Sign mismatch across models** -> classify as **mechanism underspecified** unless
   architecture and scale terms predicted the divergence.
3. **Any model in measurement degeneration/saturation regime** -> classify as
   **measurement invalid** for that claim.
4. **Both effects present** -> classify as **measurement invalid + mechanism underspecified**.

Do not use terms like "partially confirmed" unless divergence was explicitly predicted
by the pre-registered architecture/scale terms.

---

## 5. Promotion States

Use only these claim states in roadmap and doctrine docs:

- `[EXPLORATORY]` - mechanism not fully specified.
- `[MEASUREMENT_INVALID]` - operator not commensurable for the comparison.
- `[MECHANISM_UNDERSPECIFIED]` - architecture/scale terms missing or wrong.
- `[VALIDATED]` - all required fields present, falsifier not triggered across declared domain.
- `[DISPROVEN]` - falsifier triggered.

`[EMPIRICAL]` can annotate evidence type, but it does not replace the above state.

---

## 6. Repository-Wide Documentation Review Procedure

For each research-facing document:

1. Extract every promoted claim.
2. Attach claim form fields (Section 2).
3. Downgrade claims missing architecture/scale/commensurability.
4. Replace narrative conclusions with mechanism-level diagnosis.
5. Add explicit next falsification experiment where status is unresolved.

A review pass is complete only when every promoted claim in the document has a
filled claim form.

---

## 7. Non-Negotiable Prohibitions

The following are prohibited in promoted conclusions:

1. Correlation-only explanations.
2. P-value-only promotion without mechanism terms.
3. Cross-model claims without commensurability proof.
4. Heuristic thresholds not derived from machine precision, geometry, or theorem.
5. "Works in practice" as causal justification.

---

## 8. Minimum Artifact Bundle Per Claim

Each promoted claim must link:

1. Derivation note (`.md` with equations and assumptions)
2. Experiment script (exact command and version)
3. Raw results (machine-readable files)
4. Falsification log (pass/fail by model with diagnosis category)

Without this artifact bundle, the claim cannot appear as validated in mission,
vision, roadmap, or agent doctrine.
