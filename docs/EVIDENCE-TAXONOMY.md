# Evidence Taxonomy

Standard for tagging claims across all ModelCypher documentation.

---

## The 9-Label System

| Label | Meaning | Required Evidence |
|-------|---------|-------------------|
| `[PROVEN]` | Theorem-level result. Mathematical proof with assumptions stated, or standard published theorem with citation. | Proof or peer-reviewed citation |
| `[VALIDATED-ENG]` | Engineering/mechanics validation. Code path, memory behavior, artifact shape, or mechanical invariant worked under the declared environment. | Direct run artifact, command, or test; one successful run is acceptable if the claim is only engineering/mechanical |
| `[VALIDATED-EFF]` | Benchmark efficacy validation. The method improved or preserved the declared task metric under pre-registered criteria. | At least 3 seeds, seed count reported, pooled effect outside 2*SE, retained per-seed artifacts, and aggregate verdict committed |
| `[EMPIRICAL]` | Measured and reproducible, but not subjected to formal falsification testing. Single-model observations, correlations without controls, results not yet tested for generalization. | Measurement data, reproduction script |
| `[EXPLORATORY]` | Early-stage observation or hypothesis where the claim form is incomplete (missing mechanism terms, operator validity, or falsifier). | Raw observation plus explicit missing-field declaration |
| `[CONJECTURAL]` | Theoretically motivated hypothesis with insufficient evidence. Untested predictions, proposed mechanisms, claims awaiting measurement. | Theoretical motivation stated |
| `[MEASUREMENT_INVALID]` | Claim cannot be evaluated in the stated scope because the measurement operator is not commensurable or is in a degenerate regime. | Operator validity analysis + degeneration evidence |
| `[MECHANISM_UNDERSPECIFIED]` | Claim lacks required causal terms (typically architecture and/or scale), so mixed outcomes are expected and not interpretable as validation. | Explicit missing-term diagnosis + revised claim form |
| `[DISPROVEN]` | Tested and rejected. Pre-registered rejection condition met, or null-hypothesis testing showed no effect. | Falsification data |

---

## Promotion and Demotion

- `[EXPLORATORY]` -> `[CONJECTURAL]`: Claim form completed with mechanism terms and falsifier.
- `[EXPLORATORY]` -> `[EMPIRICAL]`: Reproducible measurements exist and measurement operator is declared.
- `[CONJECTURAL]` -> `[EMPIRICAL]`: Reproducible measurements exist.
- `[EMPIRICAL]` -> `[VALIDATED-ENG]`: A code, memory, artifact, or mechanics claim survived its declared engineering check.
- `[EMPIRICAL]` -> `[VALIDATED-EFF]`: A benchmark-efficacy claim survived pre-registered testing across at least 3 seeds, with pooled effect outside 2*SE and the seed count stated.
- `[VALIDATED-ENG]` -> `[VALIDATED-EFF]`: A mechanical result is later shown to produce benchmark efficacy under the efficacy rule above.
- `[VALIDATED-ENG]` or `[VALIDATED-EFF]` -> `[PROVEN]`: Mathematical proof exists (not just strong empirical evidence).
- `[EMPIRICAL]` -> `[MEASUREMENT_INVALID]`: Measurement operator shown non-commensurable or degenerate in declared comparison scope.
- `[EMPIRICAL]` -> `[MECHANISM_UNDERSPECIFIED]`: Mixed outcomes appear and required architecture/scale terms were not pre-registered.
- `[MEASUREMENT_INVALID]` -> `[EMPIRICAL]`: Operator repaired, commensurability proven, measurements reproduced.
- `[MECHANISM_UNDERSPECIFIED]` -> `[CONJECTURAL]`: Claim rewritten with explicit mechanism terms and awaits new tests.
- Any label -> `[DISPROVEN]`: Rejection condition met. Immediate. No silent reinterpretation.
- `[VALIDATED-ENG]` or `[VALIDATED-EFF]` -> `[EMPIRICAL]`: Claim fails on a new model or setting. Add scope qualifier.

Efficacy claims must carry an explicit seed count, even when the label is lower
than `[VALIDATED-EFF]`. A single-seed efficacy result is `[EMPIRICAL]` at most.
Data-rank ceiling and gradient-accumulation checks are `[VALIDATED-ENG]` unless
and until a separate efficacy claim clears the 3-seed pooled-effect rule.

---

## Tagging Format

### Section-Level

When all claims in a section share the same status:

```markdown
## Section Title [LABEL]

<!-- evidence: LABEL | scope: model(s) | date: YYYY-MM-DD -->
```

### Inline

For individual claims within a mixed section:

```markdown
CKA = 1.0 after Procrustes on training probes, by construction. [PROVEN]
```

With annotation:

```markdown
Expansion ratio correlates with reasoning quality. [EMPIRICAL: LFM2-350M only, r=0.38]
```

### Disproven Claims

Strikethrough the claim text, then tag:

```markdown
~~expansion/compression ratio = phi~~ [DISPROVEN: 2026-02-01, phi ranks 202/1014 among tested constants]
```

### Machine-Parseable Regex

```
\[(PROVEN|VALIDATED-ENG|VALIDATED-EFF|EMPIRICAL|EXPLORATORY|CONJECTURAL|MEASUREMENT_INVALID|MECHANISM_UNDERSPECIFIED|DISPROVEN)(?::([^\]]+))?\]
```

---

## Contradiction Resolution

When a disproven claim reappears as truth elsewhere:

1. **Strikethrough + tag** for claims once presented as findings. Preserves original text for context.
2. **Replace formula** when a disproven constant appears in an equation. Remove the constant, tag the corrected version.
3. **Add scope qualifier** for claims true in some contexts but stated as universal.
4. **Archival banner** for entire documents or sections that are wholly superseded.

Archival banner format:

```markdown
> **ARCHIVAL NOTE [YYYY-MM-DD]:** The findings below were subsequently [DISPROVEN].
> See `relevant-file.md` for the falsification record.
> Preserved as historical record only. Do not cite as current findings.
```

---

## Migration from Prior Labels

| Old Label | New Label | Notes |
|-----------|-----------|-------|
| `[VALIDATED]` | `[VALIDATED-ENG]` or `[VALIDATED-EFF]` | Legacy umbrella label. Choose ENG for code/memory/mechanics and EFF only for benchmark efficacy with seed count and pooled effect outside 2*SE. |
| `SUPPORTED` | `[VALIDATED-ENG]` or `[VALIDATED-EFF]` | Choose by claim type; do not preserve the old umbrella semantics. |
| `OPEN` | `[CONJECTURAL]` | More descriptive |
| `CONJECTURE` | `[CONJECTURAL]` | Normalize spelling |
| `FALSIFIED` | `[DISPROVEN]` | Parallel construction |
| `REFUTED` | `[DISPROVEN]` | Synonym; use canonical label |
| `CLOSED` | Remove; replace with evidence label + date note in prose | Closure is workflow state, not evidence state |
| `MEASURED` | `[EMPIRICAL]` | Canonical evidence term |
| `SUPERSEDED` | Keep in prose note, not evidence tag | Supersession is versioning metadata |
| `PROVEN` | `[PROVEN]` | Unchanged |
| `PAREIDOLIA` | `[DISPROVEN]` | Subsumed |
| `BREAKTHROUGH` | Remove label; tag with actual evidence level | Usually `[EMPIRICAL]` or `[CONJECTURAL]` |
