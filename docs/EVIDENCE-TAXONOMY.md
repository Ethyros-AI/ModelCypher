# Evidence Taxonomy

Standard for tagging claims across all ModelCypher documentation.

---

## The 5-Label System

| Label | Meaning | Required Evidence |
|-------|---------|-------------------|
| `[PROVEN]` | Theorem-level result. Mathematical proof with assumptions stated, or standard published theorem with citation. | Proof or peer-reviewed citation |
| `[VALIDATED]` | Null-hypothesis tested, reproduced across multiple settings or models. Falsification was attempted and failed. | p-values, CIs, multi-model/multi-seed reproduction, pre-registered criteria met |
| `[EMPIRICAL]` | Measured and reproducible, but not subjected to formal falsification testing. Single-model observations, correlations without controls, results not yet tested for generalization. | Measurement data, reproduction script |
| `[CONJECTURAL]` | Theoretically motivated hypothesis with insufficient evidence. Untested predictions, proposed mechanisms, claims awaiting measurement. | Theoretical motivation stated |
| `[DISPROVEN]` | Tested and rejected. Pre-registered rejection condition met, or null-hypothesis testing showed no effect. | Falsification data |

---

## Promotion and Demotion

- `[CONJECTURAL]` -> `[EMPIRICAL]`: Reproducible measurements exist.
- `[EMPIRICAL]` -> `[VALIDATED]`: Falsification attempted (null-hypothesis testing, multi-model audit, pre-registered pass/fail) and claim survived.
- `[VALIDATED]` -> `[PROVEN]`: Mathematical proof exists (not just strong empirical evidence).
- Any label -> `[DISPROVEN]`: Rejection condition met. Immediate. No silent reinterpretation.
- `[VALIDATED]` -> `[EMPIRICAL]`: Claim fails on a new model or setting. Add scope qualifier.

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
\[(PROVEN|VALIDATED|EMPIRICAL|CONJECTURAL|DISPROVEN)(?::([^\]]+))?\]
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
| `SUPPORTED` | `[VALIDATED]` | Equivalent semantics |
| `OPEN` | `[CONJECTURAL]` | More descriptive |
| `FALSIFIED` | `[DISPROVEN]` | Parallel construction |
| `PROVEN` | `[PROVEN]` | Unchanged |
| `PAREIDOLIA` | `[DISPROVEN]` | Subsumed |
| `BREAKTHROUGH` | Remove label; tag with actual evidence level | Usually `[EMPIRICAL]` or `[CONJECTURAL]` |
