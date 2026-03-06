# Paper Rejection: Early Entrenchment (arXiv:2603.00359)

**Paper:** How Large Language Models Get Stuck: Early Structure with Persistent Errors
**Authors:** Manna, Snyder, Tabor (2026)
**Reviewed:** 2026-03-06
**Verdict:** REJECT — fails First Principles Review Protocol §2 (incomplete claim form)

---

## Claim Summary

The paper observes that ~1/3 of BLiMP grammaticality classes fail persistently
during OPT training on the BabyLM dataset (100M words). When failure occurs, the
model establishes an erroneous likelihood separation early in training and sustains
it. The authors propose a "Bigram Hypothesis" claiming that bigram statistics bias
early learning toward wrong distinctions.

## Protocol Violations

### §2 — Missing all required claim form fields

| Field | Status |
|---|---|
| Causal operator | **MISSING** — "entrenchment" is a behavioral description, not a deterministic map |
| Equation/theorem | **MISSING** — no formal derivation |
| Architecture term | **MISSING** — OPT only, no architecture dependence specified |
| Scale term | **MISSING** — single model size, no scale analysis |
| Precision state | **MISSING** |
| Measurement operator | Partially specified (BLiMP likelihood comparison) |
| Commensurability | **MISSING** — single model, no cross-model comparison |
| Directional prediction | Partially specified (Bigram Hypothesis) |
| Falsifier | **MISSING** — Bigram Hypothesis testing described as "in progress" |

The paper scores 1.5/9 on required claim form fields. This is a hypothesis paper
with incomplete methodology, not a results paper.

### §7.1 — Correlation-only explanation

The observation (early erroneous separation → persistent failure) is purely
correlational. No mechanism is proposed for *why* the separation persists.
The likely geometric mechanism — early token directions locking the trajectory
via contractive Jacobian spectrum (||J|| < 1 → basin of attraction) — is not
mentioned or measured.

## What Is Not Novel or Actionable

- **BLiMP evaluation:** Standard benchmark, well-established.
- **OPT on BabyLM:** Single model, single dataset, no cross-architecture evidence.
- **"In progress" testing:** The Bigram Hypothesis is stated but not tested.
  The testing method is described as future work.

## What Would Make This Worth Revisiting

If someone measures:
1. The Jacobian spectrum at the point of early entrenchment — is it contractive?
2. Whether the entrenched subspace has lower intrinsic dimension than the
   non-entrenched subspace (geometric lock-in vs. behavioral lock-in)
3. Cross-architecture replication (does entrenchment occur at the same BLiMP
   classes in different architectures?)

The phenomenon is likely real. The geometric mechanism (early Jacobian contraction
trapping trajectories in local basins) is plausible. But neither is demonstrated
in this paper.

## Citation

```bibtex
@article{manna2026stuck,
  title={How Large Language Models Get Stuck: Early Structure with Persistent Errors},
  author={Manna, Alokesh and Snyder, William and Tabor, Whitney},
  journal={arXiv preprint arXiv:2603.00359},
  year={2026}
}
```
