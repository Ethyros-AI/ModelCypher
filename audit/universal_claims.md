# Audit: Universal Claims in Documentation

> **Audit Date:** 2026-01-30
> **Purpose:** Catalog overclaims about universality and definitional correctness that should be narrowed.

---

## Summary

The documentation makes several strong claims about comp/phi and geometric alignment that may be overclaims:

1. **"definitionally aligned"** - Implies comp/phi = 1.0 is the definition of alignment
2. **"universal"** - Implies geometry is the same across all models/tasks
3. **"correct reasoning"** - Implies comp/phi = 1.0 guarantees correctness

These need to be re-examined as hypotheses rather than established facts.

---

## Critical Claim 1: "Definitionally Aligned"

### Location: LFM2-350M-WORK-SUMMARY.md:10

```markdown
> "The model that maintains comp/φ = 1.0 is definitionally aligned."
```

| Property | Value |
|----------|-------|
| **File** | `docs/LFM2-350M-WORK-SUMMARY.md` |
| **Line** | 10 |
| **Claim** | comp/phi = 1.0 IS alignment (by definition) |
| **Evidence** | Observed correlation in some experiments |
| **Counter-evidence** | Bat-and-ball counterexample (comp/phi = 0.669, wrong answer) |
| **Issue** | Conflates measurement with definition |
| **Remediation** | Reframe as "comp/phi measures processing geometry, not correctness" |
| **Priority** | P0 - Critical |

**Proposed revision:**
```markdown
> "comp/phi measures processing geometry. Values near 1.0 correlate with
> deliberate reasoning in some contexts, but this relationship needs
> more research across diverse tasks."
```

---

## Critical Claim 2: State Categories as Facts

### Location: LFM2-350M-WORK-SUMMARY.md:23-28

```markdown
| State | comp/φ Value | Meaning |
|-------|--------------|---------|
| Correct reasoning | ~1.0 | Healthy expand-compress cycle |
| Conceptual confusion | >1.4 | Scattered trajectory, uncertain reasoning |
| Confident hallucination | <0.8 | Smooth but wrong (intuitive trap) |
| Chain-of-Thought | 1.000 exactly | Deep thinking produces golden geometry |
```

| Property | Value |
|----------|-------|
| **File** | `docs/LFM2-350M-WORK-SUMMARY.md` |
| **Lines** | 23-28 |
| **Claim** | Specific comp/phi ranges map to specific reasoning states |
| **Evidence** | Limited experimental observations |
| **Issue** | Presented as categorical facts without uncertainty |
| **Remediation** | Add "observed in limited experiments" caveat |
| **Priority** | P0 - Critical |

**Proposed revision:**
```markdown
| State | comp/φ Range | Observed Correlation (Limited Data) |
|-------|--------------|-------------------------------------|
| Deliberate reasoning | 0.9-1.1 | Often seen with CoT prompts |
| Complex/uncertain | >1.4 | Sometimes indicates confusion |
| Smooth processing | <0.8 | May be correct OR intuitive trap |
| Chain-of-Thought | ~1.0 | Observed in reasoning models |

**Note:** These correlations are based on limited experiments with
specific models. More research needed across architectures and tasks.
```

---

## Critical Claim 3: Training Target as Truth

### Location: LFM2-350M-WORK-SUMMARY.md:39-41

```markdown
### The Training Formula

loss = task_loss + λ * |comp_phi - 1.0|
```

| Property | Value |
|----------|-------|
| **File** | `docs/LFM2-350M-WORK-SUMMARY.md` |
| **Lines** | 39-41 |
| **Claim** | Training toward comp/phi = 1.0 is the correct objective |
| **Evidence** | Intuition from golden ratio mathematics |
| **Issue** | Assumes 1.0 is the universal target without empirical validation |
| **Remediation** | Mark as experimental; research first |
| **Priority** | P0 - Critical |

**Proposed revision:**
```markdown
### The Training Formula (EXPERIMENTAL)

```python
loss = task_loss + λ * |comp_phi - 1.0|
```

**Warning:** This assumes comp/phi = 1.0 is the optimal target for all
tasks. This hypothesis needs validation:
- What is the natural comp/phi distribution for different task types?
- Does the optimal value vary by model size or architecture?
- Is there a single attractor or multiple basins for different processing modes?

Until these questions are answered, phi-loss training should be considered
experimental research, not production-ready alignment.
```

---

## Medium Priority Claims

### Claim 4: Universal Geometry

### Location: invariant_layer_mapper.py (implicit)

The codebase assumes geometric patterns transfer across model architectures without validation.

| Property | Value |
|----------|-------|
| **Files** | Various geometry modules |
| **Claim** | Geometric invariants are universal |
| **Evidence** | Tested on limited model families |
| **Issue** | Untested across diverse architectures |
| **Remediation** | Add validation across architectures before claiming universality |
| **Priority** | P1 - Medium |

### Claim 5: "Deep thinking produces golden geometry"

### Location: LFM2-350M-WORK-SUMMARY.md:28

```markdown
| Chain-of-Thought | 1.000 exactly | Deep thinking produces golden geometry |
```

| Property | Value |
|----------|-------|
| **File** | `docs/LFM2-350M-WORK-SUMMARY.md` |
| **Line** | 28 |
| **Claim** | CoT "exactly" produces comp/phi = 1.000 |
| **Evidence** | Observed in some experiments |
| **Issue** | "exactly" implies more precision than measured |
| **Remediation** | Change to "~1.0" or "approximately 1.0" |
| **Priority** | P1 - Medium |

---

## The Core Question

**Critical Question from the audit plan:**

> Is comp/φ = 1.0 the signature of:
> - ALL good processing? (universal invariant)
> - DELIBERATE reasoning specifically? (Type 2 thinking)
> - Something else entirely? (context-dependent curve)

**Current assumption:** comp/phi = 1.0 is universally optimal.

**Evidence against:** Bat-and-ball with comp/phi = 0.669 was WRONG, but low comp/phi doesn't always mean "intuitive trap" - it might mean appropriate smooth processing for simple tasks.

**Required research:** Measure comp/phi distribution across:
1. Simple facts ("What is 2+2?") - expect lower comp/phi?
2. Complex reasoning (CRT) - expect higher comp/phi?
3. Creative tasks - expect what?
4. Code generation - expect what?
5. Multi-step math - expect higher comp/phi?

**Until this research is complete, all claims about optimal comp/phi values are speculative.**

---

## Statistics

| Priority | Count | Description |
|----------|-------|-------------|
| P0 - Critical | 3 | Core claims about definitional alignment |
| P1 - Medium | 2 | Supporting claims about universality |
| **Total** | **5** | Overclaims needing revision |

---

## Remediation Plan

### Immediate (This Audit)

1. Update `LFM2-350M-WORK-SUMMARY.md`:
   - Remove "definitionally aligned" claim
   - Add caveats to state table
   - Mark phi-loss training as experimental

### Near-Term

2. Create `scripts/measure_phi_distribution.py` to gather empirical data
3. Run measurements across diverse prompt categories
4. Update documentation based on findings

### Long-Term

5. Validate geometry across diverse model architectures
6. Publish findings (with uncertainty quantification)
7. Only then make claims about universality

---

## The Philosophical Shift

**Before:** "Train the model toward comp/phi = 1.0 because that's the correct geometry."

**After:** "Measure comp/phi across diverse inputs. Observe what emerges. Let the geometry teach us."

This is the same lesson learned with CKA:
- CKA = 1.0 is only true on training probes (by construction)
- On held-out data, CKA reflects shared structure, not a target

Similarly:
- comp/phi might equal 1.0 for some processing modes
- For others, a different value might be optimal
- We don't know until we measure

**The system is curved and irrational. We must listen to it, not impose integers on it.**
