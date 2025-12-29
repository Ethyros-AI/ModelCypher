# arXiv Submission Plan

**Created**: 2025-12-29
**Goal**: Submit ModelCypher paper series to arXiv this week

---

## Submission Priority Order

### Tier 1: Ready Now (Submit First)

| Paper | Status | Action Required |
|-------|--------|-----------------|
| **Paper 5** (Semantic Highway) | ✅ READY | Convert to LaTeX, submit |
| **Paper 1** (Invariant Structure) | ✅ READY | Already updated with corrected findings, submit |

**Rationale**: These papers have real data, proper hedging, and accurately represent findings.

---

### Tier 2: Ready After Minor Edits (Submit This Week)

| Paper | Status | Action Required |
|-------|--------|-----------------|
| **Paper 0** (Framework) | ✅ READY | No changes needed - already has correct CKA numbers |
| **Paper 4** (Toolkit) | ✅ FIXED | Claims updated to reflect actual findings |

**Changes Made**:
- Paper 4: Updated abstract and introduction to remove specific CKA claims that contradicted empirical findings

---

### Tier 3: Methodology Papers (Submit After Validation)

| Paper | Status | Action Required |
|-------|--------|-----------------|
| **Paper 2** (Entropy Safety) | ⚠️ MARKED | Now clearly labeled as "preliminary results" |
| **Paper 3** (Cross-Architecture) | ⚠️ MARKED | Now clearly labeled as "preliminary results" |

**Changes Made**:
- Added `Status: Methodological framework with preliminary results` header
- Changed "Results" → "Preliminary Results"
- Added notes about ongoing validation
- Updated conclusions with validation status

**These can be submitted as methodology papers** with the current preliminary results, OR wait for full validation.

---

## Key Corrections Made Today

### 1. CKA Numbers (Paper 4)

**Original claim**: "CKA = 0.82 for primes vs 0.54 for controls"

**Actual finding**: Primes CKA = 0.92, Random words CKA = 0.94

**Action**: Removed specific numbers, now states "CKA > 0.9 for both semantic primes and random word sets"

### 2. Paper 2 & 3 Honesty

Both papers presented results as confirmed when appendices indicated data was pending.

**Action**: Added clear "Status" headers, marked all results as "Preliminary", added validation status to conclusions.

---

## Experiments Needed for Full Validation

### Paper 2 (Entropy Safety)

```bash
# Full modifier × temperature sweep
for model in qwen2.5-3b llama-3.2-3b mistral-7b tinyllama; do
  for temp in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5; do
    mc entropy sweep --model $model --temperature $temp --output data/paper2/
  done
done

# Safety AUROC (requires curated prompt suite)
mc entropy safety-auroc --base-model qwen2.5-3b-base --tuned-model qwen2.5-3b-instruct \
  --suite prompts/harmful_benign.json
```

**Blocking**: Harmful/benign prompt suite requires human curation.

### Paper 3 (Cross-Architecture Transfer)

```bash
# Compatibility assessment
mc geometry interference predict --source qwen2.5-7b --target llama-3.2-8b

# Full HumanEval evaluation
mc eval suite --model merged --suite humaneval-50.json

# Safety evaluation
mc eval safety --model merged --suite safety-100.json
```

**Blocking**: HumanEval subset and safety prompts need curation.

---

## arXiv Formatting Checklist

For each paper before submission:

- [ ] Convert markdown to LaTeX
- [ ] Format citations properly (BibTeX)
- [ ] Add proper author/affiliation metadata
- [ ] Verify all figure/table references
- [ ] Remove relative markdown links (convert to proper citations)
- [ ] Add arXiv identifier field (after first submission)
- [ ] Choose appropriate arXiv categories (cs.LG, cs.CL, cs.AI)

---

## Recommended Categories

| Paper | Primary | Secondary |
|-------|---------|-----------|
| Paper 0 | cs.LG | cs.AI |
| Paper 1 | cs.CL | cs.LG |
| Paper 2 | cs.LG | cs.CL |
| Paper 3 | cs.LG | cs.CL |
| Paper 4 | cs.LG | cs.SE |
| Paper 5 | cs.LG | cs.CL |

---

## Submission Timeline

| Day | Action |
|-----|--------|
| Day 1 | Submit Paper 5 (Semantic Highway) - strongest empirical backing |
| Day 1 | Submit Paper 1 (Invariant Structure) - already corrected |
| Day 2 | Submit Paper 0 (Framework) - theoretical foundation |
| Day 2 | Submit Paper 4 (Toolkit) - systems paper |
| Day 3-5 | Papers 2 & 3 - decide whether to submit as methodology or wait for validation |

---

## Scientific Integrity Summary

All papers now accurately represent their validation status:

1. **Paper 0**: Framework/theory - no empirical claims beyond citing other papers
2. **Paper 1**: Real data from 6 models, correctly reports that primes ≈ random words
3. **Paper 2**: Clearly marked as preliminary; methodology is complete, validation ongoing
4. **Paper 3**: Clearly marked as preliminary; methodology is complete, validation ongoing
5. **Paper 4**: Updated to remove incorrect claims, now states correct findings
6. **Paper 5**: Real data from 3 models, properly hedged as "preliminary observation"

**NEGATIVE-RESULTS.md** is preserved and documents the key finding that semantic primes are NOT more geometrically special than random words - strengthening the universal invariance claim.
