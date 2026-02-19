# ModelCypher TODO

**Updated:** 2026-02-13

---

## Code Projects

| Project | Location | Purpose |
|---------|----------|---------|
| **ModelCypher** | `/` | Main geometric analysis toolkit for neural networks |
| **Plasma** | `/plasma/` | Tokamak plasma geometry analysis (fusion research application) |

---

## Code Tasks

### ModelCypher Core

*No open code tasks. All tracked items completed.*

**Recently completed (2026-02-13):**
- Training domain test coverage: 15 new test files covering all previously-untested modules (257 training domain tests total)
- Fixed 3 bugs in `training_notifications.py` found by new tests (missing logger, wrong class references)

### Plasma Subproject

- [ ] Complete TODO/FIXME items in `plasma/src/diiid_loader.py`
- [ ] Complete TODO/FIXME items in `plasma/src/data_loader.py`

---

## Implementation Backlog

These are integration-ready techniques from research that could become CLI features:

| Technique | Origin | Status |
|-----------|--------|--------|
| Concepts as Geometric Clusters | Research design doc | Design ready |
| Counterfactual Sensitivity | Archived code | Code exists in archive |
| Generation-Based Evaluation | Archived code | Code exists in archive |
| LoRA Isometry Ratio | Research design doc | Design ready |
| Geodesic Merge Quality | Research design doc | Design ready |

---

## Research

**All research tracking consolidated in: `docs/RESEARCH-ROADMAP.md`**

Quick links:
- Open mathematical questions: `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md`
- LFM2-350M project: `docs/LFM2-350M-WORK-SUMMARY.md`
- Failure modes: `docs/research/FAILURE-MODES.md`

---

*Source files: docs/research/*.md, plasma/src/*
