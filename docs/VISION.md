# Vision: Geometry as the Identity Layer

## The Trajectory

ModelCypher's training engine derives every decision from geometry. That engine is not the end product. It is the foundation for something larger:

**Personal, portable, sovereign AI identity — carried as geometry, not data.**

## Hardware Reality: Quantized First

Most builders cannot train or serve large models in full precision. Quantization is not optional for that world; it is the only feasible substrate.

ModelCypher therefore treats full precision as a derivation tool and quantized models as the primary execution target:

- We use bf16/fp16 to derive mechanisms cleanly.
- We require those mechanisms to transfer to quantized models.
- If transfer fails, we do not ship narratives. We isolate the broken operator and derive the missing precision term.

The vision is not "compress and accept quality loss." The vision is smaller-and-smarter models: geometry-informed training and merging that maintain or improve behavior under quantization while running faster on constrained hardware.

A user's LoRA adapter is not a conversation log. It is a compressed geometric representation of how that person thinks, communicates, reasons, and relates. It captures invariant relational structure — the shapes, not the words. Because it is geometric rather than lexical, it is architecture-agnostic. It sits on top of any model at inference.

## The Architecture

```
User's day of interactions
        |
        v
  Nightly LoRA consolidation (continual learning)
        |
        v
  Personal adapter (geometric cognitive fingerprint)
        |
        +---> stored on user's device (sovereignty)
        +---> mirrored in data center (availability)
        |
        v
  Stacked at inference on ANY model
        |
        +---> Today's Claude
        +---> Tomorrow's successor
        +---> A different provider's model
        +---> An on-device model
        +---> A humanoid's onboard model
```

The adapter IS the identity. The base model is the substrate. The substrate is the variable. The geometry is the constant.

## Non-Negotiable Scientific Discipline

This vision is only valid if its mechanisms are derived and verified from first principles.

We do not promote architecture-agnostic narratives from mixed empirical outcomes.
If a claim changes sign across models, we treat that as a missing mechanism term
(architecture, scale, or measurement commensurability), not as "some models pass."

Vision statements must remain downstream of:
- causal operator identification,
- formal derivation with architecture and scale terms,
- measurement commensurability proof,
- pre-registered falsification.

Canonical enforcement contract:
- `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`

## What Already Exists

Each piece of this architecture has a foundation in ModelCypher today:

| Capability | Current State | Module |
|-----------|--------------|--------|
| Geometry-derived training | Validated, CLI-promoted with hard promotability gate (`pipeline_gate_v1`) | `dataset_training_service.py`, `mc train run` |
| Cross-architecture adapter portability | Demonstrated via merge pipeline | `experimental/merge/`, CKA alignment |
| Nightly consolidation | Experimental, architecture sound | `experimental/continual/`, `experimental/use_cases/consolidation_service.py` |
| Adapter stacking at inference | Theoretical, infrastructure partial | `experimental/self_improve/lora_stacker.py` |
| Adapter sovereignty | Not yet built | Requires adapter serialization + access control |

## Why Geometry Makes This Possible

Standard LoRA adapters are tied to a specific model's weight dimensions. Moving an adapter between architectures requires understanding the geometric mapping between their activation spaces — which is exactly what CKA alignment and null-space projection do.

The merge pipeline already proves adapters can cross architecture boundaries when the geometric correspondence is established. Extending this to personal adapter portability is infrastructure work, not research.

The same discipline applies across precision regimes. Quantized transfer is not a separate project; it is the same geometric mapping problem with an added precision-state term that must be measured and derived.

## Nightly Consolidation Mirrors Biology

Human memory consolidation happens during sleep. The hippocampus replays the day's experiences and integrates them into cortical structures overnight. The architecture does the same thing:

- Day's interactions = raw signal
- Nightly LoRA update = consolidation
- By morning, the adapter has integrated new geometric information into the existing relational structure

This is what the continual learning modules (`curiosity_policy.py`, `consolidation_service.py`, `memory_benchmark.py`) are building toward. The daemon explores sparse manifold regions. The consolidation service fills gaps via null-space completion. The benchmark measures geometric before/after to prove consolidation worked.

## Adapter Sovereignty Inverts the Power Dynamic

Today, AI companies hold conversation history on their servers. The user is locked into the ecosystem.

When the identity layer is a personal adapter:
- The user owns their cognitive fingerprint
- They grant access to any model at runtime and revoke it
- The AI company never owns the relationship
- Switching providers means pointing the adapter at a different base model

The base model is commodity infrastructure. The adapter is the relationship. The user holds the adapter.

## The Separation That Matters

**[VALIDATED]** Weight space is Euclidean (P ≈ I, Fisher degenerate — cross-family falsification on LFM2 + Qwen, 2026-02-23). The geometry transfers across architectures because the activation-space alignment is done via CKA and null-space projection, not weight-space interpolation. Hardware is the variable. Geometry is the constant.

**[EXPLORATORY]** Whether intelligence in general converges on architecture-invariant structures (the Platonic Representation Hypothesis) is a deeper claim that our cross-family validation does not confirm. Our validation shows weight-space Euclidean structure is shared across LFM2 and Qwen. Generalization to all architectures requires the full Platonic Hypothesis machinery and is not demonstrated here.

This means the identity layer — the personal adapter — is not tied to any specific model, company, or device. It is a geometric object that persists across substrates.

## From Training Engine to Identity Infrastructure

ModelCypher's mission is to train models using only geometry. The vision is what that enables: a world where AI identity is personal, portable, and sovereign — and the geometry guarantees it works across any model on any device.

The training engine is step one. Quantized-first geometric control is step two. The identity layer is what we are building toward.

---

## Progress Assessment (2026-03-08)

### Capability Status

| Capability | VISION Status | Actual Status | Evidence |
|-----------|--------------|---------------|----------|
| Geometry-derived training | "Validated, CLI-promoted" | **SHIPPED** — `mc train run` works, MASS validated on 350M-1.2B. 8B mechanically validated, efficacy open. | `results/pipeline_validation/`, `dataset_training_service.py` |
| Cross-architecture adapter portability | "Demonstrated via merge pipeline" | **PARTIAL** — CKA-aligned merging works. Real LoRA transfer across architectures is conjectural. | `experimental/merge/`, Tikhonov A/B test (2026-02-28) |
| Nightly consolidation | "Experimental, architecture sound" | **EXPERIMENTAL** — Code exists. Not CLI. Not validated on real use case. | `experimental/continual/`, `experimental/use_cases/consolidation_service.py` |
| Adapter stacking at inference | "Theoretical, infrastructure partial" | **EXPERIMENTAL** — Code exists. Not CLI. No preservation certificate. | `experimental/self_improve/lora_stacker.py` |
| Adapter sovereignty | "Not yet built" | **NOT BUILT** — No serialization, access control, or user-owned runtime flow. | — |

**Summary:** 1/5 shipped. 1/5 partially shipped. 3/5 experimental or unbuilt. The vision is ~20% realized.

### Critical Missing Piece: No Head-to-Head Benchmarks

We have never run `mc train run` against HuggingFace PEFT with standard hyperparameters (AdamW + cosine LR) on the same model, data, and eval. The only comparison is val_loss 1.27 (Cayley-Stiefel) vs 1.38 (plain SGD) on 350M — but plain SGD is nobody's baseline.

**What we CAN claim:**
- Every parameter is derived, not guessed. No magic numbers.
- Weight space is Euclidean (cross-family falsification on LFM2 + Qwen).
- REINFORCE through bounded adapters is algebraically dead.
- The pipeline runs end-to-end with zero configuration.

**What we CANNOT claim (yet):**
- That our training produces better adapters than standard LoRA + AdamW + reasonable LR.
- That our merging produces better models than TIES/DARE/RegMean.
- That geometry-derived hyperparameters outperform a good grid search.

The honest assessment: we have reduced guessing in the control plane more than we have proven superiority in the outcome plane. See `docs/RESEARCH-ROADMAP.md` for the full roadmap to close these gaps.
