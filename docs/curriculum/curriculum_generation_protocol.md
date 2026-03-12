# Curriculum Generation Protocol: Design Specification

**Status:** Research draft — no production code yet.
**Date:** 2026-03-12

---

## Overview

A structured protocol where frontier models (Claude, GPT-4, etc.) generate
training curricula for small models. The user specifies a goal, mc profiles the
student model, the user gives the profile to a frontier model with a structured
prompt, and the frontier model returns a skill DAG + training data that mc can
ingest and execute.

**Key design decisions:**
- Prompt protocol + ingestion (no API integration in mc)
- Frontier generates new DAGs (existing hand-crafted DAG becomes a validated example)
- Research-only until validated

---

## Architecture

```
core/domain/curriculum_protocol/     <- Pure data structures, validation. No ML imports.
    student_profile.py               <- StudentProfile, SkillAssessment, GeometricProfile
    curriculum_spec.py               <- CurriculumSpec, GeneratedSkillNode
    prompt_template.py               <- PROMPT_TEMPLATE constant, build_prompt()
    response_schema.py               <- JSON schema the frontier model must return
    validation.py                    <- validate_curriculum() -> ValidationResult

core/use_cases/
    curriculum_generation_service.py  <- Orchestration: profile, prompt-build, ingest

scripts/                             <- Research scripts (not CLI commands yet)
    generate_student_profile.py
    build_curriculum_prompt.py
    ingest_curriculum.py
```

Hexagonal boundaries preserved: `core/domain/` and `core/use_cases/` have no ML
imports. Profiling uses existing adapters (`curriculum_eval_adapter.py`,
`curriculum_profiler.py`). The frontier model interaction happens entirely
outside mc.

---

## Component 1: Student Profile

mc analyzes the target model and produces a JSON document the frontier model can
read. This is the "student report card" that tells the frontier model what the
student already knows and what its geometric capacity looks like.

### Schema

```json
{
  "schema_version": "mc.student_profile.v1",
  "model_path": "/path/to/model",
  "model_id": "<content hash>",

  "geometric_profile": {
    "architecture": "LFM2ForCausalLM",
    "model_family": "lfm2",
    "parameter_count": 350000000,
    "hidden_dim": 1024,
    "num_layers": 16,
    "vocab_size": 32000,
    "context_length": 2048,
    "mean_effective_rank": 0.73,
    "mean_intrinsic_dimension": 4.2,
    "spectral_budget_remaining": 0.85
  },

  "skill_assessments": [
    {
      "skill_name": "modus_ponens",
      "accuracy": 0.45,
      "ci_lower": 0.31,
      "ci_upper": 0.60,
      "n_total": 100,
      "n_correct": 45,
      "regime": "ce",
      "is_mastered": false,
      "answer_mode": "exact"
    }
  ],

  "mastered_skills": [],
  "frontier_skills": ["modus_ponens", "disjunctive_syllogism", "single_digit_add"],
  "blocked_skills": ["modus_tollens", "carry_rule"],

  "benchmark_baselines": {
    "gsm8k": 0.0,
    "arc_easy": 0.35,
    "hellaswag": 0.28
  },

  "training_rounds_completed": 0,
  "total_training_samples_seen": 0,
  "profiled_at": "2026-03-12T10:30:00Z"
}
```

### How mc produces it (all existing infrastructure)

1. Model identity from `config.json` parsing (existing model loading)
2. Mastery evaluation via `evaluate_skill_mastery()` from
   `adapters/curriculum_eval_adapter.py`
3. PhaseScheduler state for mastered/frontier/blocked from
   `core/use_cases/curriculum/phase_scheduler.py`
4. CurriculumProfiler for geometric metrics from
   `core/use_cases/curriculum_profiler.py`
5. Benchmark baselines from `StandaloneEvaluationService` (optional)

### Stall Diagnostic (appended when re-profiling after plateau)

```json
{
  "stall_diagnostics": [
    {
      "skill_name": "carry_rule",
      "rounds_attempted": 3,
      "accuracy_history": [0.12, 0.15, 0.14],
      "training_loss_history": [2.1, 1.8, 1.75],
      "samples_trained_on": 450,
      "cka_pre_post": 0.97,
      "spectral_budget_consumed": 0.08,
      "common_failure_patterns": ["model always outputs '0' regardless of input"],
      "n_distinct_wrong_answers": 1
    }
  ]
}
```

This tells the frontier model whether the student is not learning at all (loss
flat) vs. learning but not generalizing (loss down, eval flat), and whether
geometric capacity remains.

---

## Component 2: Prompt Protocol

A deterministic template mc fills with the student profile. The frontier model
receives a single document with five sections.

### Template structure

```
# ModelCypher Curriculum Generation Protocol v1

## Your Role
You are designing a training curriculum for a neural network. You will receive
a student model profile (its current capabilities and geometric properties) and
a training goal. You must return a structured JSON curriculum that ModelCypher
can ingest and execute.

## Student Model Profile
{student_profile_json}

## Training Goal
{goal_description}

Target domain: {target_domain}
Target benchmark: {target_benchmark}

## Constraints

### Data Format
All training samples must be JSONL with this schema:
- Required: {"text": "full_text_here"}
- Optional: {"text": "...", "answer_start": N} where N is the char index
  where the answer begins
- Optional: {"text": "...", "logic_id": "skill_name"} to tag which skill
- Optional: {"text": "...", "template_id": "template_name"} for diversity
- The "text" field contains the COMPLETE example: prompt + answer concatenated.
- For reasoning traces: include chain-of-thought BEFORE the final answer.

### Skill DAG Rules
- Every skill MUST have a formal_statement: the exact logical/mathematical claim.
- Every dependency MUST have a proof_sketch: why B requires A (not "feels harder").
- The DAG MUST be acyclic. mc will reject cycles.
- Each skill needs train and eval data. Eval is held-out (min 50 samples).
- answer_mode: "exact" | "numeric" | "procedural"

### Sample Generation Rules
- QUALITY OVER QUANTITY. 100-500 curated samples per skill >> 50K random.
- Include k=2,3 skill compositions. k=4,5 will transfer.
- Include NEGATIVE examples at 10-20% (wrong reasoning, wrong answers) marked
  with {"is_negative": true} for contrastive signal.
- Include 3+ distinct TEMPLATES per skill for diversity.
- Order samples easy-to-hard for the first 30-50%, then random.
- For verifiable domains (math, logic, code): every answer must be checkable.

### What the Student Already Knows
Mastered: {mastered_skills_list}
Frontier (ready to learn): {frontier_skills_list}
Blocked (unmet prerequisites): {blocked_skills_list}

Do NOT regenerate data for mastered skills unless asked.

## Response Schema
Return EXACTLY this JSON structure:
{response_schema_json}

## Research Findings to Apply
1. 4K high-entropy compositional samples beat much larger random datasets.
2. Training on k=2,3 compositions transfers to k=4,5.
3. 1K curated examples can exceed frontier models on specific benchmarks.
4. Corrupted reasoning traces still transfer reasoning ability -- structure
   matters more than correctness of individual steps.
5. Multiple formats per concept (textbook, Q&A, exercises, chains).
6. 10-20% negative examples improve contrastive learning.
7. Coordinate difficulty with training schedule: easy material first.
```

---

## Component 3: Curriculum Schema

The JSON format the frontier model returns. Maps to existing `SkillNode` via
`GeneratedSkillNode.to_skill_node()`.

```json
{
  "schema_version": "mc.curriculum.v1",
  "curriculum_id": "gsm8k_logic_math_v1",
  "goal": "Beat 50% on GSM8K",
  "target_domain": "logic_math",
  "description": "Logic and arithmetic curriculum targeting GSM8K word problems",

  "skills": [
    {
      "name": "single_digit_add",
      "formal_statement": "(A, B in [0,9]) -> A + B = C",
      "prerequisites": [],
      "proof_sketch": "Primitive: no simpler arithmetic operation exists.",
      "branch": "math",
      "answer_mode": "numeric",
      "procedure_tokens": [],
      "verification": {
        "type": "code_execution",
        "code": "def verify(expected, generated):\n    return int(expected) == int(generated)"
      },
      "difficulty_tier": 0,
      "estimated_samples_needed": 100
    },
    {
      "name": "carry_rule",
      "formal_statement": "A + B >= 10 -> write (A+B) mod 10, carry 1",
      "prerequisites": ["single_digit_add"],
      "proof_sketch": "Carry rule requires computing A+B first (single_digit_add). The rule itself -- 'when sum exceeds one digit, write ones and carry' -- bridges lookup to positional notation.",
      "branch": "math",
      "answer_mode": "procedural",
      "procedure_tokens": ["write", "carry"],
      "verification": {
        "type": "code_execution",
        "code": "def verify(expected, generated):\n    import re\n    num = re.findall(r'\\d+', generated)\n    has_procedure = any(t in generated.lower() for t in ['write', 'carry'])\n    return num and int(num[-1]) == int(expected) and has_procedure"
      },
      "difficulty_tier": 1,
      "estimated_samples_needed": 50
    }
  ],

  "training_data": [
    {
      "skill_name": "single_digit_add",
      "filename": "single_digit_add_train.jsonl",
      "file_type": "train",
      "samples": [
        {
          "text": "What is 3 + 5? The answer is 8.",
          "answer_start": 26,
          "logic_id": "single_digit_add",
          "template_id": "question_answer",
          "is_negative": false,
          "difficulty": 1,
          "composition_k": 1
        },
        {
          "text": "What is 3 + 5? The answer is 9.",
          "answer_start": 26,
          "logic_id": "single_digit_add",
          "template_id": "question_answer",
          "is_negative": true,
          "difficulty": 1,
          "composition_k": 1
        }
      ]
    },
    {
      "skill_name": "single_digit_add",
      "filename": "single_digit_add_eval.jsonl",
      "file_type": "eval",
      "samples": []
    }
  ],

  "metadata": {
    "generator_model": "claude-opus-4-6",
    "generation_timestamp": "2026-03-12T11:00:00Z",
    "student_model_id": "abc123"
  }
}
```

### Mapping to existing SkillNode

| Curriculum field | SkillNode field | Notes |
|------------------|-----------------|-------|
| `name` | `name` | snake_case, unique |
| `formal_statement` | `formal_statement` | Direct |
| `prerequisites` | `prerequisites` | Tuple of names |
| `branch` | `branch` | Direct |
| `answer_mode` | `answer_mode` | Direct |
| `notes` (from proof_sketch) | `notes` | proof_sketch stored in notes |
| `training_data[file_type=train]` | `train_files` | Paths after writing JSONL |
| `training_data[file_type=eval]` | `eval_files` | Paths after writing JSONL |
| `verification` | N/A | Extension, not in SkillNode |
| `procedure_tokens` | N/A | Extension |
| `difficulty_tier` | N/A | Extension |

`GeneratedSkillNode` is a superset that converts down to `SkillNode` via
`.to_skill_node(train_files, eval_files)`. No modification to existing
`SkillNode` or `SkillDAG` needed.

---

## Component 4: Ingestion Validator

### Hard errors (curriculum rejected)

1. Schema version mismatch
2. Missing required fields (name, formal_statement, prerequisites, branch,
   answer_mode for each skill)
3. Non-unique or non-snake_case skill names
4. Prerequisites reference nonexistent skills (not in curriculum AND not in
   mastered set from StudentProfile)
5. Cycle detected in dependency graph (Kahn's algorithm)
6. Missing train or eval data for any skill
7. Invalid answer_mode value
8. Eval samples < 50 per skill (Clopper-Pearson CI floor)
9. Train/eval text overlap for same skill (held-out integrity)

### Warnings (accepted with caveats)

10. Train samples < 50 per skill
11. Template diversity < 3 per skill (< 3 distinct template_ids)
12. Negative example ratio outside 10-20% range
13. Vague formal_statement (< 10 characters)
14. Exact duplicate texts within a skill's training set

### Ingestion process (after validation passes)

1. Write each `training_data[].samples` to JSONL at
   `data/curriculum/{curriculum_id}/{filename}`
2. Construct `GeneratedSkillNode` objects from skills array
3. Convert to `SkillNode` objects with file paths
4. Construct `SkillDAG` (existing class, validates + topological sorts)
5. If StudentProfile had existing mastery state, merge: new DAG's root nodes
   may reference already-mastered skills from prior curriculum
6. Instantiate `PhaseScheduler` with new DAG + existing mastery state
7. Save curriculum spec to `data/curriculum/{curriculum_id}/curriculum.json`

---

## Component 5: Iteration Loop

```
1. mc profiles student model         -> student_profile.json
2. mc builds prompt from profile     -> curriculum_prompt.md
3. User gives prompt to frontier     -> curriculum.json
4. mc validates and ingests          -> SkillDAG + JSONL files
5. mc trains on next skill           -> adapter
6. mc evaluates mastery              -> PhaseScheduler advances (or not)
7. If mastered:                      -> next skill (go to 5)
8. If all skills mastered:           -> re-profile for next tier (go to 1)
9. If stalled (3+ rounds, no gain):  -> re-profile with diagnostic (go to 1)
10. If backward transfer (>10% drop): -> re-profile, add retention data (go to 1)
```

### Re-profiling triggers

- **Curriculum completion:** All skills mastered. StudentProfile shows new
  mastered set; frontier model designs next tier.
- **Training plateau:** 3+ rounds on same skill with no CI improvement.
  StudentProfile includes StallDiagnostic so frontier model can redesign data.
- **Backward transfer:** Re-evaluation of mastered skill shows >10% accuracy
  drop. Frontier model adds retention replay data (20-50 examples from
  mastered skills mixed into current training).

### Retention protocol

When backward transfer is detected, the frontier model's next curriculum
includes a `retention_replay` block: a small set of examples from previously
mastered skills mixed into training data for the current skill. This aligns
with the finding that mixed data (33% synthetic + natural) outperforms pure
synthetic.

---

## Composition with Existing Systems

| System | How it composes |
|--------|-----------------|
| **PhaseScheduler** | Ingested DAG plugs directly in. Mastery state persists across sessions via JSON. |
| **mc train run** | Existing training pipeline used as-is. JSONL data paths passed to `build_training_plan()`. |
| **CurriculumProfiler** | Post-ingestion: score generated data difficulty to verify Goldilocks zone. |
| **STaR service** | After mastery tier: STaR can generate rationalization data on verified problems. |
| **BenchmarkLoader** | Frontier model references standard benchmarks by name; mc converts via existing loaders. |
| **Mastery eval** | `evaluate_skill_mastery()` is the gate. Extended answer_modes (procedural tokens) already supported. |

---

## Verification Plan

1. **Schema round-trip:** Hand-write 3-skill curriculum JSON. Verify it converts
   to valid SkillDAG via existing constructor.

2. **Prompt protocol test:** Profile LFM2-350M, build prompt, give to Claude.
   Check whether returned JSON passes validation rules.

3. **Existing DAG reconstruction:** Give frontier model a blank profile with goal
   "teach propositional logic and basic arithmetic." Compare generated DAG to
   existing 17-node hand-crafted DAG for structural similarity.

4. **Novel domain test:** Goal outside logic+math (e.g., "teach Python list
   comprehensions"). Verify curriculum is structurally valid.

5. **Iteration test:** Mock-train first skill, update profile to show mastery,
   regenerate prompt. Verify frontier model correctly skips mastered skills.

---

## SOTA Research References

See `docs/curriculum/curriculum_sota_research.md` for full research notes.

Key papers:
- Curriculum ordering for small models: arXiv:2601.21698
- STEPS skill taxonomy: arXiv:2601.03676
- Compositional transfer: arXiv:2409.19808
- s1 curated data: Jan 2025
- Active Synthetic Data: arXiv:2512.00884
- AgentInstruct: Microsoft 2024
- Phi-4: Microsoft Dec 2024
- Absolute Zero Reasoner: NeurIPS 2025 Spotlight
