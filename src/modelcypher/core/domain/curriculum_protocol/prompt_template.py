"""Prompt template for frontier model curriculum generation.

The template is a structured document that mc fills with a StudentProfile
and goal. The frontier model receives this and returns a CurriculumSpec JSON.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.domain.curriculum_protocol.student_profile import (
        StudentProfile,
    )


RESPONSE_SCHEMA: dict = {
    "schema_version": "mc.curriculum.v1",
    "curriculum_id": "<unique_snake_case_identifier>",
    "goal": "<the training goal>",
    "target_domain": "<domain name>",
    "description": "<1-2 sentence description>",
    "skills": [
        {
            "name": "<snake_case unique identifier>",
            "formal_statement": "<precise logical/mathematical statement>",
            "prerequisites": ["<skill_name>"],
            "proof_sketch": "<why B requires A -- not 'feels harder'>",
            "branch": "<domain branch name>",
            "answer_mode": "exact|numeric|procedural",
            "procedure_tokens": ["<tokens for procedural mode>"],
            "verification": {
                "type": "code_execution|substring_match|rubric",
                "code": "<optional Python verification function as string>",
                "rubric": ["<criterion 1>"],
            },
            "difficulty_tier": 0,
            "estimated_samples_needed": 200,
        }
    ],
    "training_data": [
        {
            "skill_name": "<matches a skill name above>",
            "filename": "<skill_name>_train.jsonl",
            "file_type": "train|eval",
            "samples": [
                {
                    "text": "<complete training example: prompt + answer>",
                    "answer_start": "null or integer char index",
                    "logic_id": "<skill_name>",
                    "template_id": "<template identifier>",
                    "is_negative": False,
                    "difficulty": 1,
                    "composition_k": 1,
                }
            ],
        }
    ],
    "metadata": {
        "generator_model": "<your model name>",
        "generation_timestamp": "<ISO 8601>",
        "student_model_id": "<from student profile>",
    },
}


PROMPT_TEMPLATE = """\
# ModelCypher Curriculum Generation Protocol v1

## Your Role

You are designing a training curriculum for a neural network. You will receive
a student model profile (its current capabilities and geometric properties) and
a training goal. You must return a structured JSON curriculum that ModelCypher
can ingest and execute.

The curriculum is a skill dependency DAG: each skill has formal prerequisites
proven by dependency sketches, training data, and held-out eval data. The
training pipeline will teach skills in topological order, gated by mastery
evaluation on held-out eval sets.

---

## Student Model Profile

```json
{student_profile_json}
```

---

## Training Goal

{goal_description}

Target domain: {target_domain}
Target benchmark: {target_benchmark}

---

## Constraints

### Data Format

All training samples must use this schema:
- Required: {{"text": "full_text_here"}}
- Optional: {{"text": "...", "answer_start": N}} where N is the character
  index where the answer begins in the text
- Optional: {{"text": "...", "logic_id": "skill_name"}} to tag which skill
  a sample exercises
- Optional: {{"text": "...", "template_id": "template_name"}} for diversity
  tracking

The "text" field contains the COMPLETE training example: prompt + answer
concatenated into a single string. For reasoning traces, include
chain-of-thought BEFORE the final answer. For scratchpad formats (arithmetic),
include step-by-step work.

### Skill DAG Rules

- Every skill node MUST have a `formal_statement`: the exact logical or
  mathematical claim being taught.
- Every dependency edge MUST have a `proof_sketch`: a brief argument for why
  skill B requires skill A as a prerequisite. "It feels harder" is NOT a valid
  dependency. A valid dependency means the proof or execution of B requires A.
- The DAG MUST be acyclic. Cycles will be rejected.
- Each skill needs both training data (file_type="train") and held-out eval
  data (file_type="eval"). Eval data is never used in training.
- Eval sets MUST have at least 50 samples (minimum for reliable Clopper-Pearson
  confidence intervals).
- `answer_mode` must be one of:
  - "exact": expected substring must appear in model output
  - "numeric": last integer in output must match expected
  - "procedural": numeric match AND procedure tokens must appear in output

### Sample Generation Rules

- QUALITY OVER QUANTITY. 100-500 carefully curated samples per skill are far
  more effective than 50K random samples.
- For compositional skills (depth >= 2 in the DAG): include examples that
  combine k=2 or k=3 prerequisite skills. Research shows that training on
  simple compositions transfers to more complex ones (k=4,5) without needing
  explicit training at those levels.
- Include NEGATIVE examples (wrong reasoning, wrong answers) at 10-20% of
  samples. Mark them with {{"is_negative": true}}. These provide contrastive
  signal that improves learning.
- Include at least 3 distinct TEMPLATES per skill for surface-form diversity.
  Vary the phrasing, domain, and structure while keeping the underlying logic
  identical.
- Order samples easy-to-hard for the first 30-50% of each skill's data, then
  random ordering for the rest.
- For verifiable domains (math, logic, code): every answer must be
  programmatically checkable. Provide a verification function.

### What the Student Already Knows

Mastered skills: {mastered_skills_list}
Frontier skills (ready to learn next): {frontier_skills_list}
Blocked skills (unmet prerequisites): {blocked_skills_list}

Do NOT regenerate training data for already-mastered skills unless explicitly
asked. Focus the curriculum on: (1) frontier skills first, then (2) new skills
that extend the frontier toward the goal.

---

## Response Schema

Return EXACTLY this JSON structure. No markdown wrapping, no commentary outside
the JSON. The JSON must be parseable by `json.loads()`.

```json
{response_schema_json}
```

---

## Research Findings to Apply

These findings from recent ML research should inform your curriculum design:

1. **Compositional transfer**: Training on k=2,3 skill compositions transfers
   to k=4,5 compositions never seen in training. You do not need to enumerate
   all possible skill combinations.

2. **High-entropy sampling**: 4K diverse compositional samples beat much larger
   random datasets. Maximize skill diversity in examples, not volume.

3. **Data curation**: 1K carefully curated examples can exceed frontier models
   on specific benchmarks. Selection by difficulty, diversity, and quality
   matters more than sample count.

4. **Reasoning traces transfer structure**: Even imperfect reasoning traces
   transfer reasoning ability. The structure of multi-step work matters more
   than correctness of individual intermediate steps.

5. **Multiple formats**: Generate textbook exposition, Q&A pairs, exercises,
   and reasoning chains for each concept. Format diversity improves learning.

6. **Contrastive learning**: 10-20% negative examples (wrong answers, common
   mistakes) improve the model's ability to discriminate correct from incorrect
   reasoning.

7. **Difficulty coordination**: Order easy-to-hard for the first 30-50% of
   training, then switch to random sampling. Do not save all hard examples
   for the end.\
"""


def build_prompt(
    profile: "StudentProfile",
    goal: str,
    target_domain: str = "",
    target_benchmark: str = "",
) -> str:
    """Build the prompt document for a frontier model.

    Args:
        profile: StudentProfile with model capabilities and geometric state.
        goal: Free-text training goal (e.g., "beat 50% on GSM8K").
        target_domain: Domain name (e.g., "logic_math", "code", "physics").
        target_benchmark: Specific benchmark name (e.g., "gsm8k", "arc_easy").

    Returns:
        The complete prompt as a string, ready to give to a frontier model.
    """
    profile_json = json.dumps(profile.to_dict(), indent=2)
    schema_json = json.dumps(RESPONSE_SCHEMA, indent=2)

    mastered = ", ".join(profile.mastered_skills) if profile.mastered_skills else "(none)"
    frontier = ", ".join(profile.frontier_skills) if profile.frontier_skills else "(none)"
    blocked = ", ".join(profile.blocked_skills) if profile.blocked_skills else "(none)"

    return PROMPT_TEMPLATE.format(
        student_profile_json=profile_json,
        goal_description=goal,
        target_domain=target_domain or "(not specified)",
        target_benchmark=target_benchmark or "(not specified)",
        response_schema_json=schema_json,
        mastered_skills_list=mastered,
        frontier_skills_list=frontier,
        blocked_skills_list=blocked,
    )
