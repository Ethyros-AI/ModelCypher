"""Curriculum mastery evaluation adapter.

Implements evaluate_skill_mastery at the adapter layer (can import backends).
The core PhaseScheduler does not import this — it receives MasteryRecord objects
from outside (hexagonal boundary rule: core/use_cases cannot import from adapters).

Usage (from CLI or scripts):
    from modelcypher.adapters.curriculum_eval_adapter import evaluate_skill_mastery
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from modelcypher.core.domain.statistics import clopper_pearson_interval
from modelcypher.core.use_cases.curriculum.phase_scheduler import MasteryRecord
from modelcypher.core.use_cases.curriculum.skill_dag import SkillNode

logger = logging.getLogger(__name__)


def _extract_last_int(text: str) -> int | None:
    """Return the last integer appearing in text, or None if no integer found.

    Used for numeric answer_mode evaluation. Handles both direct answers
    ("15") and scratchpad-prefixed answers ("Ones: 7+8=15. Write 5... Answer: 15").
    """
    nums = re.findall(r"\b\d+\b", text)
    return int(nums[-1]) if nums else None


def evaluate_skill_mastery(
    model_path: str,
    skill: SkillNode,
    eval_jsonl_path: Path,
    *,
    chance_rate: float = 0.0,
) -> MasteryRecord:
    """Evaluate mastery of a skill on its held-out eval set.

    Runs inference on each problem in the eval JSONL, checks correctness,
    computes Clopper-Pearson CI, and derives regime from the CI relative
    to chance_rate.

    Correctness check depends on skill.answer_mode:
      'exact'  (default): expected substring must appear in generated text
               (case-insensitive). Used for logic skills with string answers.
      'numeric': extract last integer from both expected and generated texts;
               compare as integers. Used for arithmetic skills whose training
               data includes scratchpad steps (the model may generate intermediate
               steps; only the final numeric answer is checked for mastery).
      'procedural': final integer correct AND at least one carry-indicator token
               ("write" or "carry", case-insensitive) in generated output. Used
               when the formal claim is procedure execution, not answer recall.
               Pure memorization (correct number, no carry tokens) fails this gate.

    Args:
        model_path: Path to the model directory.
        skill: The skill node being evaluated (answer_mode field controls
            how generated output is compared to expected).
        eval_jsonl_path: Path to held-out eval JSONL. Each line:
            {"text": "prompt answer"} or {"text": "...", "answer_start": N}
        chance_rate: Random-chance baseline for this problem type.
            0.0 for free-text answers, 0.25 for 4-way multiple choice.

    Returns:
        MasteryRecord with regime derived from Clopper-Pearson CI.
        regime == 'reinforce' means ci_lower > chance_rate.
        Mastery requires n_correct == n_total (is_mastered()).
    """
    eval_path = Path(eval_jsonl_path)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval file not found for skill '{skill.name}': {eval_path}\n"
            f"Generate it first (see docs/curriculum/skill_dag.md eval requirements)."
        )

    problems = []
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    n_total = len(problems)
    if n_total <= 1:
        raise ValueError(
            f"Eval set for skill '{skill.name}' must contain at least 2 samples "
            f"to derive Clopper-Pearson confidence bounds (got n_total={n_total})."
        )

    if n_total < 50:
        # Clopper-Pearson CI at n=50, alpha=1/n=0.02 gives ~±14% at 95% confidence.
        # Below 50 samples the CI is too wide for reliable regime classification.
        logger.warning(
            "Skill '%s' eval set has only %d samples (minimum 50 for reliable CI). "
            "Results will have wide uncertainty bounds.",
            skill.name, n_total,
        )

    # Use the canonical inference path: InferenceEngine via Backend protocol.
    # max_tokens=None → InferenceEngine._derive_max_tokens() auto-derives from
    # context limit and prompt length.
    #
    # When called from scripts (not CLI entry point), the backend may not be
    # initialized yet. Auto-initialize here so scripts don't need boilerplate.
    from modelcypher.adapters.inference_engine import get_inference_engine
    from modelcypher.core.domain._backend import get_default_backend

    try:
        get_default_backend()
    except RuntimeError:
        from modelcypher.backends import detect_default_backend_type, get_backend
        from modelcypher.core.domain._backend import set_default_backend

        set_default_backend(get_backend(detect_default_backend_type()))

    engine = get_inference_engine()
    n_correct = 0

    for item in problems:
        text = item["text"]
        answer_start = item.get("answer_start")

        if answer_start is not None:
            prompt = text[:answer_start]
            expected = text[answer_start:].strip().lower()
        else:
            parts = text.rsplit("Answer:", 1)
            if len(parts) == 2:
                prompt = parts[0] + "Answer:"
                expected = parts[1].strip().lower()
            else:
                tokens = text.split()
                prompt = " ".join(tokens[:-1])
                expected = tokens[-1].strip().lower()

        try:
            result = engine.run(model=model_path, prompt=prompt, max_tokens=None)
            predicted = result.response.strip()

            if skill.answer_mode == "numeric":
                # Extract last integer from both sides. Handles direct answers
                # ("15") and scratchpad-prefixed answers ("...Answer: 15") equally.
                expected_int = _extract_last_int(expected)
                predicted_int = _extract_last_int(predicted)
                if (
                    expected_int is not None
                    and predicted_int is not None
                    and expected_int == predicted_int
                ):
                    n_correct += 1
            elif skill.answer_mode == "procedural":
                # Numeric check: final answer must be correct.
                expected_int = _extract_last_int(expected)
                predicted_int = _extract_last_int(predicted)
                numeric_correct = (
                    expected_int is not None
                    and predicted_int is not None
                    and expected_int == predicted_int
                )
                # Procedural check: model must emit at least one carry-indicator token.
                # Tokens derived from carry_rule training format:
                #   "{a} + {b} = {sum}. Write {digit}, carry 1. Answer: {sum}"
                # A model that learned the rule WILL generate these. Memorization won't.
                predicted_lower = predicted.lower()
                has_procedure = "write" in predicted_lower or "carry" in predicted_lower
                if numeric_correct and has_procedure:
                    n_correct += 1
            else:
                if expected and expected.lower() in predicted.lower():
                    n_correct += 1
        except Exception:
            logger.debug(
                "Inference failed for a problem in skill '%s'", skill.name, exc_info=True
            )

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    ci_lower, ci_upper = clopper_pearson_interval(
        n_correct=n_correct,
        n_total=n_total,
        alpha=1.0 / n_total,
    )

    if ci_lower > chance_rate:
        regime = "reinforce"
    elif ci_upper > chance_rate:
        regime = "reinforce_entropy"
    else:
        regime = "ce"

    logger.info(
        "Skill '%s' mastery eval: accuracy=%.3f CI=[%.3f, %.3f] n=%d chance=%.3f regime=%s",
        skill.name, accuracy, ci_lower, ci_upper, n_total, chance_rate, regime,
    )

    return MasteryRecord(
        skill_name=skill.name,
        regime=regime,
        accuracy=accuracy,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_total=n_total,
        n_correct=n_correct,
        chance_rate=chance_rate,
    )
