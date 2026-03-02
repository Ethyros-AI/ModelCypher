"""Curriculum CLI: Mastery-gated training progression.

Commands:
    mc curriculum status --model <path>                    # Show mastery state
    mc curriculum next --model <path>                      # Print next skill to teach
    mc curriculum eval --model <path> --skill <name>       # Evaluate mastery of one skill
    mc curriculum dag                                      # Print the skill dependency DAG
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from modelcypher.cli.output import write_error, write_output

curriculum_app = typer.Typer(no_args_is_help=True)


def _echo(text: str) -> None:
    """Write plain text to stdout."""
    write_output(text, "text")


@curriculum_app.callback()
def curriculum() -> None:
    """Mastery-gated curriculum training (logic → math)."""


@curriculum_app.command("status")
def curriculum_status(
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON instead of table"),
) -> None:
    """Show mastery status across all curriculum skill nodes.

    Reads curriculum_state.json from the model directory if it exists.
    All unevaluated skills show regime='unknown'.
    """
    from modelcypher.core.use_cases.curriculum.phase_scheduler import PhaseScheduler

    model_path = Path(model)
    scheduler = PhaseScheduler.at_model_path(model_path)
    status = scheduler.status()

    if json_output:
        write_output(status, "json", pretty=True)
        return

    mastered = status["mastered_count"]
    total = status["total_skills"]
    current = status.get("current") or "none"

    lines = [
        "",
        f"Curriculum Progress: {mastered}/{total} skills mastered",
        f"Currently teaching:  {current}",
        "",
        f"{'Skill':<28} {'Branch':<6} {'D':<3} {'Status':<12} {'Regime':<18} {'Acc':<6} {'N':<5}",
        "-" * 82,
    ]

    for skill in status["skills"]:
        status_icon = {
            "mastered": "V",
            "in_progress": ">",
            "ready": "o",
            "blocked": ".",
        }.get(skill["status"], "?")

        acc_str = f"{skill['accuracy']:.3f}" if skill["accuracy"] is not None else "—"
        n_str = str(skill["n_total"]) if skill["n_total"] else "—"

        lines.append(
            f"{status_icon} {skill['name']:<26} {skill['branch']:<6} {skill['depth']:<3} "
            f"{skill['status']:<12} {skill['regime']:<18} {acc_str:<6} {n_str:<5}"
        )

    lines.append("")
    write_output("\n".join(lines), "text")


@curriculum_app.command("next")
def curriculum_next(
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
) -> None:
    """Print the next skill node to teach.

    Shows training files and eval files. Use this to decide which dataset to
    pass to 'mc train run'.
    """
    from modelcypher.core.use_cases.curriculum.phase_scheduler import PhaseScheduler

    model_path = Path(model)
    scheduler = PhaseScheduler.at_model_path(model_path)
    node = scheduler.next_to_teach()

    if node is None:
        write_output("Curriculum complete — all skills mastered.", "text")
        return

    lines = [
        "",
        f"Next skill: {node.name}",
        f"  Branch:     {node.branch}",
        f"  Statement:  {node.formal_statement}",
    ]
    if node.prerequisites:
        lines.append(f"  Requires:   {', '.join(node.prerequisites)}")
    if node.notes:
        lines.append(f"  Notes:      {node.notes}")

    lines.append("\n  Training files:")
    for f in node.train_files:
        exists = "OK" if Path(f).exists() else "MISSING"
        lines.append(f"    [{exists}]  {f}")
    if not node.train_files:
        lines.append("    (none — generate training data first; see skill_dag.md)")

    lines.append("\n  Eval files:")
    for f in node.eval_files:
        exists = "OK" if Path(f).exists() else "MISSING"
        lines.append(f"    [{exists}]  {f}")
    if not node.eval_files:
        lines.append("    (none — generate eval data first; see skill_dag.md)")

    lines.append("")
    write_output("\n".join(lines), "text")


@curriculum_app.command("eval")
def curriculum_eval(
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    skill: str = typer.Option(..., "--skill", "-s", help="Skill name (e.g., modus_ponens)"),
    eval_file: Optional[str] = typer.Option(
        None, "--eval-file", help="Override eval JSONL path (default: from skill DAG)"
    ),
    chance_rate: float = typer.Option(
        0.0, "--chance-rate",
        help="Random-chance baseline (0.0 for free-text, 0.25 for 4-way MC)"
    ),
) -> None:
    """Evaluate mastery of a single skill and update curriculum state.

    Runs inference on the held-out eval set, computes Clopper-Pearson CI,
    derives regime (ce|reinforce_entropy|reinforce), and saves to
    <model>/curriculum_state.json.

    A skill is mastered when regime == 'reinforce'.
    """
    from modelcypher.adapters.curriculum_eval_adapter import evaluate_skill_mastery
    from modelcypher.core.use_cases.curriculum.phase_scheduler import PhaseScheduler
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG

    model_path = Path(model)

    try:
        node = CURRICULUM_DAG.get(skill)
    except KeyError:
        all_skills = sorted(n.name for n in CURRICULUM_DAG.nodes)
        write_error(
            f"Unknown skill '{skill}'. Known: {', '.join(all_skills)}",
            "text",
        )
        raise typer.Exit(code=1)

    if eval_file:
        eval_path = Path(eval_file)
    elif node.eval_files:
        eval_path = Path(node.eval_files[0])
    else:
        write_error(
            f"Skill '{skill}' has no eval files defined. Pass --eval-file.",
            "text",
        )
        raise typer.Exit(code=1)

    if not eval_path.exists():
        write_error(
            f"Eval file not found: {eval_path}\nGenerate it first (see docs/curriculum/skill_dag.md).",
            "text",
        )
        raise typer.Exit(code=1)

    write_output(
        f"\nEvaluating mastery: {skill}\n  Model: {model_path}\n  Eval:  {eval_path}\n  Running inference...",
        "text",
    )

    try:
        record = evaluate_skill_mastery(
            model_path=str(model_path),
            skill=node,
            eval_jsonl_path=eval_path,
            chance_rate=chance_rate,
        )
    except Exception as e:
        write_error(f"Eval failed: {e}", "text")
        raise typer.Exit(code=1)

    state_path = model_path / "curriculum_state.json"
    scheduler = PhaseScheduler.at_model_path(model_path)
    scheduler.update_mastery(record)

    lines = [
        "",
        f"  Accuracy:  {record.accuracy:.3f}",
        f"  CI lower:  {record.ci_lower:.3f}",
        f"  CI upper:  {record.ci_upper:.3f}",
        f"  Samples:   {record.n_total}",
        f"  Regime:    {record.regime}",
        "",
        f"  Mastered:  {'YES' if record.is_mastered() else 'NO'}",
        f"  Saved:     {state_path}",
        "",
    ]
    write_output("\n".join(lines), "text")


@curriculum_app.command("dag")
def curriculum_dag(
    branch: Optional[str] = typer.Option(
        None, "--branch", "-b", help="Filter by branch: 'logic' or 'math'"
    ),
) -> None:
    """Print the skill dependency DAG in topological order.

    Each dependency edge has a proof sketch in docs/curriculum/skill_dag.md.
    """
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG

    lines = [
        "",
        "Skill Dependency DAG (topological order)",
        "",
        f"{'Node':<28} {'Branch':<6} {'D':<3} Prerequisites",
        "-" * 75,
    ]

    for node in CURRICULUM_DAG.topological_sort():
        if branch and node.branch != branch:
            continue
        prereq_str = ", ".join(node.prerequisites) if node.prerequisites else "none"
        depth = CURRICULUM_DAG.depth(node.name)
        lines.append(f"{node.name:<28} {node.branch:<6} {depth:<3} {prereq_str}")

    lines += [
        "",
        "Proof sketches: docs/curriculum/skill_dag.md",
        "",
    ]
    write_output("\n".join(lines), "text")
