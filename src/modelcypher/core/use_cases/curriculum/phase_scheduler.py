"""PhaseScheduler: Topological curriculum traversal with mastery-gated progression.

The scheduler answers two questions at any point in training:
  1. "Which skill should the model learn next?"
  2. "Has the model mastered a given skill?"

Design:
- PhaseScheduler is a pure orchestration layer with no ML imports.
- Mastery evaluation (running the model) is done externally and fed in via
  update_mastery(). This keeps the scheduler testable and serializable.
- State persists to JSON so curriculum progress survives between sessions.

Mastery criterion:
  A skill is mastered when the model answers every item in the eval set correctly
  (n_correct == n_total, i.e., accuracy == 1.0 on the held-out set).

  Rationale: "Skill B depends on Skill A" means the formal proof of B uses A as
  a premise. Using a skill as a reliable premise requires answering every instance
  correctly — partial knowledge (e.g., 12% on chain_reasoning) cannot serve as a
  dependable premise. The eval set is small and constructed so the expected answer
  is unambiguous; a model that has genuinely internalized the rule should produce
  the correct string on every item.

  The `regime` field (ce / reinforce_entropy / reinforce) is retained for selecting
  the TRAINING OBJECTIVE (CE vs REINFORCE) and is independent of mastery. A skill
  may have regime='reinforce' but not yet be mastered (accuracy < 1.0), meaning
  REINFORCE is the right training mode but the model needs more training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from modelcypher.core.use_cases.curriculum.skill_dag import (
    CURRICULUM_DAG,
    SkillDAG,
    SkillNode,
)

logger = logging.getLogger(__name__)


@dataclass
class MasteryRecord:
    """Mastery state for a single skill node.

    Populated by external eval (e.g., curriculum_eval_adapter.evaluate_skill_mastery).
    Never populated by PhaseScheduler itself.

    Attributes:
        skill_name: Must match a node in the DAG.
        regime: 'unknown' | 'ce' | 'reinforce_entropy' | 'reinforce'
            Determines the TRAINING OBJECTIVE, not mastery:
            'ce'               = near-zero capability; CE deposits the invariant.
            'reinforce_entropy' = emerging capability; REINFORCE+entropy refines.
            'reinforce'        = above-chance capability; pure REINFORCE extracts.
        n_correct: Number of eval items answered correctly.
        n_total: Total number of eval items.
        accuracy: n_correct / n_total (0.0–1.0).
        ci_lower: Clopper-Pearson lower bound on accuracy.
        ci_upper: Clopper-Pearson upper bound on accuracy.
        chance_rate: Random-chance baseline for this problem type.
    """

    skill_name: str
    regime: str = "unknown"
    accuracy: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_total: int = 0
    n_correct: int = 0
    chance_rate: float = 0.0

    def is_mastered(self) -> bool:
        """True when the model answers every item in the eval set correctly.

        Mastery requires n_correct == n_total (accuracy == 1.0). The model
        cannot advance to the next skill in the DAG until it can reliably apply
        the current skill without error. Partial knowledge is not mastery — it
        is the training target.

        The `regime` field controls the training objective (CE vs REINFORCE)
        and is separate from this criterion.
        """
        return self.n_total > 0 and self.n_correct == self.n_total


class PhaseScheduler:
    """Traverses the curriculum DAG, teaching skills in topological order.

    State is persisted to a JSON file at ``state_path``. Between sessions,
    load an existing scheduler with PhaseScheduler.load(state_path, dag).

    Usage:
        scheduler = PhaseScheduler(dag=CURRICULUM_DAG, state_path=path)

        # Find which skill to teach now:
        node = scheduler.next_to_teach()

        # After training + eval, update mastery:
        scheduler.update_mastery(MasteryRecord(
            skill_name='modus_ponens',
            regime='reinforce',
            accuracy=0.94,
            ci_lower=0.88,
            ci_upper=0.97,
            n_total=50,
            chance_rate=0.0,
        ))

        # If mastered, advance:
        node = scheduler.next_to_teach()  # returns next node
    """

    def __init__(
        self,
        dag: SkillDAG,
        state_path: Optional[Path] = None,
        mastery: Optional[Dict[str, MasteryRecord]] = None,
    ) -> None:
        self._dag = dag
        self._state_path = state_path
        self._mastery: Dict[str, MasteryRecord] = mastery or {}

    # ── Mastery tracking ──────────────────────────────────────────────────

    def update_mastery(self, record: MasteryRecord) -> None:
        """Record mastery evaluation result for a skill.

        Call this after running the model on the held-out eval set for the skill
        and computing the regime via regime_selection.select_training_regime().
        """
        if record.skill_name not in {n.name for n in self._dag.nodes}:
            raise ValueError(
                f"Unknown skill '{record.skill_name}'. "
                f"Known skills: {sorted(n.name for n in self._dag.nodes)}"
            )
        self._mastery[record.skill_name] = record
        logger.info(
            "Mastery updated: %s | regime=%s | accuracy=%.3f | CI=[%.3f, %.3f] | n=%d",
            record.skill_name,
            record.regime,
            record.accuracy,
            record.ci_lower,
            record.ci_upper,
            record.n_total,
        )
        if self._state_path is not None:
            self.save(self._state_path)

    def mastered_skills(self) -> set[str]:
        """Set of skill names that have reached 'reinforce' regime."""
        return {
            name for name, rec in self._mastery.items() if rec.is_mastered()
        }

    def is_mastered(self, skill_name: str) -> bool:
        rec = self._mastery.get(skill_name)
        return rec is not None and rec.is_mastered()

    # ── Scheduling ────────────────────────────────────────────────────────

    def next_to_teach(self) -> Optional[SkillNode]:
        """Return the next skill to teach.

        Returns the shallowest unmastered skill whose prerequisites are all
        mastered. Returns None when all skills are mastered.

        Selection rule: shallower DAG depth is preferred (teach primitives
        before compositions). Ties broken by name for determinism.
        """
        mastered = self.mastered_skills()
        candidates = self._dag.ready_to_teach(mastered)
        if not candidates:
            all_mastered = all(
                self.is_mastered(n.name) for n in self._dag.nodes
            )
            if all_mastered:
                logger.info("Curriculum complete — all skills mastered.")
                return None
            # Prerequisites are not yet mastered, which should not happen
            # if the caller advances in topological order.
            unmastered = [
                n.name for n in self._dag.nodes if not self.is_mastered(n.name)
            ]
            logger.warning(
                "No candidates ready to teach but %d skills unmastered: %s",
                len(unmastered),
                unmastered,
            )
            return None
        return candidates[0]

    def current_phase_node(self) -> Optional[SkillNode]:
        """Return the current teaching target (same as next_to_teach)."""
        return self.next_to_teach()

    def pending_skills(self) -> List[SkillNode]:
        """Skills not yet mastered, in topological order."""
        mastered = self.mastered_skills()
        return [n for n in self._dag.topological_sort() if n.name not in mastered]

    # ── Status reporting ──────────────────────────────────────────────────

    def status(self) -> dict:
        """Full curriculum status snapshot for display / logging."""
        mastered = self.mastered_skills()
        result = {
            "mastered_count": len(mastered),
            "total_skills": len(self._dag.nodes),
            "current": None,
            "skills": [],
        }
        current = self.next_to_teach()
        if current:
            result["current"] = current.name

        for node in self._dag.topological_sort():
            rec = self._mastery.get(node.name)
            result["skills"].append({
                "name": node.name,
                "branch": node.branch,
                "depth": self._dag.depth(node.name),
                "prerequisites": list(node.prerequisites),
                "status": (
                    "mastered" if self.is_mastered(node.name)
                    else "in_progress" if node.name == (current.name if current else None)
                    else "blocked" if any(
                        p not in mastered for p in node.prerequisites
                    )
                    else "ready"
                ),
                "regime": rec.regime if rec else "unknown",
                "accuracy": round(rec.accuracy, 4) if rec else None,
                "ci_lower": round(rec.ci_lower, 4) if rec else None,
                "n_total": rec.n_total if rec else 0,
            })
        return result

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist mastery state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: asdict(rec) for name, rec in self._mastery.items()
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.debug("PhaseScheduler state saved to %s", path)

    @classmethod
    def load(cls, path: Path, dag: Optional[SkillDAG] = None) -> "PhaseScheduler":
        """Load mastery state from JSON.

        Args:
            path: Path to the JSON state file.
            dag: SkillDAG to use. Defaults to CURRICULUM_DAG.
        """
        dag = dag or CURRICULUM_DAG
        path = Path(path)
        mastery: Dict[str, MasteryRecord] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            for name, d in raw.items():
                mastery[name] = MasteryRecord(**d)
        return cls(dag=dag, state_path=path, mastery=mastery)

    @classmethod
    def at_model_path(cls, model_path: Path, dag: Optional[SkillDAG] = None) -> "PhaseScheduler":
        """Convenience: load/create scheduler state stored alongside the model.

        State file is written to <model_path>/curriculum_state.json.
        """
        state_path = Path(model_path) / "curriculum_state.json"
        return cls.load(state_path, dag=dag)


# evaluate_skill_mastery lives in adapters/curriculum_eval_adapter.py
# (core/use_cases cannot import from adapters — hexagonal boundary rule)
#
# CLI usage:
#   from modelcypher.adapters.curriculum_eval_adapter import evaluate_skill_mastery
