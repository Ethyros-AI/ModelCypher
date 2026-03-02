"""Skill dependency DAG for curriculum training.

The ordering of skills is derived from formal dependency proofs, not heuristics.
See docs/curriculum/skill_dag.md for the full specification with proof sketches.

Key property: Skill B lists Skill A as a prerequisite iff the proof of B requires A.
Training order = any topological sort of this DAG.
Mastery criterion = auto-regime detects 'reinforce' on held-out eval for that node.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SkillNode:
    """A single learnable skill with its dependency information.

    Attributes:
        name: Unique identifier (snake_case).
        formal_statement: The logical/mathematical statement being learned.
        prerequisites: Names of skills that must reach 'reinforce' regime first.
        train_files: Relative paths (from project root) to training JSONL files.
        eval_files: Relative paths to held-out eval JSONL files (not used in training).
        branch: 'logic' | 'math' — which branch of the DAG this belongs to.
        notes: Optional clarifications (not used by scheduler).
    """

    name: str
    formal_statement: str
    prerequisites: tuple[str, ...]
    train_files: tuple[str, ...]
    eval_files: tuple[str, ...]
    branch: str
    notes: str = ""


class SkillDAG:
    """Directed acyclic graph of skill dependencies.

    Edges represent provable formal dependencies (see skill_dag.md).
    Provides topological ordering and prerequisite checking for PhaseScheduler.
    """

    def __init__(self, nodes: List[SkillNode]) -> None:
        self._nodes: Dict[str, SkillNode] = {n.name: n for n in nodes}
        self._validate()

    def _validate(self) -> None:
        """Verify all prerequisites reference existing nodes."""
        for node in self._nodes.values():
            for prereq in node.prerequisites:
                if prereq not in self._nodes:
                    raise ValueError(
                        f"Node '{node.name}' lists unknown prerequisite '{prereq}'"
                    )

    @property
    def nodes(self) -> List[SkillNode]:
        return list(self._nodes.values())

    def get(self, name: str) -> SkillNode:
        return self._nodes[name]

    def predecessors(self, name: str) -> List[SkillNode]:
        """Direct prerequisites of a node."""
        return [self._nodes[p] for p in self._nodes[name].prerequisites]

    def depth(self, name: str) -> int:
        """DAG depth = length of longest path from any root to this node.

        Roots (no prerequisites) have depth 0.
        Used to order nodes: shallower nodes are taught first.
        """
        node = self._nodes[name]
        if not node.prerequisites:
            return 0
        return 1 + max(self.depth(p) for p in node.prerequisites)

    def topological_sort(self) -> List[SkillNode]:
        """Return nodes in topological order (prerequisites before dependents).

        Uses Kahn's algorithm. Nodes at the same depth are ordered by name
        for determinism.
        """
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes.values():
            for prereq in node.prerequisites:
                in_degree[node.name] = in_degree.get(node.name, 0) + 1

        # Recompute: in_degree = number of prerequisites per node
        in_degree = {}
        for name, node in self._nodes.items():
            in_degree[name] = len(node.prerequisites)

        queue: deque[str] = deque(
            sorted(n for n, d in in_degree.items() if d == 0)
        )
        result: List[SkillNode] = []

        while queue:
            name = queue.popleft()
            result.append(self._nodes[name])
            # Find all nodes that list 'name' as a prerequisite
            dependents = sorted(
                n for n, node in self._nodes.items() if name in node.prerequisites
            )
            for dep in dependents:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(result) != len(self._nodes):
            cycle_nodes = set(self._nodes) - {n.name for n in result}
            raise ValueError(f"Cycle detected in skill DAG involving: {cycle_nodes}")

        return result

    def ready_to_teach(self, mastered: set[str]) -> List[SkillNode]:
        """Return nodes whose prerequisites are all mastered.

        A node is 'ready' when all its prerequisites are in the mastered set.
        These are the candidates for the next training phase.
        """
        candidates = [
            node
            for node in self._nodes.values()
            if node.name not in mastered
            and all(p in mastered for p in node.prerequisites)
        ]
        # Order by DAG depth (shallower first), then name for determinism
        return sorted(candidates, key=lambda n: (self.depth(n.name), n.name))


# ---------------------------------------------------------------------------
# The actual DAG — derived from docs/curriculum/skill_dag.md
# ---------------------------------------------------------------------------

def build_curriculum_dag() -> SkillDAG:
    """Construct the full curriculum DAG.

    Dependencies are proven in docs/curriculum/skill_dag.md.
    Adding a new prerequisite requires updating that document first.
    """
    nodes = [
        # ── Logic Branch ──────────────────────────────────────────────────

        SkillNode(
            name="modus_ponens",
            formal_statement="(P→Q, P) ⊢ Q",
            prerequisites=(),
            train_files=(
                "data/training/phase1_inference_rules.jsonl",
                "data/training/phase1_inference_rules_balanced.jsonl",
            ),
            eval_files=(
                "data/eval/modus_ponens_eval.jsonl",
            ),
            branch="logic",
            notes="Primitive axiom of propositional logic. No smaller rule to depend on.",
        ),

        SkillNode(
            name="modus_tollens",
            formal_statement="(P→Q, ¬Q) ⊢ ¬P",
            prerequisites=("modus_ponens",),
            train_files=(
                # Phase 3 is mixed recognition; no MT-only train file yet.
                # TODO: generate data/training/modus_tollens_train.jsonl
            ),
            eval_files=(
                "data/eval/modus_tollens_eval.jsonl",
            ),
            branch="logic",
            notes=(
                "Proof: apply MP to contrapositive (¬Q→¬P). "
                "Final inference step IS an application of MP."
            ),
        ),

        SkillNode(
            name="disjunctive_syllogism",
            formal_statement="(P∨Q, ¬P) ⊢ Q",
            prerequisites=(),
            train_files=(
                # TODO: generate data/training/disj_syllogism_train.jsonl
            ),
            eval_files=(
                "data/eval/disjunctive_syllogism_eval.jsonl",
            ),
            branch="logic",
            notes=(
                "Proven independent of MP: proof uses disjunction elimination, not MP. "
                "Root node (depth 0) parallel to modus_ponens and arithmetic_add."
            ),
        ),

        SkillNode(
            name="hypothetical_syllogism",
            formal_statement="(P→Q, Q→R) ⊢ P→R",
            prerequisites=("modus_ponens",),
            train_files=(
                "data/training/phase2_rule_compositions.jsonl",
            ),
            eval_files=(
                "data/eval/hypothetical_syllogism_eval.jsonl",
            ),
            branch="logic",
            notes="Proof uses MP twice. Phase 2 data teaches this via chained conditionals.",
        ),

        SkillNode(
            name="universal_instantiation",
            formal_statement="(∀x P(x), a in domain) ⊢ P(a)",
            prerequisites=("modus_ponens",),
            train_files=(
                # TODO: generate data/training/universal_instantiation_train.jsonl
            ),
            eval_files=(
                "data/eval/universal_instantiation_eval.jsonl",
            ),
            branch="logic",
            notes=(
                "∀x P(x) acts as a universally applicable conditional. "
                "MP applies to instantiate it for a specific a."
            ),
        ),

        SkillNode(
            name="rule_recognition",
            formal_statement="Given (premises, conclusion), identify the inference rule used.",
            prerequisites=("modus_ponens", "modus_tollens", "disjunctive_syllogism"),
            train_files=(
                "data/training/phase3_rule_recognition.jsonl",
            ),
            eval_files=(
                "data/eval/rule_recognition_eval.jsonl",
            ),
            branch="logic",
            notes=(
                "Recognition requires knowing all three rules to distinguish them. "
                "Cannot recognize MT if MT is not consolidated."
            ),
        ),

        SkillNode(
            name="concise_reasoning",
            formal_statement="Apply inference rules with minimal scaffolding — answer only.",
            prerequisites=("rule_recognition",),
            train_files=(
                "data/training/phase4_conciseness.jsonl",
            ),
            eval_files=(
                "data/eval/concise_reasoning_eval.jsonl",
            ),
            branch="logic",
            notes=(
                "Compresses the expression of rules already consolidated. "
                "Cannot compress what you cannot fully execute."
            ),
        ),

        SkillNode(
            name="chain_reasoning",
            formal_statement="Multi-step deductions combining multiple inference rules.",
            prerequisites=(
                "hypothetical_syllogism",
                "modus_tollens",
                "disjunctive_syllogism",
                "universal_instantiation",
            ),
            train_files=(
                "data/training/phase5_benchmark_failures_base.jsonl",
                "data/training/phase5_benchmark_failures_p1_4.jsonl",
                "data/training/phase6_benchmark_failures_p1_5.jsonl",
            ),
            eval_files=(
                "data/eval/chain_reasoning_eval.jsonl",
            ),
            branch="logic",
            notes=(
                "By definition a composition of component rules. "
                "Phases 5-6 target benchmark failures that require chaining."
            ),
        ),

        # ── Math Branch ───────────────────────────────────────────────────

        SkillNode(
            name="arithmetic_add",
            formal_statement="Given integers A, B, compute C = A + B.",
            prerequisites=(),
            train_files=(
                # TODO: generate data/training/arithmetic_add_train.jsonl
            ),
            eval_files=(
                "data/eval/arithmetic_add_eval.jsonl",
            ),
            branch="math",
            notes="Primitive arithmetic operation. No prerequisites.",
        ),

        SkillNode(
            name="arithmetic_multiply",
            formal_statement="Given integers A, B, compute C = A × B.",
            prerequisites=("arithmetic_add",),
            train_files=(
                "data/training/retention_replay.jsonl",
            ),
            eval_files=(
                "data/eval/arithmetic_multiply_eval.jsonl",
            ),
            branch="math",
            notes=(
                "A × B = A added to itself B times (definition). "
                "retention_replay.jsonl contains multiplication facts."
            ),
        ),

        SkillNode(
            name="arithmetic_divide",
            formal_statement="Given integers A, B (B≠0), compute C = A ÷ B.",
            prerequisites=("arithmetic_multiply",),
            train_files=(
                # TODO: generate data/training/arithmetic_div_train.jsonl
            ),
            eval_files=(
                "data/eval/arithmetic_divide_eval.jsonl",
            ),
            branch="math",
            notes="A ÷ B = C iff C × B = A — checking requires multiplication.",
        ),

        # ── Cross-branch junction ─────────────────────────────────────────

        SkillNode(
            name="word_problem_1step",
            formal_statement=(
                "Given a natural language description, identify the operation "
                "and compute a single-operation answer."
            ),
            prerequisites=("arithmetic_add", "modus_ponens"),
            train_files=(
                # Populated by profile_gsm8k_difficulty.py → easy tier
                # data/training/gsm8k_easy_train.jsonl
            ),
            eval_files=(
                "data/eval/gsm8k_easy_eval.jsonl",
            ),
            branch="math",
            notes=(
                "Cross-branch: requires arithmetic (math) AND logical inference "
                "'context implies operation' (logic). Both branches must be in "
                "reinforce regime before this node."
            ),
        ),

        SkillNode(
            name="word_problem_multi",
            formal_statement=(
                "Given a natural language description requiring multiple operations, "
                "chain the operations and compute the result."
            ),
            prerequisites=("word_problem_1step", "hypothetical_syllogism"),
            train_files=(
                # Populated by profile_gsm8k_difficulty.py → medium/hard tier
                # data/training/gsm8k_medium_train.jsonl
                # data/training/gsm8k_hard_train.jsonl
            ),
            eval_files=(
                "data/eval/gsm8k_medium_eval.jsonl",
                "data/eval/gsm8k_hard_eval.jsonl",
            ),
            branch="math",
            notes=(
                "Structurally a hypothetical syllogism applied to arithmetic: "
                "step_1_result → step_2_input → ... → final_answer."
            ),
        ),

        SkillNode(
            name="algebra_linear",
            formal_statement="Given an equation with one unknown, solve for the unknown.",
            prerequisites=(
                "arithmetic_multiply",
                "arithmetic_divide",
                "modus_tollens",
            ),
            train_files=(
                # MATH dataset (HS level) + NuminaMath easy tier
                # data/training/math_hs_train.jsonl
            ),
            eval_files=(
                "data/eval/math_hs_eval.jsonl",
            ),
            branch="math",
            notes=(
                "Backward reasoning (given result, find cause) is structurally MT. "
                "Solving 3x+5=14: the final state is known; find the premise (x)."
            ),
        ),

        SkillNode(
            name="algebra_nonlinear",
            formal_statement="Competition-level math requiring multi-step algebraic reasoning.",
            prerequisites=("algebra_linear", "chain_reasoning"),
            train_files=(
                # NuminaMath full distribution
                # data/training/numina_train.jsonl
            ),
            eval_files=(
                "data/eval/numina_eval.jsonl",
            ),
            branch="math",
            notes="Requires algebraic fluency (algebra_linear) AND multi-rule chaining.",
        ),
    ]

    return SkillDAG(nodes)


# Module-level singleton — import and use directly.
CURRICULUM_DAG = build_curriculum_dag()
