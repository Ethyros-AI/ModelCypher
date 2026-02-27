# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from modelcypher.core.domain.star.problem_generator import (
    StarProblem,
    StarProblemGenerator,
)


def test_generates_unique_required_problem_fields() -> None:
    generator = StarProblemGenerator(seed=42)
    problems = generator.generate(500)

    assert len(problems) == 500
    assert len({problem.prompt for problem in problems}) == 500

    required_types = {
        "syllogistic_chain",
        "compositional_binding",
        "multi_step_arithmetic",
        "contrapositive",
        "set_exception",
    }
    generated_types = {problem.problem_type for problem in problems}
    assert required_types.issubset(generated_types)

    for problem in problems:
        record = problem.to_problem_record()
        assert record["prompt"]
        assert record["correct_answer"]
        assert record["problem_type"]
        assert int(record["difficulty"]) > 0
        assert record["verification_fn"]


def test_verifies_programmatically_computed_answers() -> None:
    generator = StarProblemGenerator(seed=123)
    problems = generator.generate(25)

    for problem in problems:
        assert problem.verify_response(f"Final answer: {problem.correct_answer}")


def test_problem_record_round_trip_reconstructs_verifier() -> None:
    generator = StarProblemGenerator(seed=7)
    problems = generator.generate(15)

    for problem in problems:
        rebuilt = StarProblem.from_problem_record(problem.to_problem_record())
        assert rebuilt.problem_id == problem.problem_id
        assert rebuilt.prompt == problem.prompt
        assert rebuilt.problem_type == problem.problem_type
        assert rebuilt.correct_answer == problem.correct_answer
        assert rebuilt.verification_fn == problem.verification_fn
        assert rebuilt.verify_response(f"Final answer: {problem.correct_answer}")
