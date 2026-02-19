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

"""STaR domain primitives."""

from modelcypher.core.domain.star.problem_generator import (
    StarProblem,
    StarProblemGenerator,
    extract_final_answer,
    normalize_text,
    parse_yes_no,
)
from modelcypher.core.domain.star.prompting import (
    FewShotExample,
    build_forward_prompt,
    build_rationalization_prompt,
    default_few_shot_examples,
)

__all__ = [
    "FewShotExample",
    "StarProblem",
    "StarProblemGenerator",
    "build_forward_prompt",
    "build_rationalization_prompt",
    "default_few_shot_examples",
    "extract_final_answer",
    "normalize_text",
    "parse_yes_no",
]
