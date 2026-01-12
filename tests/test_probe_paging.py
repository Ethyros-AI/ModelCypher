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

from modelcypher.adapters.activation_store import NPZActivationStore
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.merge.stages.probe_activation_storage import (
    PagedActivations,
    _page_activation_space,
)


def test_page_activation_space_round_trip(tmp_path) -> None:
    backend = get_default_backend()
    store = NPZActivationStore()

    activations = {
        0: backend.array([[1.0, 2.0], [3.0, 4.0]]),
        1: backend.array([[5.0, 6.0], [7.0, 8.0]]),
    }

    paged = _page_activation_space(
        activation_store=store,
        base_dir=tmp_path,
        prefix="hidden",
        activations=activations,
        backend=backend,
        cache_size=1,
    )

    assert activations == {}
    assert isinstance(paged, PagedActivations)
    assert 0 in paged
    assert 1 in paged

    first = paged[0]
    second = paged.get(1)
    assert backend.shape(first)[0] == 2
    assert backend.shape(second)[0] == 2

    assert backend.tolist(first) == [[1.0, 2.0], [3.0, 4.0]]
    assert backend.tolist(second) == [[5.0, 6.0], [7.0, 8.0]]

    paged.clear()
