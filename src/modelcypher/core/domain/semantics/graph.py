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


from collections import defaultdict


class ActivationGraphProjector:
    """
    Projects semantic concept activations into a graph topology to track co-occurrences.
    Nodes: Concept IDs
    Edges: Co-occurrence counts / weights
    """

    def __init__(self):
        # Adjacency list: node -> {neighbor: weight}
        self.adjacency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def record_co_occurrence(self, concepts: list[str]):
        """
        Increments edge weights for all pairs of concepts in the list (clique).
        """
        n = len(concepts)
        if n < 2:
            return

        for i in range(n):
            for j in range(i + 1, n):
                u, v = concepts[i], concepts[j]
                # Undirected graph
                self.adjacency[u][v] += 1.0
                self.adjacency[v][u] += 1.0

    def get_strongest_connections(self, concept_id: str, k: int = 5) -> list[tuple[str, float]]:
        if concept_id not in self.adjacency:
            return []

        neighbors = self.adjacency[concept_id]
        sorted_neighbors = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)
        return sorted_neighbors[:k]

    def get_density(self) -> float:
        """
        Graph density: 2|E| / (|V| * (|V|-1))
        """
        nodes = set(self.adjacency.keys())
        V = len(nodes)
        if V < 2:
            return 0.0

        E = sum(len(neighbors) for neighbors in self.adjacency.values()) / 2
        return (2 * E) / (V * (V - 1))
