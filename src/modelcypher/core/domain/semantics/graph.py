from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ActivationGraphProjector:
    """
    Projects semantic concept activations into a graph topology to track co-occurrences.
    Nodes: Concept IDs
    Edges: Co-occurrence counts / weights
    """
    
    def __init__(self):
        # Adjacency list: node -> {neighbor: weight}
        self.adjacency: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
    def record_co_occurrence(self, concepts: List[str]):
        """
        Increments edge weights for all pairs of concepts in the list (clique).
        """
        n = len(concepts)
        if n < 2: return
        
        for i in range(n):
            for j in range(i + 1, n):
                u, v = concepts[i], concepts[j]
                # Undirected graph
                self.adjacency[u][v] += 1.0
                self.adjacency[v][u] += 1.0
                
    def get_strongest_connections(self, concept_id: str, k: int = 5) -> List[Tuple[str, float]]:
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
        if V < 2: return 0.0
        
        E = sum(len(neighbors) for neighbors in self.adjacency.values()) / 2
        return (2 * E) / (V * (V - 1))
