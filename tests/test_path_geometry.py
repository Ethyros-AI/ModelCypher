from __future__ import annotations

from modelcypher.core.domain.path_geometry import PathGeometry, PathNode, PathSignature


def _simple_embeddings():
    return {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
        "C": [1.0, 1.0],
    }


def test_levenshtein_identical():
    nodes = [
        PathNode(gate_id="A", token_index=0, entropy=0.1),
        PathNode(gate_id="B", token_index=1, entropy=0.2),
    ]
    path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)
    result = PathGeometry.compare(path, path, gate_embeddings=_simple_embeddings())
    assert result.total_distance == 0.0
    assert result.normalized_distance == 0.0


def test_frechet_identical():
    nodes = [
        PathNode(gate_id="A", token_index=0, entropy=0.1),
        PathNode(gate_id="B", token_index=1, entropy=0.2),
        PathNode(gate_id="C", token_index=2, entropy=0.3),
    ]
    path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)
    result = PathGeometry.frechet_distance(path, path, gate_embeddings=_simple_embeddings())
    assert result.distance == 0.0
    assert result.optimal_coupling[0] == (0, 0)


def test_dtw_identical():
    nodes = [
        PathNode(gate_id="A", token_index=0, entropy=0.1),
        PathNode(gate_id="B", token_index=1, entropy=0.2),
        PathNode(gate_id="C", token_index=2, entropy=0.3),
    ]
    path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)
    result = PathGeometry.dynamic_time_warping(path, path, gate_embeddings=_simple_embeddings())
    assert result.total_cost == 0.0
    assert result.normalized_cost == 0.0
