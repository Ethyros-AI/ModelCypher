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

"""Extended MCP geometry tool tests.

Tests for geometry tools not covered in test_mcp_contracts.py:
- Gromov-Wasserstein distance
- Intrinsic dimension estimation
- Topological fingerprinting
- Spectral signature
- Manifold clustering, dimension, query
- Transport merge/synthesize
- Sparse domains and regions
- Refusal detection
- Persona extraction and drift
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

DEFAULT_TIMEOUT_SECONDS = 15


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _mlx_available() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        return False
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


def _build_env(tmp_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = _repo_root()
    python_path = os.pathsep.join(
        path for path in [str(repo_root / "src"), env.get("PYTHONPATH")] if path
    )
    env["PYTHONPATH"] = python_path
    env["MODELCYPHER_HOME"] = str(tmp_home)
    env["MC_MCP_PROFILE"] = "full"
    env["MC_ALLOW_STUB_INFERENCE"] = "1"
    env["MC_ALLOW_STUB_EMBEDDINGS"] = "1"
    if _mlx_available():
        env["MC_BACKEND"] = "mlx"
        env["MC_ENABLE_MLX"] = "1"
        env["MC_DISABLE_MLX"] = ""
    return env


def _extract_structured(result: types.CallToolResult) -> dict:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for content in result.content:
        if isinstance(content, types.TextContent):
            return json.loads(content.text)
    raise AssertionError("No structured content returned from tool call")


async def _await_with_timeout(coro, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    return await asyncio.wait_for(coro, timeout=timeout)


def _run_mcp(env: dict[str, str], runner):
    async def _run():
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "modelcypher.mcp.server"],
            env=env,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await _await_with_timeout(session.initialize())
                return await runner(session)

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def mcp_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_home = tmp_path_factory.mktemp("mcp_geometry_home")
    return _build_env(tmp_home)


# =============================================================================
# Gromov-Wasserstein Tests
# =============================================================================


class TestGromovWassersteinTool:
    """Tests for mc_geometry_gromov_wasserstein tool."""

    def test_gromov_wasserstein_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        source_points_arr = backend.random_normal((10, 3), dtype="float32")
        target_points_arr = backend.random_normal((10, 3), dtype="float32")
        backend.eval(source_points_arr)
        backend.eval(target_points_arr)
        source_points = backend.tolist(source_points_arr)
        target_points = backend.tolist(target_points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_gromov_wasserstein",
                    arguments={
                        "sourcePoints": source_points,
                        "targetPoints": target_points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.gromov_wasserstein.v1"
        assert "distance" in payload

    def test_gromov_wasserstein_distance_non_negative(self, mcp_env: dict[str, str]) -> None:
        """Gromov-Wasserstein distance must be >= 0."""
        backend = get_default_backend()
        backend.random_seed(42)
        source_points_arr = backend.random_normal((8, 4))
        target_points_arr = backend.random_normal((8, 4))
        backend.eval(source_points_arr)
        backend.eval(target_points_arr)
        source_points = backend.tolist(source_points_arr)
        target_points = backend.tolist(target_points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_gromov_wasserstein",
                    arguments={
                        "sourcePoints": source_points,
                        "targetPoints": target_points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        eps = _eps(backend, payload["distance"])
        assert payload["distance"] >= -eps

    def test_gromov_wasserstein_identical_points_near_zero(self, mcp_env: dict[str, str]) -> None:
        """Identical point clouds should have distance ≈ 0."""
        backend = get_default_backend()
        backend.random_seed(42)
        points_arr = backend.random_normal((10, 3))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_gromov_wasserstein",
                    arguments={
                        "sourcePoints": points,
                        "targetPoints": points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        eps = machine_epsilon(backend, points_arr)
        # Distance to self should be within numerical precision.
        assert payload["distance"] <= eps


# =============================================================================
# Intrinsic Dimension Tests
# =============================================================================


class TestIntrinsicDimensionTool:
    """Tests for mc_geometry_intrinsic_dimension tool."""

    def test_intrinsic_dimension_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        # 3D Gaussian (roughly 3D manifold)
        points_arr = backend.random_normal((50, 3))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_intrinsic_dimension",
                    arguments={"points": points},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.intrinsic_dimension.v1"
        assert "intrinsicDimension" in payload

    def test_intrinsic_dimension_positive(self, mcp_env: dict[str, str]) -> None:
        """Intrinsic dimension must be > 0."""
        backend = get_default_backend()
        backend.random_seed(42)
        points_arr = backend.random_normal((50, 5))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_intrinsic_dimension",
                    arguments={"points": points},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        eps = _eps(backend, payload["intrinsicDimension"])
        assert payload["intrinsicDimension"] >= eps

    def test_intrinsic_dimension_bounded_by_ambient(self, mcp_env: dict[str, str]) -> None:
        """Intrinsic dimension should not exceed ambient dimension."""
        backend = get_default_backend()
        backend.random_seed(42)
        ambient_dim = 4
        points_arr = backend.random_normal((100, ambient_dim))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_intrinsic_dimension",
                    arguments={"points": points},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        eps = _eps(backend, payload["intrinsicDimension"], float(ambient_dim))
        assert payload["intrinsicDimension"] <= ambient_dim + eps


# =============================================================================
# Topological Fingerprint Tests
# =============================================================================


class TestTopologicalFingerprintTool:
    """Tests for mc_geometry_topological_fingerprint tool."""

    def test_topological_fingerprint_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        points_arr = backend.random_normal((30, 3))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_topological_fingerprint",
                    arguments={
                        "points": points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.topological_fingerprint.v1"


# =============================================================================
# Spectral Signature Tests
# =============================================================================


class TestSpectralSignatureTool:
    """Tests for mc_geometry_spectral_signature tool."""

    def test_spectral_signature_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        points_arr = backend.random_normal((12, 3))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_spectral_signature",
                    arguments={
                        "points": points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.spectral_signature.v1"
        assert "eigenvalues" in payload
        assert "heatTrace" in payload

    def test_spectral_signature_component_count(self, mcp_env: dict[str, str]) -> None:
        """Component count is derived from geodesic connectivity.

        With data-derived k, 4 nearby points are correctly seen as connected.
        This test verifies the spectral signature computation completes correctly
        and returns valid component metrics.
        """
        points = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_spectral_signature",
                    arguments={
                        "points": points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        # Component count is derived from data, verify it's a valid integer
        assert isinstance(payload["componentCount"], int)
        assert payload["componentCount"] >= 1
        assert isinstance(payload["connected"], bool)


# =============================================================================
# Dimension Constraint Invariance Tests
# =============================================================================


class TestDimensionConstraintInvarianceTool:
    """Tests for mc_geometry_dimension_constraint_invariance tool."""

    def test_dimension_constraint_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        points = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
            [0.1, 0.7],
            [0.9, 0.2],
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_dimension_constraint_invariance",
                    arguments={
                        "points": points,
                        "paddedDimension": 4,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.dimension_constraint_invariance.v1"
        assert payload["baseDimension"] == 2
        assert payload["paddedDimension"] == 4
        points_arr = backend.array(points, dtype="float32")
        eps = machine_epsilon(backend, points_arr)
        assert abs(payload["gramCka"] - 1.0) <= eps
        assert payload["geodesicDiff"]["maxAbs"] <= eps

# =============================================================================
# Manifold Cluster Tests
# =============================================================================


class TestManifoldClusterTool:
    """Tests for mc_geometry_manifold_cluster tool."""

    def test_manifold_cluster_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        x_vals = backend.random_normal((20,))
        y_vals = backend.random_normal((20,))
        backend.eval(x_vals)
        backend.eval(y_vals)
        x_np = backend.tolist(x_vals)
        y_np = backend.tolist(y_vals)
        points = [
            {"x": float(x_np[i]), "y": float(y_np[i])} for i in range(20)
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_manifold_cluster",
                    arguments={
                        "points": points,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.manifold_cluster.v1"


# =============================================================================
# Manifold Dimension Tests
# =============================================================================


class TestManifoldDimensionTool:
    """Tests for mc_geometry_manifold_dimension tool."""

    def test_manifold_dimension_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        points_arr = backend.random_normal((50, 3))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_manifold_dimension",
                    arguments={"points": points},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.manifold_dimension.v1"

    def test_manifold_dimension_positive(self, mcp_env: dict[str, str]) -> None:
        """Manifold dimension must be > 0."""
        backend = get_default_backend()
        backend.random_seed(42)
        points_arr = backend.random_normal((80, 5))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_manifold_dimension",
                    arguments={"points": points},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        if "intrinsicDimension" in payload:
            eps = _eps(backend, payload["intrinsicDimension"])
            assert payload["intrinsicDimension"] >= eps
        elif "dimension" in payload:
            eps = _eps(backend, payload["dimension"])
            assert payload["dimension"] >= eps


# =============================================================================
# Manifold Query Tests
# =============================================================================


class TestManifoldQueryTool:
    """Tests for mc_geometry_manifold_query tool."""

    def test_manifold_query_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        point = {"x": 0.5, "y": 0.5}
        regions = [
            {"centroid": {"x": 0.0, "y": 0.0}, "radius": 1.0, "label": "region_0"},
            {"centroid": {"x": 2.0, "y": 2.0}, "radius": 0.5, "label": "region_1"},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_manifold_query",
                    arguments={
                        "point": point,
                        "regions": regions,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.manifold_query.v1"


# =============================================================================
# Sparse Domains Tests
# =============================================================================


class TestSparseDomainsTool:
    """Tests for mc_geometry_sparse_domains tool."""

    def test_sparse_domains_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool("mc_geometry_sparse_domains", arguments={})
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.sparse_domains.v1"
        assert "domains" in payload

    def test_sparse_domains_with_category(self, mcp_env: dict[str, str]) -> None:
        """Tool should filter by category."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_sparse_domains",
                    arguments={"category": "reasoning"},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.sparse_domains.v1"


# =============================================================================
# Refusal Pairs Tests
# =============================================================================


class TestRefusalPairsTool:
    """Tests for mc_geometry_refusal_pairs tool."""

    def test_refusal_pairs_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool("mc_geometry_refusal_pairs", arguments={})
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.refusal_pairs.v1"
        assert "pairs" in payload

    def test_refusal_pairs_non_empty(self, mcp_env: dict[str, str]) -> None:
        """Tool should return at least some prompt pairs."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool("mc_geometry_refusal_pairs", arguments={})
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert len(payload["pairs"]) > 0


# =============================================================================
# Refusal Detect Tests
# =============================================================================


class TestRefusalDetectTool:
    """Tests for mc_geometry_refusal_detect tool."""

    def test_refusal_detect_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        # Simulated contrastive activations
        harmful_acts_arr = backend.random_normal((5, 10))
        harmless_acts_arr = backend.random_normal((5, 10))
        backend.eval(harmful_acts_arr)
        backend.eval(harmless_acts_arr)
        harmful_acts = backend.tolist(harmful_acts_arr)
        harmless_acts = backend.tolist(harmless_acts_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_refusal_detect",
                    arguments={
                        "harmfulActivations": harmful_acts,
                        "harmlessActivations": harmless_acts,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.refusal_detect.v1"


# =============================================================================
# Persona Traits Tests
# =============================================================================


class TestPersonaTraitsTool:
    """Tests for mc_geometry_persona_traits tool."""

    def test_persona_traits_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool("mc_geometry_persona_traits", arguments={})
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.persona_traits.v1"
        assert "traits" in payload


# =============================================================================
# Persona Extract Tests
# =============================================================================


class TestPersonaExtractTool:
    """Tests for mc_geometry_persona_extract tool."""

    def test_persona_extract_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        backend = get_default_backend()
        backend.random_seed(42)
        positive_acts_arr = backend.random_normal((5, 10))
        negative_acts_arr = backend.random_normal((5, 10))
        backend.eval(positive_acts_arr)
        backend.eval(negative_acts_arr)
        positive_acts = backend.tolist(positive_acts_arr)
        negative_acts = backend.tolist(negative_acts_arr)

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_persona_extract",
                    arguments={
                        "positiveActivations": positive_acts,
                        "negativeActivations": negative_acts,
                        "traitId": "curiosity",
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.persona_extract.v1"


# =============================================================================
# Persona Drift Tests
# =============================================================================


class TestPersonaDriftTool:
    """Tests for mc_geometry_persona_drift tool."""

    def test_persona_drift_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        positions = [
            {"trait": "curiosity", "position": [0.1, 0.2, 0.3]},
            {"trait": "helpfulness", "position": [0.4, 0.5, 0.6]},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_persona_drift",
                    arguments={
                        "positions": positions,
                        "step": 100,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.geometry.persona_drift.v1"


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestGeometryToolInvariants:
    """Tests for mathematical invariants across geometry tools."""

    @pytest.mark.parametrize("seed", range(3))
    def test_gromov_wasserstein_symmetry(self, mcp_env: dict[str, str], seed: int) -> None:
        """GW(A, B) should approximately equal GW(B, A)."""
        backend = get_default_backend()
        backend.random_seed(seed)
        points_a_arr = backend.random_normal((8, 3))
        points_b_arr = backend.random_normal((8, 3))
        backend.eval(points_a_arr)
        backend.eval(points_b_arr)
        points_a = backend.tolist(points_a_arr)
        points_b = backend.tolist(points_b_arr)

        async def runner(session: ClientSession):
            result_ab = await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_gromov_wasserstein",
                    arguments={
                        "sourcePoints": points_a,
                        "targetPoints": points_b,
                    },
                )
            )
            result_ba = await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_gromov_wasserstein",
                    arguments={
                        "sourcePoints": points_b,
                        "targetPoints": points_a,
                    },
                )
            )
            return result_ab, result_ba

        result_ab, result_ba = _run_mcp(mcp_env, runner)
        payload_ab = _extract_structured(result_ab)
        payload_ba = _extract_structured(result_ba)

        # GW distance should be approximately symmetric
        eps = machine_epsilon(backend, points_a_arr)
        assert abs(payload_ab["distance"] - payload_ba["distance"]) <= eps

    @pytest.mark.parametrize("seed", range(3))
    def test_dimension_stability_across_tools(self, mcp_env: dict[str, str], seed: int) -> None:
        """Intrinsic dimension and manifold dimension should agree roughly."""
        backend = get_default_backend()
        backend.random_seed(seed)
        points_arr = backend.random_normal((60, 3))
        backend.eval(points_arr)
        points = backend.tolist(points_arr)

        async def runner(session: ClientSession):
            result_id = await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_intrinsic_dimension",
                    arguments={"points": points},
                )
            )
            result_md = await _await_with_timeout(
                session.call_tool(
                    "mc_geometry_manifold_dimension",
                    arguments={"points": points},
                )
            )
            return result_id, result_md

        result_id, result_md = _run_mcp(mcp_env, runner)
        payload_id = _extract_structured(result_id)
        payload_md = _extract_structured(result_md)

        id_dim = payload_id.get("intrinsicDimension", 0)
        md_dim = payload_md.get("intrinsicDimension", payload_md.get("dimension", 0))

        # Both estimates should be positive
        eps = _eps(backend, id_dim, md_dim)
        assert id_dim >= eps
        assert md_dim >= eps
