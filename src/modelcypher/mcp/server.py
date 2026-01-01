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

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

READ_ONLY_ANNOTATIONS = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

mcp = FastMCP("ModelCypher")
service = GeometryMetricsService()


def _load_points(path: str) -> list[list[float]]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        return [raw[key] for key in sorted(raw.keys())]
    return raw


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
def mc_geometry_gromov_wasserstein(source_file: str, target_file: str) -> dict:
    """Compute Gromov-Wasserstein distance between two point clouds."""
    source_points = _load_points(source_file)
    target_points = _load_points(target_file)
    result = service.compute_gromov_wasserstein(source_points=source_points, target_points=target_points)
    payload = service.gromov_wasserstein_payload(result)
    payload["_schema"] = "mc.geometry.gromov_wasserstein.v1"
    return payload


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
def mc_geometry_intrinsic_dimension(points_file: str) -> dict:
    """Estimate intrinsic dimension of a point cloud using TwoNN."""
    points = _load_points(points_file)
    result = service.estimate_intrinsic_dimension(points=points)
    payload = service.intrinsic_dimension_payload(result)
    payload["_schema"] = "mc.geometry.intrinsic_dimension.v1"
    return payload


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
def mc_geometry_topological_fingerprint(points_file: str) -> dict:
    """Compute topological fingerprint using persistent homology."""
    points = _load_points(points_file)
    result = service.compute_topological_fingerprint(points=points)
    payload = service.topological_fingerprint_payload(result)
    payload["_schema"] = "mc.geometry.topological_fingerprint.v1"
    return payload


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
def mc_geometry_spectral_signature(points_file: str) -> dict:
    """Compute spectral signature of a point cloud."""
    points = _load_points(points_file)
    result = service.compute_spectral_signature(points=points)
    payload = service.spectral_signature_payload(result)
    payload["_schema"] = "mc.geometry.spectral_signature.v1"
    return payload


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
