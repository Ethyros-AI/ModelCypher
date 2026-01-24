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

"""
Real-time 3D visualization of manifold geometry.

This module provides interactive visualization of:
- Activation point clouds in 3D projected space
- Curvature-colored points (red=wall, blue=funnel)
- Density-sized markers (denser regions = smaller markers)
- Animated token trajectories through concept space

Projection uses Gram transport and curvature values are derived from
Ollivier-Ricci estimates.

Requires: plotly>=5.18.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.density_estimator import DensityEstimator
from modelcypher.core.domain.geometry.dimension_cascade import CascadeResult
from modelcypher.core.support.array_utils import array_to_list

if TYPE_CHECKING:
    from modelcypher.core.domain.inference.activation_stream import ActivationFrame
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _ensure_plotly():
    """Lazily import plotly and raise helpful error if not installed."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: poetry add plotly>=5.18.0"
        )


@dataclass
class VisualizationResult:
    """Result of creating a manifold visualization.

    Attributes:
        figure: Plotly Figure object
        html: HTML string representation
        json_data: JSON-serializable data for external rendering
        point_count: Number of points in the visualization
        trajectory_length: Number of trajectory frames
    """

    figure: Any  # plotly.graph_objects.Figure
    html: str
    json_data: dict[str, Any]
    point_count: int
    trajectory_length: int = 0


class ManifoldViewer:
    """
    Real-time 3D visualization of manifold geometry.

    Creates interactive Plotly visualizations showing:
    - Point cloud of activations in 3D projected space
    - Curvature coloring: Red = walls (positive ORC), Blue = funnels (negative ORC)
    - Size by density: Denser regions have smaller markers
    - Animated token trajectories through the space

    Usage:
        viewer = ManifoldViewer(backend)

        # Create visualization from cascade result
        result = viewer.create_figure(cascade_result)

        # Add token trajectory animation
        result = viewer.add_trajectory(result.figure, frames, cascade_result)

        # Export to HTML
        viewer.export_html(result, "manifold.html")
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        point_size_min: float = 3.0,
        point_size_max: float = 15.0,
        opacity: float = 0.7,
        curvature_colorscale: str = "RdBu_r",
        trajectory_color: str = "gold",
        trajectory_width: float = 4.0,
        show_density_cloud: bool = False,
        animation_duration_ms: int = 100,
        title: str = "Manifold Geometry: 3D Shadow of High-D Concept Space",
    ) -> None:
        """
        Initialize the manifold viewer.

        Args:
            backend: Backend for tensor operations
            point_size_min: Minimum point size (dense regions)
            point_size_max: Maximum point size (sparse regions)
            opacity: Point cloud opacity
            curvature_colorscale: Plotly colorscale for curvature
            trajectory_color: Color for token trajectory
            trajectory_width: Line width for trajectory
            show_density_cloud: Whether to show volumetric density
            animation_duration_ms: Duration per frame in animation
            title: Plot title
        """
        self.backend = backend or get_default_backend()
        self.point_size_min = point_size_min
        self.point_size_max = point_size_max
        self.opacity = opacity
        self.curvature_colorscale = curvature_colorscale
        self.trajectory_color = trajectory_color
        self.trajectory_width = trajectory_width
        self.show_density_cloud = show_density_cloud
        self.animation_duration_ms = animation_duration_ms
        self.title = title
        self._density_estimator = DensityEstimator(self.backend)

    def create_figure(
        self,
        cascade_result: CascadeResult,
        target_dim: int = 3,
    ) -> VisualizationResult:
        """
        Create 3D visualization from cascade result.

        Args:
            cascade_result: Result from DimensionCascade.calibrate()
            target_dim: Target dimension for visualization (default 3)

        Returns:
            VisualizationResult with Plotly figure and metadata

        Raises:
            ValueError: If target_dim not in cascade result
        """
        go = _ensure_plotly()
        b = self.backend

        if target_dim not in cascade_result.projections:
            raise ValueError(
                f"Target dimension {target_dim} not in cascade result. "
                f"Available: {list(cascade_result.projections.keys())}"
            )

        points_nd = cascade_result.projections[target_dim]
        n_points = points_nd.shape[0]

        # Convert to lists for Plotly (no NumPy)
        points = array_to_list(b, points_nd)

        # Get curvature if available
        curvature = None
        if target_dim in cascade_result.curvatures:
            curvature = array_to_list(b, cascade_result.curvatures[target_dim])

        # Compute density for sizing (k derived from data by the estimator)
        density_result = self._density_estimator.compute(points_nd)
        density_arr = density_result.densities
        b.eval(density_arr)

        # Normalize for visualization (presentation layer)
        min_density = b.min(density_arr)
        max_density = b.max(density_arr)
        b.eval(min_density, max_density)
        range_density = max_density - min_density
        from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
        range_eps = division_epsilon(b, range_density)
        if float(b.to_scalar(range_density)) > range_eps:
            density_normalized = (density_arr - min_density) / range_density
            b.eval(density_normalized)
        else:
            density_normalized = b.ones_like(density_arr) * 0.5
            b.eval(density_normalized)
        density = array_to_list(b, density_normalized)

        # Create colors from curvature (or uniform if not available)
        if curvature is not None:
            colors = curvature
            colorbar_title = "Curvature (ORC)"
        else:
            colors = [0.5] * n_points
            colorbar_title = "Uniform"

        # Create sizes from density (inverse: higher density = smaller)
        # Map density [0,1] -> [size_max, size_min]
        size_span = self.point_size_max - self.point_size_min
        sizes_arr = b.full(density_normalized.shape, self.point_size_max) - (
            density_normalized * size_span
        )
        b.eval(sizes_arr)
        sizes = array_to_list(b, sizes_arr)

        # Create hover text
        hover_text = []
        for i in range(n_points):
            text = f"Point {i}"
            if curvature is not None:
                text += f"<br>Curvature: {curvature[i]:.4f}"
            text += f"<br>Density: {density[i]:.4f}"
            hover_text.append(text)

        # Build figure
        fig = go.Figure()

        # Get k-NN neighbor indices for topology visualization.
        neighbor_indices_raw = array_to_list(b, density_result.neighbors)
        neighbor_indices = [
            [int(idx) for idx in row] for row in neighbor_indices_raw
        ]
        neighbor_count = len(neighbor_indices[0]) if neighbor_indices else 0

        # Add point cloud
        if target_dim == 3:
            # Draw k-NN graph edges for neighborhood topology.
            # Each edge connects a point to its k nearest neighbors in the projected space
            edge_x, edge_y, edge_z = [], [], []
            for i in range(n_points):
                for j_idx in range(neighbor_count):
                    j = neighbor_indices[i, j_idx]
                    if j >= n_points:
                        continue  # Skip invalid indices
                    # Add edge as line segment with None separator
                    edge_x.extend([points[i][0], points[j][0], None])
                    edge_y.extend([points[i][1], points[j][1], None])
                    edge_z.extend([points[i][2], points[j][2], None])

            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(
                    color="rgba(100, 100, 100, 0.3)",
                    width=1,
                ),
                hoverinfo="skip",
                name="Manifold Topology (k-NN)",
            ))

            # Then add the actual points with curvature coloring
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=self.curvature_colorscale,
                    opacity=self.opacity,
                    colorbar=dict(title=colorbar_title),
                ),
                hoverinfo="text",
                text=hover_text,
                name="Activations",
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3",
                    aspectmode="data",
                ),
                title=self.title,
            )
        elif target_dim == 2:
            # Draw k-NN graph edges for neighborhood topology.
            edge_x, edge_y = [], []
            for i in range(n_points):
                for j_idx in range(neighbor_count):
                    j = neighbor_indices[i, j_idx]
                    if j >= n_points:
                        continue
                    edge_x.extend([points[i][0], points[j][0], None])
                    edge_y.extend([points[i][1], points[j][1], None])

            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    color="rgba(100, 100, 100, 0.3)",
                    width=1,
                ),
                hoverinfo="skip",
                name="Manifold Topology (k-NN)",
            ))

            # Then add points with curvature coloring
            fig.add_trace(go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=self.curvature_colorscale,
                    opacity=self.opacity,
                    colorbar=dict(title=colorbar_title),
                ),
                hoverinfo="text",
                text=hover_text,
                name="Activations",
            ))

            fig.update_layout(
                xaxis_title="PC1",
                yaxis_title="PC2",
                title=self.title,
            )
        else:
            raise ValueError(f"Visualization only supports 2D and 3D, got {target_dim}D")

        # Add metadata annotation
        fig.add_annotation(
            text=(
                f"Intrinsic dim: {cascade_result.intrinsic_dim:.1f} | "
                f"Ambient dim: {cascade_result.original_dim} | "
                f"Points: {n_points}"
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.1,
            showarrow=False,
            font=dict(size=10),
        )

        # Generate HTML
        html = fig.to_html(include_plotlyjs=True, full_html=True)

        # Generate JSON data for external rendering
        json_data = {
            "points": points.tolist(),
            "curvature": curvature.tolist() if curvature is not None else None,
            "density": density.tolist(),
            "knn_neighbors": neighbor_indices.tolist(),  # k-NN graph topology
            "intrinsic_dim": cascade_result.intrinsic_dim,
            "original_dim": cascade_result.original_dim,
            "target_dim": target_dim,
        }

        return VisualizationResult(
            figure=fig,
            html=html,
            json_data=json_data,
            point_count=n_points,
        )

    def add_trajectory(
        self,
        figure: Any,
        frames: list["ActivationFrame"],
        cascade_result: CascadeResult,
        target_dim: int = 3,
    ) -> VisualizationResult:
        """
        Add animated token trajectory to existing figure.

        Projects each activation frame through the cascade and
        adds animation frames for playback.

        Args:
            figure: Existing Plotly figure
            frames: List of ActivationFrames from ActivationStream
            cascade_result: Result from DimensionCascade.calibrate()
            target_dim: Target dimension (must match figure)

        Returns:
            Updated VisualizationResult with animation
        """
        go = _ensure_plotly()
        b = self.backend

        if not frames:
            return VisualizationResult(
                figure=figure,
                html=figure.to_html(include_plotlyjs=True, full_html=True),
                json_data={},
                point_count=0,
                trajectory_length=0,
            )

        # Get composite coupling for projection
        if target_dim not in cascade_result.couplings:
            # Compute composite from available couplings
            composite = None
            for dim in sorted(cascade_result.couplings.keys(), reverse=True):
                if dim < target_dim:
                    continue
                coupling = cascade_result.couplings[dim]
                if composite is None:
                    composite = coupling
                else:
                    composite = b.matmul(composite, coupling)
                    b.eval(composite)
                if dim == target_dim:
                    break
        else:
            composite = cascade_result.couplings[target_dim]

        # Project each frame
        trajectory = []
        for frame in frames:
            if frame.projected_3d is not None:
                point = array_to_list(b, frame.projected_3d)
            else:
                # Project using composite coupling
                hidden = frame.hidden_state
                if len(hidden.shape) == 1:
                    hidden = hidden[None, :]
                projected = b.matmul(hidden, composite)
                b.eval(projected)
                if len(projected.shape) == 2:
                    projected = projected[0]
                point = array_to_list(b, projected)
            trajectory.append(point)

        # Create animation frames
        animation_frames = []
        for i in range(len(trajectory)):
            traj_so_far = trajectory[:i+1]
            x = [p[0] for p in traj_so_far]
            y = [p[1] for p in traj_so_far]
            z = [p[2] for p in traj_so_far] if target_dim == 3 else None

            if target_dim == 3:
                frame_data = [go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines+markers",
                    line=dict(
                        color=self.trajectory_color,
                        width=self.trajectory_width,
                    ),
                    marker=dict(
                        size=8,
                        color=self.trajectory_color,
                    ),
                    name="Trajectory",
                )]
            else:
                frame_data = [go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line=dict(
                        color=self.trajectory_color,
                        width=self.trajectory_width,
                    ),
                    marker=dict(
                        size=8,
                        color=self.trajectory_color,
                    ),
                    name="Trajectory",
                )]

            animation_frames.append(go.Frame(
                data=frame_data,
                name=f"token_{i}",
            ))

        figure.frames = animation_frames

        # Add play/pause buttons
        figure.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=self.animation_duration_ms,
                                        redraw=True,
                                    ),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Token: ",
                        visible=True,
                        xanchor="right",
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [f"token_{i}"],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(i),
                            method="animate",
                        )
                        for i in range(len(animation_frames))
                    ],
                ),
            ],
        )

        # Generate updated HTML and JSON
        html = figure.to_html(include_plotlyjs=True, full_html=True)
        json_data = {
            "trajectory": [p.tolist() for p in trajectory],
            "frame_count": len(animation_frames),
        }

        return VisualizationResult(
            figure=figure,
            html=html,
            json_data=json_data,
            point_count=len(trajectory),
            trajectory_length=len(animation_frames),
        )

    def export_html(
        self,
        result: VisualizationResult,
        output_path: str | Path,
    ) -> Path:
        """
        Export visualization to HTML file.

        Args:
            result: VisualizationResult from create_figure or add_trajectory
            output_path: Path to output HTML file

        Returns:
            Path to the created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.html, encoding="utf-8")
        logger.info("Exported visualization to %s", output_path)
        return output_path

    def show(self, result: VisualizationResult) -> None:
        """
        Display visualization in browser.

        Args:
            result: VisualizationResult from create_figure or add_trajectory
        """
        result.figure.show()
