from __future__ import annotations

from typing import Protocol

from modelcypher.core.use_cases.geometry_engine import (
    LoRAAdapterGeometryMetrics,
    ProcrustesResult,
    SinkhornResult,
)


class GeometryPort(Protocol):
    def compute_lora_geometry(
        self,
        trainable_parameters: dict[str, object],
        previous_trainable_parameters: dict[str, object] | None,
        scale: float,
    ) -> LoRAAdapterGeometryMetrics: ...

    def orthogonal_procrustes(
        self,
        source_anchors: object,
        target_anchors: object,
        source_basis: object,
        target_basis: object,
        anchor_weights: list[float] | None = None,
    ) -> ProcrustesResult: ...

    def soft_procrustes_alignment(
        self,
        source_anchors: object,
        target_anchors: object,
        source_basis: object,
        target_basis: object,
    ) -> tuple[object, object, float, SinkhornResult]: ...
