from __future__ import annotations

from typing import Protocol


class Exporter(Protocol):
    def export_model(self, model_path: str, output_path: str, export_format: str) -> dict: ...
    def export_checkpoint(self, checkpoint_path: str, output_path: str, export_format: str) -> dict: ...
