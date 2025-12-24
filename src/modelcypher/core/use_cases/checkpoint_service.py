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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.exporter import Exporter
    from modelcypher.ports.storage import JobStore


class CheckpointService:
    def __init__(self, store: "JobStore", exporter: "Exporter") -> None:
        self.store = store
        self.exporter = exporter

    def list_checkpoints(self, job_id: str | None = None) -> dict:
        checkpoints = self.store.list_checkpoints(job_id)
        return {
            "checkpoints": [
                {"jobId": c.job_id, "step": c.step, "loss": c.loss, "filePath": c.file_path}
                for c in checkpoints
            ]
        }

    def delete_checkpoint(self, path: str) -> dict:
        self.store.delete_checkpoint(path)
        return {"deleted": path}

    def export_checkpoint(self, checkpoint_path: str, export_format: str, output_path: str) -> dict:
        result = self.exporter.export_checkpoint(checkpoint_path, output_path, export_format)
        return {
            "checkpoint": checkpoint_path,
            "format": export_format,
            "outputPath": result["outputPath"],
        }
