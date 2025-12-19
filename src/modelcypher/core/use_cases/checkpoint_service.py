from __future__ import annotations

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.local_exporter import LocalExporter


class CheckpointService:
    def __init__(self, store: FileSystemStore | None = None, exporter: LocalExporter | None = None) -> None:
        self.store = store or FileSystemStore()
        self.exporter = exporter or LocalExporter()

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
