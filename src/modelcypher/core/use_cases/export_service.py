from __future__ import annotations

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.adapters.local_exporter import LocalExporter


class ExportService:
    def __init__(self, store: FileSystemStore | None = None, exporter: LocalExporter | None = None) -> None:
        self.store = store or FileSystemStore()
        self.exporter = exporter or LocalExporter()

    def export_model(self, model_id: str, export_format: str, output_path: str) -> dict:
        model = self.store.get_model(model_id)
        if model is None:
            raise RuntimeError(f"Model not found: {model_id}")
        result = self.exporter.export_model(model.path, output_path, export_format)
        return {"modelID": model.id, "format": export_format, "outputPath": result["outputPath"]}

    def export_job(self, job_id: str, export_format: str, output_path: str) -> dict:
        checkpoints = self.store.list_checkpoints(job_id)
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found for job: {job_id}")
        latest = sorted(checkpoints, key=lambda c: c.step)[-1]
        result = self.exporter.export_checkpoint(latest.file_path, output_path, export_format)
        return {"modelID": job_id, "format": export_format, "outputPath": result["outputPath"]}
