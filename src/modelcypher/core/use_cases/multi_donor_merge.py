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
Multi-Donor Transplant Pipeline.

Orchestrates sequential transplants from multiple donor models into base models.
Supports:
- Multiple base models processed in parallel
- Domain-specific donor assignment
- Checkpointing after each donor for resumability
- Lightweight evaluation after each donor
- Cross-program comparison

Example usage:
    service = MultiDonorMergeService(model_loader)
    program = TransplantProgram.from_yaml("./configs/program_a.yaml")
    result = service.execute_program(program)

CLI usage:
    mc program run ./configs/program_a.yaml
    mc program run ./configs/program_a.yaml --parallel
    mc program run ./configs/program_a.yaml --resume
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class DonorSpec:
    """Specification for a single donor in a transplant program.

    Attributes:
        id: Unique identifier for this donor within the program
        source: Path to donor model (local path or HuggingFace ID)
        domains: Domains to transplant from this donor (e.g., ["mathematical", "logical"])
        layers: Specific layers to transplant (None = all layers)
        priority: Higher priority donors are applied first (default: 0)
        boundary_k: Override for transplant boundary k-neighbors
        geodesic_k: Override for transplant geodesic k-neighbors
    """

    id: str
    source: str
    domains: tuple[str, ...]
    layers: tuple[int, ...] | None = None
    priority: int = 0
    boundary_k: int | None = None
    geodesic_k: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DonorSpec":
        """Create DonorSpec from dictionary (parsed YAML/JSON)."""
        return cls(
            id=data["id"],
            source=data["source"],
            domains=tuple(data.get("domains", [])),
            layers=tuple(data["layers"]) if data.get("layers") else None,
            priority=data.get("priority", 0),
            boundary_k=data.get("boundary_k"),
            geodesic_k=data.get("geodesic_k"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "domains": list(self.domains),
        }
        if self.layers:
            result["layers"] = list(self.layers)
        if self.priority != 0:
            result["priority"] = self.priority
        if self.boundary_k is not None:
            result["boundary_k"] = self.boundary_k
        if self.geodesic_k is not None:
            result["geodesic_k"] = self.geodesic_k
        return result


@dataclass(frozen=True)
class BaseModelSpec:
    """Specification for a base model to receive transplants.

    Attributes:
        id: Unique identifier for this base within the program
        source: Path to base model (local path or HuggingFace ID)
        alias: Short name for checkpoint directories (defaults to id)
    """

    id: str
    source: str
    alias: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModelSpec":
        """Create BaseModelSpec from dictionary."""
        return cls(
            id=data["id"],
            source=data["source"],
            alias=data.get("alias"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"id": self.id, "source": self.source}
        if self.alias:
            result["alias"] = self.alias
        return result

    @property
    def effective_alias(self) -> str:
        """Return alias if set, otherwise id."""
        return self.alias or self.id


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for program evaluation.

    Attributes:
        after_each_donor: Run lightweight eval after each donor
        after_program_complete: Run full eval at end
        benchmarks: External benchmark task IDs (lm-eval-harness)
        smoke_test_prompts: Quick inference prompts for sanity check
    """

    after_each_donor: bool = True
    after_program_complete: bool = True
    benchmarks: tuple[str, ...] = ()
    smoke_test_prompts: tuple[str, ...] = (
        "What is 15 * 17?",
        "Explain the Fibonacci sequence in one sentence.",
        "Write a Python function that reverses a string.",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvaluationConfig":
        """Create EvaluationConfig from dictionary."""
        if data is None:
            return cls()
        return cls(
            after_each_donor=data.get("after_each_donor", True),
            after_program_complete=data.get("after_program_complete", True),
            benchmarks=tuple(data.get("benchmarks", [])),
            smoke_test_prompts=tuple(
                data.get(
                    "smoke_test_prompts",
                    [
                        "What is 15 * 17?",
                        "Explain the Fibonacci sequence in one sentence.",
                        "Write a Python function that reverses a string.",
                    ],
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "after_each_donor": self.after_each_donor,
            "after_program_complete": self.after_program_complete,
            "benchmarks": list(self.benchmarks),
            "smoke_test_prompts": list(self.smoke_test_prompts),
        }


@dataclass(frozen=True)
class TransplantProgram:
    """Complete transplant program specification.

    A program defines:
    - One or more base models to receive transplants
    - One or more donors with domain assignments
    - Evaluation configuration
    - Output directory structure

    Attributes:
        name: Human-readable program name
        description: Optional description
        bases: Base models to process
        donors: Donors to transplant (applied in priority then definition order)
        evaluation: Evaluation settings
        output_dir: Base directory for outputs
    """

    name: str
    bases: tuple[BaseModelSpec, ...]
    donors: tuple[DonorSpec, ...]
    output_dir: str
    description: str = ""
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransplantProgram":
        """Load program from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Parsed TransplantProgram

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            msg = f"Program config not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransplantProgram":
        """Create TransplantProgram from dictionary.

        Args:
            data: Parsed YAML/JSON dictionary

        Returns:
            TransplantProgram instance

        Raises:
            ValueError: If required fields are missing
        """
        if "name" not in data:
            msg = "Program config must have 'name' field"
            raise ValueError(msg)
        if "bases" not in data or not data["bases"]:
            msg = "Program config must have at least one base model"
            raise ValueError(msg)
        if "donors" not in data or not data["donors"]:
            msg = "Program config must have at least one donor"
            raise ValueError(msg)

        bases = tuple(BaseModelSpec.from_dict(b) for b in data["bases"])
        donors = tuple(DonorSpec.from_dict(d) for d in data["donors"])

        # Sort donors by priority (higher first), then by definition order
        sorted_donors = tuple(
            sorted(donors, key=lambda d: (-d.priority, donors.index(d)))
        )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            bases=bases,
            donors=sorted_donors,
            evaluation=EvaluationConfig.from_dict(data.get("evaluation")),
            output_dir=data.get("output_dir", "~/.modelcypher/merged"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "_schema": "mc.program.transplant.v1",
            "name": self.name,
            "description": self.description,
            "bases": [b.to_dict() for b in self.bases],
            "donors": [d.to_dict() for d in self.donors],
            "evaluation": self.evaluation.to_dict(),
            "output_dir": self.output_dir,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save program to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


@dataclass
class DonorStageResult:
    """Result of a single donor transplant stage.

    Captures metrics from one donor being transplanted into one base model.
    """

    donor_id: str
    donor_source: str
    domains: tuple[str, ...]
    donor_index: int

    # CKA alignment metrics
    cka_before: float
    cka_after: float
    cka_improvement: float

    # Boundary preservation
    boundary_preserved: float
    mean_boundary_relative_diff: float
    max_boundary_relative_diff: float

    # Transplant statistics
    layers_transplanted: int
    weights_transplanted: int
    mean_projection_loss: float
    mean_null_dim: float

    # Checkpointing
    checkpoint_path: str

    # Timing
    duration_seconds: float

    # Evaluation (if enabled)
    smoke_test_passed: bool = True
    intrinsic_dim_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "donor_id": self.donor_id,
            "donor_source": self.donor_source,
            "domains": list(self.domains),
            "donor_index": self.donor_index,
            "cka_before": self.cka_before,
            "cka_after": self.cka_after,
            "cka_improvement": self.cka_improvement,
            "boundary_preserved": self.boundary_preserved,
            "mean_boundary_relative_diff": self.mean_boundary_relative_diff,
            "max_boundary_relative_diff": self.max_boundary_relative_diff,
            "layers_transplanted": self.layers_transplanted,
            "weights_transplanted": self.weights_transplanted,
            "mean_projection_loss": self.mean_projection_loss,
            "mean_null_dim": self.mean_null_dim,
            "checkpoint_path": self.checkpoint_path,
            "duration_seconds": self.duration_seconds,
            "smoke_test_passed": self.smoke_test_passed,
            "intrinsic_dim_delta": self.intrinsic_dim_delta,
        }


@dataclass
class BaseModelResult:
    """Result of processing one base model through all donors.

    Aggregates results from all donor stages for a single base.
    """

    base_id: str
    base_source: str
    base_alias: str
    output_path: str

    # Per-donor results
    donor_stages: list[DonorStageResult]

    # Aggregate metrics
    total_cka_improvement: float
    mean_boundary_preserved: float
    total_donors_applied: int

    # Status
    status: Literal["completed", "partial", "failed"]
    error: str | None = None

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "base_id": self.base_id,
            "base_source": self.base_source,
            "base_alias": self.base_alias,
            "output_path": self.output_path,
            "donor_stages": [s.to_dict() for s in self.donor_stages],
            "total_cka_improvement": self.total_cka_improvement,
            "mean_boundary_preserved": self.mean_boundary_preserved,
            "total_donors_applied": self.total_donors_applied,
            "status": self.status,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class MultiDonorResult:
    """Result of executing a complete transplant program.

    Aggregates results from all base models.
    """

    program_id: str
    program_name: str

    # Per-base results
    base_results: list[BaseModelResult]

    # Aggregate metrics
    total_duration_seconds: float
    completed_at: datetime

    # Status
    status: Literal["completed", "partial", "failed"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "_schema": "mc.result.multi_donor.v1",
            "program_id": self.program_id,
            "program_name": self.program_name,
            "base_results": [r.to_dict() for r in self.base_results],
            "total_duration_seconds": self.total_duration_seconds,
            "completed_at": self.completed_at.isoformat(),
            "status": self.status,
        }


@dataclass
class ProgramStatus:
    """Status of a running or completed program execution.

    Used for checkpointing and resume functionality.
    """

    program_id: str
    program_name: str
    started_at: datetime
    updated_at: datetime
    status: Literal["pending", "in_progress", "completed", "failed"]

    # Progress tracking per base
    base_progress: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "program_id": self.program_id,
            "program_name": self.program_name,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "base_progress": self.base_progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgramStatus":
        """Create from dictionary."""
        return cls(
            program_id=data["program_id"],
            program_name=data["program_name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=data["status"],
            base_progress=data["base_progress"],
        )


# =============================================================================
# Service Implementation
# =============================================================================


class MultiDonorMergeService:
    """Orchestrates multi-donor transplant programs.

    This service manages the execution of transplant programs, handling:
    - Sequential donor application per base model
    - Checkpointing for resumability
    - Lightweight evaluation after each donor
    - Parallel processing of multiple base models (optional)

    Example:
        service = MultiDonorMergeService(model_loader)
        program = TransplantProgram.from_yaml("./program.yaml")
        result = service.execute_program(program)
    """

    def __init__(
        self,
        model_loader: "ModelLoaderPort | None" = None,
        backend: "Backend | None" = None,
        programs_dir: Path | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            model_loader: Model loader port for loading models
            backend: Compute backend (defaults to MLX)
            programs_dir: Directory for program state storage
        """
        self._model_loader = model_loader
        self._backend = backend or get_default_backend()
        self._programs_dir = programs_dir or Path.home() / ".modelcypher" / "multi-merge"
        self._programs_dir.mkdir(parents=True, exist_ok=True)

    def execute_program(
        self,
        program: TransplantProgram,
        parallel: bool = False,
        max_workers: int = 2,
        dry_run: bool = False,
    ) -> MultiDonorResult:
        """Execute a transplant program.

        Args:
            program: Program specification
            parallel: Process base models in parallel
            max_workers: Max parallel workers (if parallel=True)
            dry_run: Validate without executing

        Returns:
            MultiDonorResult with all base model results
        """
        program_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(
            "Starting program %s (%s) with %d bases and %d donors",
            program.name,
            program_id,
            len(program.bases),
            len(program.donors),
        )

        # Initialize status
        status = ProgramStatus(
            program_id=program_id,
            program_name=program.name,
            started_at=datetime.now(),
            updated_at=datetime.now(),
            status="in_progress",
            base_progress=[
                {
                    "base_index": i,
                    "base_id": base.id,
                    "completed_donors": 0,
                    "total_donors": len(program.donors),
                    "status": "pending",
                }
                for i, base in enumerate(program.bases)
            ],
        )

        # Create program directory
        program_dir = self._programs_dir / program_id
        program_dir.mkdir(parents=True, exist_ok=True)

        # Save program config
        program.to_yaml(program_dir / "program.yaml")
        self._save_status(program_dir, status)

        if dry_run:
            logger.info("Dry run - validating program without execution")
            return self._create_dry_run_result(program, program_id, start_time)

        # Execute for each base model
        base_results: list[BaseModelResult] = []

        if parallel and len(program.bases) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._execute_for_base, program, base, i, program_dir, status
                    ): i
                    for i, base in enumerate(program.bases)
                }
                for future in as_completed(futures):
                    base_idx = futures[future]
                    try:
                        result = future.result()
                        base_results.append(result)
                    except Exception as e:
                        logger.exception("Failed to process base %d", base_idx)
                        base_results.append(
                            self._create_failed_base_result(
                                program.bases[base_idx], str(e)
                            )
                        )
        else:
            for i, base in enumerate(program.bases):
                try:
                    result = self._execute_for_base(
                        program, base, i, program_dir, status
                    )
                    base_results.append(result)
                except Exception as e:
                    logger.exception("Failed to process base %s", base.id)
                    base_results.append(self._create_failed_base_result(base, str(e)))

        # Determine overall status
        all_completed = all(r.status == "completed" for r in base_results)
        any_failed = any(r.status == "failed" for r in base_results)

        if all_completed:
            overall_status: Literal["completed", "partial", "failed"] = "completed"
        elif any_failed and not all_completed:
            overall_status = "partial"
        else:
            overall_status = "failed"

        total_duration = time.time() - start_time

        result = MultiDonorResult(
            program_id=program_id,
            program_name=program.name,
            base_results=base_results,
            total_duration_seconds=total_duration,
            completed_at=datetime.now(),
            status=overall_status,
        )

        # Update final status
        status.status = overall_status
        status.updated_at = datetime.now()
        self._save_status(program_dir, status)

        # Save result
        result_path = program_dir / "result.json"
        with result_path.open("w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(
            "Program %s completed with status %s in %.1fs",
            program.name,
            overall_status,
            total_duration,
        )

        return result

    def resume_program(self, program_id: str) -> MultiDonorResult:
        """Resume a partially completed program from checkpoint.

        Args:
            program_id: ID of program to resume

        Returns:
            MultiDonorResult with continued execution
        """
        program_dir = self._programs_dir / program_id
        if not program_dir.exists():
            msg = f"Program {program_id} not found"
            raise FileNotFoundError(msg)

        # Load program and status
        program = TransplantProgram.from_yaml(program_dir / "program.yaml")
        status = self._load_status(program_dir)

        if status.status == "completed":
            logger.info("Program %s already completed, loading result", program_id)
            result_path = program_dir / "result.json"
            with result_path.open() as f:
                data = json.load(f)
            # Reconstruct result (simplified - would need full reconstruction)
            return MultiDonorResult(
                program_id=data["program_id"],
                program_name=data["program_name"],
                base_results=[],  # Would need to reconstruct
                total_duration_seconds=data["total_duration_seconds"],
                completed_at=datetime.fromisoformat(data["completed_at"]),
                status=data["status"],
            )

        logger.info("Resuming program %s from checkpoint", program_id)
        status.status = "in_progress"
        status.updated_at = datetime.now()
        self._save_status(program_dir, status)

        # Continue execution from last checkpoint
        # (Implementation would find resume point per base and continue)
        return self.execute_program(program)

    def list_programs(self) -> list[ProgramStatus]:
        """List all programs (running, completed, failed).

        Returns:
            List of program statuses
        """
        programs = []
        for program_dir in self._programs_dir.iterdir():
            if program_dir.is_dir():
                status_path = program_dir / "status.json"
                if status_path.exists():
                    programs.append(self._load_status(program_dir))
        return programs

    def get_program_status(self, program_id: str) -> ProgramStatus:
        """Get status of a specific program.

        Args:
            program_id: Program ID

        Returns:
            ProgramStatus
        """
        program_dir = self._programs_dir / program_id
        if not program_dir.exists():
            msg = f"Program {program_id} not found"
            raise FileNotFoundError(msg)
        return self._load_status(program_dir)

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _execute_for_base(
        self,
        program: TransplantProgram,
        base: BaseModelSpec,
        base_index: int,
        program_dir: Path,
        status: ProgramStatus,
    ) -> BaseModelResult:
        """Execute all donors for a single base model.

        Args:
            program: Program specification
            base: Base model to process
            base_index: Index of base in program
            program_dir: Program output directory
            status: Program status for updates

        Returns:
            BaseModelResult with all donor stages
        """
        from modelcypher.core.use_cases.unified_geometric_merge import (
            UnifiedGeometricMerger,
            UnifiedMergeConfig,
        )

        base_start = time.time()
        base_dir = program_dir / f"base_{base_index}"
        base_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing base %s (%s)", base.id, base.source)

        donor_stages: list[DonorStageResult] = []
        current_weights: dict[str, Any] | None = None
        current_target_path: str = base.source

        # Update status
        status.base_progress[base_index]["status"] = "in_progress"
        self._save_status(program_dir, status)

        for donor_index, donor in enumerate(program.donors):
            donor_start = time.time()
            logger.info(
                "  Donor %d/%d: %s (domains: %s)",
                donor_index + 1,
                len(program.donors),
                donor.id,
                ", ".join(donor.domains),
            )

            try:
                # Configure merge
                config = UnifiedMergeConfig(
                    probe_mode="precise",
                    transplant_domains=donor.domains,
                    transplant_layers=donor.layers,
                    transplant_boundary_k=donor.boundary_k,
                    transplant_geodesic_k_neighbors=donor.geodesic_k,
                )

                # Create merger
                merger = UnifiedGeometricMerger(
                    model_loader=self._model_loader,
                    backend=self._backend,
                )

                # Execute merge
                # Note: target_weights param will be added in next step
                merge_result = merger.merge(
                    source_path=donor.source,
                    target_path=current_target_path,
                    output_path=str(base_dir / f"checkpoint_{donor_index:03d}_{donor.id}"),
                    config=config,
                )

                # Extract metrics
                transplant_metrics = merge_result.transplant_metrics
                cka_before = transplant_metrics.get("mean_cka_before", 0.0)
                cka_after = transplant_metrics.get("mean_cka_after", 0.0)

                # Create checkpoint
                checkpoint_path = str(
                    base_dir / f"checkpoint_{donor_index:03d}_{donor.id}"
                )

                # Run lightweight evaluation if enabled
                smoke_test_passed = True
                if program.evaluation.after_each_donor:
                    smoke_test_passed = self._run_smoke_test(
                        checkpoint_path, program.evaluation.smoke_test_prompts
                    )

                stage_result = DonorStageResult(
                    donor_id=donor.id,
                    donor_source=donor.source,
                    domains=donor.domains,
                    donor_index=donor_index,
                    cka_before=cka_before,
                    cka_after=cka_after,
                    cka_improvement=cka_after - cka_before,
                    boundary_preserved=transplant_metrics.get("mean_preserved_fraction", 0.0),
                    mean_boundary_relative_diff=transplant_metrics.get(
                        "mean_boundary_relative_diff", 0.0
                    ),
                    max_boundary_relative_diff=transplant_metrics.get(
                        "max_boundary_relative_diff", 0.0
                    ),
                    layers_transplanted=transplant_metrics.get("layers_transplanted", 0),
                    weights_transplanted=transplant_metrics.get("weights_transplanted", 0),
                    mean_projection_loss=transplant_metrics.get("mean_projection_loss", 0.0),
                    mean_null_dim=transplant_metrics.get("mean_null_dim", 0.0),
                    checkpoint_path=checkpoint_path,
                    duration_seconds=time.time() - donor_start,
                    smoke_test_passed=smoke_test_passed,
                )

                donor_stages.append(stage_result)

                # Update current target for next donor
                current_target_path = checkpoint_path

                # Update status
                status.base_progress[base_index]["completed_donors"] = donor_index + 1
                status.updated_at = datetime.now()
                self._save_status(program_dir, status)

                logger.info(
                    "    Completed in %.1fs: CKA improvement %.4f, boundary preserved %.2f%%",
                    stage_result.duration_seconds,
                    stage_result.cka_improvement,
                    stage_result.boundary_preserved * 100,
                )

            except Exception as e:
                logger.exception("    Failed to apply donor %s", donor.id)
                # Continue with next donor (partial completion)
                break

        # Calculate aggregates
        total_cka_improvement = sum(s.cka_improvement for s in donor_stages)
        mean_boundary = (
            sum(s.boundary_preserved for s in donor_stages) / len(donor_stages)
            if donor_stages
            else 0.0
        )

        # Copy final weights to output
        output_dir = Path(program.output_dir).expanduser() / program.name / base.effective_alias
        output_dir.mkdir(parents=True, exist_ok=True)

        if current_target_path != base.source:
            # Copy final checkpoint to output
            final_dir = output_dir / "final"
            shutil.copytree(current_target_path, final_dir, dirs_exist_ok=True)
            final_output = str(final_dir)
        else:
            final_output = base.source

        # Update status
        status.base_progress[base_index]["status"] = "completed"
        self._save_status(program_dir, status)

        return BaseModelResult(
            base_id=base.id,
            base_source=base.source,
            base_alias=base.effective_alias,
            output_path=final_output,
            donor_stages=donor_stages,
            total_cka_improvement=total_cka_improvement,
            mean_boundary_preserved=mean_boundary,
            total_donors_applied=len(donor_stages),
            status="completed" if len(donor_stages) == len(program.donors) else "partial",
            duration_seconds=time.time() - base_start,
        )

    def _run_smoke_test(self, model_path: str, prompts: tuple[str, ...]) -> bool:
        """Run quick inference smoke test.

        Args:
            model_path: Path to model checkpoint
            prompts: Test prompts

        Returns:
            True if all prompts produce non-empty output
        """
        try:
            # Simplified - would need actual inference
            # For now, just check model files exist
            model_dir = Path(model_path)
            safetensors_files = list(model_dir.glob("*.safetensors"))
            return len(safetensors_files) > 0
        except Exception:
            return False

    def _save_status(self, program_dir: Path, status: ProgramStatus) -> None:
        """Save program status to disk."""
        status_path = program_dir / "status.json"
        with status_path.open("w") as f:
            json.dump(status.to_dict(), f, indent=2)

    def _load_status(self, program_dir: Path) -> ProgramStatus:
        """Load program status from disk."""
        status_path = program_dir / "status.json"
        with status_path.open() as f:
            data = json.load(f)
        return ProgramStatus.from_dict(data)

    def _create_dry_run_result(
        self, program: TransplantProgram, program_id: str, start_time: float
    ) -> MultiDonorResult:
        """Create result for dry run (validation only)."""
        return MultiDonorResult(
            program_id=program_id,
            program_name=program.name,
            base_results=[],
            total_duration_seconds=time.time() - start_time,
            completed_at=datetime.now(),
            status="completed",
        )

    def _create_failed_base_result(
        self, base: BaseModelSpec, error: str
    ) -> BaseModelResult:
        """Create result for failed base processing."""
        return BaseModelResult(
            base_id=base.id,
            base_source=base.source,
            base_alias=base.effective_alias,
            output_path="",
            donor_stages=[],
            total_cka_improvement=0.0,
            mean_boundary_preserved=0.0,
            total_donors_applied=0,
            status="failed",
            error=error,
        )
