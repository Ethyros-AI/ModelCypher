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
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldProfile,
    ManifoldRegion,
)
from modelcypher.ports.storage import ManifoldProfileStore
from modelcypher.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileIndexEntry:
    model_id: str
    model_name: str
    total_points: int
    region_count: int
    updated_at: datetime


class ManifoldProfilePaths:
    def __init__(self, base_path: str | Path | None = None) -> None:
        base = Path(base_path or os.environ.get("MODELCYPHER_HOME", "~/.modelcypher"))
        self.base = ensure_dir(base)
        self.manifold = ensure_dir(self.base / "manifold")
        self.profiles = ensure_dir(self.manifold / "profiles")
        self.index = self.profiles / "index.json"


class LocalManifoldProfileStore(ManifoldProfileStore):
    def __init__(self, paths: ManifoldProfilePaths | None = None) -> None:
        self.paths = paths or ManifoldProfilePaths()
        self._cache: dict[str, ManifoldProfile] = {}
        self._index: dict[str, ProfileIndexEntry] = {}
        self._index_loaded = False

    def load(self, model_id: str) -> ManifoldProfile | None:
        """Load manifold profile for a model.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        ManifoldProfile or None
            Manifold profile if found, None otherwise.
        """
        if model_id in self._cache:
            return self._cache[model_id]

        file_path = self._profile_path(model_id)
        if not file_path.exists():
            return None

        try:
            payload = self._read_json(file_path)
            profile = self._profile_from_dict(payload)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to load profile for %s: %s", model_id, exc)
            return None

        self._cache[model_id] = profile
        return profile

    def list(self, limit: int | None = None) -> list[ManifoldProfile]:
        """List manifold profiles, most recently updated first.

        Parameters
        ----------
        limit : int or None
            Maximum number of profiles to return.

        Returns
        -------
        list of ManifoldProfile
            List of manifold profiles.
        """
        self._ensure_index_loaded()
        entries = sorted(self._index.values(), key=lambda item: item.updated_at, reverse=True)
        if limit is not None:
            entries = entries[: max(1, limit)]

        profiles: list[ManifoldProfile] = []
        for entry in entries:
            profile = self.load(entry.model_id)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def save(self, profile: ManifoldProfile) -> None:
        """Save manifold profile to storage.

        Parameters
        ----------
        profile : ManifoldProfile
            Manifold profile to save.
        """
        self._ensure_index_loaded()
        file_path = self._profile_path(profile.model_id)
        payload = self._profile_to_dict(profile)
        self._write_json(file_path, payload)

        self._cache[profile.model_id] = profile
        self._index[profile.model_id] = ProfileIndexEntry(
            model_id=profile.model_id,
            model_name=profile.model_name,
            total_points=profile.total_point_count,
            region_count=len(profile.regions),
            updated_at=profile.updated_at,
        )
        self._persist_index()

        logger.info(
            "Saved profile for %s (points=%s, regions=%s)",
            profile.model_id,
            profile.total_point_count,
            len(profile.regions),
        )

    def delete(self, model_id: str) -> None:
        file_path = self._profile_path(model_id)
        if file_path.exists():
            file_path.unlink()
        self._cache.pop(model_id, None)
        self._index.pop(model_id, None)
        self._persist_index()
        logger.info("Deleted profile for %s", model_id)

    def add_point(self, point: ManifoldPoint, model_id: str, model_name: str) -> None:
        profile = self.load(model_id)
        if profile is None:
            profile = ManifoldProfile(
                id=uuid4(),
                model_id=model_id,
                model_name=model_name,
            )
        profile = self._profile_with_point(profile, point)
        self.save(profile)

    def get_statistics(self, model_id: str) -> ManifoldProfile.Statistics | None:
        profile = self.load(model_id)
        return profile.compute_statistics() if profile else None

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    def _profile_with_point(
        self, profile: ManifoldProfile, point: ManifoldPoint
    ) -> ManifoldProfile:
        updated = ManifoldProfile(
            id=profile.id,
            model_id=profile.model_id,
            model_name=profile.model_name,
            regions=list(profile.regions),
            recent_points=list(profile.recent_points) + [point],
            total_point_count=profile.total_point_count + 1,
            created_at=profile.created_at,
            updated_at=datetime.utcnow(),
            version=profile.version,
        )
        return updated

    def _ensure_index_loaded(self) -> None:
        if self._index_loaded:
            return
        self._index_loaded = True
        index_path = self.paths.index
        if index_path.exists():
            try:
                payload = self._read_json(index_path)
                self._index = {}
                for key, value in payload.items():
                    model_id = value.get("modelID") or value.get("model_id") or key
                    model_name = value.get("modelName") or value.get("model_name") or model_id
                    total_points = value.get("totalPoints")
                    if total_points is None:
                        total_points = value.get("total_points", 0)
                    region_count = value.get("regionCount")
                    if region_count is None:
                        region_count = value.get("region_count", 0)
                    updated_at = (
                        value.get("updatedAt")
                        or value.get("updated_at")
                        or datetime.utcnow().isoformat()
                    )
                    self._index[key] = ProfileIndexEntry(
                        model_id=model_id,
                        model_name=model_name,
                        total_points=int(total_points),
                        region_count=int(region_count),
                        updated_at=self._from_iso(updated_at),
                    )
                return
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Failed to load profile index, rebuilding: %s", exc)

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {}
        for file_path in self.paths.profiles.glob("*.json"):
            if file_path.name == "index.json":
                continue
            try:
                payload = self._read_json(file_path)
                profile = self._profile_from_dict(payload)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid profile file %s: %s", file_path.name, exc)
                continue
            self._index[profile.model_id] = ProfileIndexEntry(
                model_id=profile.model_id,
                model_name=profile.model_name,
                total_points=profile.total_point_count,
                region_count=len(profile.regions),
                updated_at=profile.updated_at,
            )
        self._persist_index()

    def _persist_index(self) -> None:
        if not self._index_loaded:
            return
        payload = {
            model_id: {
                "modelID": entry.model_id,
                "modelName": entry.model_name,
                "totalPoints": entry.total_points,
                "regionCount": entry.region_count,
                "updatedAt": self._to_iso(entry.updated_at),
            }
            for model_id, entry in self._index.items()
        }
        self._write_json(self.paths.index, payload)

    def _profile_path(self, model_id: str) -> Path:
        sanitized = self._sanitize_model_id(model_id)
        return self.paths.profiles / f"{sanitized}.json"

    @staticmethod
    def _sanitize_model_id(model_id: str) -> str:
        invalid = '/\\:*?"<>|'
        translation = str.maketrans({char: "_" for char in invalid})
        return model_id.translate(translation)

    @staticmethod
    def _to_iso(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _from_iso(value: str) -> datetime:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)

    def _profile_to_dict(self, profile: ManifoldProfile) -> dict[str, Any]:
        return {
            "id": str(profile.id),
            "modelID": profile.model_id,
            "modelName": profile.model_name,
            "regions": [self._region_to_dict(region) for region in profile.regions],
            "recentPoints": [self._point_to_dict(point) for point in profile.recent_points],
            "totalPointCount": profile.total_point_count,
            "createdAt": self._to_iso(profile.created_at),
            "updatedAt": self._to_iso(profile.updated_at),
            "version": profile.version,
        }

    def _profile_from_dict(self, payload: dict[str, Any]) -> ManifoldProfile:
        model_id = payload.get("modelID") or payload.get("model_id") or ""
        model_name = payload.get("modelName") or payload.get("model_name") or model_id
        profile_id = payload.get("id") or payload.get("profile_id") or payload.get("profileID")
        profile_uuid = UUID(profile_id) if profile_id else uuid4()
        recent_points = payload.get("recentPoints")
        if recent_points is None:
            recent_points = payload.get("recent_points", [])
        regions_payload = payload.get("regions", [])
        created_at = payload.get("createdAt") or payload.get("created_at")
        updated_at = payload.get("updatedAt") or payload.get("updated_at")
        total_point_count = payload.get("totalPointCount")
        if total_point_count is None:
            total_point_count = payload.get("total_point_count", 0)
        return ManifoldProfile(
            id=profile_uuid,
            model_id=model_id,
            model_name=model_name,
            regions=[self._region_from_dict(region) for region in regions_payload],
            recent_points=[self._point_from_dict(point) for point in recent_points],
            total_point_count=int(total_point_count),
            created_at=self._from_iso(created_at or datetime.utcnow().isoformat()),
            updated_at=self._from_iso(updated_at or datetime.utcnow().isoformat()),
            version=int(payload.get("version", 1)),
        )

    def _region_to_dict(self, region: ManifoldRegion) -> dict[str, Any]:
        return {
            "id": str(region.id),
            "regionType": region.region_type.value,
            "centroid": self._point_to_dict(region.centroid),
            "memberCount": region.member_count,
            "memberIDs": [str(member_id) for member_id in region.member_ids],
            "dominantGates": list(region.dominant_gates),
            "intrinsicDimension": region.intrinsic_dimension,
            "radius": region.radius,
            "updatedAt": self._to_iso(region.updated_at),
        }

    def _region_from_dict(self, payload: dict[str, Any]) -> ManifoldRegion:
        region_type = payload.get("regionType") or payload.get("region_type")
        updated_at = payload.get("updatedAt") or payload.get("updated_at")
        return ManifoldRegion(
            id=UUID(payload["id"]),
            region_type=ManifoldRegion.RegionType(region_type),
            centroid=self._point_from_dict(payload["centroid"]),
            member_count=int(payload.get("memberCount") or payload.get("member_count")),
            member_ids=[
                UUID(value) for value in payload.get("memberIDs", payload.get("member_ids", []))
            ],
            dominant_gates=list(payload.get("dominantGates", payload.get("dominant_gates", []))),
            intrinsic_dimension=payload.get(
                "intrinsicDimension", payload.get("intrinsic_dimension")
            ),
            radius=float(payload.get("radius", 0.0)),
            updated_at=self._from_iso(updated_at or datetime.utcnow().isoformat()),
        )

    def _point_to_dict(self, point: ManifoldPoint) -> dict[str, Any]:
        return {
            "id": str(point.id),
            "meanEntropy": point.mean_entropy,
            "entropyVariance": point.entropy_variance,
            "firstTokenEntropy": point.first_token_entropy,
            "gateCount": point.gate_count,
            "meanGateConfidence": point.mean_gate_confidence,
            "dominantGateCategory": point.dominant_gate_category,
            "entropyPathCorrelation": point.entropy_path_correlation,
            "assessmentStrength": point.assessment_strength,
            "promptHash": point.prompt_hash,
            "timestamp": self._to_iso(point.timestamp),
            "interventionLevel": point.intervention_level,
        }

    def _point_from_dict(self, payload: dict[str, Any]) -> ManifoldPoint:
        timestamp = payload.get("timestamp") or payload.get("time_stamp")
        return ManifoldPoint(
            id=UUID(payload["id"]),
            mean_entropy=float(payload.get("meanEntropy", payload.get("mean_entropy", 0.0))),
            entropy_variance=float(
                payload.get("entropyVariance", payload.get("entropy_variance", 0.0))
            ),
            first_token_entropy=float(
                payload.get("firstTokenEntropy", payload.get("first_token_entropy", 0.0))
            ),
            gate_count=int(payload.get("gateCount", payload.get("gate_count", 0))),
            mean_gate_confidence=float(
                payload.get("meanGateConfidence", payload.get("mean_gate_confidence", 0.0))
            ),
            dominant_gate_category=float(
                payload.get("dominantGateCategory", payload.get("dominant_gate_category", 0.0))
            ),
            entropy_path_correlation=float(
                payload.get("entropyPathCorrelation")
                or payload.get("entropy_path_correlation")
                or 0.0
            ),
            assessment_strength=float(
                payload.get("assessmentStrength") or payload.get("assessment_strength") or 0.0
            ),
            prompt_hash=payload.get("promptHash", payload.get("prompt_hash", "")),
            timestamp=self._from_iso(timestamp or datetime.utcnow().isoformat()),
            intervention_level=payload.get("interventionLevel", payload.get("intervention_level")),
        )
