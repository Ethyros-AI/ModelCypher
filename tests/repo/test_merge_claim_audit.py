# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLAIM_REGISTRY = ROOT / "results" / "repo_research_inventory" / "claim_registry.json"
REQUIRED_CLAIM_FIELDS = (
    "equation_or_theorem",
    "precision_state",
    "measurement_operator",
    "commensurability_proof",
    "directional_prediction",
    "falsifier",
)


def _load_merge_claims() -> dict[str, dict]:
    claims = json.loads(CLAIM_REGISTRY.read_text(encoding="utf-8"))
    return {
        claim["claim_id"]: claim
        for claim in claims
        if str(claim.get("claim_id", "")).startswith("CR-MRG-")
    }


def _claim_status(claim: dict) -> str | None:
    return claim.get("status") or claim.get("current_status")


def test_merge_claims_are_downgraded_until_portability_bundle_exists() -> None:
    merge_claims = _load_merge_claims()
    assert _claim_status(merge_claims["CR-MRG-001"]) == "[EXPLORATORY]"
    assert _claim_status(merge_claims["CR-MRG-002"]) == "[MEASUREMENT_INVALID]"


def test_validated_merge_claims_require_full_claim_form_fields() -> None:
    merge_claims = _load_merge_claims()
    violations: list[str] = []
    for claim_id, claim in merge_claims.items():
        if _claim_status(claim) != "[VALIDATED]":
            continue
        for field_name in REQUIRED_CLAIM_FIELDS:
            if not claim.get(field_name):
                violations.append(f"{claim_id}: missing {field_name}")

    assert not violations, "\n".join(violations)
