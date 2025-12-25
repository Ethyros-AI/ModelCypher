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

"""Property tests for HelpService schema method."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.help_service import HelpService

# Valid JSON Schema types according to JSON Schema draft-07
VALID_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}

# Valid JSON Schema draft URIs
VALID_SCHEMA_DRAFTS = {
    "http://json-schema.org/draft-04/schema#",
    "http://json-schema.org/draft-06/schema#",
    "http://json-schema.org/draft-07/schema#",
    "https://json-schema.org/draft/2019-09/schema",
    "https://json-schema.org/draft/2020-12/schema",
}


def is_valid_json_schema_type(type_value) -> bool:
    """Check if a type value is valid according to JSON Schema."""
    if isinstance(type_value, str):
        return type_value in VALID_JSON_SCHEMA_TYPES
    if isinstance(type_value, list):
        return all(t in VALID_JSON_SCHEMA_TYPES for t in type_value)
    return False


def validate_schema_structure(schema: dict) -> bool:
    """Validate that a dict follows JSON Schema structure.

    This performs structural validation without requiring the jsonschema library.
    """
    if not isinstance(schema, dict):
        return False

    # Check for $schema if present
    if "$schema" in schema:
        if schema["$schema"] not in VALID_SCHEMA_DRAFTS:
            return False

    # Check type if present
    if "type" in schema:
        if not is_valid_json_schema_type(schema["type"]):
            return False

    # Check properties if present (for object type)
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            return False
        # Recursively validate nested schemas
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict):
                if not validate_schema_structure(prop_schema):
                    return False

    # Check items if present (for array type)
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, dict):
            if not validate_schema_structure(items):
                return False
        elif isinstance(items, list):
            for item_schema in items:
                if isinstance(item_schema, dict):
                    if not validate_schema_structure(item_schema):
                        return False

    # Check required if present
    if "required" in schema:
        if not isinstance(schema["required"], list):
            return False
        if not all(isinstance(r, str) for r in schema["required"]):
            return False

    return True


# **Feature: cli-mcp-parity, Property 8: Schema returns valid JSON Schema**
# **Validates: Requirements 7.3**
@given(
    command=st.sampled_from(
        ["train start", "train_start", "model list", "model_list", "inventory"]
    ),
)
@settings(max_examples=100, deadline=None)
def test_schema_returns_valid_json_schema(command: str):
    """Property 8: For any valid command name, schema(command) returns a dict
    that is a valid JSON Schema document.

    Validates:
    1. The returned value is a dictionary
    2. The schema contains a valid $schema reference
    3. The schema has a valid type property
    4. The schema structure follows JSON Schema conventions
    """
    service = HelpService()
    schema = service.schema(command)

    # Property: schema is a dictionary
    assert isinstance(schema, dict), f"Schema should be a dict, got {type(schema)}"

    # Property: schema contains $schema key with valid draft URI
    assert "$schema" in schema, "Schema should contain $schema key"
    assert schema["$schema"] in VALID_SCHEMA_DRAFTS, (
        f"$schema should be a valid JSON Schema draft URI, got {schema['$schema']}"
    )

    # Property: schema has a valid type
    assert "type" in schema, "Schema should contain type key"
    assert is_valid_json_schema_type(schema["type"]), (
        f"type should be a valid JSON Schema type, got {schema['type']}"
    )

    # Property: schema structure is valid
    assert validate_schema_structure(schema), "Schema structure should be valid JSON Schema"


@given(
    command=st.sampled_from(["train start", "model list", "inventory"]),
)
@settings(max_examples=100, deadline=None)
def test_schema_has_consistent_structure(command: str):
    """Test that schema returns consistent structure across calls."""
    service = HelpService()

    schema1 = service.schema(command)
    schema2 = service.schema(command)

    # Property: same command returns same schema
    assert schema1 == schema2, "Schema should be consistent across calls"


@given(
    invalid_command=st.text(min_size=1, max_size=30).filter(
        lambda s: s.lower().replace(" ", "_").replace("-", "_")
        not in {"train_start", "trainstart", "model_list", "modellist", "inventory"}
    ),
)
@settings(max_examples=50, deadline=None)
def test_schema_rejects_invalid_command(invalid_command: str):
    """Test that schema raises ValueError for unknown commands."""
    service = HelpService()

    with pytest.raises(ValueError) as exc_info:
        service.schema(invalid_command)

    # Property: error message mentions the command
    assert "not found" in str(exc_info.value).lower() or invalid_command in str(exc_info.value)


@given(
    command=st.sampled_from(["train start", "model list", "inventory"]),
)
@settings(max_examples=100, deadline=None)
def test_schema_properties_have_valid_types(command: str):
    """Test that all properties in schema have valid types."""
    service = HelpService()
    schema = service.schema(command)

    def check_properties(s: dict) -> bool:
        """Recursively check that all properties have valid types."""
        if "properties" in s:
            for prop_name, prop_schema in s["properties"].items():
                if isinstance(prop_schema, dict):
                    if "type" in prop_schema:
                        if not is_valid_json_schema_type(prop_schema["type"]):
                            return False
                    if not check_properties(prop_schema):
                        return False
        if "items" in s and isinstance(s["items"], dict):
            if "type" in s["items"]:
                if not is_valid_json_schema_type(s["items"]["type"]):
                    return False
            if not check_properties(s["items"]):
                return False
        return True

    assert check_properties(schema), "All properties should have valid JSON Schema types"
