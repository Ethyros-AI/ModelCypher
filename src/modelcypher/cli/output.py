from __future__ import annotations

import sys
from typing import Any

import yaml

from modelcypher.utils.json import dump_json, json_default


def write_output(data: Any, output_format: str, pretty: bool = False) -> None:
    if output_format == "json":
        sys.stdout.write(dump_json(data, pretty=pretty))
        sys.stdout.write("\n")
        return
    if output_format == "yaml":
        yaml.safe_dump(data, sys.stdout, sort_keys=True)
        return
    sys.stdout.write(str(data))
    sys.stdout.write("\n")


def write_error(error: dict, output_format: str, pretty: bool = False) -> None:
    payload = {"error": error}
    if output_format == "json":
        sys.stdout.write(dump_json(payload, pretty=pretty))
        sys.stdout.write("\n")
        return
    if output_format == "yaml":
        yaml.safe_dump(payload, sys.stdout, sort_keys=True)
        return
    sys.stderr.write(f"Error: {error.get('title')}\n{error.get('detail')}\n")
