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

import sys
from typing import TYPE_CHECKING, Any

from modelcypher.utils.json import dump_json

if TYPE_CHECKING:
    from modelcypher.core.domain.agent_protocol import AgentEnvelope


def write_output(
    data: Any,
    output_format: str = "json",
    pretty: bool = False,
) -> None:
    """Write output to stdout in the specified format.

    Args:
        data: Data to write (dict, list, or string for text format)
        output_format: "json", "yaml", or "text"
        pretty: Whether to pretty-print JSON output
    """
    if output_format == "text":
        if isinstance(data, str):
            sys.stdout.write(data)
        else:
            sys.stdout.write(dump_json(data, pretty=True))
    elif output_format == "yaml":
        try:
            import yaml
            sys.stdout.write(yaml.dump(data, default_flow_style=False))
        except ImportError:
            sys.stdout.write(dump_json(data, pretty=True))
    else:
        sys.stdout.write(dump_json(data, pretty=pretty))
    sys.stdout.write("\n")


def write_agent_output(
    envelope: "AgentEnvelope",
    output_format: str = "json",
    pretty: bool = False,
    text_result: str | None = None,
) -> None:
    """Write an AgentEnvelope to stdout.

    In JSON/YAML mode, writes the full envelope structure.
    In text mode, writes ``text_result`` (if provided) followed by a
    diagnostics summary section.
    """
    if output_format == "text":
        parts: list[str] = []
        if text_result:
            parts.append(text_result)
        diag = envelope.diagnostics
        parts.append("")
        parts.append(f"Summary: {diag.summary}")
        if diag.observations:
            parts.append("")
            for obs in diag.observations:
                parts.append(f"  - {obs}")
        if diag.recommendations:
            parts.append("")
            parts.append("Next steps:")
            for rec in diag.recommendations:
                line = f"  [{rec.action}] {rec.reason}"
                if rec.command:
                    line += f"\n    $ {rec.command}"
                parts.append(line)
        if envelope.next_actions:
            parts.append("")
            parts.append("Exact next commands:")
            for action in envelope.next_actions:
                parts.append(f"  [{action.name}] {action.reason}")
                parts.append(f"    $ {action.command}")
        write_output("\n".join(parts), output_format, pretty)
    else:
        write_output(envelope.to_dict(), output_format, pretty)


def write_error(
    error: Any,
    output_format: str = "json",
    pretty: bool = False,
    exit_code: int = 1,
) -> None:
    payload = {"error": error, "exitCode": exit_code}
    write_output(payload, output_format=output_format, pretty=pretty)
