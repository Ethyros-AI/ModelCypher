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
from pathlib import Path

import typer

from modelcypher.cli.output import write_output
from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory

from .common import get_context

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("build-eval-dataset")
    def build_eval_dataset(
        ctx: typer.Context,
        output_path: str = typer.Argument(..., help="Path to output JSONL file"),
    ) -> None:
        """Build evaluation dataset from UnifiedAtlas probe support texts.

        Extracts support texts from probes matching the specified filters and
        writes them to a JSONL file suitable for perplexity evaluation with
        `mc eval run`.

        Example usage:
            mc geometry research build-eval-dataset atlas-eval.jsonl
        """
        context = get_context(ctx)

        include_name = True
        include_description = True
        max_texts_per_probe: int | None = None

        probes = UnifiedAtlasInventory.all_probes()

        # Collect texts
        texts: list[str] = []
        probe_count = 0
        for probe in probes:
            probe_texts: list[str] = []

            if include_name and probe.name:
                probe_texts.append(probe.name)
            if include_description and probe.description:
                probe_texts.append(probe.description)

            if probe.support_texts:
                support_texts = list(probe.support_texts)
                if max_texts_per_probe is not None:
                    support_texts = support_texts[:max_texts_per_probe]
                for text in support_texts:
                    if text and text.strip():
                        probe_texts.append(text.strip())

            if probe_texts:
                probe_count += 1
                texts.extend(probe_texts)

        # Write JSONL
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w") as f:
            for text in texts:
                f.write(json.dumps({"text": text}) + "\n")

        # Build summary payload
        domains_used = sorted(
            set(p.domain.value if hasattr(p.domain, "value") else str(p.domain) for p in probes)
        )
        sources_used = sorted(
            set(p.source.value if hasattr(p.source, "value") else str(p.source) for p in probes)
        )

        payload = {
            "_schema": "mc.geometry.research.build_eval_dataset.v1",
            "outputPath": str(output_path),
            "totalTexts": len(texts),
            "probeCount": probe_count,
            "domainsUsed": domains_used,
            "sourcesUsed": sources_used,
            "includesName": include_name,
            "includesDescription": include_description,
            "maxTextsPerProbe": max_texts_per_probe,
        }

        if context.output_format == "text":
            lines = [
                "EVALUATION DATASET CREATED",
                f"Output: {output_path}",
                f"Total texts: {len(texts)}",
                f"Probes used: {probe_count}",
                f"Domains: {', '.join(domains_used)}",
                f"Sources: {', '.join(sources_used)}",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
