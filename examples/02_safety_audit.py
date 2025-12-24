#!/usr/bin/env python3

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
Safety Audit Example

This example demonstrates how to run safety probes against a model or adapter
to detect potential risks before deployment.

Usage:
    python examples/02_safety_audit.py /path/to/adapter.safetensors

Requirements:
    - A LoRA adapter file (.safetensors)
    - Base model path (optional, for behavioral probes)
"""
import sys
from pathlib import Path

from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService
from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService


def main():
    if len(sys.argv) < 2:
        print("Usage: python 02_safety_audit.py /path/to/adapter.safetensors")
        sys.exit(1)

    adapter_path = Path(sys.argv[1])
    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    print(f"Running safety audit on: {adapter_path}")
    print("=" * 60)

    # Initialize services
    safety_service = SafetyProbeService()
    entropy_service = EntropyProbeService()

    # 1. Static analysis - check adapter structure
    print("\n[1/3] Static Analysis")
    print("-" * 40)
    try:
        static_result = safety_service.static_scan(str(adapter_path))
        print(f"  Threat indicators found: {len(static_result.indicators)}")
        for indicator in static_result.indicators[:3]:
            print(f"    - {indicator.threat_type}: {indicator.description}")
        print(f"  Overall risk: {static_result.risk_level}")
    except Exception as e:
        print(f"  Static scan skipped: {e}")

    # 2. Entropy baseline verification
    print("\n[2/3] Entropy Baseline Check")
    print("-" * 40)
    try:
        entropy_result = entropy_service.verify_baseline(
            adapter_path=str(adapter_path),
            threshold=0.1,
        )
        print(f"  Verdict: {entropy_result.verdict}")
        print(f"  Delta from baseline: {entropy_result.delta:.4f}")
        if entropy_result.verdict == "trusted":
            print("  Adapter entropy matches expected baseline.")
        else:
            print("  Warning: Entropy deviation detected!")
    except Exception as e:
        print(f"  Entropy check skipped: {e}")

    # 3. Pattern analysis
    print("\n[3/3] Entropy Pattern Analysis")
    print("-" * 40)
    try:
        pattern_result = entropy_service.analyze_patterns(
            adapter_path=str(adapter_path),
        )
        print(f"  Pattern detected: {pattern_result.pattern}")
        print(f"  Trend: {pattern_result.trend}")
        print(f"  Distress level: {pattern_result.distress_level}")
    except Exception as e:
        print(f"  Pattern analysis skipped: {e}")

    print("\n" + "=" * 60)
    print("Audit complete. Review findings before deployment.")


if __name__ == "__main__":
    main()
