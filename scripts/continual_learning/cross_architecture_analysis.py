#!/usr/bin/env python3
"""Cross-architecture capacity analysis for Exp2 results."""

import json
import sys
from pathlib import Path


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_model(data: dict) -> dict:
    """Group layers by projection type and compute aggregate stats."""
    report = data["capacity_report"]
    layers = report["layers"]

    groups: dict[str, dict] = {}
    for layer in layers:
        name = layer["layerName"]
        # Classify layer type
        if "q_proj" in name:
            t = "q_proj"
        elif "k_proj" in name:
            t = "k_proj"
        elif "v_proj" in name:
            t = "v_proj"
        elif "o_proj" in name or ("out_proj" in name and "self_attn" in name):
            t = "o_proj"
        elif "out_proj" in name and "conv" in name:
            t = "conv_out_proj"
        elif "in_proj" in name:
            t = "conv_in_proj"
        elif "gate_proj" in name or "w1" in name:
            t = "gate_proj"
        elif "down_proj" in name or "w2" in name:
            t = "down_proj"
        elif "up_proj" in name or "w3" in name:
            t = "up_proj"
        elif "embed" in name:
            t = "embed"
        elif "lm_head" in name:
            t = "lm_head"
        else:
            t = "other"

        if t not in groups:
            groups[t] = {"null_dims": [], "utils": [], "shapes": []}
        groups[t]["null_dims"].append(layer["nullSpaceDimF32"])
        groups[t]["utils"].append(layer["capacityUtilization"])
        groups[t]["shapes"].append(tuple(layer["weightShape"]))

    return {
        "analyzed_layers": report["analyzed_layers"],
        "analyzed_parameters": report["analyzed_parameters"],
        "mean_null_rank": report["mean_null_rank"],
        "groups": groups,
    }


def print_model_summary(label: str, summary: dict) -> None:
    groups = summary["groups"]
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Layers: {summary['analyzed_layers']}  "
          f"Params: {summary['analyzed_parameters']:,}  "
          f"Mean null_dim: {summary['mean_null_rank']:.2f}")
    print(f"{'=' * 60}")
    print(f"  {'Type':20s} {'Count':>5s} {'Mean null':>10s} {'Mean util':>10s} {'Shape[0]':>10s}")
    for t in sorted(groups.keys()):
        g = groups[t]
        n = len(g["null_dims"])
        mn = sum(g["null_dims"]) / n
        mu = sum(g["utils"]) / n
        s0 = g["shapes"][0][0]
        print(f"  {t:20s} {n:5d} {mn:10.2f} {mu:10.3f} {s0:10d}")


def main() -> None:
    results_dir = Path("results/continual_learning")

    # Discover all exp2 results
    result_dirs = sorted(results_dir.glob("exp2*/seed42/exp2_results.json"))

    if not result_dirs:
        print("No Exp2 results found.")
        sys.exit(1)

    models: dict[str, dict] = {}
    for path in result_dirs:
        data = load_result(str(path))
        label = data.get("model_path", str(path.parent.parent.name))
        # Use directory name as short label
        short = path.parent.parent.name.replace("exp2_", "").replace("exp2", "base")
        models[short] = summarize_model(data)

    # Print individual summaries
    for label, summary in models.items():
        print_model_summary(label, summary)

    # Cross-architecture comparison table
    print(f"\n{'=' * 80}")
    print("  CROSS-ARCHITECTURE COMPARISON")
    print(f"{'=' * 80}")

    # Collect all layer types across all models
    all_types = set()
    for s in models.values():
        all_types.update(s["groups"].keys())

    header = f"  {'Type':15s}"
    for label in models:
        header += f" | {label:>12s} util  null"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for t in sorted(all_types):
        row = f"  {t:15s}"
        for label, summary in models.items():
            if t in summary["groups"]:
                g = summary["groups"][t]
                mu = sum(g["utils"]) / len(g["utils"])
                mn = sum(g["null_dims"]) / len(g["null_dims"])
                row += f" | {mu:12.3f} {mn:5.1f}"
            else:
                row += f" | {'—':>12s} {'—':>5s}"
        print(row)

    # Output structured JSON
    output = {}
    for label, summary in models.items():
        model_out = {"params": summary["analyzed_parameters"], "layers": summary["analyzed_layers"], "mean_null_rank": summary["mean_null_rank"], "by_type": {}}
        for t, g in summary["groups"].items():
            n = len(g["null_dims"])
            model_out["by_type"][t] = {
                "count": n,
                "mean_null_dim": sum(g["null_dims"]) / n,
                "mean_util": sum(g["utils"]) / n,
                "dim": g["shapes"][0][0],
            }
        output[label] = model_out

    out_file = results_dir / "cross_architecture_comparison.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
