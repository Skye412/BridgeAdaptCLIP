"""Create frozen-protocol tables from completed DACL10K external runs."""

import argparse
import json
from pathlib import Path


MODELS = ("row0", "v20", "v21")
MAIN_TASKS = ("Bridge4", "Crack", "Spalling", "Corrosion", "Efflorescence")
OPEN_TASKS = ("All-Damage", "Seen-Damage", "Unseen-Damage")


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def summarize(root):
    root = Path(root)
    reports = {
        model: json.loads((root / model / "metrics.json").read_text(encoding="utf-8"))
        for model in MODELS
    }
    reference_support = reports["row0"]["metrics"]
    for model in MODELS[1:]:
        for task in (*MAIN_TASKS, *OPEN_TASKS):
            if reports[model]["metrics"][task]["support"] != reference_support[task]["support"]:
                raise ValueError(f"Support mismatch for {model}/{task}")

    lines = [
        "# DACL10K-v2 Official-Validation External Evaluation",
        "",
        "All models are frozen and DACL10K is used only for external evaluation.",
        "Pixel metrics use a streaming 65,536-bin score histogram.",
        "",
        "| Model | Bridge4 AP | Bridge4 AUROC | F1max | Crack AP | Spalling AP | Corrosion AP | Efflorescence AP | Macro AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        metrics = reports[model]["metrics_percent"]
        lines.append(
            f"| {model} | {fmt(metrics['Bridge4']['P-AP'])} | "
            f"{fmt(metrics['Bridge4']['P-AUROC'])} | {fmt(metrics['Bridge4']['P-F1max'])} | "
            f"{fmt(metrics['Crack']['P-AP'])} | {fmt(metrics['Spalling']['P-AP'])} | "
            f"{fmt(metrics['Corrosion']['P-AP'])} | {fmt(metrics['Efflorescence']['P-AP'])} | "
            f"{fmt(metrics['Bridge4-Macro']['P-AP'])} |"
        )
    lines.extend([
        "", "| Model | All-Damage AP | Seen-Damage AP | Unseen-Damage AP |",
        "|---|---:|---:|---:|",
    ])
    for model in MODELS:
        metrics = reports[model]["metrics_percent"]
        lines.append(
            f"| {model} | {fmt(metrics['All-Damage']['P-AP'])} | "
            f"{fmt(metrics['Seen-Damage']['P-AP'])} | {fmt(metrics['Unseen-Damage']['P-AP'])} |"
        )
    lines.extend(["", "## Support", ""])
    for task in (*MAIN_TASKS, *OPEN_TASKS):
        support = reference_support[task]["support"]
        lines.append(f"- {task}: `{json.dumps(support, sort_keys=True)}`")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    combined = {
        "protocol_id": "dacl10k-external-eval-v1",
        "models": {model: reports[model]["metrics"] for model in MODELS},
        "support_verified_identical": True,
    }
    (root / "combined_metrics.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root")
    summarize(parser.parse_args().experiment_root)
