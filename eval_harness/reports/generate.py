from __future__ import annotations

from typing import Any


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2%}"
    return str(value)


def render_text_report(name: str, metrics: dict[str, Any]) -> str:
    width = 72
    lines = ["=" * width, "Experiment", name, "=" * width, "Counts"]
    for key, value in metrics.get("counts", {}).items():
        lines.append(f"  {key:<32} {value}")
    lines.append("Performance")
    for key, value in metrics.items():
        if key in {
            "counts",
            "length",
            "correct_length",
            "per_source",
            "correct_trajectories_per_problem",
        }:
            continue
        lines.append(f"  {key:<32} {_format(value)}")
    if "length" in metrics:
        lines.append("Length (completion tokens)")
        for key, value in metrics["length"].items():
            display = "N/A" if value is None else f"{value:,.2f}"
            lines.append(f"  {key:<32} {display}")
    if "correct_length" in metrics:
        lines.append("Correct Trajectory Length")
        for key, value in metrics["correct_length"].items():
            display = "N/A" if value is None else f"{value:,.2f}"
            lines.append(f"  {key:<32} {display}")
    if "correct_trajectories_per_problem" in metrics:
        lines.append("Correct Trajectories Per Problem")
        for key, value in metrics["correct_trajectories_per_problem"].items():
            lines.append(f"  {key + ' correct':<32} {value}")
    if "per_source" in metrics:
        lines.append("Per Source")
        for key, value in metrics["per_source"].items():
            lines.append(f"  {key:<32} {_format(value)}")
    lines.append("=" * width)
    return "\n".join(lines) + "\n"
