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
            "scienceworld_per_type",
            "scienceworld_panel",
        }:
            continue
        display = ("N/A" if value is None else f"{value:.2f}/100") if key == "scienceworld_score" else _format(value)
        lines.append(f"  {key:<32} {display}")
    if "scienceworld_panel" in metrics:
        lines.append("ScienceWorld Panel")
        for key, value in metrics["scienceworld_panel"].items():
            lines.append(f"  {key:<32} {value}")
        lines.append("ScienceWorld Per Task Type (score /100; success %)")
        for task, values in metrics["scienceworld_per_type"].items():
            score = "N/A" if values["score"] is None else f"{values['score']:.2f}"
            lines.append(f"  {task}: score={score}, success={_format(values['success_rate'])}, resolved={values['resolved']}/{values['planned']}")
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
