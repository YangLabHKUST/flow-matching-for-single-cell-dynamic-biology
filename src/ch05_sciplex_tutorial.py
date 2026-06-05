from __future__ import annotations

import json
from pathlib import Path
import textwrap

import pandas as pd


def wrapped_method_label(method, method_labels: dict[str, str], width: int = 23) -> str:
    label = method_labels.get(method, str(method))
    return "\n".join(textwrap.wrap(label, width=width))


def short_compound_label(name, width: int = 18, aliases: dict[str, str] | None = None) -> str:
    aliases = {"Aminoglutethimide": "Aminoglutethimide"} if aliases is None else aliases
    text = aliases.get(str(name), str(name)).replace(" (", "\n(")
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [part])
    return "\n".join(lines[:2])


def metric_table_for_split(summary, split_name, method_order, method_labels: dict[str, str]):
    frame = pd.DataFrame(summary).loc[pd.DataFrame(summary)["split_name"].eq(split_name)].copy()
    available = set(frame["method"].astype(str))
    missing = [method_labels[m] for m in method_order if m not in available]
    ordered = [m for m in method_order if m in available]
    frame["_order"] = frame["method"].map({method: i for i, method in enumerate(ordered)})
    frame = frame.loc[frame["method"].isin(ordered)].sort_values("_order").drop(columns="_order")
    frame["method_label"] = frame["method"].map(method_labels)
    return frame.reset_index(drop=True), missing


def metric_value_table(frame):
    frame = pd.DataFrame(frame)
    if "MMD" in frame.columns and "Sliced W2" in frame.columns:
        out = frame[["method_label", "MMD", "Sliced W2"]].copy()
    else:
        out = frame[["method_label", "program_readout_mmd", "program_readout_sliced_w2"]].copy().rename(
            columns={
                "program_readout_mmd": "MMD",
                "program_readout_sliced_w2": "Sliced W2",
            }
        )
    return out.rename(columns={"method_label": "method"})


def make_metric_display_table(rows, source_label, method_labels: dict[str, str]):
    frame = pd.DataFrame(rows)
    frame["method_label"] = frame["method"].map(method_labels)
    frame["metric_display_source"] = source_label
    return frame


def make_metric_display_table_from_summary(
    source_path,
    split_name,
    method_order,
    expected_display,
    method_labels: dict[str, str],
    *,
    project_root=None,
):
    source_path = Path(source_path)
    payload = json.loads(source_path.read_text())
    rows = payload.get("key_metrics", {}).get("sciplex_summary", payload.get("sciplex_metrics_summary", []))
    raw = pd.DataFrame(rows)
    required = {"split_name", "method", "program_readout_mmd", "program_readout_sliced_w2"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Metric display source is missing required columns: {source_path}")
    frame = raw.loc[raw["split_name"].eq(split_name) & raw["method"].isin(method_order)].copy()
    frame["_order"] = frame["method"].map({method: i for i, method in enumerate(method_order)})
    frame = frame.sort_values("_order").drop(columns="_order")
    if frame["method"].tolist() != list(method_order):
        raise ValueError(f"Metric display source does not contain all requested methods for {split_name}")
    frame["MMD"] = frame["program_readout_mmd"].astype(float).round(4)
    frame["Sliced W2"] = frame["program_readout_sliced_w2"].astype(float).round(3)
    frame["method_label"] = frame["method"].map(method_labels)
    if project_root is None:
        metric_display_source = str(source_path)
    else:
        metric_display_source = str(source_path.relative_to(Path(project_root)))
    frame["metric_display_source"] = metric_display_source
    expected = pd.DataFrame(expected_display)
    got = frame[["method", "MMD", "Sliced W2"]].reset_index(drop=True)
    want = expected[["method", "MMD", "Sliced W2"]].reset_index(drop=True)
    if not got.equals(want):
        raise ValueError(f"Display metrics do not match the manuscript table for {split_name}:\n{got}\n!=\n{want}")
    return frame
