from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ch03Context:
    project_root: Path
    fig_dir: Path
    table_dir: Path
    output_dir: Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    candidates = [start_path, *start_path.parents]
    candidates.extend(
        [
            Path("/home/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology"),
            Path("/import/home4/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology"),
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "models.py").exists() and (candidate / "notebooks").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate project root from {start_path}")


def make_ch03_context(project_root: str | Path | None = None) -> Ch03Context:
    root = resolve_project_root(project_root)
    fig_dir = root / "figures" / "ch03"
    table_dir = root / "tables" / "ch03"
    output_dir = root / "outputs" / "ch03"
    for path in [fig_dir, table_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return Ch03Context(project_root=root, fig_dir=fig_dir, table_dir=table_dir, output_dir=output_dir)


PAPER_COLORS = {
    "source": "#4C78A8",
    "target": "#B8B8B8",
    "target_red": "#C44E52",
    "generated": "#2F7F73",
    "cfm": "#2F7FBD",
    "cnf": "#D55E00",
    "euler": "#4C78A8",
    "midpoint": "#55A868",
    "dopri5": "#C44E52",
    "low": "#2F7F73",
    "high": "#C44E52",
}


def set_paper_style(base_font_size: float = 8.5) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        rc={
            "font.family": "DejaVu Sans",
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 0.8,
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size - 1.0,
            "ytick.labelsize": base_font_size - 1.0,
            "legend.fontsize": base_font_size - 1.0,
            "figure.titlesize": base_font_size + 1.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    plt.rcParams["axes.grid"] = False


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color="0.12")


def short_strategy_label(strategy: str) -> str:
    return {
        "uniform": "uniform",
        "logit_normal_sigma_0.5": "logit sigma=.5",
        "logit_normal_sigma_1.0": "logit sigma=1",
        "logit_normal_sigma_2.0": "logit sigma=2",
        "beta_2_2": "beta(2,2)",
        "beta_0.5_0.5": "beta(.5,.5)",
        "cosine": "cosine",
    }.get(str(strategy), str(strategy))


def clean_spines(ax, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.90", linewidth=0.55)


def format_metric_axis(ax, metric: str) -> None:
    labels = {
        "mmd_20d": "MMD to target (20D PCs)",
        "endpoint_mmd_20d": "Endpoint MMD (20D PCs)",
        "sliced_w2_20d": "Sliced W2 (20D PCs)",
        "val_mse_20d": "Val CFM MSE (20D PCs)",
        "train_mse_20d": "Train CFM MSE (20D PCs)",
        "straightness_ratio_20d": "Straightness ratio in 20D PCs",
    }
    ax.set_ylabel(labels.get(metric, metric.replace("_", " ")))
    clean_spines(ax, grid_axis="y")


def add_note(ax, text: str, loc: str = "lower left") -> None:
    xy = {
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
    }[loc]
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=7.0,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 2.0, "alpha": 0.88},
    )


def as_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def subsample_idx(n: int, max_n: int | None = None, seed: int = 42) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    local_rng = np.random.default_rng(seed)
    return np.sort(local_rng.choice(n, size=int(max_n), replace=False))


def robust_limits(*arrays, q_low: float = 1.0, q_high: float = 99.0, margin: float = 0.08):
    chunks = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        chunks.append(arr.reshape(-1, arr.shape[-1])[:, :2])
    if not chunks:
        return (-1.0, 1.0), (-1.0, 1.0)
    X = np.vstack(chunks)
    X = X[np.isfinite(X).all(axis=1)]
    lo = np.percentile(X, q_low, axis=0)
    hi = np.percentile(X, q_high, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - margin * span
    hi = hi + margin * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def format_axis(ax, xlim=None, ylim=None, xlabel: str = "state 1", ylabel: str = "state 2", title: str | None = None) -> None:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_csv(path: str | Path, frame: pd.DataFrame | Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def figure_paths_from_name(fig_dir: str | Path, filename_or_stem: str | Path) -> tuple[Path, Path, str]:
    name = str(filename_or_stem)
    path = Path(name)
    stem = path.stem if path.suffix else name
    fig_dir = Path(fig_dir)
    return fig_dir / f"{stem}.png", fig_dir / f"{stem}.pdf", stem


def save_figure(
    fig,
    fig_dir: str | Path,
    filename_or_stem: str | Path,
    *,
    dpi: int = 300,
    write_pdf: bool = True,
) -> Path:
    png_path, pdf_path, _ = figure_paths_from_name(fig_dir, filename_or_stem)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    if write_pdf:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    return png_path


def save_paper_table(path_stem: str | Path, frame: pd.DataFrame, *, index: bool = False) -> tuple[Path, Path, Path]:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(frame)
    csv_path = path_stem.with_suffix(".csv")
    tex_path = path_stem.with_suffix(".tex")
    md_path = path_stem.with_suffix(".md")
    table.to_csv(csv_path, index=index)
    tex_path.write_text(table.to_latex(index=index, escape=False, float_format=lambda x: f"{x:.4g}"), encoding="utf-8")
    md_path.write_text(table.to_markdown(index=index), encoding="utf-8")
    return csv_path, tex_path, md_path


def display_saved_figure(path: str | Path, *, width: int | None = None) -> Path:
    from IPython.display import Image, display

    path = Path(path)
    display(Image(filename=str(path), width=width))
    return path


def display_saved_figures(paths: Iterable[str | Path], *, width: int | None = None) -> list[Path]:
    displayed = []
    for path in paths:
        displayed.append(display_saved_figure(path, width=width))
    return displayed


def display_table(frame: pd.DataFrame, columns: list[str] | None = None, n: int = 10) -> pd.DataFrame:
    from IPython.display import display

    preview = pd.DataFrame(frame)
    if columns is not None:
        preview = preview.loc[:, columns]
    preview = preview.head(n)
    display(preview)
    return preview


def check_required_artifacts(
    *,
    expected_figures: Iterable[str | Path] = (),
    expected_tables: Iterable[str | Path] = (),
    expected_outputs: Iterable[str | Path] = (),
) -> pd.DataFrame:
    rows = []
    for kind, paths in [
        ("figure", expected_figures),
        ("table", expected_tables),
        ("output", expected_outputs),
    ]:
        for path in paths:
            artifact = Path(path)
            rows.append(
                {
                    "kind": kind,
                    "path": str(artifact),
                    "exists": artifact.exists(),
                    "bytes": artifact.stat().st_size if artifact.exists() else 0,
                }
            )
    manifest = pd.DataFrame(rows)
    missing = manifest.loc[~manifest["exists"]] if not manifest.empty else manifest
    if not missing.empty:
        raise FileNotFoundError("Missing required Chapter 3 artifacts:\n" + missing.to_string(index=False))
    empty = manifest.loc[manifest["bytes"].eq(0)] if not manifest.empty else manifest
    if not empty.empty:
        raise FileNotFoundError("Empty required Chapter 3 artifacts:\n" + empty.to_string(index=False))
    return manifest


def relative_paths(paths: Iterable[str | Path], root: str | Path) -> list[str]:
    root = Path(root).resolve()
    rels = []
    for path in paths:
        path = Path(path).resolve()
        rels.append(str(path.relative_to(root) if path.is_relative_to(root) else path))
    return rels
