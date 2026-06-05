#!/usr/bin/env python
"""Run Ch4 Exp 9b WFR-FM sampling-depth sensitivity on EB snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ch04_wfrfm")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.baselines.wfrfm import (
    INTERNAL_IMPLEMENTATION_NOTE,
    build_wfrfm_dataframe,
    compare_growth_tables,
    evaluate_growth_by_bin,
    make_sampling_indices,
    standardize_train_eval,
    train_wfrfm_model,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "ch04"
FIG_DIR = PROJECT_ROOT / "figures" / "ch04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "trajectorynet_eb" / "eb_velocity_v5.npz")
    parser.add_argument("--dim", type=int, default=5, choices=[5, 20])
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--equal-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--raw-seed", type=int, default=11)
    parser.add_argument("--k-bins", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-hiddens", type=int, default=2)
    parser.add_argument("--eval-cells-per-time", type=int, default=600)
    parser.add_argument("--max-cells-per-time", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default=None, help="Append suffix to output stems, e.g. smoke or quick_cap120.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_eb(path: Path, dim: int) -> dict:
    z = np.load(path, allow_pickle=True)
    pcs = np.asarray(z["pcs"], dtype=np.float32)
    phate = np.asarray(z["phate"], dtype=np.float32)
    labels = np.asarray(z["sample_labels"], dtype=np.float32)
    scaler20 = StandardScaler()
    pcs20 = scaler20.fit_transform(pcs[:, :20]).astype(np.float32)
    train_features = pcs20[:, :dim].astype(np.float32)
    return {
        "pcs": pcs,
        "pcs20_standardized": pcs20,
        "train_features": train_features,
        "phate": phate,
        "labels": labels,
    }


def fixed_eval_indices(labels: np.ndarray, cells_per_time: int, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for t in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == t)
        n = min(len(idx), int(cells_per_time))
        selected.append(np.sort(rng.choice(idx, size=n, replace=False)))
    return np.concatenate(selected).astype(int)


def build_state_bins(pcs20: np.ndarray, k_bins: int, seed: int = 42) -> np.ndarray:
    k = min(int(k_bins), pcs20.shape[0])
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(pcs20).astype(int)


def plot_sensitivity(growth: pd.DataFrame, comparison: pd.DataFrame, out_path: Path) -> None:
    raw = growth[growth["setting"] == "raw_observed_depth"].copy()
    eq = growth[growth["setting"] == "equal_depth"].groupby(["eval_time", "state_bin"], as_index=False)["mean_g"].mean()
    raw_mat = raw.pivot_table(index="eval_time", columns="state_bin", values="mean_g", aggfunc="mean").sort_index()
    eq_mat = eq.pivot_table(index="eval_time", columns="state_bin", values="mean_g", aggfunc="mean").reindex_like(raw_mat)
    diff_mat = raw_mat - eq_mat

    vmax = float(np.nanmax(np.abs(pd.concat([raw_mat, eq_mat, diff_mat]).to_numpy())))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), constrained_layout=True)
    panels = [
        (axes[0, 0], raw_mat, "raw observed-depth mean growth"),
        (axes[0, 1], eq_mat, "equal-depth mean growth, averaged seeds"),
        (axes[1, 0], diff_mat, "raw minus equal-depth mean growth"),
    ]
    for ax, mat, title in panels:
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("state bin")
        ax.set_ylabel("eval time")
        ax.set_xticks(np.arange(mat.shape[1]), [str(c) for c in mat.columns])
        ax.set_yticks(np.arange(mat.shape[0]), [f"{x:g}" for x in mat.index])
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    if not comparison.empty:
        grouped = comparison.groupby("equal_seed", as_index=False).agg(
            spearman_growth_rank=("spearman_growth_rank", "mean"),
            top_expanding_overlap_k3=("top_expanding_overlap_k3", "mean"),
            top_shrinking_overlap_k3=("top_shrinking_overlap_k3", "mean"),
        )
        x = np.arange(len(grouped))
        width = 0.25
        ax.bar(x - width, grouped["spearman_growth_rank"], width=width, label="Spearman rank")
        ax.bar(x, grouped["top_expanding_overlap_k3"], width=width, label="top expanding overlap")
        ax.bar(x + width, grouped["top_shrinking_overlap_k3"], width=width, label="top shrinking overlap")
        ax.set_xticks(x, grouped["equal_seed"].astype(str))
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("equal-depth seed")
    ax.set_title("raw vs equal-depth growth-field agreement")

    fig.suptitle("WFR-FM growth readout is sensitive to EB snapshot sampling depth", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def suffix_stem(stem: str, suffix: str | None) -> str:
    clean = (suffix or "").strip()
    if not clean:
        return stem
    clean = clean[1:] if clean.startswith("_") else clean
    return f"{stem}_{clean}"


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.smoke:
        args.n_epochs = 5
        args.equal_seeds = [42]
        args.max_cells_per_time = 80
        args.eval_cells_per_time = min(args.eval_cells_per_time, 120)
        args.chunk_size = min(args.chunk_size, 80)
        if args.output_suffix is None:
            args.output_suffix = "smoke"
    if args.delta is None:
        args.delta = 2.0 if args.dim == 5 else 4.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    eb = load_eb(args.data, args.dim)
    labels = eb["labels"]
    raw_counts = {str(int(t) if float(t).is_integer() else t): int(np.sum(labels == t)) for t in sorted(np.unique(labels))}
    equal_depth_count = min(raw_counts.values())
    if args.max_cells_per_time is not None:
        equal_depth_count = min(equal_depth_count, int(args.max_cells_per_time))

    state_bins_all = build_state_bins(eb["pcs20_standardized"], args.k_bins)
    eval_idx = fixed_eval_indices(labels, args.eval_cells_per_time)
    eval_original_features = eb["train_features"][eval_idx]
    eval_times = labels[eval_idx]
    eval_bins = state_bins_all[eval_idx]

    runs = [("raw_observed_depth", args.raw_seed)] + [("equal_depth", seed) for seed in args.equal_seeds]
    growth_tables = []
    run_records = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for setting, seed in runs:
        indices = make_sampling_indices(labels, setting=setting, seed=seed, max_cells_per_time=args.max_cells_per_time)
        train_x, eval_x, norm = standardize_train_eval(eb["train_features"][indices], eval_original_features)
        df = build_wfrfm_dataframe(train_x, labels[indices], np.arange(train_x.shape[0]))
        model, info = train_wfrfm_model(
            df,
            dim=args.dim,
            seed=seed,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            delta=args.delta,
            hidden_dim=args.hidden_dim,
            n_hiddens=args.n_hiddens,
            chunk_size=args.chunk_size,
            device=device,
        )
        gt = evaluate_growth_by_bin(
            model,
            eval_x,
            eval_times,
            eval_bins,
            setting=setting,
            seed=seed,
            device=device,
        )
        growth_tables.append(gt)
        run_records.append(
            {
                "setting": setting,
                "seed": int(seed),
                "n_train_cells": int(len(indices)),
                "train_timepoint_counts": {
                    str(int(t) if float(t).is_integer() else t): int(np.sum(labels[indices] == t))
                    for t in sorted(np.unique(labels[indices]))
                },
                "normalization": norm,
                **info,
            }
        )

    growth = pd.concat(growth_tables, ignore_index=True)
    comparison = compare_growth_tables(growth, top_k=3)

    growth_path = OUT_DIR / f"{suffix_stem('table4_6c_wfrfm_growth_by_bin', args.output_suffix)}.csv"
    comparison_path = OUT_DIR / f"{suffix_stem('table4_6d_wfrfm_sampling_sensitivity', args.output_suffix)}.csv"
    summary_path = OUT_DIR / f"{suffix_stem('wfrfm_sampling_sensitivity_summary', args.output_suffix)}.json"
    fig_path = FIG_DIR / f"{suffix_stem('fig4_11d_wfrfm_growth_sensitivity', args.output_suffix)}.png"
    growth.to_csv(growth_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    plot_sensitivity(growth, comparison, fig_path)

    summary = {
        "experiment": "Exp 9b WFR-FM sampling-depth sensitivity",
        "implementation": "internal_minimal_wfrfm",
        "external_baseline_runtime_dependency": False,
        "internal_implementation_note": INTERNAL_IMPLEMENTATION_NOTE,
        "data_path": str(args.data),
        "output_suffix": args.output_suffix,
        "dim": int(args.dim),
        "delta": float(args.delta),
        "epochs": int(args.n_epochs),
        "batch_size": int(args.batch_size),
        "chunk_size": int(args.chunk_size),
        "use_mini_batch": True,
        "raw_timepoint_counts": raw_counts,
        "equal_depth_count": int(equal_depth_count),
        "equal_seeds": [int(x) for x in args.equal_seeds],
        "raw_seed": int(args.raw_seed),
        "k_bins": int(args.k_bins),
        "eval_cells_per_time": int(args.eval_cells_per_time),
        "max_cells_per_time": None if args.max_cells_per_time is None else int(args.max_cells_per_time),
        "raw_observed_depth_was_capped": bool(args.max_cells_per_time is not None),
        "sampling_depth_caveat": (
            "raw_observed_depth was capped by max_cells_per_time, so this run is a capped quick/smoke diagnostic rather than a full observed-depth comparison."
            if args.max_cells_per_time is not None
            else "raw_observed_depth uses all observed cells per timepoint; no max_cells_per_time cap was applied."
        ),
        "smoke": bool(args.smoke),
        "device": str(device),
        "runtime_sec": float(time.time() - start),
        "run_records": run_records,
        "relative_mass_audit": INTERNAL_IMPLEMENTATION_NOTE,
        "caveats": [
            "This is an algorithm-level sensitivity diagnostic, not a claim that WFR-FM is wrong.",
            "EB observed snapshot cell counts are treated as sampling-depth proxies, not calibrated biological census.",
            "Without calibrated census, proliferation, apoptosis, or lineage evidence, inferred growth is a model-dependent hypothesis.",
            "Smoke/quick settings trade estimator stability for runtime.",
        ],
        "outputs": {
            "growth_by_bin_csv": str(growth_path),
            "sensitivity_csv": str(comparison_path),
            "figure_png": str(fig_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({"outputs": summary["outputs"], "summary_json": str(summary_path), "runtime_sec": summary["runtime_sec"]}, indent=2))


if __name__ == "__main__":
    main()
