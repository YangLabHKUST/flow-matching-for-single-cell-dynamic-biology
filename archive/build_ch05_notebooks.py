from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def append_ch04_cache_section() -> None:
    path = NOTEBOOK_DIR / "04_coupling_statespace_assumptions.ipynb"
    nb = nbf.read(path, as_version=4)
    marker = "sci-Plex 3 A549 data cache for Chapter 5"
    if any(marker in "".join(cell.get("source", "")) for cell in nb.cells):
        nbf.write(nb, path)
        return
    nb.cells.extend(
        [
            md(
                """
## sci-Plex 3 A549 data cache for Chapter 5

This section prepares and caches the sci-Plex 3 A549 data assets used by Chapter 5. It only downloads, normalizes metadata, and writes cache files under `data/`; it does not run Chapter 5 models or generate Chapter 5 figures/tables.
"""
            ),
            code(
                """
from src.data import load_or_prepare_sciplex3_a549, load_lincs_smiles_corpus

CH05_SCIPLEX_DOWNLOAD = os.environ.get("CH05_SCIPLEX_DOWNLOAD", "1") == "1"
CH05_ALLOW_SYNTHETIC_SCIPLEX = os.environ.get("CH05_ALLOW_SYNTHETIC_SCIPLEX", "0") == "1"

sciplex_cache = load_or_prepare_sciplex3_a549(
    data_dir=PROJECT_ROOT / "data" / "sciplex3_a549",
    lincs_smiles_dir=PROJECT_ROOT / "data" / "chemcpa_lincs_smiles",
    download=CH05_SCIPLEX_DOWNLOAD,
    synthetic_if_missing=CH05_ALLOW_SYNTHETIC_SCIPLEX,
    hvg_top_n=1000,
    seed=DEFAULT_SEED,
)
lincs_smiles_cache = load_lincs_smiles_corpus(
    cache_dir=PROJECT_ROOT / "data" / "chemcpa_lincs_smiles",
    download=CH05_SCIPLEX_DOWNLOAD,
)

print("sci-Plex cache paths:", sciplex_cache.paths)
print("sci-Plex summary:", sciplex_cache.summary)
print("LINCS SMILES corpus:", lincs_smiles_cache.path, "n_smiles=", len(lincs_smiles_cache.smiles))
display(sciplex_cache.cell_counts.head(20))
"""
            ),
        ]
    )
    nbf.write(nb, path)


def build_ch05_notebook() -> None:
    cells = [
        md(
            """
# Chapter 5: Biological Designs Evaluation

This notebook evaluates reusable biological design patterns for flow matching: multi-timepoint EB bridge sharing and condition-aware sci-Plex perturbation response prediction. Heavy data preparation for sci-Plex is intentionally cached by the Chapter 4 notebook section `sci-Plex 3 A549 data cache for Chapter 5`.
"""
        ),
        md(
            """
## 0. Setup

Imports, paths, environment-controlled quick/full mode, seeds, device, output directories, and save helpers.
"""
        ),
        code(
            """
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ch05")

from pathlib import Path
import sys
import json
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except Exception as exc:
    raise ImportError("Chapter 5 experiments require PyTorch.") from exc

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path("/home/xmabs/flow_matching_for_dynamic_biology/flow_matching_for_dynamic_biology")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    load_or_prepare_sciplex3_a549,
    load_lincs_smiles_corpus,
    make_sciplex_split,
    make_sciplex_pca_state_table,
    compute_rdkit2d_with_external_norm,
)
from src.ch05_experiments import (
    set_global_seed,
    load_eb_ch05,
    run_eb_pairwise_vs_shared,
    run_eb_skip_pair_ablation,
    run_eb_section51_main_suite,
    summarize_eb_section51_main_suite,
    build_ch05_section51_main_text_results,
    choose_heldout_compound,
    sciplex_split_counts,
    evaluate_sciplex_split,
    aggregate_metric_table,
)

DEFAULT_SEED = int(os.environ.get("CH05_SEED", "42"))
QUICK_MODE = os.environ.get("CH05_QUICK", "1") == "1"
TRAINING_STEPS = int(os.environ.get("CH05_TRAINING_STEPS", "1500" if QUICK_MODE else "6000"))
BATCH_SIZE = int(os.environ.get("CH05_BATCH_SIZE", "128" if QUICK_MODE else "256"))
NFE = int(os.environ.get("CH05_NFE", "16" if QUICK_MODE else "32"))
EB_MAX_CELLS_PER_TIME = int(os.environ.get("CH05_EB_MAX_CELLS_PER_TIME", "220" if QUICK_MODE else "900"))
SCIPLEX_DOWNLOAD_IN_CH05 = os.environ.get("CH05_SCIPLEX_DOWNLOAD_IN_CH05", "0") == "1"
SCIPLEX_SYNTHETIC_IF_MISSING = os.environ.get("CH05_ALLOW_SYNTHETIC_SCIPLEX", "0") == "1"
MAX_EVAL_GROUPS = os.environ.get("CH05_MAX_EVAL_GROUPS", "")
MAX_EVAL_GROUPS = None if MAX_EVAL_GROUPS == "" else int(MAX_EVAL_GROUPS)
EB_SKIP_ABLATION_SEEDS = [
    int(part.strip())
    for part in os.environ.get("CH05_EB_SKIP_ABLATION_SEEDS", str(DEFAULT_SEED)).split(",")
    if part.strip()
]
SECTION51_MAIN_SUITE_SEEDS = [
    int(part.strip())
    for part in os.environ.get("CH05_SECTION51_MAIN_SUITE_SEEDS", "42,43,44").split(",")
    if part.strip()
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "ch05"
TABLE_DIR = PROJECT_ROOT / "tables" / "ch05"
OUT_DIR = PROJECT_ROOT / "outputs" / "ch05"
for path in [FIG_DIR, TABLE_DIR, OUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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

def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    return path

def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path

def save_figure(fig, filename):
    path = FIG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

set_global_seed(DEFAULT_SEED)
RUN_SUMMARY = {
    "quick_mode": bool(QUICK_MODE),
    "seed": int(DEFAULT_SEED),
    "device": str(DEVICE),
    "training_steps": int(TRAINING_STEPS),
    "batch_size": int(BATCH_SIZE),
    "nfe": int(NFE),
    "paths": {"figures": str(FIG_DIR), "tables": str(TABLE_DIR), "outputs": str(OUT_DIR)},
}
print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {DEVICE}; quick={QUICK_MODE}; steps={TRAINING_STEPS}; batch={BATCH_SIZE}; nfe={NFE}")
print(f"EB skip-pair ablation seeds: {EB_SKIP_ABLATION_SEEDS}")
print(f"Section 5.1 main-suite seeds: {SECTION51_MAIN_SUITE_SEEDS}")
"""
        ),
        md("## 1. EB data load"),
        code(
            """
EB_PATH = DATA_DIR / "trajectorynet_eb" / "eb_velocity_v5.npz"
eb = load_eb_ch05(EB_PATH, max_cells_per_time=EB_MAX_CELLS_PER_TIME, seed=DEFAULT_SEED, n_pc=20)
RUN_SUMMARY["eb_data"] = {
    "path": str(EB_PATH),
    "max_cells_per_time": int(EB_MAX_CELLS_PER_TIME),
    "counts": eb["counts"],
    "training_space": "standardized PC-20",
    "display_space": "PHATE 2D",
    "cluster_labels": "KMeans k=8 fit in PC-20",
}
display(eb["counts"])
"""
        ),
        md("## 2. Exp 1 EB pairwise vs shared"),
        code(
            """
eb_metrics, eb_diag, eb_cache = run_eb_pairwise_vs_shared(
    eb,
    training_steps=TRAINING_STEPS,
    batch_size=BATCH_SIZE,
    nfe=NFE,
    seed=DEFAULT_SEED,
    device=DEVICE,
)
save_csv(TABLE_DIR / "tab_5_1_multi_timepoint.csv", eb_metrics)
save_csv(TABLE_DIR / "tab_5_1_diag_velocity_jump.csv", eb_diag)

method_order = ["pairwise_local_bridges", "shared_adjacent_only", "shared_adjacent_skip"]
method_labels = ["Pairwise", "Shared adj", "Shared adj+skip"]
method_colors = ["#4C78A8", "#54A24B", "#F58518"]
target_rows = [("hidden_t2", "hidden_t2"), ("seen_t4", "seen_t4")]
metric_cols = [("mmd_rbf", "MMD RBF"), ("sliced_w2", "Sliced W2"), ("centroid_l2", "Centroid L2")]

fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True)
for row_idx, (target, target_label) in enumerate(target_rows):
    target_df = eb_metrics[eb_metrics["target"].eq(target)].set_index("method").reindex(method_order)
    for col_idx, (metric, metric_label) in enumerate(metric_cols):
        ax = axes[row_idx, col_idx]
        ax.bar(method_labels, target_df[metric].to_numpy(), color=method_colors)
        ax.set_title(f"{target_label} / {metric_label}")
        ax.set_ylabel(metric_label)
        ax.tick_params(axis="x", rotation=20)
fig.suptitle("EB multi-timepoint recovery: global distribution metrics")
save_figure(fig, "fig_5_1_eb_pairwise_vs_shared.png")

RUN_SUMMARY["eb_metrics"] = eb_metrics.to_dict(orient="records")
RUN_SUMMARY["eb_diagnostics"] = eb_diag.to_dict(orient="records")
display(eb_metrics)
display(eb_diag)
"""
        ),
        md("## 2b. EB skip-pair exposure ablation"),
        code(
            """
eb_skip_metrics, eb_skip_diag, eb_skip_cache = run_eb_skip_pair_ablation(
    eb,
    batch_size=BATCH_SIZE,
    nfe=NFE,
    seed=DEFAULT_SEED,
    seeds=EB_SKIP_ABLATION_SEEDS,
    device=DEVICE,
)
save_csv(TABLE_DIR / "tab_5_1_skip_pair_ablation.csv", eb_skip_metrics)
save_csv(TABLE_DIR / "tab_5_1_skip_pair_ablation_diag.csv", eb_skip_diag)

variant_order = [
    "shared_adjacent_only_6000",
    "shared_adjacent_only_9000",
    "shared_adjacent_only_12000",
    "shared_skip_uniform_6000",
    "shared_skip_uniform_12000",
    "shared_skip_adj2_skip1_9000",
    "shared_skip_adj3_skip1_8000",
    "shared_skip_medium_only_9000",
]
variant_labels = [
    "adj 6k",
    "adj 9k",
    "adj 12k",
    "skip uni 6k",
    "skip uni 12k",
    "skip 2:1 9k",
    "skip 3:1 8k",
    "skip med 9k",
]
family_by_variant = (
    eb_skip_metrics[["variant", "variant_family"]]
    .drop_duplicates()
    .set_index("variant")["variant_family"]
    .to_dict()
)
family_colors = {"adjacent_only": "#54A24B", "skip": "#F58518"}
ablation_plot = (
    eb_skip_metrics
    .groupby(["variant", "variant_family", "target"], observed=False)[["mmd_rbf", "sliced_w2", "centroid_l2"]]
    .mean()
    .reset_index()
)
panel_specs = [
    ("seen_t4", "sliced_w2", "seen_t4 Sliced W2"),
    ("seen_t4", "centroid_l2", "seen_t4 Centroid L2"),
    ("hidden_t2", "mmd_rbf", "hidden_t2 MMD RBF"),
    ("hidden_t2", "sliced_w2", "hidden_t2 Sliced W2"),
]
fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), sharex=True)
for ax, (target, metric, title) in zip(axes.ravel(), panel_specs):
    panel = (
        ablation_plot[ablation_plot["target"].eq(target)]
        .set_index("variant")
        .reindex(variant_order)
        .reset_index()
    )
    values = panel[metric].to_numpy()
    colors = [family_colors[family_by_variant[v]] for v in variant_order]
    bars = ax.bar(np.arange(len(variant_order)), values, color=colors, alpha=0.88)
    for family in ["adjacent_only", "skip"]:
        family_mask = panel["variant_family"].eq(family)
        if family_mask.any():
            best_local_idx = int(panel.loc[family_mask, metric].astype(float).idxmin())
            bars[best_local_idx].set_edgecolor("black")
            bars[best_local_idx].set_linewidth(1.8)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(len(variant_order)))
    ax.set_xticklabels(variant_labels, rotation=30, ha="right")
fig.suptitle("EB skip-pair ablation: rollout and hidden-time global metrics")
save_figure(fig, "fig_5_1_skip_pair_ablation.png")

RUN_SUMMARY["eb_skip_pair_ablation"] = eb_skip_cache["decision_summary"]
display(eb_skip_metrics)
display(eb_skip_diag)
print(json.dumps(json_ready(eb_skip_cache["decision_summary"]), indent=2))
"""
        ),
        md("## 2c. Section 5.1 unified main suite"),
        code(
            """
section51_metrics, section51_diag, section51_cache = run_eb_section51_main_suite(
    eb,
    batch_size=BATCH_SIZE,
    nfe=NFE,
    seed=DEFAULT_SEED,
    seeds=SECTION51_MAIN_SUITE_SEEDS,
    device=DEVICE,
)
section51_summary, section51_diag_summary = summarize_eb_section51_main_suite(section51_metrics, section51_diag)
section51_main_text_results, section51_summary_payload = build_ch05_section51_main_text_results(
    section51_summary,
    section51_diag_summary,
)
save_csv(TABLE_DIR / "tab_5_1_main_suite.csv", section51_metrics)
save_csv(TABLE_DIR / "tab_5_1_main_suite_summary.csv", section51_summary)
save_csv(TABLE_DIR / "tab_5_1_main_suite_diag.csv", section51_diag)
save_csv(TABLE_DIR / "tab_5_1_main_suite_diag_summary.csv", section51_diag_summary)
save_csv(TABLE_DIR / "tab_5_1_main_text_results.csv", section51_main_text_results)

suite_variant_order = [
    "pairwise_local_bridges_6000",
    "shared_adjacent_only_6000",
    "shared_skip_uniform_6000",
    "shared_skip_adj2_skip1_9000",
]
suite_variant_labels = ["Pairwise", "Shared adj", "Skip uniform", "Skip 2:1"]
suite_metric_cols = [("mmd_rbf", "MMD RBF"), ("sliced_w2", "Sliced W2"), ("centroid_l2", "Centroid L2")]
suite_target_rows = [("hidden_t2", "hidden_t2"), ("seen_t4", "seen_t4")]
suite_colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]

fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6), sharex=True)
for row_idx, (target, target_label) in enumerate(suite_target_rows):
    target_df = (
        section51_summary[section51_summary["target"].eq(target)]
        .set_index("variant")
        .reindex(suite_variant_order)
    )
    for col_idx, (metric, metric_label) in enumerate(suite_metric_cols):
        ax = axes[row_idx, col_idx]
        ax.bar(suite_variant_labels, target_df[f"{metric}_mean"].to_numpy(), color=suite_colors)
        ax.set_title(f"{target_label} / {metric_label}")
        ax.set_ylabel(metric_label)
        ax.tick_params(axis="x", rotation=25)
fig.suptitle("Section 5.1 unified EB main suite: global distribution metrics")
save_figure(fig, "fig_5_1_main_suite.png")

RUN_SUMMARY["section_5_1_main_suite"] = {
    "metrics_table": str((TABLE_DIR / "tab_5_1_main_suite.csv").relative_to(PROJECT_ROOT)),
    "summary_table": str((TABLE_DIR / "tab_5_1_main_suite_summary.csv").relative_to(PROJECT_ROOT)),
    "diag_table": str((TABLE_DIR / "tab_5_1_main_suite_diag.csv").relative_to(PROJECT_ROOT)),
    "diag_summary_table": str((TABLE_DIR / "tab_5_1_main_suite_diag_summary.csv").relative_to(PROJECT_ROOT)),
    "main_text_results_table": str((TABLE_DIR / "tab_5_1_main_text_results.csv").relative_to(PROJECT_ROOT)),
    "figure": str((FIG_DIR / "fig_5_1_main_suite.png").relative_to(PROJECT_ROOT)),
    **section51_summary_payload,
}
section51_claim_parts = [
    "hidden_t2_main_comparison",
    "seen_t4_long_horizon",
    "hidden_t2_skip_tradeoff",
    "velocity_jump_diagnostic",
]
display(section51_summary)
display(section51_diag_summary)
display(section51_main_text_results)
print(json.dumps(json_ready(RUN_SUMMARY["section_5_1_main_suite"]), indent=2))
"""
        ),
        md("## 3. sci-Plex data audit + split-aware preprocessing"),
        code(
            """
try:
    sciplex = load_or_prepare_sciplex3_a549(
        data_dir=DATA_DIR / "sciplex3_a549",
        lincs_smiles_dir=DATA_DIR / "chemcpa_lincs_smiles",
        download=SCIPLEX_DOWNLOAD_IN_CH05,
        synthetic_if_missing=SCIPLEX_SYNTHETIC_IF_MISSING,
        hvg_top_n=1000,
        seed=DEFAULT_SEED,
    )
except FileNotFoundError as exc:
    raise FileNotFoundError(
        "sci-Plex cache is missing. Run the final Chapter 4 cache section first, "
        "or set CH05_SCIPLEX_DOWNLOAD_IN_CH05=1 for this notebook."
    ) from exc

metadata = sciplex.metadata.reset_index(drop=True).copy()
source_text = str(sciplex.summary.get("source", ""))
if bool(sciplex.summary.get("is_synthetic", False)) or "synthetic" in source_text.lower():
    save_json(OUT_DIR / "real_data_audit.json", {
        "status": "failed",
        "reason": "sci-Plex cache is synthetic or synthetic-labeled",
        "summary": sciplex.summary,
    })
    raise ValueError("Chapter 5 full run refuses synthetic sci-Plex data. Rebuild the Chapter 4 cache section with real A549 data.")
heldout_compound, heldout_reason = choose_heldout_compound(metadata)
split_a = make_sciplex_split("random", metadata, test_fraction=0.2, seed=DEFAULT_SEED)
split_b = make_sciplex_split("heldout_highest_dose", metadata, seed=DEFAULT_SEED)
split_c = make_sciplex_split("heldout_compound", metadata, heldout_compound=heldout_compound, seed=DEFAULT_SEED)
splits = {
    "Split A random sanity": split_a,
    "Split B held-out highest dose": split_b,
    "Split C held-out compound": split_c,
}
split_table = sciplex_split_counts(splits)
save_csv(TABLE_DIR / "tab_5_2_sciplex_splits.csv", split_table)

states = {}
for split_name, split_meta in splits.items():
    states[split_name] = make_sciplex_pca_state_table(sciplex.adata, split_meta, n_pcs=30, hvg_top_n=1000)

cell_counts = sciplex.cell_counts.copy()
vehicle_count = int(metadata["is_vehicle"].sum())
compound_count = int(metadata.loc[~metadata["is_vehicle"], "compound"].nunique())
RUN_SUMMARY["sciplex_data"] = {
    "paths": sciplex.paths,
    "summary": sciplex.summary,
    "K_compounds": compound_count,
    "vehicle_count": vehicle_count,
    "compound_dose_counts_head": cell_counts.head(40),
    "heldout_compound": heldout_compound,
    "heldout_compound_reason": heldout_reason,
    "split_counts": split_table,
    "pca_explained_variance": {name: state.pca_explained_variance_ratio for name, state in states.items()},
}
real_data_audit = {
    "status": "ok",
    "source": sciplex.summary.get("source"),
    "source_url": sciplex.summary.get("source_url"),
    "is_synthetic": bool(sciplex.summary.get("is_synthetic", False)),
    "K_compounds": compound_count,
    "compound_list": sciplex.summary.get("compound_list", sorted(metadata.loc[~metadata["is_vehicle"], "compound"].astype(str).unique().tolist())),
    "vehicle_count": vehicle_count,
    "dose_values": sciplex.summary.get("dose_values", sorted(map(float, metadata.loc[~metadata["is_vehicle"], "dose"].dropna().unique().tolist()))),
    "missing_smiles_count": sciplex.summary.get("missing_smiles_count"),
    "obs_schema_used": sciplex.summary.get("obs_schema_used"),
    "subset_rule": sciplex.summary.get("subset_rule"),
    "split_counts": split_table,
}
save_json(OUT_DIR / "real_data_audit.json", real_data_audit)
print("K compounds:", compound_count, "vehicle cells:", vehicle_count)
print("Held-out compound:", heldout_compound, heldout_reason)
display(cell_counts.head(30))
display(split_table)
"""
        ),
        md("## 4. RDKit2D preprocessing audit"),
        code(
            """
lincs = load_lincs_smiles_corpus(cache_dir=DATA_DIR / "chemcpa_lincs_smiles", download=SCIPLEX_DOWNLOAD_IN_CH05)
compound_smiles = (
    metadata.loc[~metadata["is_vehicle"], ["compound", "SMILES"]]
    .drop_duplicates()
    .sort_values("compound")
    .reset_index(drop=True)
)
rdkit_cache_path = OUT_DIR / "rdkit2d_compound_features.npz"
rdkit_diag_path = OUT_DIR / "rdkit2d_diagnostics.json"
if rdkit_cache_path.exists() and rdkit_diag_path.exists():
    z = np.load(rdkit_cache_path, allow_pickle=True)
    cached_compounds = z["compounds"].astype(str).tolist()
    current_compounds = compound_smiles["compound"].astype(str).tolist()
    if cached_compounds == current_compounds:
        rdkit_features = z["features"].astype(np.float32)
        rdkit_diagnostics = json.loads(rdkit_diag_path.read_text())
        if int(rdkit_diagnostics.get("D_RDKit", rdkit_features.shape[1])) != int(rdkit_features.shape[1]):
            rdkit_features = None
            rdkit_diagnostics = None
    else:
        rdkit_cache_path.unlink()
        rdkit_diag_path.unlink()
        rdkit_features = None
        rdkit_diagnostics = None
else:
    rdkit_features = None
    rdkit_diagnostics = None
if rdkit_features is None:
    rdkit_result = compute_rdkit2d_with_external_norm(
        compound_smiles["SMILES"].tolist(),
        external_smiles=lincs.smiles,
    )
    rdkit_features = rdkit_result.features
    rdkit_diagnostics = rdkit_result.diagnostics
    rdkit_diagnostics["D_RDKit"] = int(rdkit_features.shape[1])
    np.savez_compressed(
        rdkit_cache_path,
        compounds=compound_smiles["compound"].astype(str).to_numpy(),
        features=rdkit_features,
    )
    save_json(rdkit_diag_path, rdkit_diagnostics)
rdkit_by_compound = {
    str(compound): rdkit_features[i]
    for i, compound in enumerate(compound_smiles["compound"].astype(str).tolist())
}
rdkit_audit = pd.DataFrame([rdkit_diagnostics])
save_csv(OUT_DIR / "rdkit2d_audit.csv", rdkit_audit)
RUN_SUMMARY["rdkit2d"] = {
    **rdkit_diagnostics,
    "D_RDKit": int(rdkit_features.shape[1]),
    "lincs_smiles_path": str(lincs.path),
    "lincs_smiles_count": len(lincs.smiles),
    "lincs_invalid_count": int(lincs.n_invalid),
}
display(rdkit_audit)
"""
        ),
        md("## 5. Exp 2 Split A random sanity"),
        code(
            """
split_a_metrics, split_a_cache = evaluate_sciplex_split(
    states["Split A random sanity"].X_pca,
    states["Split A random sanity"].metadata,
    rdkit_by_compound=rdkit_by_compound,
    split_name="Split A random sanity",
    training_steps=TRAINING_STEPS,
    batch_size=BATCH_SIZE,
    nfe=NFE,
    seed=DEFAULT_SEED,
    device=DEVICE,
    max_eval_groups=MAX_EVAL_GROUPS,
)
display(aggregate_metric_table(split_a_metrics))
"""
        ),
        md("## 6. Exp 3 Split B held-out highest dose"),
        code(
            """
split_b_metrics, split_b_cache = evaluate_sciplex_split(
    states["Split B held-out highest dose"].X_pca,
    states["Split B held-out highest dose"].metadata,
    rdkit_by_compound=rdkit_by_compound,
    split_name="Split B held-out highest dose",
    training_steps=TRAINING_STEPS,
    batch_size=BATCH_SIZE,
    nfe=NFE,
    seed=DEFAULT_SEED + 1,
    device=DEVICE,
    max_eval_groups=MAX_EVAL_GROUPS,
)
display(aggregate_metric_table(split_b_metrics))
"""
        ),
        md("## 7. Exp 4 Split C held-out compound"),
        code(
            """
split_c_metrics, split_c_cache = evaluate_sciplex_split(
    states["Split C held-out compound"].X_pca,
    states["Split C held-out compound"].metadata,
    rdkit_by_compound=rdkit_by_compound,
    split_name="Split C held-out compound",
    training_steps=TRAINING_STEPS,
    batch_size=BATCH_SIZE,
    nfe=NFE,
    seed=DEFAULT_SEED + 2,
    device=DEVICE,
    max_eval_groups=MAX_EVAL_GROUPS,
)
display(aggregate_metric_table(split_c_metrics))
"""
        ),
        md("## 8. Merge tables, save figures"),
        code(
            """
sciplex_metrics = pd.concat([split_a_metrics, split_b_metrics, split_c_metrics], ignore_index=True)
sciplex_summary = aggregate_metric_table(sciplex_metrics)
save_csv(OUT_DIR / "sciplex_metrics_by_group.csv", sciplex_metrics)
save_csv(OUT_DIR / "sciplex_metrics_summary.csv", sciplex_summary)

fig, ax = plt.subplots(figsize=(10.5, 4.2))
plot_df = sciplex_summary.copy()
plot_df["label"] = plot_df["split_name"].str.replace("Split ", "S", regex=False) + "\\n" + plot_df["method"]
ax.bar(plot_df["label"], plot_df["program_readout_sliced_w2"], color="#4C78A8")
ax.set_ylabel("Sliced W2 in split-aware PCA-30")
ax.set_title("sci-Plex perturbation response metrics by split and method")
ax.tick_params(axis="x", rotation=75)
save_figure(fig, "fig_5_3_sciplex_heldout_compound_summary.png")

heldout_keys = [key for key in split_c_cache["predictions"] if key[0] == heldout_compound]
if not heldout_keys:
    heldout_keys = list(split_c_cache["predictions"].keys())
representative_key = sorted(heldout_keys, key=lambda x: x[1])[-1]
panel = split_c_cache["predictions"][representative_key]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
panels = [
    ("vehicle", panel["vehicle_as_prediction"], "#8E8E8E"),
    ("ground truth", panel["target"], "#F58518"),
    ("M4 chemistry", panel["M4_chemistry_aware"], "#54A24B"),
    ("M3 no chemistry", panel["M3_no_chemistry"], "#4C78A8"),
]
for ax, (title, pts, color) in zip(axes, panels):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=7, alpha=0.55, linewidths=0, color=color)
    ax.set_title(title)
    ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
fig.suptitle(f"Split C held-out {representative_key[0]} dose={representative_key[1]:g}")
save_figure(fig, "fig_5_3_sciplex_heldout_compound.png")

RUN_SUMMARY["sciplex_metrics_summary"] = sciplex_summary
RUN_SUMMARY["sciplex_representative_heldout"] = {"compound": representative_key[0], "dose": representative_key[1]}
display(sciplex_summary)
"""
        ),
        md("## 9. Write run_summary and final file existence checks"),
        code(
            """
required_paths = [
    FIG_DIR / "fig_5_1_eb_pairwise_vs_shared.png",
    FIG_DIR / "fig_5_1_skip_pair_ablation.png",
    FIG_DIR / "fig_5_1_main_suite.png",
    FIG_DIR / "fig_5_3_sciplex_heldout_compound.png",
    TABLE_DIR / "tab_5_1_multi_timepoint.csv",
    TABLE_DIR / "tab_5_1_diag_velocity_jump.csv",
    TABLE_DIR / "tab_5_1_skip_pair_ablation.csv",
    TABLE_DIR / "tab_5_1_skip_pair_ablation_diag.csv",
    TABLE_DIR / "tab_5_1_main_suite.csv",
    TABLE_DIR / "tab_5_1_main_suite_summary.csv",
    TABLE_DIR / "tab_5_1_main_suite_diag.csv",
    TABLE_DIR / "tab_5_1_main_suite_diag_summary.csv",
    TABLE_DIR / "tab_5_1_main_text_results.csv",
    TABLE_DIR / "tab_5_2_sciplex_splits.csv",
    OUT_DIR / "sciplex_metrics_by_group.csv",
    OUT_DIR / "sciplex_metrics_summary.csv",
    OUT_DIR / "real_data_audit.json",
    OUT_DIR / "run_summary.json",
]

metric_frames = {
    "eb_metrics": eb_metrics,
    "eb_diag": eb_diag,
    "eb_skip_metrics": eb_skip_metrics,
    "eb_skip_diag": eb_skip_diag,
    "section51_metrics": section51_metrics,
    "section51_diag": section51_diag,
    "section51_summary": section51_summary,
    "section51_diag_summary": section51_diag_summary,
    "section51_main_text_results": section51_main_text_results,
    "sciplex_metrics": sciplex_metrics,
    "sciplex_summary": sciplex_summary,
}
finite_checks = {}
for name, frame in metric_frames.items():
    numeric = frame.select_dtypes(include=[np.number])
    finite_checks[name] = bool(np.isfinite(numeric.to_numpy()).all()) if numeric.size else True

RUN_SUMMARY["key_metrics"] = {
    "eb_hidden_mmd_best": eb_metrics.loc[eb_metrics["target"].eq("hidden_t2")].sort_values("mmd_rbf").head(1).to_dict(orient="records"),
    "eb_hidden_sliced_w2_best": eb_metrics.loc[eb_metrics["target"].eq("hidden_t2")].sort_values("sliced_w2").head(1).to_dict(orient="records"),
    "eb_seen_sliced_w2_best": eb_metrics.loc[eb_metrics["target"].eq("seen_t4")].sort_values("sliced_w2").head(1).to_dict(orient="records"),
    "eb_seen_centroid_l2_best": eb_metrics.loc[eb_metrics["target"].eq("seen_t4")].sort_values("centroid_l2").head(1).to_dict(orient="records"),
    "sciplex_summary": sciplex_summary,
}
RUN_SUMMARY["finite_metric_checks"] = finite_checks
RUN_SUMMARY["expected_artifacts"] = [str(path.relative_to(PROJECT_ROOT)) for path in required_paths]
if bool(RUN_SUMMARY["sciplex_data"]["summary"].get("is_synthetic", False)):
    raise ValueError("Synthetic sci-Plex data reached final summary; refusing to write final run_summary.")
if "synthetic" in str(RUN_SUMMARY["sciplex_data"]["summary"].get("source", "")).lower():
    raise ValueError("Synthetic-labeled sci-Plex source reached final summary; refusing to write final run_summary.")
save_json(OUT_DIR / "run_summary.json", RUN_SUMMARY)

missing = []
for path in required_paths:
    if not path.exists() or path.stat().st_size <= 0:
        missing.append(str(path))
if missing:
    raise FileNotFoundError(f"Missing or empty required artifacts: {missing}")
if not all(finite_checks.values()):
    raise ValueError(f"Non-finite numeric metrics detected: {finite_checks}")

print("Required artifacts:")
for path in required_paths:
    print(path.relative_to(PROJECT_ROOT), path.stat().st_size)
display(pd.DataFrame({"metric_frame": list(finite_checks), "all_finite": list(finite_checks.values())}))
"""
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbf.write(nb, NOTEBOOK_DIR / "05_biological_designs_evaluation.ipynb")


if __name__ == "__main__":
    append_ch04_cache_section()
    build_ch05_notebook()
