from __future__ import annotations

import numpy as np


def _as_frame(data):
    if hasattr(data, "to_frame"):
        frame = data.to_frame()
    else:
        frame = data.copy()
    if "state_1" not in frame.columns and "x0" in frame.columns:
        frame["state_1"] = frame["x0"]
    if "state_2" not in frame.columns and "x1" in frame.columns:
        frame["state_2"] = frame["x1"]
    return frame


def _state_limits(frame, pad: float = 0.25):
    x = frame["state_1"].to_numpy()
    y = frame["state_2"].to_numpy()
    return (float(x.min() - pad), float(x.max() + pad)), (float(y.min() - pad), float(y.max() + pad))


def _format_state_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("state 1")
    ax.set_ylabel("state 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _covariance_ellipse(ax, coords, color, linestyle="-", scale: float = 3.4, alpha: float = 0.17):
    from matplotlib.patches import Ellipse

    if len(coords) < 3:
        return None
    center = coords.mean(axis=0)
    cov = np.cov(coords.T)
    vals, vecs = np.linalg.eigh(cov + np.eye(2) * 1e-6)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ellipse = Ellipse(
        center,
        width=scale * np.sqrt(vals[0]),
        height=scale * np.sqrt(vals[1]),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.7,
        linestyle=linestyle,
    )
    ax.add_patch(ellipse)
    return center


def _flow_arrow(ax, start, end, color="0.18", rad: float = 0.0, alpha: float = 0.78, linewidth: float = 1.9):
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)
    return arrow


def _time_palette(times, cmap_name: str = "viridis"):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    if len(times) == 1:
        return {times[0]: cmap(0.65)}
    return {t: cmap(i / (len(times) - 1)) for i, t in enumerate(times)}


def plot_snapshots(X, time, ax=None, title: str = "Population snapshots"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(X[:, 0], X[:, 1], c=time, s=8, cmap="viridis", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("state 1")
    ax.set_ylabel("state 2")
    return ax, sc


def plot_pairs(x0, x1, ax=None, max_arrows: int = 100, title: str = "Endpoint pairs"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    n = min(max_arrows, len(x0))
    idx = np.linspace(0, len(x0) - 1, n).astype(int)
    ax.scatter(x0[:, 0], x0[:, 1], s=8, label="source")
    ax.scatter(x1[:, 0], x1[:, 1], s=8, label="target")
    for i in idx:
        ax.annotate("", xy=x1[i], xytext=x0[i], arrowprops={"arrowstyle": "->", "alpha": 0.25})
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_static_to_dynamic_snapshots(
    data,
    ax=None,
    max_points_per_time: int | None = 450,
    title: str = "Static atlas to dynamic population snapshots",
):
    """Plot observed population clouds at each sampled timepoint."""
    import matplotlib.pyplot as plt

    frame = _as_frame(data)
    times = sorted(frame["time"].unique())
    colors = _time_palette(times)
    xlim, ylim = _state_limits(frame)

    if ax is None:
        fig, axes = plt.subplots(1, len(times), figsize=(2.25 * len(times), 2.65), sharex=True, sharey=True)
    else:
        fig = ax.figure
        axes = np.asarray([ax])
    axes = np.asarray(axes).reshape(-1)

    for axis, t in zip(axes, times):
        subset = frame[frame["time"] == t]
        if max_points_per_time is not None and len(subset) > max_points_per_time:
            subset = subset.sample(max_points_per_time, random_state=0)
        axis.scatter(
            subset["state_1"],
            subset["state_2"],
            s=7,
            alpha=0.65,
            color=colors[t],
            linewidths=0,
        )
        axis.text(0.05, 0.92, f"t={t:.2f}", transform=axis.transAxes, fontsize=9, weight="bold")
        _format_state_axis(axis, xlim, ylim)
    for axis in axes[len(times) :]:
        axis.set_visible(False)

    fig.suptitle(title, y=1.03, fontsize=13)
    fig.text(
        0.5,
        -0.03,
        "Static snapshots show which states are present. Dynamic biology asks how population occupancy changes.",
        ha="center",
        fontsize=9,
    )
    return fig, axes


def plot_snapshots_not_movies(
    data,
    hidden_paths,
    axes=None,
    max_points_per_time: int | None = 300,
    max_paths: int = 45,
    title: str = "Observed snapshots versus unobserved cell movies",
):
    """Contrast unpaired observed snapshots with hypothetical hidden paths."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frame = _as_frame(data)
    hidden = hidden_paths.copy()
    combined = frame[["state_1", "state_2"]].copy()
    hidden_states = hidden[["state_1", "state_2"]].copy()
    combined = np.vstack([combined.to_numpy(), hidden_states.to_numpy()])
    xlim = (float(combined[:, 0].min() - 0.25), float(combined[:, 0].max() + 0.25))
    ylim = (float(combined[:, 1].min() - 0.25), float(combined[:, 1].max() + 0.25))
    times = sorted(frame["time"].unique())
    colors = _time_palette(times)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    else:
        axes = np.asarray(axes)
        fig = axes[0].figure

    ax_obs, ax_hidden = np.asarray(axes).reshape(-1)[:2]
    for axis in (ax_obs, ax_hidden):
        _format_state_axis(axis, xlim, ylim)

    for t in times:
        subset = frame[frame["time"] == t]
        if max_points_per_time is not None and len(subset) > max_points_per_time:
            subset = subset.sample(max_points_per_time, random_state=int(round(float(t) * 1000)) + 11)
        ax_obs.scatter(
            subset["state_1"],
            subset["state_2"],
            s=7,
            alpha=0.62,
            color=colors[t],
            linewidths=0,
            label=f"t={t:.2f}",
        )
        ax_hidden.scatter(
            subset["state_1"],
            subset["state_2"],
            s=5,
            alpha=0.24,
            color=colors[t],
            linewidths=0,
        )

    path_ids = hidden["path_id"].drop_duplicates().head(max_paths)
    for path_id in path_ids:
        path = hidden[hidden["path_id"] == path_id].sort_values("time")
        ax_hidden.plot(
            path["state_1"],
            path["state_2"],
            color="0.35",
            linewidth=0.8,
            alpha=0.45,
            linestyle=(0, (3, 2)),
        )

    ax_obs.set_title("Observed unpaired snapshots", fontsize=11)
    ax_hidden.set_title("Hypothetical same-cell movies (unobserved)", fontsize=11)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[t],
            markeredgecolor="none",
            markersize=6,
            label=f"t={t:.2f}",
        )
        for t in times
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        title="snapshot time",
        title_fontsize=8,
    )
    fig.suptitle(title, y=0.985, fontsize=13)
    return fig, np.asarray([ax_obs, ax_hidden])


def plot_distribution_flow_problem(
    data,
    ax=None,
    max_points_per_time: int | None = 500,
    title: str = "Empirical distributions and desired population flow",
):
    """Show empirical marginals with branching-aware idealized flow arrows."""
    import matplotlib.pyplot as plt

    frame = _as_frame(data)
    times = sorted(frame["time"].unique())
    colors = _time_palette(times)
    xlim, ylim = _state_limits(frame)
    branch_colors = {"major": "#2C7FB8", "rare": "#D95F02", "trunk": "0.25"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 4.55))
    else:
        fig = ax.figure
    _format_state_axis(ax, xlim, ylim)

    centers_by_time: dict[float, np.ndarray] = {}
    centers_by_branch: dict[tuple[float, str], np.ndarray] = {}
    for t in times:
        subset = frame[frame["time"] == t]
        if max_points_per_time is not None and len(subset) > max_points_per_time:
            subset = subset.sample(max_points_per_time, random_state=int(round(float(t) * 1000)) + 23)
        ax.scatter(
            subset["state_1"],
            subset["state_2"],
            s=6,
            alpha=0.25,
            color=colors[t],
            linewidths=0,
        )

        centers_by_time[t] = subset[["state_1", "state_2"]].to_numpy().mean(axis=0)
        fate_values = set(subset["fate_label"])
        if {"major", "rare"} & fate_values:
            groups = [("major", subset[subset["fate_label"] == "major"]), ("rare", subset[subset["fate_label"] == "rare"])]
        else:
            groups = [("trunk", subset)]
        for label, group in groups:
            if group.empty:
                continue
            coords = group[["state_1", "state_2"]].to_numpy()
            linestyle = "--" if label == "rare" else "-"
            center = _covariance_ellipse(ax, coords, colors[t], linestyle=linestyle, scale=3.2, alpha=0.13)
            if center is not None:
                centers_by_branch[(t, label)] = center

    if len(times) >= 3:
        trunk_times = times[:3]
        trunk_points = [centers_by_time[t] for t in trunk_times if t in centers_by_time]
        for start, end in zip(trunk_points[:-1], trunk_points[1:]):
            _flow_arrow(ax, start, end, color="0.23", rad=0.0, linewidth=1.7)

    branch_time = times[2] if len(times) > 2 else times[-1]
    split_source = centers_by_time.get(branch_time)
    if split_source is not None:
        major_targets = [centers_by_branch[key] for key in centers_by_branch if key[1] == "major" and key[0] > branch_time]
        rare_targets = [centers_by_branch[key] for key in centers_by_branch if key[1] == "rare" and key[0] > branch_time]
        major_targets = sorted(major_targets, key=lambda point: point[0])
        rare_targets = sorted(rare_targets, key=lambda point: point[0])
        if major_targets:
            _flow_arrow(ax, split_source, major_targets[0], color=branch_colors["major"], rad=0.08, linewidth=2.2)
            for start, end in zip(major_targets[:-1], major_targets[1:]):
                _flow_arrow(ax, start, end, color=branch_colors["major"], rad=0.04, linewidth=1.9)
        if rare_targets:
            _flow_arrow(ax, split_source, rare_targets[0], color=branch_colors["rare"], rad=-0.08, linewidth=2.2)
            for start, end in zip(rare_targets[:-1], rare_targets[1:]):
                _flow_arrow(ax, start, end, color=branch_colors["rare"], rad=-0.04, linewidth=1.9)

    ax.text(
        0.03,
        0.95,
        r"learn $v_\theta(x,t,c)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="0.18",
        bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 3.0},
    )
    ax.text(
        0.97,
        0.07,
        "branch-specific marginals",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.28",
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.5},
    )
    ax.set_title(title, fontsize=13)
    return fig, ax


def _obs_frame(adata_or_frame):
    import pandas as pd

    if hasattr(adata_or_frame, "obs"):
        return adata_or_frame.obs.copy()
    return pd.DataFrame(adata_or_frame).copy()


def _categorical_codes(values):
    import pandas as pd

    series = pd.Series(values)
    codes, uniques = pd.factorize(series.astype(str), sort=True)
    return codes, [str(u) for u in uniques]


def plot_qc_summary(adata, color: str = "condition", max_bins: int = 50):
    import matplotlib.pyplot as plt

    X = np.asarray(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    totals = X.sum(axis=1)
    detected = (X > 0).sum(axis=1)
    obs = adata.obs.copy()
    groups = sorted(obs[color].astype(str).unique()) if color in obs else ["all"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for group in groups:
        mask = np.ones(adata.n_obs, dtype=bool) if group == "all" else obs[color].astype(str).to_numpy() == group
        axes[0].hist(totals[mask], bins=max_bins, alpha=0.55, label=group)
        axes[1].hist(detected[mask], bins=max_bins, alpha=0.55, label=group)
    axes[0].set_xlabel("total counts per cell")
    axes[0].set_ylabel("cells")
    axes[1].set_xlabel("detected genes per cell")
    axes[1].set_ylabel("cells")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Count-table diagnostics before state-space selection", fontsize=13)
    return fig, axes


def plot_pca_variance(adata, n_components: int | None = None):
    import matplotlib.pyplot as plt

    ratio = np.asarray(adata.uns.get("pca_variance_ratio", []), dtype=float)
    if n_components is not None:
        ratio = ratio[:n_components]
    pcs = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(pcs, ratio, color="#4C78A8", alpha=0.86)
    ax.plot(pcs, np.cumsum(ratio), color="#F58518", marker="o", linewidth=1.7, label="cumulative")
    if len(pcs):
        ax.axvline(len(pcs), color="0.25", linestyle="--", linewidth=1.0)
        ax.text(len(pcs), max(np.cumsum(ratio)) * 0.96, f"d={len(pcs)}", ha="right", va="top", fontsize=9)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("PCA defines a compact modeling state space")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_embedding_by_metadata(adata, basis: str = "X_pca", color: str = "time", max_points: int | None = 3000):
    import matplotlib.pyplot as plt

    if basis not in adata.obsm:
        raise KeyError(f"{basis!r} not found in adata.obsm")
    coords = np.asarray(adata.obsm[basis])
    if coords.shape[1] < 2:
        raise ValueError(f"{basis!r} must have at least two dimensions to plot")
    obs = adata.obs.copy()
    idx = np.arange(adata.n_obs)
    if max_points is not None and len(idx) > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))

    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    values = obs[color].to_numpy()[idx] if color in obs else np.zeros(len(idx))
    try:
        numeric = values.astype(float)
        sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=numeric, s=9, cmap="viridis", alpha=0.72, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color)
    except (ValueError, TypeError):
        codes, labels = _categorical_codes(values)
        sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=codes, s=9, cmap="tab10", alpha=0.72, linewidths=0)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=sc.cmap(sc.norm(i)), markersize=6, label=lab)
            for i, lab in enumerate(labels)
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, title=color)
    prefix = "PC" if basis == "X_pca" else basis
    ax.set_xlabel(f"{prefix} 1")
    ax.set_ylabel(f"{prefix} 2")
    ax.set_title("Observed snapshot timepoints in the chosen state space")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_celltype_by_time(adata_or_frame, condition_key: str = "condition", label_key: str = "fate_label"):
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = _obs_frame(adata_or_frame)
    grouped = (
        frame.groupby(["time", condition_key, label_key], observed=False)
        .size()
        .reset_index(name="n_cells")
    )
    grouped["total"] = grouped.groupby(["time", condition_key], observed=False)["n_cells"].transform("sum")
    grouped["fraction"] = grouped["n_cells"] / grouped["total"].clip(lower=1)
    conditions = sorted(grouped[condition_key].astype(str).unique())
    labels = sorted(grouped[label_key].astype(str).unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions), 3.8), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    palette = dict(zip(labels, plt.get_cmap("Set2").colors[: len(labels)]))
    for ax, condition in zip(axes, conditions):
        sub = grouped[grouped[condition_key].astype(str) == condition]
        pivot = sub.pivot_table(index="time", columns=label_key, values="fraction", fill_value=0, observed=False)
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for label in labels:
            values = pivot[label].to_numpy() if label in pivot else np.zeros(len(pivot))
            ax.bar(x, values, bottom=bottom, color=palette[label], label=label)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(t):.2f}" if isinstance(t, (float, int)) else str(t) for t in pivot.index])
        ax.set_title(condition)
        ax.set_xlabel("time")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("fraction of cells")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("Diagnostic labels are downstream annotations", fontsize=13)
    return fig, axes


def plot_batch_time_diagnostics(adata_or_frame, batch_key: str = "batch"):
    import matplotlib.pyplot as plt

    frame = _obs_frame(adata_or_frame)
    times = sorted(frame["time"].unique())
    batches = sorted(frame[batch_key].astype(str).unique())
    conditions = sorted(frame["condition"].astype(str).unique()) if "condition" in frame else []
    batch_table = (
        frame.assign(**{batch_key: frame[batch_key].astype(str)})
        .groupby(["time", batch_key], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=times, columns=batches, fill_value=0)
    )
    batch_prop = batch_table.div(batch_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    im = axes[0].imshow(batch_prop.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(batches)))
    axes[0].set_xticklabels(batches, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(times)))
    axes[0].set_yticklabels([f"{float(t):.2f}" for t in times])
    axes[0].set_xlabel("batch")
    axes[0].set_ylabel("time")
    axes[0].set_title("batch composition")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="fraction")

    if conditions:
        cond_table = (
            frame.assign(condition=frame["condition"].astype(str))
            .groupby(["time", "condition"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(index=times, columns=conditions, fill_value=0)
        )
        cond_prop = cond_table.div(cond_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        bottom = np.zeros(len(cond_prop))
        x = np.arange(len(cond_prop))
        colors = plt.get_cmap("Set1").colors
        for i, condition in enumerate(conditions):
            values = cond_prop[condition].to_numpy()
            axes[1].bar(x, values, bottom=bottom, label=condition, color=colors[i % len(colors)], alpha=0.78)
            bottom += values
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{float(t):.2f}" for t in times])
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("time")
        axes[1].set_title("condition composition")
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Batch-time diagnostics before dynamic modeling", fontsize=13)
    return fig, axes


def plot_sampler_examples(batch, dataset=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1.0, 1.0, 0.95]})
    t0 = batch["t0"][0]
    t1 = batch["t1"][0]
    x0 = np.asarray(batch["x0"])
    x1 = np.asarray(batch["x1"])
    if x0.shape[1] < 2 or x1.shape[1] < 2:
        raise ValueError("plot_sampler_examples needs at least two feature dimensions")

    condition = batch.get("condition")
    if isinstance(condition, dict):
        source_condition = condition["source"][0]
        target_condition = condition["target"][0]
    elif condition is None:
        source_condition = target_condition = None
    else:
        source_condition = target_condition = condition[0]

    if dataset is not None:
        source_pool = dataset.cells_at(t0, condition=source_condition)
        target_pool = dataset.cells_at(t1, condition=target_condition)
        axes[0].scatter(source_pool[:, 0], source_pool[:, 1], s=8, color="0.78", linewidths=0, label="pool")
        axes[1].scatter(target_pool[:, 0], target_pool[:, 1], s=8, color="0.78", linewidths=0, label="pool")
    axes[0].scatter(x0[:, 0], x0[:, 1], s=22, color="#4C78A8", alpha=0.82, linewidths=0, label="sampled x0")
    axes[1].scatter(x1[:, 0], x1[:, 1], s=22, color="#F58518", alpha=0.82, linewidths=0, label="sampled x1")
    axes[0].set_title(f"source snapshot t={float(t0):.2f}")
    axes[1].set_title(f"target snapshot t={float(t1):.2f}")
    for ax in axes[:2]:
        ax.set_xlabel("state dim 1")
        ax.set_ylabel("state dim 2")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].axis("off")
    lines = []
    for key in ["x0", "x1", "t0", "t1", "condition", "labels", "idx0", "idx1"]:
        value = batch.get(key)
        if isinstance(value, dict):
            shape = "{" + ", ".join(f"{k}: {np.asarray(v).shape}" for k, v in value.items()) + "}"
        else:
            shape = str(np.asarray(value, dtype=object).shape) if value is not None else "None"
        lines.append(f"{key}: {shape}")
    note = "\n\nshown with X_toy_state for visualization\nsame contract applies to HVG-PCA batches"
    axes[2].text(0.02, 0.98, "\n".join(lines) + note, va="top", ha="left", family="monospace", fontsize=9)
    axes[2].set_title("batch contract")
    fig.suptitle("PairSampler assembles batches; it does not infer couplings", fontsize=13)
    return fig, axes


def _ch03_limits(*arrays, pad: float = 0.35):
    points = [np.asarray(a, dtype=float)[:, :2] for a in arrays if a is not None and len(a)]
    if not points:
        return (-1.0, 1.0), (-1.0, 1.0)
    stacked = np.vstack(points)
    x_pad = pad + 0.04 * float(np.ptp(stacked[:, 0]))
    y_pad = pad + 0.04 * float(np.ptp(stacked[:, 1]))
    return (
        float(stacked[:, 0].min() - x_pad),
        float(stacked[:, 0].max() + x_pad),
    ), (
        float(stacked[:, 1].min() - y_pad),
        float(stacked[:, 1].max() + y_pad),
    )


def _ch03_axis(ax, xlim, ylim, title: str):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("state 1")
    ax.set_ylabel("state 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ch03_subsample(n: int, max_n: int, seed: int):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _ch03_arrows(ax, x0, x1, indices=None, color="0.25", alpha=0.26, linewidth=0.8, linestyle="-"):
    if indices is None:
        indices = np.arange(len(x0))
    for i in indices:
        ax.annotate(
            "",
            xy=x1[i, :2],
            xytext=x0[i, :2],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "alpha": alpha,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "shrinkA": 1,
                "shrinkB": 1,
            },
        )


def _ch03_top_coupling_edges(pi, max_edges: int = 80):
    flat = np.asarray(pi, dtype=float).reshape(-1)
    if flat.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    max_edges = min(max_edges, flat.size)
    order = np.argpartition(flat, -max_edges)[-max_edges:]
    order = order[np.argsort(flat[order])[::-1]]
    i, j = np.unravel_index(order, np.asarray(pi).shape)
    mass = flat[order]
    return i, j, mass


def plot_ch03_unpaired_marginals_transport_question(
    source,
    target,
    seed: int = 42,
    max_points: int = 650,
    max_arrows: int = 45,
    source_label: str = "source",
    target_label: str = "target",
    title: str = "From paired trajectories to unpaired marginals",
):
    import matplotlib.pyplot as plt

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    rng = np.random.default_rng(seed)
    src_idx = _ch03_subsample(len(source), max_points, seed)
    tgt_idx = _ch03_subsample(len(target), max_points, seed + 1)
    xlim, ylim = _ch03_limits(source[:, :2], target[:, :2])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    for ax in axes[1:]:
        _ch03_axis(ax, xlim, ylim, "")
        ax.scatter(source[src_idx, 0], source[src_idx, 1], s=9, color="#4C78A8", alpha=0.52, linewidths=0)
        ax.scatter(target[tgt_idx, 0], target[tgt_idx, 1], s=9, color="#F58518", alpha=0.52, linewidths=0)

    # Counterfactual schematic: it is deliberately not an EB lineage panel.
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    paired_n = min(max_arrows, 14)
    y0 = np.linspace(0.18, 0.82, paired_n)
    y1 = np.clip(y0 + rng.normal(scale=0.055, size=paired_n), 0.12, 0.88)
    x0 = np.column_stack([np.full(paired_n, 0.23), y0])
    x1 = np.column_stack([np.full(paired_n, 0.78), y1])
    axes[0].scatter(x0[:, 0], x0[:, 1], s=24, color="#4C78A8", alpha=0.76, linewidths=0)
    axes[0].scatter(x1[:, 0], x1[:, 1], s=24, color="#F58518", alpha=0.76, linewidths=0)
    _ch03_arrows(axes[0], x0, x1, color="0.20", alpha=0.35, linewidth=0.9, linestyle="--")
    axes[0].set_title("A. Paired world", fontsize=10)
    axes[0].text(
        0.03,
        0.96,
        "hypothetical\nnot observed",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )

    axes[1].set_title("B. Snapshot world", fontsize=10)
    axes[1].text(
        0.03,
        0.96,
        f"EB empirical marginals\n{source_label} to {target_label}\nno row-wise pairing",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )

    _covariance_ellipse(axes[2], source[src_idx, :2], "#4C78A8", alpha=0.13)
    _covariance_ellipse(axes[2], target[tgt_idx, :2], "#F58518", alpha=0.13)
    start = source[:, :2].mean(axis=0)
    end = target[:, :2].mean(axis=0)
    _flow_arrow(axes[2], start, end, color="0.18", linewidth=2.2, rad=0.05)
    axes[2].set_title("C. Distributional transport", fontsize=10)
    axes[2].text(
        0.50,
        0.90,
        r"$T_\#\mu_0 \approx \mu_1$",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=10,
        color="0.18",
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )

    fig.suptitle(title, y=1.03, fontsize=13)
    return fig, axes


def plot_ch03_static_ot_endpoint_transport(
    X0,
    X1,
    C,
    pi_ot,
    pi_ind=None,
    target_labels=None,
    X0_plot=None,
    X1_plot=None,
    target_region_probability=None,
    target_region_mask=None,
    target_region_label: str = "A",
    max_arrows: int = 90,
    seed: int = 42,
    title: str = "Static OT as endpoint-level probabilistic transport",
):
    import matplotlib.pyplot as plt

    from matplotlib.patches import Ellipse

    from .ot import barycentric_projection, fate_probabilities

    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    X0_plot = X0[:, :2] if X0_plot is None else np.asarray(X0_plot, dtype=float)
    X1_plot = X1[:, :2] if X1_plot is None else np.asarray(X1_plot, dtype=float)
    C = np.asarray(C, dtype=float)
    pi_ot = np.asarray(pi_ot, dtype=float)
    pi_ind = np.full_like(pi_ot, 1.0 / pi_ot.size) if pi_ind is None else np.asarray(pi_ind, dtype=float)
    xlim, ylim = _ch03_limits(X0_plot[:, :2], X1_plot[:, :2])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))
    axes = axes.reshape(-1)

    for ax in [axes[0], axes[3], axes[4], axes[5]]:
        _ch03_axis(ax, xlim, ylim, "")
        ax.scatter(X0_plot[:, 0], X0_plot[:, 1], s=10, color="#4C78A8", alpha=0.46, linewidths=0)
        ax.scatter(X1_plot[:, 0], X1_plot[:, 1], s=10, color="#F58518", alpha=0.38, linewidths=0)

    i, j, mass = _ch03_top_coupling_edges(pi_ot, max_edges=max_arrows)
    widths = 0.45 + 3.6 * mass / np.clip(mass.max(), 1e-15, None) if len(mass) else []
    for src_i, tgt_j, width in zip(i, j, widths):
        axes[0].annotate(
            "",
            xy=X1_plot[tgt_j, :2],
            xytext=X0_plot[src_i, :2],
            arrowprops={"arrowstyle": "->", "color": "0.22", "alpha": 0.13, "linewidth": float(width) * 0.72},
        )
    axes[0].set_title("A. EB endpoint arrows in PHATE", fontsize=10)
    axes[0].text(
        0.03,
        0.96,
        "OT mass computed in PC space\narrows drawn in PHATE",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )

    im0 = axes[1].imshow(C, aspect="auto", cmap="magma")
    axes[1].set_title("B. Cost matrix C_ij", fontsize=10)
    axes[1].set_xlabel("target index")
    axes[1].set_ylabel("source index")
    fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

    im1 = axes[2].imshow(pi_ot, aspect="auto", cmap="viridis")
    axes[2].set_title("C. Transport plan P*_ij", fontsize=10)
    axes[2].set_xlabel("target index")
    axes[2].set_ylabel("source index")
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

    if target_region_probability is not None:
        color = np.asarray(target_region_probability, dtype=float)
        sc = axes[3].scatter(X0_plot[:, 0], X0_plot[:, 1], c=color, s=18, cmap="plasma", alpha=0.82, linewidths=0)
        if target_region_mask is not None:
            target_region_mask = np.asarray(target_region_mask, dtype=bool)
            region = X1_plot[target_region_mask]
            if len(region) >= 3:
                center = region[:, :2].mean(axis=0)
                spread = np.std(region[:, :2], axis=0)
                ellipse = Ellipse(
                    center,
                    width=max(2.2 * spread[0], 1e-3),
                    height=max(2.2 * spread[1], 1e-3),
                    facecolor="#F58518",
                    edgecolor="#B85C00",
                    alpha=0.15,
                    linewidth=1.5,
                    linestyle="--",
                )
                axes[3].add_patch(ellipse)
                axes[3].scatter(region[:, 0], region[:, 1], s=18, facecolors="none", edgecolors="#B85C00", alpha=0.55, linewidths=0.8)
                axes[3].text(
                    center[0],
                    center[1],
                    f"region {target_region_label}",
                    fontsize=8,
                    color="#7A3B00",
                    weight="bold",
                    ha="center",
                    va="center",
                )
        fig.colorbar(sc, ax=axes[3], fraction=0.046, pad=0.04, label=fr"$q_{target_region_label}(x_i)$")
        axes[3].set_title("D. Target-region probability under OT", fontsize=10)
        axes[3].text(
            0.03,
            0.96,
            "endpoint-level probability\nnot observed lineage",
            transform=axes[3].transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
        )
    elif target_labels is not None:
        fate = fate_probabilities(pi_ot, np.asarray(target_labels))
        prob_cols = [col for col in fate.columns if col.startswith("prob_")]
        color_col = "prob_rare" if "prob_rare" in prob_cols else prob_cols[-1]
        color = fate[color_col].to_numpy(dtype=float)
        sc = axes[3].scatter(X0_plot[:, 0], X0_plot[:, 1], c=color, s=18, cmap="plasma", alpha=0.82, linewidths=0)
        fig.colorbar(sc, ax=axes[3], fraction=0.046, pad=0.04, label=color_col.replace("prob_", "p(") + ")")
        axes[3].set_title("D. Label probability on source cells", fontsize=10)
    else:
        axes[3].set_title("D. Source cells under OT", fontsize=10)

    bary = barycentric_projection(X1_plot[:, :2], pi_ot)
    axes[4].scatter(bary[:, 0], bary[:, 1], s=14, color="#54A24B", alpha=0.82, linewidths=0, label="barycentric push-forward")
    axes[4].legend(frameon=False, fontsize=8, loc="lower right")
    axes[4].set_title("E. Barycentric push-forward in PHATE", fontsize=10)

    i_ot, j_ot, _ = _ch03_top_coupling_edges(pi_ot, max_edges=max_arrows // 2)
    draw_n = min(max_arrows // 2, len(i_ot))
    _ch03_arrows(axes[5], X0_plot[i_ot[:draw_n]], X1_plot[j_ot[:draw_n]], color="#4C78A8", alpha=0.18, linewidth=0.7)
    rng = np.random.default_rng(seed)
    random_i = rng.integers(0, len(X0), size=draw_n)
    random_j = rng.integers(0, len(X1), size=draw_n)
    _ch03_arrows(axes[5], X0_plot[random_i], X1_plot[random_j], color="#D95F02", alpha=0.13, linewidth=0.6, linestyle="--")
    axes[5].set_title("F. Same marginals, different couplings", fontsize=10)
    axes[5].text(
        0.03,
        0.96,
        "blue: OT edges\norange: independent pairs",
        transform=axes[5].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )

    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    return fig, axes


def plot_ch03_endpoint_coupling_is_not_dynamics(
    x0_pair,
    x1_pair,
    straight,
    curved,
    stochastic,
    background=None,
    max_lines: int = 80,
    highlight_indices=None,
    title: str = "Endpoint coupling is not dynamics",
):
    import matplotlib.pyplot as plt

    x0_pair = np.asarray(x0_pair, dtype=float)
    x1_pair = np.asarray(x1_pair, dtype=float)
    path_arrays = list(straight.values()) + list(curved.values()) + list(stochastic.values())
    xlim, ylim = _ch03_limits(
        x0_pair[:, :2],
        x1_pair[:, :2],
        *[np.asarray(a)[:, :2] for a in path_arrays],
        None if background is None else np.asarray(background)[:, :2],
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.95), sharex=True, sharey=True)
    idx = _ch03_subsample(len(x0_pair), max_lines, seed=0)
    if highlight_indices is None:
        highlight_indices = np.linspace(0, len(x0_pair) - 1, min(4, len(x0_pair))).astype(int)
    highlight_indices = np.asarray(highlight_indices, dtype=int)
    highlight_colors = ["#E45756", "#54A24B", "#B279A2", "#F58518", "#72B7B2"]

    for ax in axes:
        _ch03_axis(ax, xlim, ylim, "")
        ax.scatter(x0_pair[:, 0], x0_pair[:, 1], s=13, color="#4C78A8", alpha=0.72, linewidths=0)
        ax.scatter(x1_pair[:, 0], x1_pair[:, 1], s=13, color="#F58518", alpha=0.72, linewidths=0)

    _ch03_arrows(axes[0], x0_pair, x1_pair, indices=idx, color="0.35", alpha=0.30, linewidth=0.8, linestyle="--")
    axes[0].set_title("A. Endpoint coupling", fontsize=10)
    axes[0].text(0.05, 0.94, "who is paired with whom?", transform=axes[0].transAxes, va="top", fontsize=8)

    def _plot_paths(ax, paths, color, alpha=0.40):
        taus = sorted(paths)
        stacked = np.stack([np.asarray(paths[tau], dtype=float)[:, :2] for tau in taus], axis=0)
        for k in idx:
            ax.plot(stacked[:, k, 0], stacked[:, k, 1], color=color, alpha=alpha, linewidth=0.85)
        for tau in taus[1:-1]:
            pts = np.asarray(paths[tau], dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.20, linewidths=0)

    def _plot_highlights(ax, paths=None):
        if paths is not None:
            taus = sorted(paths)
            stacked = np.stack([np.asarray(paths[tau], dtype=float)[:, :2] for tau in taus], axis=0)
        for rank, k in enumerate(highlight_indices):
            color = highlight_colors[rank % len(highlight_colors)]
            if paths is None:
                ax.plot([x0_pair[k, 0], x1_pair[k, 0]], [x0_pair[k, 1], x1_pair[k, 1]], color=color, alpha=0.92, linewidth=2.0)
            else:
                ax.plot(stacked[:, k, 0], stacked[:, k, 1], color=color, alpha=0.92, linewidth=2.0)
            ax.scatter(x0_pair[k, 0], x0_pair[k, 1], s=34, color=color, edgecolors="white", linewidths=0.6, zorder=5)
            ax.scatter(x1_pair[k, 0], x1_pair[k, 1], s=34, color=color, edgecolors="white", linewidths=0.6, zorder=5)

    _plot_highlights(axes[0], None)

    _plot_paths(axes[1], straight, "#4C78A8", alpha=0.30)
    _plot_highlights(axes[1], straight)
    axes[1].set_title("B. Straight bridges", fontsize=10)
    if background is not None:
        bg = np.asarray(background, dtype=float)
        axes[2].scatter(bg[:, 0], bg[:, 1], s=5, color="0.72", alpha=0.13, linewidths=0)
    _plot_paths(axes[2], curved, "#54A24B", alpha=0.40)
    _plot_highlights(axes[2], curved)
    axes[2].set_title("C. Curved bridges", fontsize=10)
    axes[2].text(0.04, 0.94, "faint EB density background", transform=axes[2].transAxes, va="top", fontsize=8)
    for tau, pts in sorted(stochastic.items()):
        pts = np.asarray(pts, dtype=float)
        color = "0.55" if tau in (0.0, 1.0) else "#B279A2"
        alpha = 0.16 if tau not in (0.0, 1.0) else 0.08
        axes[3].scatter(pts[:, 0], pts[:, 1], s=10, color=color, alpha=alpha, linewidths=0)
    # Draw a few repeated stochastic samples for the same highlighted endpoints.
    from .paths import brownian_bridge_path

    taus = sorted(stochastic)
    for rank, k in enumerate(highlight_indices[:3]):
        color = highlight_colors[rank % len(highlight_colors)]
        x0 = x0_pair[[k]]
        x1 = x1_pair[[k]]
        for sample_seed in range(6):
            bridge = np.vstack(
                [
                    brownian_bridge_path(
                        x0,
                        x1,
                        np.array([tau]),
                        sigma=0.18,
                        seed=1000 + 37 * rank + sample_seed * 11 + int(round(100 * tau)),
                    )[0]
                    for tau in taus
                ]
            )
            axes[3].plot(bridge[:, 0], bridge[:, 1], color=color, alpha=0.28, linewidth=0.9)
    _plot_highlights(axes[3], stochastic)
    axes[3].set_title("D. Stochastic bridge samples", fontsize=10)

    fig.suptitle(title, y=1.04, fontsize=13)
    return fig, axes


def plot_ch03_dynamic_ot_density_velocity(
    x0_pair,
    x1_pair,
    economical_paths,
    detour_paths,
    energy_table=None,
    time_clouds=None,
    adjacent_arrows=None,
    max_arrows: int = 60,
    title: str = "Dynamic OT as density path and low-action transport",
):
    import matplotlib.pyplot as plt

    x0_pair = np.asarray(x0_pair, dtype=float)
    x1_pair = np.asarray(x1_pair, dtype=float)
    all_paths = list(economical_paths.values()) + list(detour_paths.values())
    time_arrays = [] if time_clouds is None else [np.asarray(v, dtype=float)[:, :2] for v in time_clouds.values()]
    arrow_arrays = [] if adjacent_arrows is None else [np.asarray(a, dtype=float)[:, :2] for pair in adjacent_arrows for a in pair]
    xlim, ylim = _ch03_limits(
        x0_pair[:, :2],
        x1_pair[:, :2],
        *[np.asarray(a)[:, :2] for a in all_paths],
        *time_arrays,
        *arrow_arrays,
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))
    idx = _ch03_subsample(len(x0_pair), max_arrows, seed=11)

    for ax in axes:
        _ch03_axis(ax, xlim, ylim, "")

    if time_clouds is None:
        axes[0].scatter(x0_pair[:, 0], x0_pair[:, 1], s=10, color="#4C78A8", alpha=0.55, linewidths=0)
        axes[0].scatter(x1_pair[:, 0], x1_pair[:, 1], s=10, color="#F58518", alpha=0.50, linewidths=0)
    else:
        times = sorted(time_clouds)
        palette = _time_palette(times, cmap_name="viridis")
        for t in times:
            pts = np.asarray(time_clouds[t], dtype=float)
            sample_idx = _ch03_subsample(len(pts), 700, seed=17 + len(str(t)))
            axes[0].scatter(pts[sample_idx, 0], pts[sample_idx, 1], s=6, color=palette[t], alpha=0.26, linewidths=0, label=str(t))
            center = pts[:, :2].mean(axis=0)
            axes[0].text(center[0], center[1], str(t), fontsize=8, weight="bold", color="0.20")
        axes[0].legend(frameon=False, fontsize=7, title="EB time", title_fontsize=7, loc="best")
    axes[0].set_title("A. EB empirical density path", fontsize=10)

    if time_clouds is not None:
        times = sorted(time_clouds)
        palette = _time_palette(times, cmap_name="viridis")
        for t in times:
            pts = np.asarray(time_clouds[t], dtype=float)
            sample_idx = _ch03_subsample(len(pts), 420, seed=23 + len(str(t)))
            axes[1].scatter(pts[sample_idx, 0], pts[sample_idx, 1], s=5, color=palette[t], alpha=0.17, linewidths=0)
    if adjacent_arrows is not None:
        for starts, ends in adjacent_arrows:
            starts = np.asarray(starts, dtype=float)
            ends = np.asarray(ends, dtype=float)
            n = min(len(starts), max_arrows)
            draw = _ch03_subsample(len(starts), n, seed=29)
            for k in draw:
                axes[1].annotate(
                    "",
                    xy=ends[k, :2],
                    xytext=starts[k, :2],
                    arrowprops={"arrowstyle": "->", "color": "0.20", "alpha": 0.25, "linewidth": 0.8},
                )
    else:
        _ch03_arrows(axes[1], x0_pair, x1_pair, indices=idx, color="0.30", alpha=0.25, linewidth=0.8)
    axes[1].set_title("B. Adjacent-time velocity intuition", fontsize=10)
    axes[1].text(
        0.03,
        0.96,
        "schematic adjacent-time\nOT-like arrows\nnot observed lineage",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )

    _ch03_axis(axes[2], xlim, ylim, "C. Low-action path vs detour")
    for paths, color, label, alpha, line_idx in [
        (economical_paths, "#54A24B", "economical", 0.32, idx),
        (detour_paths, "#E45756", "detour", 0.26, idx[: max(8, len(idx) // 3)]),
    ]:
        taus = sorted(paths)
        stacked = np.stack([np.asarray(paths[tau], dtype=float)[:, :2] for tau in taus], axis=0)
        for k in line_idx:
            axes[2].plot(stacked[:, k, 0], stacked[:, k, 1], color=color, alpha=alpha, linewidth=0.8)
        axes[2].plot([], [], color=color, label=label)
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    if energy_table is not None:
        try:
            lines = [
                f"{row['path_family']}: {float(row['energy_proxy']):.2f}"
                for _, row in energy_table.iterrows()
            ]
            axes[2].text(
                0.03,
                0.96,
                "energy proxy\n" + "\n".join(lines),
                transform=axes[2].transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
            )
        except Exception:
            pass
    fig.suptitle(title, y=1.03, fontsize=13)
    return fig, axes


def _ch03_toy_vector_field(x, t):
    x = np.asarray(x, dtype=float)
    t = float(t)
    drift = np.zeros_like(x)
    drift[:, 0] = 1.15 + 0.18 * np.cos(np.pi * t) - 0.08 * x[:, 1]
    drift[:, 1] = 0.46 * np.tanh(x[:, 0]) - 0.20 * x[:, 1] + 0.16 * np.sin(2 * np.pi * t)
    return drift


def _ch03_euler_integrate(x0, n_steps: int = 64, vector_field=None):
    vector_field = _ch03_toy_vector_field if vector_field is None else vector_field
    x = np.asarray(x0, dtype=float).copy()
    traj = [x.copy()]
    times = np.linspace(0.0, 1.0, int(n_steps) + 1)
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = t1 - t0
        x = x + dt * vector_field(x, t0)
        traj.append(x.copy())
    return np.stack(traj, axis=0), times


def plot_ch03_nf_to_cnf(
    initial_cloud=None,
    trajectories=None,
    vector_field=None,
    seed: int = 42,
    title: str = "From discrete flows to continuous neural transport",
):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    base = rng.normal(loc=(-1.35, -0.05), scale=(0.24, 0.20), size=(220, 2))

    def f1(x):
        return x @ np.array([[1.05, 0.20], [-0.12, 0.92]]) + np.array([0.55, 0.12])

    def f2(x):
        y = x.copy()
        y[:, 1] = y[:, 1] + 0.32 * np.sin(2.2 * y[:, 0])
        y[:, 0] = y[:, 0] + 0.28
        return y

    def f3(x):
        y = x.copy()
        r = np.linalg.norm(y, axis=1)
        theta = 0.55 * np.exp(-0.30 * r**2)
        c, s = np.cos(theta), np.sin(theta)
        out = np.column_stack([c * y[:, 0] - s * y[:, 1], s * y[:, 0] + c * y[:, 1]])
        return out + np.array([0.62, -0.03])

    clouds = [base]
    for transform in [f1, f2, f3]:
        clouds.append(transform(clouds[-1]))

    def toy_v(x, t):
        x = np.asarray(x, dtype=float)
        return np.column_stack(
            [
                1.10 - 0.18 * x[:, 1] + 0.12 * np.cos(np.pi * t),
                0.58 * np.sin(1.25 * x[:, 0]) - 0.18 * x[:, 1] + 0.10 * np.sin(2 * np.pi * t),
            ]
        )

    traj, times = _ch03_euler_integrate(base, n_steps=80, vector_field=toy_v)

    xlim, ylim = _ch03_limits(*(cloud[:, :2] for cloud in clouds), *(traj[k, :, :2] for k in [0, 25, 50, 80]), pad=0.28)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.95), sharex=True, sharey=True)
    for ax in axes:
        _ch03_axis(ax, xlim, ylim, "")

    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518"]
    for k, cloud in enumerate(clouds):
        axes[0].scatter(cloud[:, 0], cloud[:, 1], s=8, color=colors[k], alpha=0.38, linewidths=0)
        if k < len(clouds) - 1:
            _flow_arrow(axes[0], cloud.mean(axis=0), clouds[k + 1].mean(axis=0), color="0.25", linewidth=1.4)
            mid = 0.5 * (cloud.mean(axis=0) + clouds[k + 1].mean(axis=0))
            axes[0].text(mid[0], mid[1] + 0.10, f"$f_{k+1}$", fontsize=9, color="0.20", ha="center")
    axes[0].text(
        0.50,
        0.94,
        r"$x_K=f_K\circ\cdots\circ f_1(x_0)$",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )
    axes[0].set_title("A. Normalizing flow as layer composition", fontsize=10)

    grid_x, grid_y = np.meshgrid(np.linspace(xlim[0], xlim[1], 12), np.linspace(ylim[0], ylim[1], 10))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    v = toy_v(grid, 0.35)
    axes[1].quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], color="0.25", alpha=0.58, width=0.004)
    point = np.array([[-0.65, -0.25]])
    step = point + 0.20 * toy_v(point, 0.35)
    axes[1].scatter(point[:, 0], point[:, 1], s=34, color="#4C78A8", zorder=5)
    axes[1].scatter(step[:, 0], step[:, 1], s=34, color="#F58518", zorder=5)
    axes[1].annotate("", xy=step[0], xytext=point[0], arrowprops={"arrowstyle": "->", "color": "#E45756", "linewidth": 2.0})
    axes[1].text(
        0.50,
        0.93,
        r"$x_{k+1}=x_k+\Delta t\,u_\theta(x_k)$",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )
    axes[1].set_title("B. Residual layer as Euler step", fontsize=10)

    idx = _ch03_subsample(traj.shape[1], 42, seed=seed + 2)
    for k in idx:
        axes[2].plot(traj[:, k, 0], traj[:, k, 1], color="#4C78A8", alpha=0.46, linewidth=1.0)
    axes[2].scatter(traj[0, :, 0], traj[0, :, 1], s=7, color="#4C78A8", alpha=0.22, linewidths=0)
    axes[2].scatter(traj[-1, :, 0], traj[-1, :, 1], s=7, color="#F58518", alpha=0.28, linewidths=0)
    axes[2].text(
        0.50,
        0.93,
        r"$dX_t/dt=v_\theta(X_t,t,c)$",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )
    axes[2].set_title("C. CNF as ODE flow", fontsize=10)

    show = [0, len(times) // 2, len(times) - 1]
    density_colors = ["#4C78A8", "#54A24B", "#F58518"]
    labels = [r"$\rho_0$", r"$\rho_t$", r"$\rho_1$"]
    for color, step, label in zip(density_colors, show, labels):
        pts = traj[step]
        _covariance_ellipse(axes[3], pts[:, :2], color, alpha=0.12, scale=3.2)
        axes[3].scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.25, linewidths=0, label=label)
    axes[3].legend(frameon=False, fontsize=8, loc="lower right")
    axes[3].text(
        0.50,
        0.93,
        r"$\rho_t=(\psi_t^\theta)_\#\rho_0$",
        transform=axes[3].transAxes,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0},
    )
    axes[3].text(0.03, 0.08, "conceptual schematic", transform=axes[3].transAxes, fontsize=8, color="0.30")
    axes[3].set_title("D. Induced density path", fontsize=10)
    fig.suptitle(title, y=1.04, fontsize=13)
    return fig, axes


def plot_ch03_cnf_training_bottleneck(
    title: str = "From simulation-heavy CNF training to simulation-free velocity regression",
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    left_steps = [
        "sample x",
        "solve learned ODE",
        "track log-density",
        "compute likelihood or endpoint loss",
        "backprop through solver",
    ]
    right_steps = [
        "sample endpoint / path condition",
        "sample time t",
        "construct path point",
        "compute target local velocity",
        r"regress $v_\theta(x,t)$",
    ]
    panels = [
        (axes[0], left_steps, "#4C78A8", "Traditional CNF training", "ODE rollout inside training loop\nlog-density and divergence tracking\nadjoint sensitivity"),
        (axes[1], right_steps, "#54A24B", "Flow Matching training", "no learned ODE solve in training loss\nODE solver still used for sampling\nsame CNF vector field"),
    ]
    for ax, steps, color, panel_title, note in panels:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        y_positions = np.linspace(0.82, 0.22, len(steps))
        centers = []
        for y, step in zip(y_positions, steps):
            box = FancyBboxPatch(
                (0.18, y - 0.045),
                0.64,
                0.09,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.1,
                edgecolor=color,
                facecolor="white",
            )
            ax.add_patch(box)
            ax.text(0.50, y, step, ha="center", va="center", fontsize=9, color="0.15")
            centers.append((0.50, y))
        for start, end in zip(centers[:-1], centers[1:]):
            arrow = FancyArrowPatch(
                (start[0], start[1] - 0.052),
                (end[0], end[1] + 0.052),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="0.32",
            )
            ax.add_patch(arrow)
        ax.text(0.50, 0.95, panel_title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(
            0.50,
            0.06,
            note,
            ha="center",
            va="center",
            fontsize=8.5,
            color="0.28",
            bbox={"facecolor": "white", "edgecolor": "0.84", "pad": 3.0},
        )
    axes[0].text(0.50, 0.13, r"learn $v_\theta$", ha="center", fontsize=10, color="#4C78A8")
    axes[1].text(0.50, 0.13, r"learn the same $v_\theta$", ha="center", fontsize=10, color="#54A24B")
    fig.suptitle(title, y=0.99, fontsize=13)
    return fig, axes


def plot_ch03_real_eb_time_marginals(eb_dataset, max_points_per_time: int = 900):
    import matplotlib.pyplot as plt

    times = list(eb_dataset.timepoints)
    colors = _time_palette(times, cmap_name="viridis")
    xlim, ylim = _ch03_limits(eb_dataset.X[:, :2], pad=0.5)
    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    _ch03_axis(ax, xlim, ylim, "Optional EB PHATE time marginals")
    for i, t in enumerate(times):
        X = eb_dataset.cells_at(t, condition="eb_npz")
        idx = _ch03_subsample(len(X), max_points_per_time, seed=100 + i)
        ax.scatter(X[idx, 0], X[idx, 1], s=6, color=colors[t], alpha=0.38, linewidths=0, label=f"time {t}")
        center = X[idx, :2].mean(axis=0)
        ax.text(center[0], center[1], str(t), fontsize=9, weight="bold", color="0.18")
    ax.legend(frameon=False, fontsize=8, title="sample_labels", title_fontsize=8, loc="best")
    fig.suptitle("Real time-course snapshots remain unpaired marginals", y=0.99, fontsize=13)
    return fig, ax


def _ch04_limits(*arrays, pad: float = 0.25):
    chunks = [np.asarray(a, dtype=float).reshape(-1, 2) for a in arrays if a is not None and np.asarray(a).size]
    X = np.vstack(chunks)
    span = np.maximum(X.max(axis=0) - X.min(axis=0), 1e-6)
    lo = X.min(axis=0) - pad * span
    hi = X.max(axis=0) + pad * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _ch04_robust_limits(*arrays, q_low: float = 1.0, q_high: float = 99.0, margin: float = 0.08):
    """Return shared 2D limits using robust percentiles, not raw outliers."""
    chunks = [np.asarray(a, dtype=float).reshape(-1, 2) for a in arrays if a is not None and np.asarray(a).size]
    if not chunks:
        raise ValueError("At least one nonempty 2D array is required")
    X = np.vstack(chunks)
    X = X[np.isfinite(X).all(axis=1)]
    if X.size == 0:
        raise ValueError("No finite points available for axis limits")
    p_lo = np.percentile(X, float(q_low), axis=0)
    p_hi = np.percentile(X, float(q_high), axis=0)
    q1 = np.percentile(X, 25.0, axis=0)
    q3 = np.percentile(X, 75.0, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    fence_lo = q1 - 1.5 * iqr
    fence_hi = q3 + 1.5 * iqr
    in_fence = np.all((X >= fence_lo) & (X <= fence_hi), axis=1)
    if in_fence.sum() >= max(4, min(20, X.shape[0] // 4)):
        core = X[in_fence]
        c_lo = np.percentile(core, float(q_low), axis=0)
        c_hi = np.percentile(core, float(q_high), axis=0)
        lo = np.maximum(p_lo, c_lo)
        hi = np.minimum(p_hi, c_hi)
    else:
        lo = p_lo
        hi = p_hi
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - float(margin) * span
    hi = hi + float(margin) * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _ch04_clip_vectors(vectors, max_norm: float | None = None):
    """Clip vector norms so a few large arrows do not dominate a plot."""
    vectors = np.asarray(vectors, dtype=float)
    if max_norm is None:
        norms = np.linalg.norm(vectors, axis=-1)
        positive = norms[np.isfinite(norms) & (norms > 0)]
        max_norm = float(np.percentile(positive, 90)) if positive.size else 1.0
    max_norm = max(float(max_norm), 1e-12)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.clip(norms, 1e-12, None))
    return vectors * scale


def _ch04_local_average_vectors(points, vectors, xlim, ylim, bins: int = 7, min_count: int = 1):
    """Average conditional velocities in populated local neighborhoods."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 2)
    if points.shape != vectors.shape:
        raise ValueError("points and vectors must have matching (n, 2) shapes")
    bins = int(bins)
    x_edges = np.linspace(float(xlim[0]), float(xlim[1]), bins + 1)
    y_edges = np.linspace(float(ylim[0]), float(ylim[1]), bins + 1)
    ix = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    iy = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    valid = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins) & np.isfinite(points).all(axis=1) & np.isfinite(vectors).all(axis=1)
    centers = []
    avg_vectors = []
    counts = []
    for bx in range(bins):
        for by in range(bins):
            mask = valid & (ix == bx) & (iy == by)
            count = int(mask.sum())
            if count < int(min_count):
                continue
            centers.append([(x_edges[bx] + x_edges[bx + 1]) / 2, (y_edges[by] + y_edges[by + 1]) / 2])
            avg_vectors.append(vectors[mask].mean(axis=0))
            counts.append(count)
    return np.asarray(centers, dtype=float), np.asarray(avg_vectors, dtype=float), np.asarray(counts, dtype=int)


def _ch04_axis(ax, xlim, ylim, title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PHATE state 1 (standardized)")
    ax.set_ylabel("PHATE state 2 (standardized)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ch04_subsample(n: int, max_points: int | None, seed: int = 42):
    if max_points is None or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_points), replace=False))


def _ch04_box(ax, xy, text, color="#4C78A8", width=0.24, height=0.12, fontsize=9):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="0.15")
    return box


def _ch04_arrow(ax, start, end, color="0.3", linewidth=1.2, mutation_scale=12, alpha=0.9):
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(arrow)
    return arrow


def plot_training_vs_sampling_schematic(title: str = "Flow Matching changes the training objective, not the sampler"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    panels = [
        (
            "Simulation-dependent\nrollout training",
            ["sample endpoints", "roll out learned ODE", "endpoint / likelihood loss", "backprop through solver"],
            "#4C78A8",
        ),
        (
            "FM local regression",
            ["sample chosen pair", "sample path time s", "construct x_s", "regress target velocity"],
            "#54A24B",
        ),
        (
            "Post-training sampling",
            ["draw source samples", "integrate learned ODE", "compare endpoint distribution"],
            "#E45756",
        ),
    ]
    for ax, (heading, steps, color) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.93, heading, ha="center", va="center", fontsize=11, weight="bold", color=color)
        ys = np.linspace(0.75, 0.25, len(steps))
        for y, step in zip(ys, steps):
            _ch04_box(ax, (0.5, y), step, color=color, width=0.68, height=0.11)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            _ch04_arrow(ax, (0.5, y0 - 0.06), (0.5, y1 + 0.06), color="0.35")
    fig.suptitle(title, y=1.02, fontsize=13)
    return fig, axes


def plot_velocity_targets(X0, X1, x_s=None, u_s=None, max_points: int = 350, max_arrows: int = 80):
    import matplotlib.pyplot as plt

    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    xlim, ylim = _ch04_limits(X0, X1, x_s, pad=0.15)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _ch04_axis(ax, xlim, ylim, "Endpoint pairs induce local velocity targets")
    i0 = _ch04_subsample(len(X0), max_points, seed=10)
    i1 = _ch04_subsample(len(X1), max_points, seed=11)
    ax.scatter(X0[i0, 0], X0[i0, 1], s=8, color="#4C78A8", alpha=0.35, linewidths=0, label="source snapshot")
    ax.scatter(X1[i1, 0], X1[i1, 1], s=8, color="#F58518", alpha=0.35, linewidths=0, label="target snapshot")
    if x_s is not None and u_s is not None:
        x_s = np.asarray(x_s, dtype=float)
        u_s = np.asarray(u_s, dtype=float)
        idx = _ch04_subsample(len(x_s), min(max_arrows, len(x_s)), seed=12)
        ax.quiver(
            x_s[idx, 0],
            x_s[idx, 1],
            u_s[idx, 0],
            u_s[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=8,
            width=0.004,
            color="#54A24B",
            alpha=0.72,
            label="constructed conditional velocity",
        )
    else:
        idx = _ch04_subsample(min(len(X0), len(X1)), min(max_arrows, len(X0), len(X1)), seed=13)
        for i in idx:
            ax.plot([X0[i, 0], X1[i, 0]], [X0[i, 1], X1[i, 1]], color="0.35", linewidth=0.6, alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return fig, ax


def plot_conditional_vs_marginal_velocity(x_s, u_s, center=None, radius_quantile: float = 0.22, max_arrows: int = 180):
    import matplotlib.pyplot as plt

    x_s = np.asarray(x_s, dtype=float)
    u_s = np.asarray(u_s, dtype=float)
    if center is None:
        center = np.median(x_s, axis=0)
    center = np.asarray(center, dtype=float)
    dist = np.linalg.norm(x_s - center[None, :], axis=1)
    radius = np.quantile(dist, radius_quantile)
    local = dist <= max(radius, 1e-6)
    idx = np.flatnonzero(local)
    if len(idx) > max_arrows:
        idx = idx[_ch04_subsample(len(idx), max_arrows, seed=21)]
    mean_v = u_s[idx].mean(axis=0) if len(idx) else np.zeros(2)
    xlim, ylim = _ch04_limits(x_s, x_s[idx] if len(idx) else x_s, pad=0.18)
    fig, ax = plt.subplots(figsize=(6.0, 5.1))
    _ch04_axis(ax, xlim, ylim, "Many conditional velocities average to a marginal vector")
    ax.scatter(x_s[:, 0], x_s[:, 1], s=5, color="0.78", alpha=0.35, linewidths=0)
    ax.scatter(x_s[idx, 0], x_s[idx, 1], s=12, color="#4C78A8", alpha=0.48, linewidths=0)
    if len(idx):
        ax.quiver(
            x_s[idx, 0],
            x_s[idx, 1],
            u_s[idx, 0],
            u_s[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=9,
            width=0.0035,
            color="#72B7B2",
            alpha=0.56,
        )
    ax.quiver(
        [center[0]],
        [center[1]],
        [mean_v[0]],
        [mean_v[1]],
        angles="xy",
        scale_units="xy",
        scale=3.5,
        width=0.012,
        color="#E45756",
        alpha=0.95,
        label="marginal vector field estimate",
    )
    ax.legend(frameon=False, fontsize=8, loc="best")
    return fig, ax


def plot_cfm_object_ladder(X0, X1, x_s=None, u_s=None, grid=None, vector_field=None):
    import matplotlib.pyplot as plt

    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    x_s = None if x_s is None else np.asarray(x_s, dtype=float)
    u_s = None if u_s is None else np.asarray(u_s, dtype=float)
    xlim, ylim = _ch04_robust_limits(X0, X1, x_s, q_low=1, q_high=99, margin=0.10)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.45), sharex=True, sharey=True, constrained_layout=True)
    titles = [
        "Single conditional path\nand constructed velocity",
        "Many conditional paths crossing and mixing",
        "Marginal vector field:\nposterior/local average",
    ]
    for ax, title in zip(axes, titles):
        _ch04_axis(ax, xlim, ylim, title)
        ax.scatter(X0[:, 0], X0[:, 1], s=5, color="#4C78A8", alpha=0.14, linewidths=0)
        ax.scatter(X1[:, 0], X1[:, 1], s=5, color="#F58518", alpha=0.14, linewidths=0)
        ax.set_ylabel("")
    axes[0].set_ylabel("PHATE state 2 (standardized)")

    n = min(len(X0), len(X1))
    one_idx = int(_ch04_subsample(n, 1, seed=33)[0])
    axes[0].plot([X0[one_idx, 0], X1[one_idx, 0]], [X0[one_idx, 1], X1[one_idx, 1]], color="0.22", alpha=0.78, linewidth=1.5)
    mid = 0.52 * X1[one_idx] + 0.48 * X0[one_idx]
    vel = X1[one_idx] - X0[one_idx]
    axes[0].quiver(
        [mid[0]],
        [mid[1]],
        [vel[0]],
        [vel[1]],
        angles="xy",
        scale_units="xy",
        scale=3.5,
        width=0.012,
        color="#54A24B",
        label="constructed conditional velocity",
    )
    axes[0].scatter([X0[one_idx, 0]], [X0[one_idx, 1]], s=42, color="#4C78A8", edgecolor="white", linewidth=0.6, zorder=3)
    axes[0].scatter([X1[one_idx, 0]], [X1[one_idx, 1]], s=42, color="#F58518", edgecolor="white", linewidth=0.6, zorder=3)
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    idx = _ch04_subsample(n, min(95, n), seed=32)
    for i in idx:
        axes[1].annotate(
            "",
            xy=X1[i],
            xytext=X0[i],
            arrowprops={"arrowstyle": "->", "color": "0.22", "alpha": 0.18, "linewidth": 0.58},
        )

    if grid is not None and vector_field is not None:
        gx, gy = grid
        vf = _ch04_clip_vectors(np.asarray(vector_field))
        axes[2].quiver(gx, gy, vf[..., 0], vf[..., 1], color="#E45756", alpha=0.88, scale=18, width=0.006)
    elif x_s is not None and u_s is not None:
        faint_idx = _ch04_subsample(len(x_s), min(70, len(x_s)), seed=31)
        axes[2].quiver(
            x_s[faint_idx, 0],
            x_s[faint_idx, 1],
            u_s[faint_idx, 0],
            u_s[faint_idx, 1],
            angles="xy",
            scale_units="xy",
            color="#72B7B2",
            alpha=0.16,
            scale=13,
            width=0.0025,
        )
        centers, avg, counts = _ch04_local_average_vectors(x_s, u_s, xlim=xlim, ylim=ylim, bins=6, min_count=4)
        if len(centers):
            keep = np.argsort(counts)[-min(18, len(counts)) :]
            avg = _ch04_clip_vectors(avg[keep])
            centers = centers[keep]
            counts = counts[keep]
            axes[2].scatter(centers[:, 0], centers[:, 1], s=np.clip(counts * 8, 24, 110), color="#E45756", alpha=0.20, linewidths=0)
            axes[2].quiver(
                centers[:, 0],
                centers[:, 1],
                avg[:, 0],
                avg[:, 1],
                angles="xy",
                scale_units="xy",
                color="#E45756",
                alpha=0.94,
                scale=7,
                width=0.010,
                label="posterior/local average",
            )
            axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Conditional velocities become a marginal vector field by local averaging", y=1.06, fontsize=13)
    return fig, axes


def plot_velocity_mlp_architecture(x_dim: int = 2, hidden_dim: int = 128, condition_dim: int = 0, n_parameters: int | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.84, bottom=0.14)
    ax.set_xlim(0, 1.24)
    ax.set_ylim(0, 1)
    ax.axis("off")
    inputs = [
        ("state x_s", (0.12, 0.72), "#4C78A8"),
        ("time s", (0.12, 0.50), "#F58518"),
        ("condition c\noptional", (0.12, 0.28), "#B279A2"),
    ]
    for label, xy, color in inputs:
        _ch04_box(ax, xy, label, color=color, width=0.18, height=0.13)
        _ch04_arrow(ax, (xy[0] + 0.10, xy[1]), (0.33, 0.50), color=color)
    _ch04_box(ax, (0.40, 0.50), "concat\n[x_s, s, c]", color="#72B7B2", width=0.18, height=0.17)
    _ch04_arrow(ax, (0.50, 0.50), (0.61, 0.50), color="0.35")
    _ch04_box(ax, (0.70, 0.50), f"MLP\nhidden={hidden_dim}", color="#54A24B", width=0.22, height=0.20)
    _ch04_arrow(ax, (0.82, 0.50), (0.98, 0.50), color="0.35")
    output_label = f"output velocity\nv_theta(x_s,s,c) in R^d\nR^{x_dim} example"
    _ch04_box(ax, (1.10, 0.50), output_label, color="#E45756", width=0.23, height=0.21, fontsize=8.4)
    note = f"Output shape == input state shape. Here the EB PHATE example has d={x_dim}."
    if n_parameters is not None:
        note += f" Trainable parameters: {int(n_parameters):,}."
    ax.text(0.62, 0.12, note, ha="center", va="center", fontsize=10, color="0.25")
    fig.suptitle("VelocityMLP maps local state and interpolation time to velocity", y=0.98, fontsize=13)
    return fig, ax


def plot_batch_prediction_vs_target(x_s, u_s, pred, max_arrows: int = 120):
    import matplotlib.pyplot as plt

    x_s = np.asarray(x_s, dtype=float)
    u_s = np.asarray(u_s, dtype=float)
    pred = np.asarray(pred, dtype=float)
    idx = _ch04_subsample(len(x_s), min(max_arrows, len(x_s)), seed=41)
    xlim, ylim = _ch04_limits(x_s, pad=0.18)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, sharey=True)
    for ax, vec, title, color in [
        (axes[0], u_s, "Path-induced target velocity", "#54A24B"),
        (axes[1], pred, "Model-predicted velocity", "#E45756"),
    ]:
        _ch04_axis(ax, xlim, ylim, title)
        ax.scatter(x_s[:, 0], x_s[:, 1], s=6, color="0.78", alpha=0.4, linewidths=0)
        ax.quiver(
            x_s[idx, 0],
            x_s[idx, 1],
            vec[idx, 0],
            vec[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=10,
            width=0.004,
            color=color,
            alpha=0.72,
        )
    return fig, axes


def plot_training_speed_comparison(history):
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.55))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.25, wspace=0.26)
    colors = {
        "cfm_local_regression": "#54A24B",
        "simulation_dependent_rollout_endpoint": "#E45756",
    }
    summary = history.groupby("method", sort=False).agg(
        sec_per_step=("sec_per_step", "mean"),
        nfe_train_per_step=("nfe_train_per_step", "mean"),
    )
    x = np.arange(len(summary))
    labels = ["CFM local\nregression" if m == "cfm_local_regression" else "solver-in-loop\nproxy" for m in summary.index]
    colors_ordered = [colors.get(m, "0.4") for m in summary.index]
    axes[0].bar(x, summary["sec_per_step"], color=colors_ordered, width=0.58)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("seconds per training step")
    axes[0].set_title("Panel A: wall-clock cost")
    axes[1].bar(x, summary["nfe_train_per_step"], color=colors_ordered, width=0.58)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("network evaluations per step")
    axes[1].set_title("Panel B: NFE inside training loss")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.90", linewidth=0.8)
        for patch in ax.patches:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, height, f"{height:.3g}", ha="center", va="bottom", fontsize=8)
    fig.text(
        0.5,
        0.055,
        "CFM loss uses one network evaluation and no learned-ODE rollout; the pedagogical solver-in-loop proxy repeatedly calls the velocity network through numerical integration.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.28",
    )
    fig.suptitle("Training-time cost: local regression versus solver-in-loop baseline", y=0.96, fontsize=13)
    return fig, axes


def plot_vector_field_2d(
    model,
    xlim,
    ylim,
    device="cpu",
    grid_size: int = 24,
    s_value: float = 0.5,
    ax=None,
    color="#54A24B",
    clip_quantile: float = 90.0,
    normalize: bool = False,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    import torch

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.8))
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(pts, dtype=torch.float32, device=device)
        s = torch.full((x.shape[0], 1), float(s_value), dtype=torch.float32, device=device)
        v = model(x, s).detach().cpu().numpy().reshape(grid_size, grid_size, 2)
    if was_training:
        model.train()
    norms = np.linalg.norm(v.reshape(-1, 2), axis=1)
    positive = norms[norms > 0]
    max_norm = float(np.percentile(positive, clip_quantile)) if positive.size else 1.0
    v_plot = _ch04_clip_vectors(v, max_norm=max_norm)
    if normalize:
        n = np.linalg.norm(v_plot, axis=-1, keepdims=True)
        median_norm = float(np.percentile(n[n > 0], 60)) if np.any(n > 0) else max_norm
        v_plot = v_plot / np.clip(n, 1e-12, None) * min(median_norm, max_norm)
    ax.quiver(gx, gy, v_plot[..., 0], v_plot[..., 1], color=color, alpha=0.72, scale=24)
    _ch04_axis(ax, xlim, ylim, title or f"Learned Eulerian field on plotted EB region (s={s_value:.2f})")
    return ax, (gx, gy), v


def plot_sampled_trajectories(traj, target=None, times=(0.0, 0.25, 0.5, 0.75, 1.0), max_points: int = 450, max_lines: int = 35):
    import matplotlib.pyplot as plt

    traj = np.asarray(traj, dtype=float)
    target = None if target is None else np.asarray(target, dtype=float)
    xlim, ylim = _ch04_robust_limits(traj.reshape(-1, traj.shape[-1])[:, :2], target, q_low=1, q_high=99, margin=0.10)
    fig, axes = plt.subplots(1, len(times), figsize=(2.75 * len(times), 3.35), sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    step_positions = np.round(np.asarray(times) * (traj.shape[0] - 1)).astype(int)
    for ax, tau, step in zip(axes, times, step_positions):
        X = traj[step]
        idx = _ch04_subsample(len(X), max_points, seed=50 + int(step))
        if target is not None:
            tidx = _ch04_subsample(len(target), max_points, seed=60)
            ax.scatter(target[tidx, 0], target[tidx, 1], s=5, color="#F58518", alpha=0.18, linewidths=0)
        ax.scatter(X[idx, 0], X[idx, 1], s=6, color="#4C78A8", alpha=0.46, linewidths=0)
        line_idx = _ch04_subsample(traj.shape[1], min(max_lines, traj.shape[1]), seed=70)
        for j in line_idx:
            ax.plot(traj[: step + 1, j, 0], traj[: step + 1, j, 1], color="0.25", linewidth=0.45, alpha=0.22)
        _ch04_axis(ax, xlim, ylim, f"s={tau:.2f}")
    fig.suptitle("Generated samples at s=0,...,1 from integrating the learned velocity field", y=1.03, fontsize=13)
    return fig, axes


def plot_endpoint_fit(X_source, X_pred, X_target, max_points: int = 650):
    import matplotlib.pyplot as plt

    X_source = np.asarray(X_source, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    xlim, ylim = _ch04_limits(X_source, X_pred, X_target, pad=0.15)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True, sharey=True)
    panels = [
        (X_source, "Source EB time 1", "#4C78A8", 81),
        (X_pred, "Generated endpoint", "#54A24B", 82),
        (X_target, "Target EB time 2", "#F58518", 83),
    ]
    for ax, X, title, color, seed in zip(axes, *zip(*panels)):
        idx = _ch04_subsample(len(X), max_points, seed=seed)
        _ch04_axis(ax, xlim, ylim, title)
        ax.scatter(X[idx, 0], X[idx, 1], s=7, color=color, alpha=0.45, linewidths=0)
    fig.suptitle("Endpoint agreement in the pedagogical PHATE state space", y=1.03, fontsize=13)
    return fig, axes


def plot_training_vs_sampling_pushforward(
    x_s,
    u_s,
    target,
    generated,
    traj,
    max_points: int = 500,
    max_arrows: int = 110,
    max_lines: int = 35,
):
    import matplotlib.pyplot as plt

    x_s = np.asarray(x_s, dtype=float)
    u_s = np.asarray(u_s, dtype=float)
    target = np.asarray(target, dtype=float)
    generated = np.asarray(generated, dtype=float)
    traj = np.asarray(traj, dtype=float)
    xlim, ylim = _ch04_robust_limits(
        x_s,
        target,
        generated,
        traj.reshape(-1, traj.shape[-1])[:, :2],
        q_low=2,
        q_high=98,
        margin=0.10,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65), sharex=True, sharey=True, constrained_layout=True)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("PHATE state 1 (standardized)")
    axes[0].set_ylabel("PHATE state 2 (standardized)")

    qidx = _ch04_subsample(len(x_s), min(max_arrows, len(x_s)), seed=110)
    axes[0].scatter(x_s[:, 0], x_s[:, 1], s=5, color="0.72", alpha=0.32, linewidths=0)
    axes[0].quiver(
        x_s[qidx, 0],
        x_s[qidx, 1],
        u_s[qidx, 0],
        u_s[qidx, 1],
        angles="xy",
        scale_units="xy",
        scale=9,
        width=0.004,
        color="#54A24B",
        alpha=0.65,
    )
    axes[0].set_title("Training: local regression samples")

    tidx = _ch04_subsample(len(target), max_points, seed=111)
    gidx = _ch04_subsample(len(generated), max_points, seed=112)
    axes[1].scatter(target[tidx, 0], target[tidx, 1], s=6, color="#F58518", alpha=0.22, linewidths=0, label="target")
    axes[1].scatter(generated[gidx, 0], generated[gidx, 1], s=6, color="#4C78A8", alpha=0.46, linewidths=0, label="generated")
    line_idx = _ch04_subsample(traj.shape[1], min(max_lines, traj.shape[1]), seed=113)
    for j in line_idx:
        axes[1].plot(traj[:, j, 0], traj[:, j, 1], color="0.25", linewidth=0.45, alpha=0.22)
    axes[1].set_title("Sampling: integrate learned ODE")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Training-time local targets versus sampling-time EB pushforward", y=1.03, fontsize=13)
    return fig, axes


def plot_solver_step_sensitivity(results, target=None, max_points: int = 550):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = list(results)
    arrays = [np.asarray(item["samples"], dtype=float) for item in panels]
    xlim, ylim = _ch04_robust_limits(*(arrays + ([target] if target is not None else [])), q_low=1, q_high=99, margin=0.10)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.85 * n, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.22, wspace=0.08)
    axes = np.asarray(axes).reshape(-1)
    for ax, item, X in zip(axes, panels, arrays):
        if target is not None:
            tidx = _ch04_subsample(len(target), max_points, seed=91)
            ax.scatter(target[tidx, 0], target[tidx, 1], s=5, color="#F58518", alpha=0.18, linewidths=0)
        idx = _ch04_subsample(len(X), max_points, seed=92)
        ax.scatter(X[idx, 0], X[idx, 1], s=6, color="#4C78A8", alpha=0.45, linewidths=0)
        title = f"{item.get('sampler', '')}\nK={item.get('steps', 'adaptive')}, NFE={item.get('nfe', '?')}"
        _ch04_axis(ax, xlim, ylim, title)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C78A8", markeredgecolor="none", markersize=6, label="generated"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F58518", markeredgecolor="none", markersize=6, label="target"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Sampling-time numerical integration diagnostic: solver step count sensitivity", y=0.96, fontsize=13)
    return fig, axes


def plot_nfe_vs_error(solver_table, y: str = "endpoint_sliced_w2"):
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(solver_table).copy()
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    markers = {"euler": "o", "midpoint": "s", "dopri5": "^"}
    colors = {"euler": "#4C78A8", "midpoint": "#54A24B", "dopri5": "#E45756"}
    for sampler, group in table.groupby("sampler", sort=False):
        group = group.sort_values("nfe")
        ax.plot(
            group["nfe"],
            group[y],
            marker=markers.get(sampler, "o"),
            color=colors.get(sampler, "0.25"),
            linewidth=1.6,
            label=sampler,
        )
    ax.set_xlabel("number of function evaluations (NFE)")
    ax.set_ylabel(y.replace("_", " "))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sampling trade-off: NFE versus endpoint error", y=1.02, fontsize=13)
    return fig, ax


def plot_training_loop_diagram():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.8, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        ("chosen\nendpoint pair", (0.12, 0.55), "#4C78A8"),
        ("sample s", (0.30, 0.55), "#F58518"),
        ("construct x_s", (0.48, 0.55), "#72B7B2"),
        ("target u_s", (0.66, 0.70), "#54A24B"),
        ("VelocityMLP", (0.66, 0.40), "#B279A2"),
        ("local MSE", (0.84, 0.55), "#E45756"),
    ]
    for text, xy, color in nodes:
        _ch04_box(ax, xy, text, color=color, width=0.15, height=0.13)
    for start, end in [((0.20, 0.55), (0.22, 0.55)), ((0.38, 0.55), (0.40, 0.55)), ((0.56, 0.58), (0.58, 0.67)), ((0.56, 0.52), (0.58, 0.43)), ((0.74, 0.70), (0.77, 0.58)), ((0.74, 0.40), (0.77, 0.52))]:
        _ch04_arrow(ax, start, end, color="0.35")
    ax.text(
        0.50,
        0.15,
        "Training is local velocity regression along a path induced by the chosen coupling.",
        ha="center",
        va="center",
        fontsize=10,
        color="0.25",
    )
    fig.suptitle("Minimal endpoint-pair CFM training loop", y=0.98, fontsize=13)
    return fig, ax


def _ch05_pair_panel(ax, X0, X1, pair_x0, pair_x1, title, line_color="0.72", max_points: int = 450, max_lines: int = 90):
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    pair_x0 = np.asarray(pair_x0, dtype=float)
    pair_x1 = np.asarray(pair_x1, dtype=float)
    xlim, ylim = _ch04_robust_limits(X0, X1, pair_x0, pair_x1, q_low=1, q_high=99, margin=0.10)
    _ch04_axis(ax, xlim, ylim, title)
    i0 = _ch04_subsample(len(X0), max_points, seed=501)
    i1 = _ch04_subsample(len(X1), max_points, seed=502)
    ax.scatter(X0[i0, 0], X0[i0, 1], s=7, color="#4C78A8", alpha=0.30, linewidths=0, label="source")
    ax.scatter(X1[i1, 0], X1[i1, 1], s=7, color="#E45756", alpha=0.28, linewidths=0, label="target")
    line_idx = _ch04_subsample(len(pair_x0), min(max_lines, len(pair_x0)), seed=503)
    for j in line_idx:
        ax.plot([pair_x0[j, 0], pair_x1[j, 0]], [pair_x0[j, 1], pair_x1[j, 1]], color=line_color, linewidth=0.65, alpha=0.42)
    return ax


def plot_ch05_pairing_panels(X0, X1, panels, max_points: int = 450, max_lines: int = 90, title: str | None = None):
    """Draw source/target clouds with sampled endpoint chords for one or more pair samplers."""
    import matplotlib.pyplot as plt

    panels = list(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.35), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, panel in zip(axes, panels):
        _ch05_pair_panel(
            ax,
            X0,
            X1,
            panel["x0"],
            panel["x1"],
            panel.get("title", "endpoint pairs"),
            line_color=panel.get("color", "0.72"),
            max_points=max_points,
            max_lines=max_lines,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.18, wspace=0.18)
    fig.suptitle(title or "Endpoint coupling changes which chords train the same CFM loss", y=0.97, fontsize=13)
    return fig, axes


def plot_ch05_sinkhorn_epsilon_heatmaps(pi_by_epsilon, diagnostics=None, max_size: int = 80):
    """Show how Sinkhorn epsilon changes coupling concentration."""
    import matplotlib.pyplot as plt

    items = list(pi_by_epsilon.items())
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.05 * n, 3.45), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    diag_lookup = {}
    if diagnostics is not None:
        try:
            import pandas as pd

            frame = pd.DataFrame(diagnostics)
            diag_lookup = {float(row["epsilon"]): row for _, row in frame.iterrows() if "epsilon" in row}
        except Exception:
            diag_lookup = {}
    for ax, (epsilon, pi) in zip(axes, items):
        pi = np.asarray(pi, dtype=float)
        row_idx = _ch04_subsample(pi.shape[0], min(max_size, pi.shape[0]), seed=510)
        col_idx = _ch04_subsample(pi.shape[1], min(max_size, pi.shape[1]), seed=511)
        sub = pi[np.ix_(row_idx, col_idx)]
        image = np.log10(sub + 1e-12)
        ax.imshow(image, cmap="viridis", aspect="auto")
        subtitle = f"epsilon={float(epsilon):g}"
        row = diag_lookup.get(float(epsilon))
        if row is not None and "expected_cost_normalized" in row:
            subtitle += f"\nnorm cost={float(row['expected_cost_normalized']):.2g}"
            if "sinkhorn_converged" in row and not bool(row["sinkhorn_converged"]):
                subtitle += "\ncheck diagnostics"
        elif row is not None and "expected_cost" in row:
            subtitle += f"\ncost={float(row['expected_cost']):.2g}"
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("target index")
        ax.set_ylabel("source index")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Normalized Sinkhorn epsilon controls coupling sharpness and diffusion", y=1.04, fontsize=13)
    return fig, axes


def plot_ch05_otcfm_pipeline():
    """Compact schematic for the Chapter 5 OT-CFM training pipeline."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        ("same EB\nsnapshots", (0.10, 0.57), "#4C78A8"),
        ("choose pair\nsampler", (0.27, 0.57), "#72B7B2"),
        ("linear path\nx_s", (0.44, 0.57), "#54A24B"),
        ("same target\nx1 - x0", (0.61, 0.57), "#F58518"),
        ("same\nVelocityMLP", (0.78, 0.57), "#B279A2"),
        ("different learned\npath geometry", (0.94, 0.57), "#E45756"),
    ]
    for text, xy, color in nodes:
        _ch04_box(ax, xy, text, color=color, width=0.135, height=0.18, fontsize=8.8)
    for start, end in zip(nodes[:-1], nodes[1:]):
        _ch04_arrow(ax, (start[1][0] + 0.075, start[1][1]), (end[1][0] - 0.075, end[1][1]), color="0.35", linewidth=1.1)
    ax.text(0.27, 0.26, "random product coupling", color="0.42", ha="center", fontsize=9)
    ax.text(0.27, 0.16, "or PCA-space OT coupling", color="#008A7A", ha="center", fontsize=9)
    ax.text(0.60, 0.22, "CFM loss and network architecture are unchanged", ha="center", fontsize=10, color="0.25")
    fig.suptitle("OT-CFM changes endpoint sampling, not the local velocity-regression loss", y=0.96, fontsize=13)
    return fig, ax


def _ch05_eval_vector_field(model, xlim, ylim, device="cpu", grid_size: int = 22, s_value: float = 0.5):
    import torch

    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(pts, dtype=torch.float32, device=device)
        s = torch.full((x.shape[0], 1), float(s_value), dtype=torch.float32, device=device)
        v = model(x, s).detach().cpu().numpy().reshape(grid_size, grid_size, 2)
    if was_training:
        model.train()
    return xs, ys, _ch04_clip_vectors(v)


def plot_ch05_streamplot_comparison(models, trajectories, X0, X1, device="cpu", title: str | None = None):
    """Compare learned vector fields and sampled ODE trajectories."""
    import matplotlib.pyplot as plt

    model_items = list(models.items())
    traj_items = dict(trajectories)
    arrays = [X0, X1]
    arrays.extend([np.asarray(t).reshape(-1, np.asarray(t).shape[-1]) for t in traj_items.values()])
    xlim, ylim = _ch04_robust_limits(*arrays, q_low=1, q_high=99, margin=0.10)
    fig, axes = plt.subplots(1, len(model_items), figsize=(5.1 * len(model_items), 4.35), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    colors = {"random_cfm": "0.48", "ot_cfm": "#008A7A", "one_round_reflow": "#5369A6"}
    for ax, (name, model) in zip(axes, model_items):
        _ch04_axis(ax, xlim, ylim, name.replace("_", " "))
        i0 = _ch04_subsample(len(X0), 380, seed=521)
        i1 = _ch04_subsample(len(X1), 380, seed=522)
        ax.scatter(np.asarray(X0)[i0, 0], np.asarray(X0)[i0, 1], s=5, color="#4C78A8", alpha=0.16, linewidths=0)
        ax.scatter(np.asarray(X1)[i1, 0], np.asarray(X1)[i1, 1], s=5, color="#E45756", alpha=0.15, linewidths=0)
        xs, ys, v = _ch05_eval_vector_field(model, xlim, ylim, device=device, grid_size=23, s_value=0.5)
        ax.streamplot(xs, ys, v[..., 0], v[..., 1], color=colors.get(name, "0.35"), density=1.0, linewidth=0.75, arrowsize=0.8)
        traj = np.asarray(traj_items.get(name))
        if traj.size:
            idx = _ch04_subsample(traj.shape[1], min(24, traj.shape[1]), seed=523)
            for j in idx:
                ax.plot(traj[:, j, 0], traj[:, j, 1], color=colors.get(name, "0.35"), linewidth=0.6, alpha=0.35)
    fig.suptitle(title or "The same CFM loss can learn different path geometries", y=1.02, fontsize=13)
    return fig, axes


def plot_ch05_reflow_comparison(traj_before, traj_after, target=None):
    """Compare first-round OT-CFM paths with one-round reflow paths."""
    import matplotlib.pyplot as plt

    traj_before = np.asarray(traj_before, dtype=float)
    traj_after = np.asarray(traj_after, dtype=float)
    target = None if target is None else np.asarray(target, dtype=float)
    xlim, ylim = _ch04_robust_limits(
        traj_before.reshape(-1, traj_before.shape[-1]),
        traj_after.reshape(-1, traj_after.shape[-1]),
        target,
        q_low=1,
        q_high=99,
        margin=0.10,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.35), sharex=True, sharey=True)
    panels = [(traj_before, "OT-CFM before reflow", "#008A7A"), (traj_after, "one-round reflow", "#5369A6")]
    for ax, (traj, panel_title, color) in zip(axes, panels):
        _ch04_axis(ax, xlim, ylim, panel_title)
        if target is not None:
            tidx = _ch04_subsample(len(target), 450, seed=531)
            ax.scatter(target[tidx, 0], target[tidx, 1], s=5, color="#E45756", alpha=0.16, linewidths=0, label="target")
        idx = _ch04_subsample(traj.shape[1], min(34, traj.shape[1]), seed=532)
        for j in idx:
            ax.plot(traj[:, j, 0], traj[:, j, 1], color=color, linewidth=0.85, alpha=0.42)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, color=color, alpha=0.32, linewidths=0, label="generated")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Reflow rectifies sampled solver paths, not biological lineage truth", y=1.02, fontsize=13)
    return fig, axes


def plot_ch05_metric_bars(metric_table, metrics=None, title: str | None = None):
    """Plot Chapter 5 path geometry metrics as small multiples."""
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(metric_table).copy()
    if metrics is None:
        metrics = ["endpoint_mmd", "sliced_w2", "path_length", "path_energy", "straightness", "off_manifold_knn"]
    metrics = [m for m in metrics if m in table.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 3.8), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    colors = {"random_cfm": "0.68", "ot_cfm": "#008A7A", "one_round_reflow": "#5369A6"}
    for ax, metric in zip(axes, metrics):
        values = table[metric].astype(float).to_numpy()
        methods = table["method"].astype(str).to_numpy()
        x = np.arange(len(methods))
        ax.bar(x, values, color=[colors.get(m, "0.45") for m in methods], width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
        ax.set_title(metric.replace("_", " "), fontsize=9.5)
        ax.grid(axis="y", color="0.91", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title or "Endpoint fit alone does not determine path geometry", y=1.05, fontsize=13)
    return fig, axes


def plot_ch05_solver_sensitivity(solver_table):
    """Plot Euler step count sensitivity for Chapter 5 models."""
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(solver_table).copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.05), constrained_layout=True)
    colors = {"random_cfm": "0.55", "ot_cfm": "#008A7A", "one_round_reflow": "#5369A6"}
    for method, group in table.groupby("method", sort=False):
        group = group.sort_values("n_steps")
        color = colors.get(str(method), "0.35")
        axes[0].plot(group["nfe"], group["endpoint_sliced_w2"], marker="o", color=color, linewidth=1.5, label=str(method).replace("_", " "))
        axes[1].plot(group["nfe"], group["coarse_step_endpoint_error"], marker="o", color=color, linewidth=1.5, label=str(method).replace("_", " "))
    axes[0].set_ylabel("endpoint sliced W2")
    axes[1].set_ylabel("distance to 64-step endpoint")
    for ax in axes:
        ax.set_xlabel("NFE")
        ax.grid(color="0.91", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Solver-step sensitivity is part of the model report", y=1.04, fontsize=13)
    return fig, axes


def plot_ch05_branch_diagnostics(X0, X1, panels=None, diagnostics=None, kind: str = "pairs"):
    """Draw controlled toy branch leakage or rare-fate diagnostic bars."""
    import matplotlib.pyplot as plt
    import pandas as pd

    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    if kind == "pairs":
        panels = list(panels or [])
        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.25), sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        xlim, ylim = _ch04_robust_limits(X0, X1, q_low=1, q_high=99, margin=0.12)
        colors = {"random": "0.68", "ot": "#008A7A"}
        for ax, panel in zip(axes, panels):
            _ch04_axis(ax, xlim, ylim, panel.get("title", "toy pairs"))
            ax.set_xlabel("toy state 1")
            ax.set_ylabel("toy state 2")
            ax.scatter(X0[:, 0], X0[:, 1], s=9, color="#4C78A8", alpha=0.28, linewidths=0, label="t=0.50")
            ax.scatter(X1[:, 0], X1[:, 1], s=9, color="#E45756", alpha=0.24, linewidths=0, label="t=1.00")
            pair_x0 = np.asarray(panel["x0"], dtype=float)
            pair_x1 = np.asarray(panel["x1"], dtype=float)
            idx = _ch04_subsample(len(pair_x0), min(95, len(pair_x0)), seed=541)
            color = panel.get("color", colors.get(panel.get("method", ""), "0.45"))
            for j in idx:
                ax.plot([pair_x0[j, 0], pair_x1[j, 0]], [pair_x0[j, 1], pair_x1[j, 1]], color=color, linewidth=0.65, alpha=0.36)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=9)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.18, wspace=0.10)
        fig.suptitle("Controlled toy labels expose branch mixing in endpoint pair choices", y=0.96, fontsize=13)
        return fig, axes

    table = pd.DataFrame(diagnostics).copy()
    metrics = ["branch_leakage", "target_fate_mass_error", "rare_fate_mass_error", "sampled_rare_target_fraction"]
    metrics = [m for m in metrics if m in table.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.15 * len(metrics), 3.75), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    colors = {"random": "0.68", "ot": "#008A7A"}
    for ax, metric in zip(axes, metrics):
        methods = table["method"].astype(str).to_numpy()
        x = np.arange(len(methods))
        ax.bar(x, table[metric].astype(float), color=[colors.get(m, "0.45") for m in methods], width=0.58)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(metric.replace("_", " "), fontsize=10)
        ax.grid(axis="y", color="0.91", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Controlled toy pair-sampler diagnostics: rare fate mass, not lineage recall", y=1.05, fontsize=13)
    return fig, axes


CH06_COLORS = {
    "raw": "#6B6B6B",
    "pca": "#4C78A8",
    "program": "#59A14F",
    "viz": "#F58518",
    "balanced": "#4C78A8",
    "growth": "#D6275F",
    "prior": "#7B61A7",
    "target": "#E45756",
}


def _ch06_clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="0.92", linewidth=0.7)


def _ch06_subsample(n: int, max_n: int, seed: int = 42):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_n), replace=False))


def plot_ch06_state_variable_choices(
    toy,
    eb=None,
    max_cells: int = 500,
    title: str = "Same cells, different state variables",
):
    """Show raw/log, PCA, program, toy visualization, and EB anchor spaces."""
    import matplotlib.pyplot as plt

    X_log = np.asarray(toy["X_log"], dtype=float)
    X_pca = np.asarray(toy["X_pca"], dtype=float)
    X_program = np.asarray(toy["X_program"], dtype=float)
    X_viz = np.asarray(toy["X_viz"], dtype=float)
    time = np.asarray(toy.get("time", np.zeros(X_log.shape[0])), dtype=float)
    program_names = list(toy.get("program_names", ["program 1", "program 2", "program 3", "program 4"]))
    idx = _ch06_subsample(X_log.shape[0], max_cells, seed=11)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), constrained_layout=True)
    axes = axes.reshape(2, 3)

    heat = X_log[idx[: min(80, len(idx))], : min(32, X_log.shape[1])]
    im = axes[0, 0].imshow(heat, aspect="auto", cmap="Greys")
    axes[0, 0].set_title("Raw/log expression matrix")
    axes[0, 0].set_xlabel("pseudo genes")
    axes[0, 0].set_ylabel("same cells")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03, label="log expr")

    sc = axes[0, 1].scatter(X_pca[idx, 0], X_pca[idx, 1], c=time[idx], s=12, cmap="viridis", alpha=0.72, linewidths=0)
    axes[0, 1].set_title("Training state: PCA")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    fig.colorbar(sc, ax=axes[0, 1], fraction=0.046, pad=0.03, label="time")

    x_prog = 2 if len(program_names) > 2 else 0
    y_prog = 3 if len(program_names) > 3 else min(1, X_program.shape[1] - 1)
    axes[0, 2].scatter(X_program[idx, x_prog], X_program[idx, y_prog], c=time[idx], s=12, cmap="viridis", alpha=0.72, linewidths=0)
    axes[0, 2].set_title("Training/readout state: gene programs")
    axes[0, 2].set_xlabel(program_names[x_prog])
    axes[0, 2].set_ylabel(program_names[y_prog])

    axes[1, 0].scatter(X_viz[idx, 0], X_viz[idx, 1], c=time[idx], s=12, cmap="viridis", alpha=0.72, linewidths=0)
    axes[1, 0].set_title("Visualization only: toy state")
    axes[1, 0].set_xlabel("toy state 1")
    axes[1, 0].set_ylabel("toy state 2")
    axes[1, 0].set_aspect("equal", adjustable="box")

    axes[1, 1].axis("off")
    lines = [
        "State x is chosen by the researcher.",
        "PCA and program scores are different training geometries.",
        "Toy 2D and EB PHATE are display spaces here.",
        "Program readout is explicit, not implied by a latent vector field.",
    ]
    axes[1, 1].text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10)
    axes[1, 1].set_title("Claim boundary")

    ax = axes[1, 2]
    if eb is not None:
        phate = np.asarray(eb["phate_plot"], dtype=float)
        labels = np.asarray(eb["labels"], dtype=float)
        idx_eb = _ch06_subsample(phate.shape[0], min(max_cells, phate.shape[0]), seed=13)
        ax.scatter(phate[idx_eb, 0], phate[idx_eb, 1], c=labels[idx_eb], s=8, cmap="plasma", alpha=0.58, linewidths=0)
        ax.set_xlabel("PHATE 1 (raw display coordinate)")
        ax.set_ylabel("PHATE 2 (raw display coordinate)")
        ax.set_title("EB PHATE visualization anchor")
        ax.text(
            0.02,
            0.03,
            "EB PCs are used for cost/coupling diagnostics.\n"
            "PHATE is display space, not training/readout space.\n"
            "no raw gene matrix; no EB gene-program readout",
            transform=ax.transAxes,
            fontsize=7.6,
            color="0.22",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 2.2, "alpha": 0.92},
        )
    else:
        ax.axis("off")
        ax.text(0.02, 0.95, "EB anchor unavailable", va="top")

    for axis in axes.reshape(-1):
        if axis.axison:
            _ch06_clean_axis(axis)
    fig.suptitle(title, fontsize=14)
    return fig, axes


def plot_ch06_training_vs_readout(state_spaces, readout_table, trajectories=None):
    """Separate native training trajectories from shared biological readout."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    table = pd.DataFrame(readout_table).copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.45))
    fig.subplots_adjust(left=0.07, right=0.79, top=0.83, bottom=0.16, wspace=0.32)
    ax0, ax1 = axes
    pca = state_spaces["pca"]
    idx = _ch06_subsample(pca["X0_train"].shape[0], min(120, pca["X0_train"].shape[0]), seed=17)
    ax0.scatter(pca["X0_train"][:, 0], pca["X0_train"][:, 1], s=10, color=CH06_COLORS["pca"], alpha=0.28, linewidths=0, label="source")
    ax0.scatter(pca["X1_train"][:, 0], pca["X1_train"][:, 1], s=10, color=CH06_COLORS["target"], alpha=0.24, linewidths=0, label="target")
    if trajectories and "pca_otcfm" in trajectories:
        traj = np.asarray(trajectories["pca_otcfm"], dtype=float)
        for j in idx[: min(50, traj.shape[1])]:
            ax0.plot(traj[:, j, 0], traj[:, j, 1], color="0.28", linewidth=0.65, alpha=0.32)
    else:
        for j in idx[:50]:
            ax0.plot([pca["X0_train"][j, 0], pca["X1_train"][j % pca["X1_train"].shape[0], 0]], [pca["X0_train"][j, 1], pca["X1_train"][j % pca["X1_train"].shape[0], 1]], color="0.4", linewidth=0.6, alpha=0.22)
    ax0.set_title("Left: PCA training state")
    ax0.set_xlabel("standardized PC1")
    ax0.set_ylabel("standardized PC2")
    ax0.legend(frameon=False, fontsize=8)

    colors = {
        "progenitor_trunk": "#4C78A8",
        "transition": "#F58518",
        "major_fate": "#59A14F",
        "rare_fate": "#D6275F",
    }
    for (method, program), group in table.groupby(["method", "program"], sort=False):
        group = group.sort_values("bridge_time")
        linestyle = "-" if "program" in str(method) else "--"
        ax1.plot(
            group["bridge_time"],
            group["mean_score"],
            marker="o",
            linewidth=1.35,
            linestyle=linestyle,
            color=colors.get(str(program), "0.35"),
            label="_nolegend_",
        )
    ax1.set_title("Right: shared program-score readout")
    ax1.set_xlabel("bridge time")
    ax1.set_ylabel("mean program score")
    program_handles = [
        Line2D([0], [0], color=color, lw=1.8, label=name)
        for name, color in colors.items()
    ]
    method_handles = [
        Line2D([0], [0], color="0.25", lw=1.8, linestyle="--", label="pca_otcfm"),
        Line2D([0], [0], color="0.25", lw=1.8, linestyle="-", label="program_otcfm"),
    ]
    legend1 = ax1.legend(
        handles=program_handles,
        title="program",
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax1.add_artist(legend1)
    ax1.legend(
        handles=method_handles,
        title="method",
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02),
        borderaxespad=0.0,
    )
    for ax in axes:
        _ch06_clean_axis(ax)
    fig.suptitle("Training space and biological readout are different objects", fontsize=13)
    return fig, axes


def plot_ch06_training_visualization_readout():
    """Draw a compact table/diagram distinguishing training, visualization, and readout spaces."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.04, 0.56, 0.25, 0.26, "Training state", "PCA, gene program,\nlog expression", CH06_COLORS["pca"]),
        (0.38, 0.56, 0.25, 0.26, "Visualization", "X_toy_state, EB PHATE\nfor plots only", CH06_COLORS["viz"]),
        (0.71, 0.56, 0.25, 0.26, "Biological readout", "program scores,\nmarkers, fate proportions", CH06_COLORS["program"]),
        (
            0.07,
            0.08,
            0.86,
            0.27,
            "EB anchor boundary",
            "pcs can define a cost/coupling diagnostic; PHATE displays arrows only.\n"
            "This npz has no raw gene matrix and no EB gene-program readout.\n"
            "PHATE is display space, not training/readout space.",
            "0.45",
        ),
    ]
    for x, y, w, h, head, body, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8))
        ax.text(x + 0.02, y + h - 0.06, head, fontsize=11, weight="bold", color=color, va="top")
        body_font = 8.5 if head == "EB anchor boundary" else 9.5
        ax.text(x + 0.02, y + h - 0.13, body, fontsize=body_font, color="0.20", va="top")
    for start, end in [((0.29, 0.69), (0.38, 0.69)), ((0.63, 0.69), (0.71, 0.69))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="0.35"))
    ax.text(0.50, 0.42, "A claim is valid only after specifying which map produced it.", ha="center", fontsize=10, color="0.25")
    fig.suptitle("Training, visualization, and readout spaces must be named separately", fontsize=13)
    return fig, ax


def plot_ch06_representation_to_flow_pipeline():
    """Code-generated schematic for representation-to-flow assumptions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    labels = [
        ("raw counts\nlog expression", CH06_COLORS["raw"]),
        ("PCA or\ngene program", CH06_COLORS["pca"]),
        ("distance\nmatrix", "0.35"),
        ("OT\ncoupling", CH06_COLORS["program"]),
        ("endpoint\npairs", "0.35"),
        ("target\nvelocity", CH06_COLORS["viz"]),
        ("learned\nflow", CH06_COLORS["balanced"]),
        ("readout\ncheck", CH06_COLORS["program"]),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.04, 0.86, len(labels))
    width = 0.105
    for i, ((label, color), x) in enumerate(zip(labels, xs)):
        ax.add_patch(Rectangle((x, 0.42), width, 0.28, facecolor="white", edgecolor=color, linewidth=1.7))
        ax.text(x + width / 2, 0.56, label, ha="center", va="center", fontsize=9.5, color="0.18")
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + width, 0.56), (xs[i + 1], 0.56), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="0.38"))
    ax.text(0.50, 0.22, "Changing the representation changes cost, coupling, endpoint pairs, and CFM targets.", ha="center", fontsize=10, color="0.25")
    fig.suptitle("Representation choice propagates through the flow pipeline", fontsize=13)
    return fig, ax


def plot_ch06_cost_coupling_comparison(
    C_pca,
    C_program,
    pi_pca,
    pi_program,
    X0_viz,
    X1_viz,
    sample_pca,
    sample_program,
    annotation: str = "",
):
    """Compare toy PCA and program costs, couplings, and sampled arrows in toy 2D."""
    import matplotlib.pyplot as plt

    C_pca = np.asarray(C_pca, dtype=float)
    C_program = np.asarray(C_program, dtype=float)
    pi_pca = np.asarray(pi_pca, dtype=float)
    pi_program = np.asarray(pi_program, dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.8))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.875, bottom=0.08, hspace=0.44, wspace=0.40)
    panels = [
        (axes[0, 0], C_pca, "PCA normalized cost", "magma"),
        (axes[0, 1], C_program, "Program normalized cost", "magma"),
        (axes[1, 0], pi_pca, "PCA Sinkhorn coupling", "viridis"),
        (axes[1, 1], pi_program, "Program Sinkhorn coupling", "viridis"),
    ]
    for ax, mat, title, cmap in panels:
        view = mat[: min(120, mat.shape[0]), : min(120, mat.shape[1])]
        im = ax.imshow(view, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("target cells")
        ax.set_ylabel("source cells")
        fig.colorbar(im, ax=ax, fraction=0.034, pad=0.025)

    arrow_panels = [
        (axes[2, 0], sample_pca, "PCA coupling displayed in toy state", CH06_COLORS["pca"]),
        (axes[2, 1], sample_program, "Program coupling displayed in toy state", CH06_COLORS["program"]),
    ]
    xlim, ylim = _ch03_limits(X0_viz, X1_viz, pad=0.25)
    for ax, sample, title, color in arrow_panels:
        _ch03_axis(ax, xlim, ylim, title)
        ax.set_xlabel("toy state 1")
        ax.set_ylabel("toy state 2")
        ax.scatter(X0_viz[:, 0], X0_viz[:, 1], s=8, color=CH06_COLORS["pca"], alpha=0.22, linewidths=0)
        ax.scatter(X1_viz[:, 0], X1_viz[:, 1], s=8, color=CH06_COLORS["target"], alpha=0.18, linewidths=0)
        idx0 = np.asarray(sample["idx0"], dtype=int)
        idx1 = np.asarray(sample["idx1"], dtype=int)
        keep = _ch06_subsample(len(idx0), min(90, len(idx0)), seed=23)
        for j in keep:
            a = X0_viz[idx0[j]]
            b = X1_viz[idx1[j]]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, alpha=0.35, linewidth=0.65)
    if annotation:
        fig.text(0.5, 0.925, annotation, ha="center", va="bottom", fontsize=9.5, color="0.22")
    fig.suptitle("Changing representation changes cost and coupling for the same toy cells", fontsize=13, y=0.975)
    return fig, axes


def plot_ch06_eb_representation_coupling_anchor(eb, eb_result):
    """Show EB pcs/phate coupling diagnostics and PHATE-displayed sampled arrows."""
    import matplotlib.pyplot as plt

    C_pcs = eb_result["C_pcs"]
    C_phate = eb_result["C_phate"]
    pi_pcs = eb_result["pi_pcs"]
    pi_phate = eb_result["pi_phate"]
    sample_pcs = eb_result["sample_pcs"]
    sample_phate = eb_result["sample_phate"]
    X0_plot = eb["X0_plot"]
    X1_plot = eb["X1_plot"]

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.10, hspace=0.46, wspace=0.30)
    panels = [
        (axes[0, 0], C_pcs, "EB pcs cost/coupling diagnostic", "magma"),
        (axes[0, 1], pi_pcs, "EB pcs-induced coupling", "viridis"),
        (axes[1, 0], C_phate, "PHATE diagnostic cost (not training recommendation)", "magma"),
        (axes[1, 1], pi_phate, "PHATE-induced diagnostic coupling", "viridis"),
    ]
    for ax, mat, title, cmap in panels:
        view = mat[: min(120, mat.shape[0]), : min(120, mat.shape[1])]
        im = ax.imshow(view, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("target cells")
        ax.set_ylabel("source cells")
        fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)

    for ax, sample, title, color in [
        (axes[0, 2], sample_pcs, "pcs-induced coupling displayed in PHATE", CH06_COLORS["pca"]),
        (axes[1, 2], sample_phate, "PHATE-induced diagnostic coupling displayed in PHATE", CH06_COLORS["viz"]),
    ]:
        xlim, ylim = _ch03_limits(X0_plot, X1_plot, pad=0.25)
        _ch03_axis(ax, xlim, ylim, title)
        ax.scatter(X0_plot[:, 0], X0_plot[:, 1], s=8, color=CH06_COLORS["pca"], alpha=0.24, linewidths=0, label="source")
        ax.scatter(X1_plot[:, 0], X1_plot[:, 1], s=8, color=CH06_COLORS["target"], alpha=0.20, linewidths=0, label="target")
        idx0 = np.asarray(sample["idx0"], dtype=int)
        idx1 = np.asarray(sample["idx1"], dtype=int)
        keep = _ch06_subsample(len(idx0), min(85, len(idx0)), seed=29)
        for j in keep:
            a = X0_plot[idx0[j]]
            b = X1_plot[idx1[j]]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, alpha=0.32, linewidth=0.65)
        ax.text(
            0.02,
            0.03,
            "displayed coupling, not observed velocity",
            transform=ax.transAxes,
            fontsize=8,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "0.86", "pad": 2.0, "alpha": 0.90},
        )
    fig.text(
        0.5,
        0.885,
        "PHATE coupling is a contrastive diagnostic, not a recommended training geometry.",
        ha="center",
        fontsize=9,
        color="0.25",
    )
    fig.suptitle("Real EB anchor: representation changes coupling before model training", fontsize=13, y=0.965)
    return fig, axes


def plot_ch06_euclidean_vs_manifold_paths(diagnostic):
    """Draw Euclidean chord and graph-geodesic diagnostic."""
    import matplotlib.pyplot as plt

    points = np.asarray(diagnostic["points"], dtype=float)
    start = np.asarray(diagnostic["start"], dtype=float)
    end = np.asarray(diagnostic["end"], dtype=float)
    graph_path = np.asarray(diagnostic["graph_path"], dtype=float)
    straight_path = np.asarray(diagnostic["straight_path"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(points[:, 0], points[:, 1], s=9, color="0.72", alpha=0.55, linewidths=0, label="observed states")
    ax.plot(straight_path[:, 0], straight_path[:, 1], color=CH06_COLORS["growth"], linewidth=2.1, label="Euclidean chord")
    ax.plot(graph_path[:, 0], graph_path[:, 1], color=CH06_COLORS["program"], linewidth=2.1, label="kNN graph path")
    ax.scatter([start[0], end[0]], [start[1], end[1]], s=48, color=[CH06_COLORS["pca"], CH06_COLORS["target"]], zorder=5)
    ax.text(
        0.02,
        0.98,
        "straight off-manifold: {:.3f}\ngraph off-manifold: {:.3f}".format(
            float(diagnostic["straight_off_manifold"]),
            float(diagnostic["graph_off_manifold"]),
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 3.0},
    )
    ax.set_title("Straight in representation space is a modeling assumption")
    ax.set_xlabel("state 1")
    ax.set_ylabel("state 2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9)
    _ch06_clean_axis(ax)
    return fig, ax


def plot_ch06_balanced_unbalanced_prior_boundaries(X0_viz, X1_viz, labels1=None):
    """Three-panel conceptual boundary for balanced flow, growth/source, and priors."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    X0_viz = np.asarray(X0_viz, dtype=float)
    X1_viz = np.asarray(X1_viz, dtype=float)
    labels1 = np.asarray(labels1).astype(str) if labels1 is not None else np.full(X1_viz.shape[0], "target")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True, sharex=True, sharey=True)
    xlim, ylim = _ch03_limits(X0_viz, X1_viz, pad=0.28)
    titles = ["A. Balanced flow", "B. Unbalanced growth/death", "C. Priors and uncertainty"]
    for ax, title in zip(axes, titles):
        _ch03_axis(ax, xlim, ylim, title)
        ax.set_xlabel("toy state 1")
        ax.set_ylabel("toy state 2")
        ax.scatter(X0_viz[:, 0], X0_viz[:, 1], s=10, color=CH06_COLORS["pca"], alpha=0.20, linewidths=0)
        ax.scatter(X1_viz[:, 0], X1_viz[:, 1], s=10, color=CH06_COLORS["target"], alpha=0.18, linewidths=0)

    c0 = X0_viz.mean(axis=0)
    c1 = X1_viz.mean(axis=0)
    axes[0].add_patch(FancyArrowPatch(c0, c1, arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color=CH06_COLORS["balanced"]))
    axes[0].text(0.05, 0.92, "normalized probability mass = 1\nredistribution only", transform=axes[0].transAxes, fontsize=9, va="top", bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.5})

    major = X1_viz[labels1 == "major"]
    rare = X1_viz[labels1 == "rare"]
    for group, scale, color, label in [(major, 220, CH06_COLORS["growth"], "major growth"), (rare, 120, CH06_COLORS["program"], "rare weight")]:
        if len(group):
            center = group.mean(axis=0)
            axes[1].scatter(center[0], center[1], s=scale, facecolor="none", edgecolor=color, linewidth=2.2, label=label)
            axes[1].add_patch(FancyArrowPatch(c0, center, arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=color))
    axes[1].text(0.05, 0.92, "abundance mass changes by construction\nrequires source/growth evidence", transform=axes[1].transAxes, fontsize=9, va="top", bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.5})
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    rng = np.random.default_rng(42)
    for offset in rng.normal(scale=0.07, size=(14, 2)):
        axes[2].plot([c0[0], c1[0] + offset[0]], [c0[1], c1[1] + offset[1]], color="0.45", alpha=0.17, linewidth=1.0)
    axes[2].add_patch(FancyArrowPatch(c0 + np.array([0.0, 0.1]), c1 + np.array([0.25, 0.0]), arrowstyle="-|>", mutation_scale=16, linewidth=1.7, color=CH06_COLORS["prior"], linestyle="--"))
    axes[2].text(0.05, 0.92, "stochasticity and RNA/GRN priors\nare assumptions to ablate", transform=axes[2].transAxes, fontsize=9, va="top", bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.5})
    fig.suptitle("Default FM is balanced probability flow, not automatic growth recovery", fontsize=13)
    return fig, axes
