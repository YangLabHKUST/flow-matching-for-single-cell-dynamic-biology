"""Self-contained WFR-FM components used by Chapter 4 Exp 9b.

This module reproduces the minimal WFR-FM training path needed for the paper's
sampling-depth sensitivity diagnostic.  It intentionally uses all-ones UOT
masses, matching the audited baseline training path used for Exp 9b; timepoint
sample-size ratios are not used in either UOT construction or batch sampling.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

import numpy as np
import ot
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


INTERNAL_IMPLEMENTATION_NOTE = (
    "Internal minimal WFR-FM implementation for Ch4 Exp 9b: learns a displacement "
    "vector field v_net and scalar growth-rate field g_net; constructs WFR/UOT "
    "plans with all-ones source/target masses a and b; relative_mass/sample-size "
    "ratios are not used in UOT plans or get_batch. This preserves the audited "
    "training setting used for the sampling-depth diagnostic and should not be "
    "interpreted as using calibrated biological census mass."
)


class VelocityNet(nn.Module):
    def __init__(self, in_out_dim: int, hidden_dim: int, n_hiddens: int, activation: str = "Tanh"):
        super().__init__()
        dims = [int(in_out_dim) + 1] + [int(hidden_dim)] * int(n_hiddens) + [int(in_out_dim)]
        act = _activation(activation)
        self.net = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], dims[i + 1]), act) for i in range(len(dims) - 2)])
        self.out = nn.Linear(dims[-2], dims[-1])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t.expand(x.shape[0], 1)
        state = torch.cat((t, x), dim=1)
        h = state
        for layer in self.net:
            h = layer(h)
        return self.out(h)


class GrowthNet(nn.Module):
    def __init__(self, in_out_dim: int, hidden_dim: int, activation: str = "Tanh"):
        super().__init__()
        act = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(int(in_out_dim) + 1, int(hidden_dim)),
            act,
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            _activation(activation),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            _activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t.expand(x.shape[0], 1)
        return self.net(torch.cat((t, x), dim=1))


class WFRFMNet(nn.Module):
    def __init__(self, in_out_dim: int, hidden_dim: int, n_hiddens: int, activation: str = "leakyrelu"):
        super().__init__()
        self.in_out_dim = int(in_out_dim)
        self.hidden_dim = int(hidden_dim)
        self.v_net = VelocityNet(in_out_dim, hidden_dim, n_hiddens, activation)
        self.g_net = GrowthNet(in_out_dim, hidden_dim, activation)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.v_net(t, x).float(), self.g_net(t, x).float()


FNet = WFRFMNet


@dataclass(frozen=True)
class UOTPlans:
    gamma0_plans: list[np.ndarray]
    gamma1_plans: list[np.ndarray]
    sampling_info: list[dict | None]


def _activation(name: str) -> nn.Module:
    if name == "Tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unknown activation: {name}")


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def compute_wfr_cost_matrix(x_source: np.ndarray, x_target: np.ndarray, delta: float) -> np.ndarray:
    dist = ot.dist(np.asarray(x_source), np.asarray(x_target), metric="euclidean")
    cos_sq = np.cos(np.minimum(dist / (2 * float(delta)), np.pi / 2)) ** 2
    return -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))


def compute_uot_plans(
    X: list[np.ndarray],
    t_train: list[float],
    delta: float = 1.0,
    use_mini_batch_uot: bool = False,
    chunk_size: int = 1000,
    device: str | torch.device | None = None,
    show_progress: bool = True,
) -> UOTPlans:
    gamma0_plans = []
    gamma1_plans = []
    sampling_info_plans = []
    dev = _device(device)
    iterator = range(len(t_train) - 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing UOT plans...")

    for i in iterator:
        x_source, x_target = X[i], X[i + 1]
        n_source, n_target = x_source.shape[0], x_target.shape[0]
        a = np.ones(n_source, dtype=np.float64)
        b = np.ones(n_target, dtype=np.float64)
        cost_matrix = compute_wfr_cost_matrix(x_source, x_target, delta)

        if not use_mini_batch_uot:
            G = _uot_plan_torch(a, b, cost_matrix, dev)
            sampling_info_plans.append(None)
        else:
            group_number = n_source // int(chunk_size) + 1
            G = np.zeros((n_source, n_target), dtype=np.float64)
            source_perm = np.arange(n_source)
            target_perm = np.arange(n_target)
            np.random.shuffle(source_perm)
            np.random.shuffle(target_perm)
            source_groups = np.array_split(source_perm, group_number)
            target_groups = np.array_split(target_perm, group_number)
            gamma0_sub_plans = []
            for src_idx, tgt_idx in zip(source_groups, target_groups):
                sub_cost = cost_matrix[np.ix_(src_idx, tgt_idx)]
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                G_sub = _uot_plan_torch(sub_a, sub_b, sub_cost, dev)
                G[np.ix_(src_idx, tgt_idx)] = G_sub
                g_sub_sum_1 = G_sub.sum(1)
                gamma0_sub = ((sub_a / (g_sub_sum_1 + 1e-12))[:, None]) * G_sub
                gamma0_sub_plans.append(gamma0_sub.astype(np.float32))
            sampling_info_plans.append(
                {"sub_plans": gamma0_sub_plans, "source_groups": source_groups, "target_groups": target_groups}
            )

        g_sum_1 = G.sum(1)
        g_sum_0 = G.sum(0)
        gamma0_plans.append((((a / (g_sum_1 + 1e-12))[:, None]) * G).astype(np.float32))
        gamma1_plans.append(((b / (g_sum_0 + 1e-12)) * G).astype(np.float32))

    return UOTPlans(gamma0_plans=gamma0_plans, gamma1_plans=gamma1_plans, sampling_info=sampling_info_plans)


def _uot_plan_torch(a: np.ndarray, b: np.ndarray, cost_matrix: np.ndarray, device: torch.device) -> np.ndarray:
    a_t = torch.from_numpy(np.asarray(a)).to(device)
    b_t = torch.from_numpy(np.asarray(b)).to(device)
    c_t = torch.from_numpy(np.asarray(cost_matrix)).to(device)
    return ot.unbalanced.mm_unbalanced(a_t, b_t, c_t, reg_m=[1.0, 1.0]).detach().cpu().numpy()


def sample_from_ot_plan(
    ot_plan: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    batch_size: int,
    sampling_info: dict | None = None,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev = _device(device)
    if sampling_info is None:
        pi = torch.from_numpy(ot_plan.astype(np.float32)).to(dev)
        row_sums = pi.sum(axis=1)
        total_sum = row_sums.sum()
        if total_sum < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        i_samples = torch.multinomial(row_sums / total_sum, num_samples=int(batch_size), replacement=True)
        selected_rows = pi[i_samples]
        conditional = selected_rows / (row_sums[i_samples].unsqueeze(1) + 1e-12)
        j_samples = torch.multinomial(conditional, num_samples=1).squeeze(1)
        i = i_samples.cpu().numpy()
        j = j_samples.cpu().numpy()
    else:
        g_subs = sampling_info["sub_plans"]
        source_groups = sampling_info["source_groups"]
        target_groups = sampling_info["target_groups"]
        block_masses = [g.sum() for g in g_subs]
        total_mass = float(sum(block_masses))
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        block_probs = torch.tensor(block_masses, dtype=torch.float32, device=dev) / total_mass
        sampled_groups = torch.multinomial(block_probs, num_samples=int(batch_size), replacement=True)
        final_i = torch.empty(int(batch_size), dtype=torch.int64, device=dev)
        final_j = torch.empty(int(batch_size), dtype=torch.int64, device=dev)
        g_subs_gpu = [torch.from_numpy(g).to(dev) for g in g_subs]
        source_gpu = [torch.from_numpy(idx).to(dev) for idx in source_groups]
        target_gpu = [torch.from_numpy(idx).to(dev) for idx in target_groups]
        unique_groups, counts = torch.unique(sampled_groups, return_counts=True)
        for group_idx, count in zip(unique_groups, counts):
            g_sub = g_subs_gpu[group_idx]
            sub_row_sums = g_sub.sum(axis=1)
            if sub_row_sums.sum() < 1e-9:
                continue
            i_local = torch.multinomial(sub_row_sums / sub_row_sums.sum(), num_samples=count.item(), replacement=True)
            selected_rows = g_sub[i_local]
            conditional = selected_rows / (sub_row_sums[i_local].unsqueeze(1) + 1e-12)
            j_local = torch.multinomial(conditional, num_samples=1).squeeze(1)
            mask = sampled_groups == group_idx
            final_i[mask] = source_gpu[group_idx][i_local]
            final_j[mask] = target_gpu[group_idx][j_local]
        i = final_i.cpu().numpy()
        j = final_j.cpu().numpy()
    return x0[i], x1[j], i, j


def compute_xt_ut_gt(
    t_relative: torch.Tensor,
    delta_t: float,
    x0: torch.Tensor,
    x1: torch.Tensor,
    mass0: torch.Tensor,
    mass1: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.norm(x1 - x0, dim=1) < torch.pi * float(delta)
    t_relative = t_relative[index]
    x0 = x0[index]
    x1 = x1[index]
    mass0 = mass0[index]
    mass1 = mass1[index]

    diff = x1 - x0
    norm = torch.norm(diff, dim=1, keepdim=True)
    norm_vector = diff / (norm + 1e-9)
    tau = torch.tan(norm / (2 * float(delta)))
    scale = torch.sqrt(mass0 * mass1 / (1 + tau**2))
    omega = 2 * float(delta) * tau * scale
    omega_vector = omega * norm_vector

    A = mass1 + mass0 - 2 * scale
    B = mass0 - scale
    inv_sqrt = 2 * float(delta) / (omega + 1e-9)
    xt_samp = x0 + omega_vector * (
        inv_sqrt * (torch.arctan((A * t_relative - B) * inv_sqrt) - torch.arctan(-B * inv_sqrt))
    )
    masst_samp = A * t_relative**2 - 2 * B * t_relative + mass0
    dmasst_dt = 2 * A * t_relative - 2 * B
    gt_samp = dmasst_dt / masst_samp * (1 / float(delta_t))
    ut_samp = omega_vector / masst_samp * (1 / float(delta_t))
    return xt_samp, gt_samp, ut_samp, masst_samp / mass0, index


def get_batch(
    X: list[np.ndarray],
    t_train: list[float],
    batch_size: int,
    gamma0_plans: list[np.ndarray],
    gamma1_plans: list[np.ndarray],
    delta: float,
    sampling_info_plans: list[dict | None],
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = _device(device)
    ts, xts, uts, gts, massts = [], [], [], [], []
    for t in range(len(t_train) - 1):
        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]
        x0_np, x1_np, idx_0, idx_1 = sample_from_ot_plan(
            gamma0_plan, X[t], X[t + 1], batch_size, sampling_info_plans[t], device=dev
        )
        x0 = torch.from_numpy(x0_np).float().to(dev)
        x1 = torch.from_numpy(x1_np).float().to(dev)
        mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(dev)
        mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(dev)
        delta_t = float(t_train[t + 1] - t_train[t])
        t_relative = torch.rand(x0.shape[0], 1, device=dev, dtype=x0.dtype)
        t_samp = delta_t * t_relative
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_relative, delta_t, x0, x1, mass0, mass1, delta
        )
        ts.append(t_samp[index] + float(t_train[t]))
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)
    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts), torch.cat(massts)


def train_wfrfm_model(
    df: pd.DataFrame,
    *,
    dim: int,
    seed: int,
    n_epochs: int,
    batch_size: int,
    delta: float,
    hidden_dim: int = 64,
    n_hiddens: int = 2,
    lr_v: float = 5e-3,
    lr_g: float = 5e-3,
    eta_min: float = 1e-5,
    use_mini_batch: bool = True,
    chunk_size: int = 512,
    device: str | torch.device | None = None,
    show_progress: bool = True,
) -> tuple[WFRFMNet, dict]:
    set_all_seeds(seed)
    dev = _device(device)
    time_labels = df["samples"].to_numpy(dtype=np.float32)
    data = df.iloc[:, 1 : int(dim) + 1].to_numpy(dtype=np.float32)
    t_train = sorted(np.unique(time_labels).tolist())
    x_selected = [data[time_labels == t] for t in t_train]

    plans = compute_uot_plans(
        x_selected,
        t_train,
        delta=delta,
        use_mini_batch_uot=use_mini_batch,
        chunk_size=chunk_size,
        device=dev,
        show_progress=show_progress,
    )
    model = WFRFMNet(in_out_dim=dim, hidden_dim=hidden_dim, n_hiddens=n_hiddens, activation="leakyrelu").to(dev)
    opt_v = torch.optim.Adam(model.v_net.parameters(), lr=lr_v)
    opt_g = torch.optim.Adam(model.g_net.parameters(), lr=lr_g)
    sched_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=n_epochs, eta_min=eta_min)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=n_epochs, eta_min=eta_min)

    v_losses, g_losses, losses = [], [], []
    start = time.time()
    iterator = range(int(n_epochs))
    if show_progress:
        iterator = tqdm(iterator, desc="Begin flow and growth matching...", unit="epoch")
    for epoch in iterator:
        opt_v.zero_grad()
        opt_g.zero_grad()
        t, xt, ut, gt, masst = get_batch(
            x_selected,
            t_train,
            batch_size,
            plans.gamma0_plans,
            plans.gamma1_plans,
            delta,
            plans.sampling_info,
            device=dev,
        )
        vt = model.v_net(t, xt)
        gt_pred = model.g_net(t, xt)
        vloss = torch.mean((vt - ut) ** 2 * masst)
        gloss = torch.mean((gt_pred - gt) ** 2 * masst)
        loss = vloss + gloss
        loss.backward()
        opt_v.step()
        opt_g.step()
        sched_v.step()
        sched_g.step()
        v_losses.append(float(vloss.detach().cpu()))
        g_losses.append(float(gloss.detach().cpu()))
        losses.append(float(loss.detach().cpu()))
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{losses[-1]:.6f}", "vloss": f"{v_losses[-1]:.6f}", "gloss": f"{g_losses[-1]:.6f}"})

    return model, {
        "runtime_sec": float(time.time() - start),
        "final_loss": losses[-1] if losses else None,
        "final_vloss": v_losses[-1] if v_losses else None,
        "final_gloss": g_losses[-1] if g_losses else None,
        "n_loss_steps": len(losses),
    }


def make_sampling_indices(
    labels: np.ndarray,
    setting: str,
    seed: int,
    max_cells_per_time: int | None = None,
) -> np.ndarray:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected = []
    unique_times = np.array(sorted(np.unique(labels).tolist()))
    counts = {t: int(np.sum(labels == t)) for t in unique_times}
    if setting == "raw_observed_depth":
        for t in unique_times:
            idx = np.flatnonzero(labels == t)
            if max_cells_per_time is not None and len(idx) > max_cells_per_time:
                idx = rng.choice(idx, size=int(max_cells_per_time), replace=False)
            selected.append(np.sort(idx))
    elif setting == "equal_depth":
        depth = min(counts.values())
        if max_cells_per_time is not None:
            depth = min(depth, int(max_cells_per_time))
        for t in unique_times:
            idx = np.flatnonzero(labels == t)
            selected.append(np.sort(rng.choice(idx, size=depth, replace=False)))
    else:
        raise ValueError(f"Unknown sampling setting: {setting}")
    return np.concatenate(selected).astype(int)


def build_wfrfm_dataframe(features: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    indices = np.asarray(indices, dtype=int)
    x = np.asarray(features, dtype=np.float32)[indices]
    t = np.asarray(labels, dtype=np.float32)[indices]
    order = np.lexsort((indices, t))
    x = x[order]
    t = t[order]
    data = {"samples": t}
    data.update({f"x{i}": x[:, i].astype(np.float32) for i in range(x.shape[1])})
    return pd.DataFrame(data)


def standardize_train_eval(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((eval_x - mean) / std).astype(np.float32), {
        "mean": mean.ravel().tolist(),
        "std": std.ravel().tolist(),
    }


def evaluate_growth_by_bin(
    model: nn.Module,
    eval_features: np.ndarray,
    eval_times: np.ndarray,
    state_bins: np.ndarray,
    *,
    setting: str,
    seed: int,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> pd.DataFrame:
    dev = _device(device)
    model.eval()
    x_all = torch.as_tensor(eval_features, dtype=torch.float32, device=dev)
    times = np.asarray(eval_times, dtype=np.float32)
    bins = np.asarray(state_bins)
    rows = []
    with torch.no_grad():
        for eval_time in sorted(np.unique(times).tolist()):
            mask_time = times == eval_time
            g_vals = np.empty(int(mask_time.sum()), dtype=np.float32)
            x_time = x_all[mask_time]
            for start in range(0, x_time.shape[0], batch_size):
                stop = min(start + batch_size, x_time.shape[0])
                t = torch.full((stop - start, 1), float(eval_time), dtype=torch.float32, device=dev)
                g_vals[start:stop] = model.g_net(t, x_time[start:stop]).detach().cpu().numpy().reshape(-1)
            bins_time = bins[mask_time]
            for b in sorted(np.unique(bins).tolist()):
                g = g_vals[bins_time == b]
                if g.size == 0:
                    values = {"mean_g": np.nan, "median_g": np.nan, "std_g": np.nan, "frac_positive": np.nan}
                else:
                    values = {
                        "mean_g": float(np.mean(g)),
                        "median_g": float(np.median(g)),
                        "std_g": float(np.std(g)),
                        "frac_positive": float(np.mean(g > 0)),
                    }
                rows.append(
                    {
                        "setting": setting,
                        "seed": int(seed),
                        "eval_time": float(eval_time),
                        "state_bin": int(b),
                        "n_eval_cells": int(g.size),
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(pd.Series(a).rank(method="average").to_numpy(), pd.Series(b).rank(method="average").to_numpy())


def compare_growth_tables(growth: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    rows = []
    raw = growth[growth["setting"] == "raw_observed_depth"]
    equal = growth[growth["setting"] == "equal_depth"]
    for (seed, eval_time), eq in equal.groupby(["seed", "eval_time"]):
        raw_t = raw[raw["eval_time"] == eval_time]
        merged = raw_t[["state_bin", "mean_g"]].merge(eq[["state_bin", "mean_g"]], on="state_bin", suffixes=("_raw", "_equal"))
        a = merged["mean_g_raw"].to_numpy(dtype=float)
        b = merged["mean_g_equal"].to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        k = min(int(top_k), int(valid.sum()))
        raw_expand = merged.loc[valid].sort_values("mean_g_raw", ascending=False)["state_bin"].tolist()
        eq_expand = merged.loc[valid].sort_values("mean_g_equal", ascending=False)["state_bin"].tolist()
        raw_shrink = merged.loc[valid].sort_values("mean_g_raw", ascending=True)["state_bin"].tolist()
        eq_shrink = merged.loc[valid].sort_values("mean_g_equal", ascending=True)["state_bin"].tolist()
        denom = max(k, 1)
        rows.append(
            {
                "equal_seed": int(seed),
                "eval_time": float(eval_time),
                "spearman_growth_rank": _spearman(a, b),
                "pearson_growth": _pearson(a, b),
                "sign_agreement": float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))) if valid.any() else np.nan,
                "top_expanding_overlap_k3": float(len(set(raw_expand[:k]) & set(eq_expand[:k])) / denom),
                "top_shrinking_overlap_k3": float(len(set(raw_shrink[:k]) & set(eq_shrink[:k])) / denom),
                "mean_abs_delta_g": float(np.mean(np.abs(a[valid] - b[valid]))) if valid.any() else np.nan,
                "max_abs_delta_g": float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dataframe_digest(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()
