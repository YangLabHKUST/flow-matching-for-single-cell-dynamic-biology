from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_counts(X, target_sum: float = 1e4):
    X = np.asarray(X, dtype=float)
    scale = X.sum(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return X / scale * target_sum


def log1p(X):
    return np.log1p(X)


def pca(X, n_components: int = 20):
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, random_state=42)
        return model.fit_transform(X)
    except Exception:
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        return u[:, :n_components] * s[:n_components]


def _pca_with_variance(X, n_components: int, seed: int):
    X = np.asarray(X, dtype=float)
    n_components = min(int(n_components), X.shape[0], X.shape[1])
    X = X - X.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, random_state=seed)
        transformed = model.fit_transform(X)
        ratio = model.explained_variance_ratio_
    except Exception:
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        transformed = u[:, :n_components] * s[:n_components]
        denom = np.sum(s**2)
        ratio = (s[:n_components] ** 2) / denom if denom > 0 else np.zeros(n_components)
    return transformed.astype(np.float32), np.asarray(ratio, dtype=float)


def _select_hvgs_by_variance(X, n_hvgs: int) -> np.ndarray:
    n_hvgs = min(int(n_hvgs), X.shape[1])
    variance = np.var(X, axis=0)
    order = np.argsort(variance)[::-1]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[order[:n_hvgs]] = True
    return mask


def _try_scanpy_hvg(adata, n_hvgs: int, layer: str | None):
    try:
        import scanpy as sc

        tmp = adata.copy()
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvgs, layer=layer, flavor="seurat")
        mask = np.asarray(tmp.var["highly_variable"].to_numpy(), dtype=bool)
        if mask.sum() > 0:
            return mask
    except Exception:
        return None
    return None


def preprocess_adata_minimal(
    adata,
    n_hvgs: int = 80,
    n_pcs: int = 20,
    count_layer: str = "counts",
    pca_key: str = "X_pca",
    seed: int = 42,
):
    """Return a processed AnnData copy with log-normalized HVGs and PCA."""
    processed = adata.copy()
    if count_layer in processed.layers:
        counts = np.asarray(processed.layers[count_layer])
    else:
        counts = np.asarray(processed.X)
        processed.layers[count_layer] = counts.copy()

    normalized = normalize_counts(counts)
    logged = log1p(normalized)
    hvg_mask = _try_scanpy_hvg(processed, n_hvgs=n_hvgs, layer=count_layer)
    if hvg_mask is None:
        hvg_mask = _select_hvgs_by_variance(logged, n_hvgs=n_hvgs)

    processed.var["highly_variable"] = hvg_mask
    processed.layers["log_normalized"] = logged.astype(np.float32)
    processed.uns["hvg_count"] = int(hvg_mask.sum())
    processed.uns["preprocess_note"] = "Library-size normalized, log1p transformed, HVG-selected, then PCA."
    processed.uns["processed_matrix"] = "layers['log_normalized'][:, var['highly_variable']]"

    hvg_matrix = logged[:, hvg_mask]
    X_pca, variance_ratio = _pca_with_variance(hvg_matrix, n_components=n_pcs, seed=seed)
    processed.obsm[pca_key] = X_pca
    processed.uns["pca_variance_ratio"] = variance_ratio
    processed.uns["pca_n_components"] = int(X_pca.shape[1])
    return processed


def representation_table(adata) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "representation": "raw counts",
            "location": "X and layers['counts']" if "counts" in adata.layers else "X",
            "shape": str(tuple(int(x) for x in adata.X.shape)),
            "meaning": "Unnormalized pseudo-count matrix; kept for QC and reproducibility.",
            "used_for_training": False,
        }
    )
    if "log_normalized" in adata.layers:
        hvg_mask = np.asarray(adata.var.get("highly_variable", np.ones(adata.n_vars, dtype=bool)), dtype=bool)
        rows.append(
            {
                "representation": "log-normalized HVG matrix",
                "location": "layers['log_normalized'][:, var['highly_variable']]",
                "shape": str((int(adata.n_obs), int(hvg_mask.sum()))),
                "meaning": "Library-size normalized and log1p transformed high-variance genes.",
                "used_for_training": False,
            }
        )
    else:
        rows.append(
            {
                "representation": "processed count matrix",
                "location": "not computed",
                "shape": "",
                "meaning": "Run preprocess_adata_minimal to define a log-normalized HVG matrix.",
                "used_for_training": False,
            }
        )
    if "X_pca" in adata.obsm:
        rows.append(
            {
                "representation": "X_pca",
                "location": "obsm['X_pca']",
                "shape": str(tuple(int(x) for x in adata.obsm["X_pca"].shape)),
                "meaning": "Compact PCA state space used as the default modeling representation.",
                "used_for_training": True,
            }
        )
    if "X_toy_state" in adata.obsm:
        rows.append(
            {
                "representation": "X_toy_state",
                "location": "obsm['X_toy_state']",
                "shape": str(tuple(int(x) for x in adata.obsm["X_toy_state"].shape)),
                "meaning": "Original two-dimensional toy snapshot coordinates for diagnostics.",
                "used_for_training": False,
            }
        )
    return pd.DataFrame(rows)
