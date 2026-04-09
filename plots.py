# plots.py
"""
Plotting functions for synthetic data evaluation.

Two tiers:
  - Building-block functions (plot_corr_matrices, plot_pca_projection, etc.)
    that draw a single panel or figure.
  - Paper figure functions (paper_fig_*) that compose building blocks into
    self-contained, titled figures ready for export.

All figure functions return a matplotlib Figure — the caller decides whether
to display() or savefig().
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from scipy.stats import entropy


# ── Shared style ───��──────────────────────────────���──────────────────────────

METHOD_COLORS = {"bootstrap": "#4878CF", "gmm": "#F28E2B", "cvae": "#59A14F", "iid_columnwise": "#B07AA1"}
METHOD_ORDER  = ["bootstrap", "gmm", "cvae", "iid_columnwise"]
DATASET_COLORS = {"HIV": "#1f77b4", "breast_cancer": "#ff7f0e", "diabetes": "#2ca02c"}


# ── Internal helpers ─���───────────────────────────────��───────────────────────

def _compute_klds(X_real, X_syn, bins=30):
    """KL(real || syn) per feature via histogram."""
    klds = []
    for j in range(X_real.shape[1]):
        lo = min(X_real[:, j].min(), X_syn[:, j].min())
        hi = max(X_real[:, j].max(), X_syn[:, j].max())
        if hi == lo:
            klds.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(X_real[:, j], bins=edges, density=True)
        q, _ = np.histogram(X_syn[:, j],  bins=edges, density=True)
        p = p + 1e-10;  p /= p.sum()
        q = q + 1e-10;  q /= q.sum()
        klds.append(float(entropy(p, q)))
    return np.array(klds)


def _add_ellipse(ax, x, y, edgecolor, n_std=2.0, lw=2):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    order      = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle      = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h       = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(
        (x.mean(), y.mean()), width=w, height=h, angle=angle,
        facecolor="none", edgecolor=edgecolor, lw=lw,
    ))


def _get_full_items(all_figs, dataset, method, required_key):
    """
    Return items from all_figs matching dataset / method / full feature mode,
    sorted by fraction ascending.
    """
    items = [
        item for item in all_figs
        if item.get("dataset")       == dataset
        and item.get("method")       == method
        and item.get("feature_mode") == "full"
        and required_key in item
    ]
    return sorted(items, key=lambda x: x.get("frac", 0))


def _subplot_label(item):
    frac = item.get("frac")
    n0   = item.get("n0", "?")
    n1   = item.get("n1", "?")
    if frac is not None:
        return f"frac={frac:.1f}  (n0={n0}, n1={n1})"
    return f"n0={n0}, n1={n1}"


# ═══���══════════════════════════════════════════════════════════════════════════
# Building-block functions
# ═════════════��════════════════════════════���═══════════════════════════════════

# ── 1. Correlation matrices ──────────────────────────────────────────────────

def plot_corr_matrices(X_real, X_syn, title=None):
    """Three panels: real | synthetic | absolute difference."""
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn  = np.corrcoef(X_syn,  rowvar=False)
    diff      = np.abs(corr_real - corr_syn)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, mat, label, cmap, vmin, vmax in [
        (axs[0], corr_real, "Real",       "RdBu_r", -1, 1),
        (axs[1], corr_syn,  "Synthetic",  "RdBu_r", -1, 1),
        (axs[2], diff,      "|Real − Syn|", "Oranges", 0, 1),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(label, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title or "Correlation structure: real vs synthetic",
                 fontsize=12, y=1.02)
    return fig


# ── 2. PCA projection ──────���──────────────────────────────────────���─────────

def plot_pca_projection(X_real, y_real, X_syn, y_syn, max_syn_display=300,
                        ax=None, title=None):
    """PCA scatter with 2-SD confidence ellipses."""
    pca    = PCA(n_components=2).fit(X_real)
    Z_real = pca.transform(X_real)
    Z_syn  = pca.transform(X_syn)

    if len(Z_syn) > max_syn_display:
        idx   = np.random.default_rng(0).choice(len(Z_syn), max_syn_display, replace=False)
        Z_syn = Z_syn[idx]
        y_syn = y_syn[idx]

    df = pd.DataFrame({
        "PC1":    np.concatenate([Z_real[:, 0], Z_syn[:, 0]]),
        "PC2":    np.concatenate([Z_real[:, 1], Z_syn[:, 1]]),
        "class":  np.concatenate([y_real, y_syn]).astype(str),
        "source": ["Real"] * len(Z_real) + ["Synthetic"] * len(Z_syn),
    })

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df, x="PC1", y="PC2",
        hue="class", style="source",
        palette={"0": "#4878CF", "1": "#E15759"},
        markers={"Real": "o", "Synthetic": "^"},
        alpha=0.55, s=40, ax=ax, legend=standalone,
    )

    for (src, cls), color in {
        ("Real",      "0"): "#4878CF",
        ("Real",      "1"): "#E15759",
        ("Synthetic", "0"): "#76A8E0",
        ("Synthetic", "1"): "#F0908F",
    }.items():
        d = df[(df["source"] == src) & (df["class"] == cls)]
        _add_ellipse(ax, d["PC1"], d["PC2"], edgecolor=color)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    ax.set_title(title or "PCA: real vs synthetic", fontsize=10)
    ax.axhline(0, color="lightgray", lw=0.6)
    ax.axvline(0, color="lightgray", lw=0.6)
    ax.grid(True, alpha=0.2)

    if standalone:
        return fig


# ── 3. KLD per feature ───────��────────────────────────────────���─────────────

def plot_kld_per_feature(X_real, X_syn, feature_names=None, bins=30,
                         top_n=15, title=None):
    """Bar chart of KLD per feature sorted worst-first."""
    klds  = _compute_klds(X_real, X_syn, bins)
    p     = len(klds)
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]
    order = np.argsort(klds)[::-1]
    if top_n is not None:
        order = order[:top_n]

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.5), 4),
                           constrained_layout=True)
    ax.bar(range(len(order)), klds[order],
           color="#E15759", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KL divergence  KL(real \u2225 syn)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title or f"Per-feature KLD (top {len(order)}, worst first)", fontsize=12)
    return fig


def plot_kld_per_feature_by_method(kld_dict, feature_names=None, top_n=15,
                                   title=None):
    """Grouped bar chart comparing KLD across methods on the same feature set."""
    methods = [m for m in METHOD_ORDER if m in kld_dict]
    if not methods:
        raise ValueError("kld_dict has no recognised method keys.")

    p      = len(next(iter(kld_dict.values())))
    names  = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]
    mean_k = np.mean([kld_dict[m] for m in methods], axis=0)
    order  = np.argsort(mean_k)[::-1]
    if top_n is not None:
        order = order[:top_n]

    x     = np.arange(len(order))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.6), 4.5),
                           constrained_layout=True)
    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, kld_dict[method][order],
               width=width * 0.9, label=method,
               color=METHOD_COLORS.get(method, f"C{i}"),
               alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KL divergence  KL(real \u2225 syn)", fontsize=11)
    ax.set_title(title or f"Per-feature KLD by method (top {len(order)}, worst first)",
                 fontsize=12)
    ax.legend(title="Method", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


# ── 4. Ablation curve ───────────���───────────────────────────────────────────

def plot_ablation_curve(df, dataset, feature_mode="forward",
                        metric_col="rf_sep_mean", error_col="rf_sep_sd",
                        title=None):
    """Metric vs k for forward / reverse ablation."""
    sub = df[(df["dataset"] == dataset) & (df["feature_mode"] == feature_mode)].copy()
    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset}, feature_mode={feature_mode}")

    x_col = "subset_param" if "subset_param" in sub.columns else "k"
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, metric_col])

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for method in [m for m in METHOD_ORDER if m in sub["method"].unique()]:
        d     = sub[sub["method"] == method].sort_values(x_col)
        color = METHOD_COLORS.get(method)
        if error_col and error_col in d.columns:
            ax.errorbar(d[x_col], d[metric_col], yerr=d[error_col],
                        marker="o", capsize=4, label=method,
                        color=color, linewidth=2, markersize=6)
        else:
            ax.plot(d[x_col], d[metric_col],
                    marker="o", label=method, color=color, linewidth=2, markersize=6)

    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="chance (0.5)")
    ax.set_ylim(0.48, 1.02)

    default_xlabel = ("Features kept (top k)" if feature_mode == "forward"
                      else "Top-k features dropped")
    ax.set_xlabel(default_xlabel, fontsize=11)
    ax.set_ylabel(metric_col.replace("_", " "), fontsize=11)
    ax.set_title(title or f"{dataset} \u2014 {feature_mode} ablation ({metric_col})",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


# ═══════════════════════════════���══════════════════════════════════════════════
# Grid / multi-panel convenience functions (for exploratory notebooks)
# ══════���══════════════════════════════════════════════��════════════════════════

def plot_corr_fraction_grid(all_figs, dataset, method):
    """Correlation triptychs stacked by sample fraction."""
    items = _get_full_items(all_figs, dataset, method, "X_real")
    if not items:
        return None

    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for row_axes, item in zip(axes, items):
        X_r, X_s = item["X_real"], item["X_syn"]
        corr_r = np.corrcoef(X_r, rowvar=False)
        corr_s = np.corrcoef(X_s, rowvar=False)
        diff   = np.abs(corr_r - corr_s)

        for ax, mat, label, cmap, vmin, vmax in [
            (row_axes[0], corr_r, "Real",       "RdBu_r", -1, 1),
            (row_axes[1], corr_s, "Synthetic",  "RdBu_r", -1, 1),
            (row_axes[2], diff,   "|Real \u2212 Syn|", "Oranges", 0, 1),
        ]:
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(label, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        row_axes[0].set_ylabel(_subplot_label(item), fontsize=10, labelpad=8)

    fig.suptitle(f"{dataset} | {method} \u2014 correlation structure across sample sizes",
                 fontsize=13, fontweight="bold")
    return fig


def plot_pca_fraction_grid(all_figs, dataset, method):
    """PCA projections tiled by sample fraction."""
    items = _get_full_items(all_figs, dataset, method, "X_real")
    if not items:
        return None

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, items):
        plot_pca_projection(
            item["X_real"], item["y_real"],
            item["X_syn"],  item["y_syn"],
            ax=ax, title=_subplot_label(item),
        )

    fig.suptitle(f"{dataset} | {method} \u2014 PCA across sample sizes",
                 fontsize=13, fontweight="bold")
    return fig


def plot_kld_fraction_grid(all_figs, dataset, method, feature_names=None, top_n=15):
    """KLD bar charts stacked by sample fraction."""
    items = _get_full_items(all_figs, dataset, method, "kld_array")
    if not items:
        return None

    p     = len(items[0]["kld_array"])
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]

    mean_kld = np.mean([item["kld_array"] for item in items], axis=0)
    order    = np.argsort(mean_kld)[::-1]
    if top_n is not None:
        order = order[:top_n]

    n = len(items)
    fig, axes = plt.subplots(n, 1,
                             figsize=(max(10, len(order) * 0.5), 3.5 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, items):
        klds = item["kld_array"]
        ax.bar(range(len(order)), klds[order],
               color="#E15759", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("KLD", fontsize=10)
        ax.set_title(_subplot_label(item), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{dataset} | {method} \u2014 KLD per feature across sample sizes",
                 fontsize=13, fontweight="bold")
    return fig


# ═══��═══════════════════════════════════════════════════════���══════════════════
# Paper figure functions — self-contained, one question per figure
# ════════════════��═══════════════════════════════���═════════════════════════════

def paper_fig_rf_sep_vs_frac(df, datasets=None):
    """
    Fig: RF separability vs sample fraction for all methods.
    Returns a list of individual figures, one per dataset.
    """
    df_full = df[df["feature_mode"] == "full"]
    if datasets is None:
        datasets = list(df_full["dataset"].unique())

    figs = []
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        sub = df_full[df_full["dataset"] == ds]
        for m in METHOD_ORDER:
            d = sub[sub["method"] == m].sort_values("frac")
            if d.empty:
                continue
            color = METHOD_COLORS[m]
            ax.plot(d["frac"], d["rf_sep_mean"], marker="o",
                    label=m, color=color)
            ax.fill_between(d["frac"],
                            d["rf_sep_mean"] - d["rf_sep_sd"],
                            d["rf_sep_mean"] + d["rf_sep_sd"],
                            alpha=0.15, color=color)

        ax.axhline(0.5, ls="--", color="black", lw=0.8, label="chance")
        ax.set_title(f"{ds} — Can a classifier distinguish real from synthetic?",
                     fontsize=11)
        ax.set_xlabel("Fraction of real data")
        ax.set_ylabel("RF Separation")
        ax.set_ylim(0.45, 1.05)
        ax.legend(fontsize=8)
        figs.append(fig)

    return figs


def paper_fig_kld_vs_rf_sep(df):
    """
    Fig: scatter of mean KLD vs RF Sep at each (dataset, method, frac).
    Tests whether marginal fidelity predicts functional indistinguishability.
    """
    df_full = df[df["feature_mode"] == "full"]
    ds_markers = {"HIV": "o", "breast_cancer": "s", "diabetes": "^"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHOD_ORDER:
        for ds in df_full["dataset"].unique():
            sub = df_full[(df_full["method"] == m) & (df_full["dataset"] == ds)]
            ax.scatter(sub["kld_mean"], sub["rf_sep_mean"],
                       c=METHOD_COLORS.get(m, "gray"),
                       marker=ds_markers.get(ds, "o"),
                       alpha=0.7, s=60, label=f"{m} / {ds}")

    ax.axhline(0.5, ls="--", color="black", lw=0.8)
    ax.set_xlabel("Mean KLD (lower = more similar marginals)")
    ax.set_ylabel("RF Sep (0.5 = indistinguishable)")
    ax.set_title("Do marginal fidelity and classifier separability agree?",
                 fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def paper_fig_ablation(df, dataset, title=None):
    """
    Fig: forward and reverse ablation side by side for one dataset.
    Shows whether distinguishability is concentrated or distributed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for ax, mode, xlabel in [
        (axes[0], "forward", "Features kept (top k)"),
        (axes[1], "reverse", "Top-k features dropped"),
    ]:
        sub = df[(df["dataset"] == dataset) & (df["feature_mode"] == mode)].copy()
        x_col = "subset_param"
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub = sub.dropna(subset=[x_col, "rf_sep_mean"])

        for m in METHOD_ORDER:
            d = sub[sub["method"] == m].sort_values(x_col)
            if d.empty:
                continue
            color = METHOD_COLORS[m]
            ax.errorbar(d[x_col], d["rf_sep_mean"],
                        yerr=d.get("rf_sep_sd", None),
                        marker="o", capsize=4, label=m,
                        color=color, linewidth=2, markersize=6)

        ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6)
        ax.set_ylim(0.48, 1.02)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("RF Separation", fontsize=11)
        ax.set_title(f"{mode.title()} ablation", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(
        title or f"{dataset} \u2014 Is distinguishability concentrated in a few features?",
        fontsize=13,
    )
    return fig
