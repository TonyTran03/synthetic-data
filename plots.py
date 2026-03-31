# plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from scipy.stats import entropy


# ── Shared style ──────────────────────────────────────────────────────────────

METHOD_COLORS = {"bootstrap": "#4878CF", "gmm": "#F28E2B", "cvae": "#59A14F"}
METHOD_ORDER  = ["bootstrap", "gmm", "cvae"]


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _get_full_items(all_figs, dataset, method, plot_key):
    """
    Return items from all_figs matching dataset / method / full feature mode,
    sorted by fraction ascending. Filters to items that have plot_key in fig dict.
    """
    items = [
        item for item in all_figs
        if item.get("dataset")      == dataset
        and item.get("method")      == method
        and item.get("feature_mode") == "full"
        and plot_key in item.get("fig", {})
    ]
    return sorted(items, key=lambda x: x.get("frac", 0))


def _subplot_label(item):
    frac = item.get("frac")
    n0   = item.get("n0", "?")
    n1   = item.get("n1", "?")
    if frac is not None:
        return f"frac={frac:.1f}  (n0={n0}, n1={n1})"
    return f"n0={n0}, n1={n1}"


# ── 1. Correlation matrices — single and grid ─────────────────────────────────

def plot_corr_matrices(X_real, X_syn):
    """
    Three panels: real | synthetic | absolute difference.
    """
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn  = np.corrcoef(X_syn,  rowvar=False)
    diff      = np.abs(corr_real - corr_syn)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, mat, title, cmap, vmin, vmax in [
        (axs[0], corr_real, "Real",                              "RdBu_r", -1, 1),
        (axs[1], corr_syn,  "Synthetic",                         "RdBu_r", -1, 1),
        (axs[2], diff,      "Absolute difference\n(syn − real)", "Oranges",  0, 1),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Pairwise correlation structure", fontsize=13, y=1.02)
    return fig


def plot_corr_fraction_grid(all_figs, dataset, method):
    """
    One figure showing correlation difference heatmaps for all fractions.
    Each row is one fraction, showing: real | synthetic | absolute difference.
    """
    items = _get_full_items(all_figs, dataset, method, "corr")
    if not items:
        print(f"No corr figures found for {dataset} | {method}")
        return None

    n    = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for row_axes, item in zip(axes, items):
        corr_fig = item["fig"]["corr"]
        # extract data from stored figure axes
        # re-compute from scratch using stored raw arrays is not possible here,
        # so we display the pre-rendered images by extracting from the stored figure
        for col_ax, src_ax in zip(row_axes, corr_fig.get_axes()[:3]):
            for img in src_ax.get_images():
                col_ax.imshow(img.get_array(), cmap=img.get_cmap(),
                              vmin=img.norm.vmin, vmax=img.norm.vmax,
                              aspect="auto")
            col_ax.set_title(src_ax.get_title(), fontsize=10)
            col_ax.axis("off")

        row_axes[0].set_ylabel(_subplot_label(item), fontsize=10, labelpad=8)

    fig.suptitle(f"{dataset} | {method} — correlation structure across sample sizes",
                 fontsize=13, fontweight="bold")
    return fig


# ── 2. PCA projection — single and grid ──────────────────────────────────────

def plot_pca_projection(X_real, y_real, X_syn, y_syn, max_syn_display=300,
                        ax=None, title=None):
    """
    PCA scatter with 2-SD confidence ellipses.
    If ax is provided, draws into that axes (for grid use).
    If ax is None, creates and returns a new figure.
    """
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
    ax.set_title(title or "PCA — real vs synthetic", fontsize=10)
    ax.axhline(0, color="lightgray", lw=0.6)
    ax.axvline(0, color="lightgray", lw=0.6)
    ax.grid(True, alpha=0.2)

    if standalone:
        return fig


def plot_pca_fraction_grid(all_figs, dataset, method):
    """
    One figure showing PCA projections for all fractions as a column of subplots.
    Requires X_real/X_syn to be stored — uses stored pca fig data if available,
    otherwise skips. To use this properly, store raw arrays in all_figs
    (see note below).

    NOTE: PCA cannot be re-rendered from a stored matplotlib figure.
    This function works when all_figs stores 'X_real', 'y_real', 'X_syn', 'y_syn'
    alongside the fig dict. Add these to figs.append in run_experiment:
        'X_real': X_use, 'y_real': y_small,
        'X_syn':  X_syn, 'y_syn':  y_syn,
    """
    items = [
        item for item in all_figs
        if item.get("dataset")       == dataset
        and item.get("method")       == method
        and item.get("feature_mode") == "full"
        and "X_real" in item
    ]
    items = sorted(items, key=lambda x: x.get("frac", 0))

    if not items:
        print(f"No raw arrays found for {dataset} | {method} — "
              f"add X_real/y_real/X_syn/y_syn to figs.append in run_experiment")
        return None

    n    = len(items)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, items):
        plot_pca_projection(
            item["X_real"], item["y_real"],
            item["X_syn"],  item["y_syn"],
            ax=ax, title=_subplot_label(item),
        )

    fig.suptitle(f"{dataset} | {method} — PCA across sample sizes",
                 fontsize=13, fontweight="bold")
    return fig


# ── 3. KLD per feature — single and grid ─────────────────────────────────────

def plot_kld_per_feature(X_real, X_syn, feature_names=None, bins=30,
                         top_n=15, method=None, frac=None):
    """
    Bar chart of KLD per feature sorted worst → best.
    """
    klds  = _compute_klds(X_real, X_syn, bins)
    p     = len(klds)
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]
    order = np.argsort(klds)[::-1]
    if top_n is not None:
        order = order[:top_n]

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.5), 4), constrained_layout=True)
    ax.bar(range(len(order)), klds[order],
           color="#E15759", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KL divergence  KL(real ∥ syn)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    parts = ["Per-feature KLD (sorted worst → best)"]
    if method: parts.append(f"method={method}")
    if frac:   parts.append(f"frac={frac:.1f}")
    ax.set_title(" | ".join(parts), fontsize=12)
    return fig


def plot_kld_fraction_grid(all_figs, dataset, method, feature_names=None, top_n=15):
    """
    One figure showing KLD bars for all fractions as a column of subplots.
    Features ordered consistently by mean KLD across fractions.
    """
    items = _get_full_items(all_figs, dataset, method, "kld_array")
    if not items:
        print(f"No kld_array found for {dataset} | {method}")
        return None

    p     = len(items[0]["fig"]["kld_array"])
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]

    # consistent order across all fractions
    mean_kld = np.mean([item["fig"]["kld_array"] for item in items], axis=0)
    order    = np.argsort(mean_kld)[::-1]
    if top_n is not None:
        order = order[:top_n]

    n     = len(items)
    fig, axes = plt.subplots(n, 1,
                             figsize=(max(10, len(order) * 0.5), 3.5 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, items):
        klds = item["fig"]["kld_array"]
        ax.bar(range(len(order)), klds[order],
               color="#E15759", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("KLD", fontsize=10)
        ax.set_title(_subplot_label(item), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{dataset} | {method} — KLD per feature across sample sizes",
                 fontsize=13, fontweight="bold")
    return fig


def plot_kld_per_feature_by_method(kld_dict, feature_names=None, top_n=15):
    """
    Grouped bar chart comparing KLD across methods on the same feature set.
    kld_dict: {method_name: kld_array}
    """
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

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.6), 4.5), constrained_layout=True)
    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, kld_dict[method][order],
               width=width * 0.9, label=method,
               color=METHOD_COLORS.get(method, f"C{i}"),
               alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KL divergence  KL(real ∥ syn)", fontsize=11)
    ax.set_title("Per-feature KLD by method (sorted by mean, worst → best)", fontsize=12)
    ax.legend(title="Method", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


# ── 4. Ablation curve ─────────────────────────────────────────────────────────

def plot_ablation_curve(df, dataset, feature_mode="forward",
                        metric_col="rf_sep_mean", error_col="rf_sep_sd"):
    """
    Metric vs. k for forward / reverse ablation.
    y-axis fixed to [0.48, 1.02] with dashed reference at 0.5 (chance).
    """
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
    ax.set_xlabel(
        "Features kept (top k)" if feature_mode == "forward" else "Top-k features dropped",
        fontsize=11,
    )
    ax.set_ylabel(metric_col.replace("_", " "), fontsize=11)
    ax.set_title(f"{dataset} — {feature_mode} ablation", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig