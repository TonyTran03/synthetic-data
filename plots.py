# plots.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
def plot_corr_matrices(X_real, X_syn):

    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn = np.corrcoef(X_syn, rowvar=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axs[0].imshow(corr_real, vmin=-1, vmax=1)
    axs[0].set_title("Real corr")

    im1 = axs[1].imshow(corr_syn, vmin=-1, vmax=1)
    axs[1].set_title("Synthetic corr")

    plt.colorbar(im1, ax=axs)
    return fig

import seaborn as sns

from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


def add_confidence_ellipse(ax, x, y, edgecolor="black", label=None, n_std=2.0, lw=2):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2:
        return

    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        lw=lw,
        label=label,
    )
    ax.add_patch(ellipse)

def plot_pca_projection(X_real, y_real, X_syn, y_syn):
    # Fit PCA on real data only
    pca = PCA(n_components=2)
    pca.fit(X_real)

    Z_real = pca.transform(X_real)
    Z_syn = pca.transform(X_syn)

    Z = np.vstack([Z_real, Z_syn])
    y = np.concatenate([y_real, y_syn])
    source = np.array(["Real"] * len(X_real) + ["Synthetic"] * len(X_syn))

    df_plot = pd.DataFrame({
        "PC1": Z[:, 0],
        "PC2": Z[:, 1],
        "class": y.astype(str),
        "source": source,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df_plot,
        x="PC1",
        y="PC2",
        hue="class",
        style="source",
        palette={"0": "tab:blue", "1": "tab:red"},
        markers={"Real": "o", "Synthetic": "^"},
        alpha=0.65,
        s=70,
        ax=ax,
    )

    combo_colors = {
        ("Real", "0"): "tab:blue",
        ("Real", "1"): "tab:red",
        ("Synthetic", "0"): "deepskyblue",
        ("Synthetic", "1"): "salmon",
    }

    for src in ["Real", "Synthetic"]:
        for cls in ["0", "1"]:
            d = df_plot[(df_plot["source"] == src) & (df_plot["class"] == cls)]
            add_confidence_ellipse(
                ax,
                d["PC1"],
                d["PC2"],
                edgecolor=combo_colors[(src, cls)],
                label=f"{src} class {cls}",
                n_std=2.0,
                lw=2,
            )

    ax.set_title("PCA projection by class and source")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.axhline(0, color="lightgray", lw=1)
    ax.axvline(0, color="lightgray", lw=1)
    ax.grid(True, alpha=0.25)

    return fig

def plot_flat_overlap(X_real, X_syn):
    fig, ax = plt.subplots()
    ax.hist(
        X_real.ravel(),
        bins=50,
        density=True,
        alpha=0.5,
        label="Real"
    )

    ax.hist(
        X_syn.ravel(),
        bins=50,
        density=True,
        alpha=0.5,
        label="Syn"
    )

    ax.legend()
    ax.set_title("Value overlap")

    return fig


from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd

def plot_ablation_curve(
    df,
    dataset,
    feature_mode="forward",
    metric_col="rf_sep_mean",
    error_col="rf_sep_sd",
):
    sub = df[
        (df["dataset"] == dataset) &
        (df["feature_mode"] == feature_mode)
    ].copy()

    if sub.empty:
        raise ValueError(f"No rows found for dataset={dataset}, feature_mode={feature_mode}")

    # old schema used k, new clean schema uses subset_param
    if "subset_param" in sub.columns:
        x_col = "subset_param"
    elif "k" in sub.columns:
        x_col = "k"
    else:
        raise KeyError("Neither 'subset_param' nor 'k' exists in the dataframe.")

    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, metric_col])

    fig, ax = plt.subplots(figsize=(7, 4))

    method_order = ["bootstrap", "gmm", "cvae"]
    methods_present = [m for m in method_order if m in sub["method"].unique()]

    for method in methods_present:
        d = sub[sub["method"] == method].copy().sort_values(x_col)

        if error_col is not None and error_col in d.columns:
            ax.errorbar(
                d[x_col],
                d[metric_col],
                yerr=d[error_col],
                marker="o",
                capsize=3,
                label=method,
            )
        else:
            ax.plot(
                d[x_col],
                d[metric_col],
                marker="o",
                label=method,
            )

    if feature_mode == "forward":
        ax.set_xlabel("Top k kept")
    elif feature_mode == "reverse":
        ax.set_xlabel("Top k dropped")
    else:
        ax.set_xlabel(x_col)

    ax.set_ylabel(metric_col)
    ax.set_title(f"{dataset} | {feature_mode} ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig