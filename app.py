import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

st.set_page_config(page_title="Synthetic Fidelity Dashboard", layout="wide")

# =========================================================
# PATHS
# =========================================================
CSV_PATH = "results/synthetic_results_clean.csv"
SUMMARY_FIGS_PATH = "results/summary_figs.pkl"
ALL_FIGS_PATH = "results/all_figs.pkl"

# =========================================================
# LOADERS
# =========================================================
@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    numeric_cols = [
        "n0", "n1", "seed",
        "subset_param",
        "n_features_used", "n_features_total",
        "rf_sep_mean", "rf_sep_sd",
        "rf_auc_mean", "rf_auc_sd",
        "corr_mean_abs_diff", "corr_max_abs_diff",
        "prop_significant",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_resource
def load_pickle(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


df = load_results(CSV_PATH)
summary_figs = load_pickle(SUMMARY_FIGS_PATH)
all_figs = load_pickle(ALL_FIGS_PATH)

# =========================================================
# HELPERS
# =========================================================
METHOD_ORDER = ["bootstrap", "gmm", "cvae"]
DATASET_ORDER = ["breast_cancer", "diabetes"]
FEATURE_MODE_ORDER = ["full", "drop_one", "forward", "reverse"]


def ordered_values(values, preferred_order):
    vals = list(pd.Series(values).dropna().unique())
    return [x for x in preferred_order if x in vals] + [x for x in vals if x not in preferred_order]


def padded_limits(series, pad_frac=0.08, default=(0.0, 1.0)):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return default

    lo = float(s.min())
    hi = float(s.max())

    if lo == hi:
        pad = 0.05 if lo == 0 else abs(lo) * pad_frac
        return lo - pad, hi + pad

    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def metric_limits(df_sub, value_col, error_col=None, default=(0.0, 1.0)):
    vals = pd.to_numeric(df_sub[value_col], errors="coerce")
    if error_col is not None and error_col in df_sub.columns:
        err = pd.to_numeric(df_sub[error_col], errors="coerce").fillna(0)
        vals = pd.concat([vals - err, vals + err], ignore_index=True)
    return padded_limits(vals, default=default)


def subset_xlim(df_sub):
    s = pd.to_numeric(df_sub["subset_param"], errors="coerce").dropna()
    if s.empty:
        return None
    lo, hi = float(s.min()), float(s.max())
    return (lo - 0.5, hi + 0.5)


def get_summary_fig(summary_figs, dataset, mode):
    if summary_figs is None:
        return None
    key = f"ablation_{dataset}_{mode}"
    return summary_figs.get(key)


def filter_all_figs(figs, dataset=None, method=None, feature_mode=None):
    if figs is None:
        return []

    out = []
    for item in figs:
        if dataset is not None and item.get("dataset") != dataset:
            continue
        if method is not None and item.get("method") != method:
            continue
        if feature_mode is not None and item.get("feature_mode") != feature_mode:
            continue
        out.append(item)
    return out


def render_saved_diag_figure(fig_item, plot_key):
    fig_obj = fig_item.get("fig")
    if isinstance(fig_obj, dict):
        if plot_key in fig_obj:
            st.pyplot(fig_obj[plot_key], clear_figure=False)
        else:
            st.info(f"Plot key '{plot_key}' not found for this figure.")
    else:
        st.pyplot(fig_obj, clear_figure=False)


# =========================================================
# PLOTTING FROM CSV
# =========================================================
def plot_full_compare(df_sub, metric_col="rf_sep_mean", error_col="rf_sep_sd", ylim=None):
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(10, 3.5), sharey=False)

    if len(DATASET_ORDER) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASET_ORDER):
        dset = df_sub[(df_sub["dataset"] == dataset) & (df_sub["feature_mode"] == "full")].copy()

        if dset.empty:
            ax.text(0.5, 0.5, f"No rows for {dataset}", ha="center", va="center")
            ax.set_axis_off()
            continue

        dset["sample_label"] = dset["n0"].astype("Int64").astype(str) + "-" + dset["n1"].astype("Int64").astype(str)
        dset = dset.sort_values(["n0", "n1"])

        for method in METHOD_ORDER:
            d = dset[dset["method"] == method]
            if d.empty:
                continue

            ax.errorbar(
                d["sample_label"],
                d[metric_col],
                yerr=d[error_col] if error_col in d.columns else None,
                marker="o",
                capsize=4,
                label=method,
            )

        ax.set_title(dataset)
        ax.set_xlabel("(n0, n1)")
        ax.set_ylabel("RF separation")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
        if ylim is not None:
            ax.set_ylim(*ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)

    fig.suptitle("Full feature set baseline", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_drop_one_compare(df_sub, metric_col="rf_sep_mean", error_col="rf_sep_sd", ylim=None, xlim=None):
    fig, axes = plt.subplots(len(DATASET_ORDER), len(METHOD_ORDER), figsize=(16, 7), sharex=False, sharey=False)

    for i, dataset in enumerate(DATASET_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            ax = axes[i, j]
            d = df_sub[
                (df_sub["dataset"] == dataset) &
                (df_sub["feature_mode"] == "drop_one") &
                (df_sub["method"] == method)
            ].copy()

            if d.empty:
                ax.text(0.5, 0.5, "No rows", ha="center", va="center")
                ax.set_axis_off()
                continue

            d["subset_param"] = pd.to_numeric(d["subset_param"], errors="coerce")
            d = d.dropna(subset=["subset_param", metric_col]).sort_values("subset_param")

            ax.errorbar(
                d["subset_param"],
                d[metric_col],
                yerr=d[error_col] if error_col in d.columns else None,
                marker="o",
                capsize=3,
            )

            if i == 0:
                ax.set_title(method)
            if j == 0:
                ax.set_ylabel(f"{dataset}\nRF separation")
            else:
                ax.set_ylabel("RF separation")

            ax.set_xlabel("Dropped feature index")
            ax.grid(True, alpha=0.3)

            if ylim is not None:
                ax.set_ylim(*ylim)
            if xlim is not None:
                ax.set_xlim(*xlim)

    fig.suptitle("Drop-one ablation", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def render_ablation_summary_figures(summary_figs, mode):
    st.subheader(f"{mode.capitalize()} summary figures")
    for dataset in DATASET_ORDER:
        fig = get_summary_fig(summary_figs, dataset, mode)
        if fig is not None:
            st.pyplot(fig, clear_figure=False)
        else:
            st.info(f"No summary figure found for {dataset} | {mode}")


def render_mode_table(df_sub, feature_mode):
    sub = df_sub[df_sub["feature_mode"] == feature_mode].copy()

    cols = [
        "dataset", "method", "feature_mode",
        "n0", "n1",
        "subset_label", "subset_param",
        "n_features_used",
        "rf_sep_mean", "rf_sep_sd",
        "rf_auc_mean", "rf_auc_sd",
        "corr_mean_abs_diff", "corr_max_abs_diff",
        "prop_significant",
    ]
    cols = [c for c in cols if c in sub.columns]

    sort_cols = [c for c in ["dataset", "method", "subset_param"] if c in sub.columns]

    st.dataframe(
        sub[cols].sort_values(sort_cols),
        use_container_width=True,
    )


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Controls")

feature_mode = st.sidebar.selectbox(
    "Feature mode",
    ordered_values(df["feature_mode"], FEATURE_MODE_ORDER)
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
This dashboard is organized by **feature mode** first.

That makes it easier to compare **datasets against each other**
for the same subgrouping strategy.
"""
)

# =========================================================
# FILTERED DATA
# =========================================================
df_mode = df[df["feature_mode"] == feature_mode].copy()

rf_ylim = metric_limits(df_mode, "rf_sep_mean", error_col="rf_sep_sd", default=(0.45, 0.75))
drop_xlim = subset_xlim(df_mode) if feature_mode == "drop_one" else None

# =========================================================
# HEADER
# =========================================================
st.title("Synthetic Fidelity Dashboard")
st.subheader(f"Feature mode: {feature_mode}")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows", len(df_mode))
with c2:
    st.metric("Datasets", df_mode["dataset"].nunique())
with c3:
    st.metric("Methods", df_mode["method"].nunique())

st.markdown("---")

# =========================================================
# MAIN CONTENT BY FEATURE MODE
# =========================================================
if feature_mode == "full":
    st.header("Full-data baseline")
    st.caption("Small baseline view only. This is context before subgrouping / ablation.")
    st.pyplot(plot_full_compare(df_mode, ylim=rf_ylim))

    with st.expander("Show full-mode table"):
        render_mode_table(df, "full")

elif feature_mode == "drop_one":
    st.header("Drop-one comparison")
    st.caption("Compare whether one feature dominates the distinguishability signal.")
    st.pyplot(plot_drop_one_compare(df_mode, ylim=rf_ylim, xlim=drop_xlim))

    with st.expander("Show drop-one table"):
        render_mode_table(df, "drop_one")

elif feature_mode in {"forward", "reverse"}:
    st.header(f"{feature_mode.capitalize()} comparison")
    st.caption(
        "These are the summary ablation figures produced by your original helper "
        "`evaluate_abl(df)`."
    )
    render_ablation_summary_figures(summary_figs, feature_mode)

    with st.expander(f"Show {feature_mode} table"):
        render_mode_table(df, feature_mode)

# =========================================================
# DIAGNOSTIC FIGURES FROM evaluate_all(...)
# =========================================================
st.markdown("---")
st.header("Representative diagnostic figures")
st.caption("These come from the saved per-run figures returned by `evaluate_all(...)`.")

diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)

with diag_col1:
    diag_dataset = st.selectbox("Dataset", ordered_values(df["dataset"], DATASET_ORDER))
with diag_col2:
    diag_method = st.selectbox("Method", ordered_values(df["method"], METHOD_ORDER))
with diag_col3:
    diag_plot_key = st.selectbox("Plot type", ["pca", "corr", "overlap"])
with diag_col4:
    diag_mode = st.selectbox("Diagnostic feature mode", ordered_values(df["feature_mode"], FEATURE_MODE_ORDER), index=FEATURE_MODE_ORDER.index(feature_mode) if feature_mode in FEATURE_MODE_ORDER else 0)

diag_candidates = filter_all_figs(
    all_figs,
    dataset=diag_dataset,
    method=diag_method,
    feature_mode=diag_mode,
)

if not diag_candidates:
    st.info("No matching saved figures found.")
else:
    labels = [item.get("subset_label", "") for item in diag_candidates]
    chosen_idx = st.selectbox(
        "Subset label",
        options=list(range(len(diag_candidates))),
        format_func=lambda i: labels[i] if labels[i] else f"entry_{i}",
    )
    render_saved_diag_figure(diag_candidates[chosen_idx], diag_plot_key)