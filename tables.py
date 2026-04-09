# tables.py
"""
Reporting tables for synthetic data evaluation.

One function per experiment type:
    make_table_full()      → Table 1: main results, fixed n, all features
    make_table_sample_size() → Table 2: effect of sample size (n sweep)
    make_table_forward()   → Table 3: forward ablation (keep top k)
    make_table_reverse()   → Table 4: reverse ablation (drop top k)
    make_table_drop_one()  → Table 5: drop-one feature sensitivity

All return a styled pandas DataFrame ready for display() or export.
Call save_all_tables(df_all, outdir) to write every table to CSV at once.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Column definitions ────────────────────────────────────────────────────────

# Maps raw column name → (display name, higher/lower is better, format string)
METRIC_SPEC = {
    "rf_sep_mean":        ("RF Sep ↓",       "lower", "{:.3f}"),
    "rf_sep_sd":          ("RF Sep SD",       None,    "±{:.3f}"),
    "disc_f1_mean":       ("Disc F1 ↓",      "lower", "{:.3f}"),
    "disc_f1_sd":         ("Disc F1 SD",      None,    "±{:.3f}"),
    "tstr_f1":            ("TSTR F1 ↑",      "higher", "{:.3f}"),
    "trtr_f1":            ("TRTR F1 (base)", None,     "{:.3f}"),
    "utility_gap":        ("Util Gap ↓",     "lower", "{:.3f}"),
    "kld_mean":           ("KLD Mean ↓",     "lower", "{:.3f}"),
    "kld_max":            ("KLD Max ↓",      "lower", "{:.3f}"),
    "corr_mean_abs_diff": ("Corr Diff ↓",   "lower", "{:.3f}"),
    "prop_significant":   ("Prop Sig ↓",    "lower", "{:.3f}"),
}

# Columns included in the main summary table (Table 1 / Table 2)
SUMMARY_METRICS = [
    "rf_sep_mean", "rf_sep_sd",
    "disc_f1_mean", "disc_f1_sd",
    "tstr_f1", "trtr_f1", "utility_gap",
    "kld_mean",
    "corr_mean_abs_diff",
    "prop_significant",
]

METHOD_ORDER   = ["bootstrap", "gmm", "cvae", "iid_columnwise"]
DATASET_ORDER  = ["HIV", "breast_cancer", "diabetes"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _keep_present(cols, df):
    """Return only cols that actually exist in df."""
    return [c for c in cols if c in df.columns]


def _method_sort(df):
    df = df.copy()
    df["_m_order"] = df["method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    ).fillna(99)
    return df.sort_values(["_m_order"]).drop(columns=["_m_order"])


def _dataset_sort(df):
    df = df.copy()
    df["_d_order"] = df["dataset"].map(
        {d: i for i, d in enumerate(DATASET_ORDER)}
    ).fillna(99)
    return df.sort_values(["_d_order", "_m_order"] if "_m_order" in df.columns
                          else ["_d_order"]).drop(columns=["_d_order"])


def _format_mean_sd(df_out, mean_col, sd_col, new_col):
    """Combine mean ± SD into a single display string."""
    if mean_col in df_out.columns and sd_col in df_out.columns:
        df_out[new_col] = df_out.apply(
            lambda r: f"{r[mean_col]:.3f} ± {r[sd_col]:.3f}"
            if pd.notna(r[mean_col]) else "—",
            axis=1,
        )
        df_out = df_out.drop(columns=[mean_col, sd_col])
    return df_out


def _rename_metrics(df):
    rename = {k: v[0] for k, v in METRIC_SPEC.items() if k in df.columns}
    return df.rename(columns=rename)


def _style(df):
    """
    Apply background gradient to numeric columns so the
    'best' cell per column is visually obvious.
    Returns a Styler (use display() or .to_html()).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return df.style

    return (
        df.style
        .format(precision=3, na_rep="—")
        .background_gradient(subset=numeric_cols, cmap="RdYlGn_r", axis=0)
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "11px"), ("text-align", "center")]},
            {"selector": "td", "props": [("font-size", "11px"), ("text-align", "center")]},
        ])
    )


# ── Table 1: Main results — full features, fixed n ───────────────────────────

def make_table_full(df_all,frac=None):
    """
    Table 1: one row per (dataset, method).
    Filtered to feature_mode='full' and optionally to a specific (n0, n1).

    Parameters
    ----------
    df_all : pd.DataFrame  — master results dataframe
    n0, n1 : int or None   — if provided, filter to that sample size

    Returns
    -------
    styled pd.DataFrame
    """
    sub = df_all[df_all["feature_mode"] == "full"].copy()

    if frac is not None:
        sub = sub[sub["frac"] == frac]

    if sub.empty:
        print("make_table_full: no rows found after filtering.")
        return pd.DataFrame()

    keep_cols = ["dataset", "method", "n0", "n1"] + _keep_present(SUMMARY_METRICS, sub)
    sub = sub[keep_cols]

    # Combine RF sep mean ± sd into one column
    sub = _format_mean_sd(sub, "rf_sep_mean", "rf_sep_sd", "RF Sep (mean ± SD)")
    sub = _format_mean_sd(sub, "disc_f1_mean", "disc_f1_sd", "Disc F1 (mean ± SD)")

    # Sort
    sub["_m"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    sub["_d"] = sub["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
    sub = sub.sort_values(["_d", "_m"]).drop(columns=["_m", "_d"])

    sub = _rename_metrics(sub)
    sub = sub.rename(columns={"dataset": "Dataset", "method": "Method",
                               "n0": "n₀", "n1": "n₁"})

    print(f"Table 1 — Full feature results ({len(sub)} rows)")
    return sub.reset_index(drop=True)


# ── Table 2: Sample size sweep ────────────────────────────────────────────────

def make_table_sample_size(df_all):
    """
    Table 2: how does quality change with n?
    Rows: (dataset, method, n0, n1), feature_mode='full' only.
    """
    sub = df_all[df_all["feature_mode"] == "full"].copy()

    if sub.empty:
        print("make_table_sample_size: no full-mode rows found.")
        return pd.DataFrame()

    keep_cols = (["dataset", "method", "n0", "n1"]
                 + _keep_present(SUMMARY_METRICS, sub))
    sub = sub[keep_cols]

    sub = _format_mean_sd(sub, "rf_sep_mean", "rf_sep_sd", "RF Sep (mean ± SD)")
    sub = _format_mean_sd(sub, "disc_f1_mean", "disc_f1_sd", "Disc F1 (mean ± SD)")

    sub["_m"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    sub["_d"] = sub["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
    sub = sub.sort_values(["_d", "n0", "n1", "_m"]).drop(columns=["_m", "_d"])

    sub = _rename_metrics(sub)
    sub = sub.rename(columns={"dataset": "Dataset", "method": "Method",
                               "n0": "n₀", "n1": "n₁"})

    print(f"Table 2 — Sample size sweep ({len(sub)} rows)")
    return sub.reset_index(drop=True)


# ── Table 3: Forward ablation (keep top k features) ──────────────────────────

def make_table_forward(df_all, dataset=None):
    """
    Table 3: forward ablation — keep top k features.
    Rows: (dataset, method, k), sorted by dataset → k → method.

    Parameters
    ----------
    dataset : str or None — filter to one dataset, or None for all
    """
    sub = df_all[df_all["feature_mode"] == "forward"].copy()

    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]

    if sub.empty:
        print("make_table_forward: no forward-mode rows found.")
        return pd.DataFrame()

    # k lives in subset_param (clean schema) or k (old schema)
    k_col = "subset_param" if "subset_param" in sub.columns else "k"
    sub["k"] = pd.to_numeric(sub[k_col], errors="coerce")

    keep_cols = (["dataset", "method", "k", "n_features_used"]
                 + _keep_present(SUMMARY_METRICS, sub))
    sub = sub[[c for c in keep_cols if c in sub.columns]]

    sub = _format_mean_sd(sub, "rf_sep_mean", "rf_sep_sd", "RF Sep (mean ± SD)")
    sub = _format_mean_sd(sub, "disc_f1_mean", "disc_f1_sd", "Disc F1 (mean ± SD)")

    sub["_m"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    sub["_d"] = sub["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
    sub = sub.sort_values(["_d", "k", "_m"]).drop(columns=["_m", "_d"])

    sub = _rename_metrics(sub)
    sub = sub.rename(columns={
        "dataset": "Dataset", "method": "Method",
        "k": "k (features kept)", "n_features_used": "# Features",
    })

    print(f"Table 3 — Forward ablation ({len(sub)} rows)")
    return sub.reset_index(drop=True)


# ── Table 4: Reverse ablation (drop top k features) ──────────────────────────

def make_table_reverse(df_all, dataset=None):
    """
    Table 4: reverse ablation — drop top k features.
    Rows: (dataset, method, k), sorted by dataset → k → method.
    """
    sub = df_all[df_all["feature_mode"] == "reverse"].copy()

    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]

    if sub.empty:
        print("make_table_reverse: no reverse-mode rows found.")
        return pd.DataFrame()

    k_col = "subset_param" if "subset_param" in sub.columns else "k"
    sub["k"] = pd.to_numeric(sub[k_col], errors="coerce")

    keep_cols = (["dataset", "method", "k", "n_features_used"]
                 + _keep_present(SUMMARY_METRICS, sub))
    sub = sub[[c for c in keep_cols if c in sub.columns]]

    sub = _format_mean_sd(sub, "rf_sep_mean", "rf_sep_sd", "RF Sep (mean ± SD)")
    sub = _format_mean_sd(sub, "disc_f1_mean", "disc_f1_sd", "Disc F1 (mean ± SD)")

    sub["_m"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    sub["_d"] = sub["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
    sub = sub.sort_values(["_d", "k", "_m"]).drop(columns=["_m", "_d"])

    sub = _rename_metrics(sub)
    sub = sub.rename(columns={
        "dataset": "Dataset", "method": "Method",
        "k": "k (features dropped)", "n_features_used": "# Features",
    })

    print(f"Table 4 — Reverse ablation ({len(sub)} rows)")
    return sub.reset_index(drop=True)


# ── Table 5: Drop-one sensitivity ────────────────────────────────────────────

def make_table_drop_one(df_all, dataset=None, sort_by="rf_sep_mean"):
    """
    Table 5: drop-one feature sensitivity.
    Each row = one feature dropped, showing which features matter most.
    Sorted by sort_by descending (worst synthetic quality first).

    Parameters
    ----------
    dataset  : str or None — filter to one dataset
    sort_by  : str         — metric column to rank features by
    """
    sub = df_all[df_all["feature_mode"] == "drop_one"].copy()

    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]

    if sub.empty:
        print("make_table_drop_one: no drop_one-mode rows found.")
        return pd.DataFrame()

    # dropped feature index lives in subset_param or drop_idx
    idx_col = "subset_param" if "subset_param" in sub.columns else "drop_idx"
    sub["dropped_feature_idx"] = pd.to_numeric(sub[idx_col], errors="coerce")

    # Pull the dropped feature name if kept_feature_names is available
    # (kept_feature_names lists what was KEPT, so dropped = the one missing)
    # Easier: use subset_label which encodes "drop_idx_j"
    if "subset_label" in sub.columns:
        sub["dropped_feature"] = sub["subset_label"]
    else:
        sub["dropped_feature"] = sub["dropped_feature_idx"].apply(
            lambda x: f"feature_{int(x)}" if pd.notna(x) else "?"
        )

    keep_cols = (["dataset", "method", "dropped_feature"]
                 + _keep_present(SUMMARY_METRICS, sub))
    sub = sub[[c for c in keep_cols if c in sub.columns]]

    sub = _format_mean_sd(sub, "rf_sep_mean", "rf_sep_sd", "RF Sep (mean ± SD)")
    sub = _format_mean_sd(sub, "disc_f1_mean", "disc_f1_sd", "Disc F1 (mean ± SD)")

    # Sort: worst quality first (highest sep = most distinguishable)
    sort_col = sort_by if sort_by in sub.columns else "RF Sep (mean ± SD)"
    if sort_col in sub.columns and pd.api.types.is_numeric_dtype(sub[sort_col]):
        sub["_m"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
        sub["_d"] = sub["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
        sub = sub.sort_values(["_d", sort_col, "_m"],
                              ascending=[True, False, True]).drop(columns=["_m", "_d"])
    else:
        sub["_m"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
        sub = sub.sort_values(["dropped_feature", "_m"]).drop(columns=["_m"])

    sub = _rename_metrics(sub)
    sub = sub.rename(columns={
        "dataset": "Dataset", "method": "Method",
        "dropped_feature": "Dropped Feature",
    })

    print(f"Table 5 — Drop-one sensitivity ({len(sub)} rows)")
    return sub.reset_index(drop=True)


# ── Export all tables to CSV ──────────────────────────────────────────────────

def save_all_tables(df_all, outdir="../results"):
    """
    Build and save all five tables as CSVs.
    Returns a dict of {table_name: DataFrame}.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table1_full":        make_table_full(df_all),
        "table2_sample_size": make_table_sample_size(df_all),
        "table3_forward":     make_table_forward(df_all),
        "table4_reverse":     make_table_reverse(df_all),
        "table5_drop_one":    make_table_drop_one(df_all),
    }

    for name, tbl in tables.items():
        if tbl is not None and not tbl.empty:
            path = outdir / f"{name}.csv"
            tbl.to_csv(path, index=False)
            print(f"  Saved: {path}")

    return tables
