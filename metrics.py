# metrics.py
"""
Pure computation — no plotting, no side effects.
All figure generation lives in plots.py.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu, entropy


# ── Helpers ───────────────────────────────────────────────────────────────────

def strat_samp(idx0, idx1, n0, n1, rng):
    a = rng.choice(idx0, size=n0, replace=False)
    b = rng.choice(idx1, size=n1, replace=False)
    return np.concatenate([a, b])


def stratified_subsample(X, y, n0, n1, seed=42):
    """
    Draw n0 class-0 and n1 class-1 rows from X without replacement.
    Clamps to available count if request exceeds dataset size.
    """
    rng  = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    if n0 > len(idx0):
        print(f"  [subsample] n0={n0} > available {len(idx0)} — clamping")
        n0 = len(idx0)
    if n1 > len(idx1):
        print(f"  [subsample] n1={n1} > available {len(idx1)} — clamping")
        n1 = len(idx1)

    idx = strat_samp(idx0, idx1, n0, n1, rng)
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def print_dataset_summary(load_fns):
    """Print class counts for each dataset."""
    print(f"{'Dataset':<20} {'p':>5} {'Class 0':>10} {'Class 1':>10} {'Total':>8}")
    print("-" * 56)
    for fn in load_fns:
        d = fn()
        y = d["y"]
        print(f"{d['dataset']:<20} {d['X'].shape[1]:>5} "
              f"{int((y==0).sum()):>10} {int((y==1).sum()):>10} {len(y):>8}")


# ── RF discriminator ──────────────────────────────────────────────────────────

def one_stochastic_experiment(
    X_real, y_real, X_syn, y_syn,
    holdout_neg=3, holdout_pos=12,
    train_neg=20,  train_pos=20,
    seed=42, n_estimators=5,
):
    rng      = np.random.default_rng(seed)
    real_neg = np.where(y_real == 0)[0]
    real_pos = np.where(y_real == 1)[0]
    syn_neg  = np.where(y_syn  == 0)[0]
    syn_pos  = np.where(y_syn  == 1)[0]

    holdout_neg = max(1, min(holdout_neg, len(real_neg) // 2, len(syn_neg) // 2))
    holdout_pos = max(1, min(holdout_pos, len(real_pos) // 2, len(syn_pos) // 2))

    test_real_idx = strat_samp(real_neg, real_pos, holdout_neg, holdout_pos, rng)
    test_syn_idx  = strat_samp(syn_neg,  syn_pos,  holdout_neg, holdout_pos, rng)

    rem_real     = np.setdiff1d(np.arange(X_real.shape[0]), test_real_idx)
    rem_syn      = np.setdiff1d(np.arange(X_syn.shape[0]),  test_syn_idx)
    rem_real_neg = rem_real[y_real[rem_real] == 0]
    rem_real_pos = rem_real[y_real[rem_real] == 1]
    rem_syn_neg  = rem_syn[y_syn[rem_syn]   == 0]
    rem_syn_pos  = rem_syn[y_syn[rem_syn]   == 1]

    train_neg = max(1, min(train_neg, len(rem_real_neg), len(rem_syn_neg)))
    train_pos = max(1, min(train_pos, len(rem_real_pos), len(rem_syn_pos)))

    train_real_idx = strat_samp(rem_real_neg, rem_real_pos, train_neg, train_pos, rng)
    train_syn_idx  = strat_samp(rem_syn_neg,  rem_syn_pos,  train_neg, train_pos, rng)

    X_train = np.vstack([X_real[train_real_idx], X_syn[train_syn_idx]])
    s_train = np.concatenate([
        np.zeros(len(train_real_idx)), np.ones(len(train_syn_idx))
    ]).astype(int)

    X_test  = np.vstack([X_real[test_real_idx], X_syn[test_syn_idx]])
    s_test  = np.concatenate([
        np.zeros(len(test_real_idx)), np.ones(len(test_syn_idx))
    ]).astype(int)

    rf    = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train, s_train)
    probs = rf.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(s_test, probs)

    return {
        "rf_auc_raw": auc,
        "rf_auc_sep": max(auc, 1 - auc),
        "disc_f1":    f1_score(s_test, rf.predict(X_test), average="binary", zero_division=0),
    }


def run_many_rf_trials(X_real, y_real, X_syn, y_syn, trials=10):
    aucs, seps, f1s = [], [], []
    for t in range(trials):
        if t == 0 or (t + 1) % 10 == 0 or t == trials - 1:
            print(f"  RF trial {t+1}/{trials}")
        out = one_stochastic_experiment(
            X_real, y_real, X_syn, y_syn,
            holdout_neg=3, holdout_pos=12,
            train_neg=20,  train_pos=20,
            seed=t, n_estimators=5,
        )
        aucs.append(out["rf_auc_raw"])
        seps.append(out["rf_auc_sep"])
        f1s.append(out["disc_f1"])

    return {
        "rf_auc_mean":  np.mean(aucs),
        "rf_auc_sd":    np.std(aucs),
        "rf_sep_mean":  np.mean(seps),
        "rf_sep_sd":    np.std(seps),
        "disc_f1_mean": np.mean(f1s),
        "disc_f1_sd":   np.std(f1s),
    }


# ── TSTR utility ──────────────────────────────────────────────────────────────

def tstr_f1(X_real, y_real, X_syn, y_syn, seed=42, n_estimators=100):
    """Train on synthetic, test on real. TRTR baseline via stratified CV."""
    rf_syn = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf_syn.fit(X_syn, y_syn)
    tstr = f1_score(y_real, rf_syn.predict(X_real), average="binary", zero_division=0)

    n_splits = max(2, min(5, int((y_real == 0).sum()), int((y_real == 1).sum())))
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trtr_scores = []
    for train_idx, test_idx in skf.split(X_real, y_real):
        rf_real = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        rf_real.fit(X_real[train_idx], y_real[train_idx])
        trtr_scores.append(
            f1_score(y_real[test_idx], rf_real.predict(X_real[test_idx]),
                     average="binary", zero_division=0)
        )

    trtr = np.mean(trtr_scores)
    return {"tstr_f1": tstr, "trtr_f1": trtr, "utility_gap": trtr - tstr}


# ── Structural fidelity ───────────────────────────────────────────────────────

def correlation_diff(X_real, X_syn):
    diff = np.abs(np.corrcoef(X_real, rowvar=False) - np.corrcoef(X_syn, rowvar=False))
    return {
        "corr_mean_abs_diff": float(diff.mean()),
        "corr_max_abs_diff":  float(diff.max()),
    }


def kld_per_feature(X_real, X_syn, bins=30):
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
    klds = np.array(klds)
    return {
        "kld_mean":        float(klds.mean()),
        "kld_max":         float(klds.max()),
        "kld_per_feature": klds,
    }


def per_feature_tests(X_real, X_syn, alpha=0.05):
    pvals = np.array([
        mannwhitneyu(X_real[:, j], X_syn[:, j], alternative="two-sided").pvalue
        for j in range(X_real.shape[1])
    ])
    return {
        "pval_mean":        float(pvals.mean()),
        "pval_median":      float(np.median(pvals)),
        "prop_significant": float((pvals < alpha).mean()),
    }


# ── Realism (nearest-neighbour distances) ────────────────────────────────────

def nn_distances(X_real, X_syn, k=1):
    """
    Nearest-neighbour diagnostics for realism.

    For each synthetic sample, find its distance to the nearest real sample.
    Returns summary statistics of those distances plus the raw array.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_real)
    dists, _ = nn.kneighbors(X_syn)
    dists = dists[:, -1]  # k-th neighbour distance

    # Also compute real-to-real baseline for comparison
    nn_real = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_real.fit(X_real)
    dists_real, _ = nn_real.kneighbors(X_real)
    dists_real = dists_real[:, -1]  # exclude self (index 0)

    return {
        "nn_dist_mean":       float(dists.mean()),
        "nn_dist_median":     float(np.median(dists)),
        "nn_dist_std":        float(dists.std()),
        "nn_dist_max":        float(dists.max()),
        "nn_real_mean":       float(dists_real.mean()),
        "nn_real_median":     float(np.median(dists_real)),
        "nn_ratio_mean":      float(dists.mean() / dists_real.mean()) if dists_real.mean() > 0 else np.nan,
        "nn_dists_syn":       dists,
        "nn_dists_real":      dists_real,
    }


# ── Master evaluation ─────────────────────────────────────────────────────────

def evaluate_all(X_real, y_real, X_syn, y_syn):
    """
    Compute all metrics. Returns (metrics_dict, kld_per_feature_array).
    No figures are created — plotting is the caller's responsibility.
    """
    metrics = {}
    metrics.update(run_many_rf_trials(X_real, y_real, X_syn, y_syn))
    metrics.update(tstr_f1(X_real, y_real, X_syn, y_syn))
    metrics.update(correlation_diff(X_real, X_syn))
    metrics.update(per_feature_tests(X_real, X_syn))

    kld_result = kld_per_feature(X_real, X_syn)
    metrics["kld_mean"] = kld_result["kld_mean"]
    metrics["kld_max"]  = kld_result["kld_max"]

    nn_result = nn_distances(X_real, X_syn)
    metrics["nn_dist_mean"]  = nn_result["nn_dist_mean"]
    metrics["nn_dist_median"] = nn_result["nn_dist_median"]
    metrics["nn_ratio_mean"] = nn_result["nn_ratio_mean"]

    return metrics, kld_result["kld_per_feature"]