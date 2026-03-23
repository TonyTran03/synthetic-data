import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu


def strat_samp(idx0, idx1, n0, n1, rng):
    a = rng.choice(idx0, size=n0, replace=False)
    b = rng.choice(idx1, size=n1, replace=False)
    return np.concatenate([a, b])

def stratified_subsample(X, y, n0, n1, seed=42):
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    idx = strat_samp(idx0, idx1, n0, n1, rng)
    rng.shuffle(idx)

    return X[idx], y[idx], idx



def one_stochastic_experiment(
    X_real, y_real_hiv,
    X_syn,  y_syn_hiv,
    holdout_neg=3, holdout_pos=12,
    train_neg=20,  train_pos=20,
    seed=42,
    n_estimators=200,
):
    rng = np.random.default_rng(seed)

    real_neg = np.where(y_real_hiv == 0)[0]
    real_pos = np.where(y_real_hiv == 1)[0]
    syn_neg  = np.where(y_syn_hiv  == 0)[0]
    syn_pos  = np.where(y_syn_hiv  == 1)[0]

    test_real_idx = strat_samp(real_neg, real_pos, holdout_neg, holdout_pos, rng)
    test_syn_idx  = strat_samp(syn_neg,  syn_pos,  holdout_neg, holdout_pos, rng)

    rem_real = np.setdiff1d(np.arange(X_real.shape[0]), test_real_idx)
    rem_syn  = np.setdiff1d(np.arange(X_syn.shape[0]),  test_syn_idx)

    rem_real_neg = rem_real[y_real_hiv[rem_real] == 0]
    rem_real_pos = rem_real[y_real_hiv[rem_real] == 1]
    rem_syn_neg  = rem_syn[y_syn_hiv[rem_syn] == 0]
    rem_syn_pos  = rem_syn[y_syn_hiv[rem_syn] == 1]

    train_real_idx = strat_samp(rem_real_neg, rem_real_pos, train_neg, train_pos, rng)
    train_syn_idx  = strat_samp(rem_syn_neg,  rem_syn_pos,  train_neg, train_pos, rng)

    X_train = np.vstack([X_real[train_real_idx], X_syn[train_syn_idx]])
    s_train = np.concatenate([
        np.zeros(len(train_real_idx), dtype=int),
        np.ones(len(train_syn_idx), dtype=int)
    ])

    X_test = np.vstack([X_real[test_real_idx], X_syn[test_syn_idx]])
    s_test = np.concatenate([
        np.zeros(len(test_real_idx), dtype=int),
        np.ones(len(test_syn_idx), dtype=int)
    ])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed
    )
    rf.fit(X_train, s_train)

    p_syn = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(s_test, p_syn)
    sep = max(auc, 1 - auc)

    return {
        "rf_auc_raw": auc,
        "rf_auc_sep": sep,
    }

def run_many_rf_trials(X_real, y_real, X_syn, y_syn, trials=200):
    aucs = []
    seps = []

    for t in range(trials):
        if t == 0 or (t + 1) % 10 == 0 or t == trials - 1:
            print(f"RF trial {t+1}/{trials}")        
        out = one_stochastic_experiment(
            X_real, y_real,
            X_syn, y_syn,
            holdout_neg=3, holdout_pos=12,
            train_neg=20, train_pos=20,
            seed=t,
            n_estimators=200,
        )
        aucs.append(out["rf_auc_raw"])
        seps.append(out["rf_auc_sep"])

    aucs = np.array(aucs)
    seps = np.array(seps)

    return {
        "rf_auc_mean": aucs.mean(),
        "rf_auc_sd": aucs.std(),
        "rf_sep_mean": seps.mean(),
        "rf_sep_sd": seps.std(),
    }

def correlation_diff(X_real, X_syn):
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn = np.corrcoef(X_syn, rowvar=False)

    diff = np.abs(corr_real - corr_syn)

    return {
        "corr_mean_abs_diff": diff.mean(),
        "corr_max_abs_diff": diff.max(),
    }


def per_feature_tests(X_real, X_syn, alpha=0.05):
    pvals = []

    for j in range(X_real.shape[1]):
        _, p = mannwhitneyu(
            X_real[:, j],
            X_syn[:, j],
            alternative="two-sided"
        )
        pvals.append(p)

    pvals = np.array(pvals)

    return {
        "pval_mean": pvals.mean(),
        "pval_median": np.median(pvals),
        "prop_significant": (pvals < alpha).mean(),
    }


### ALL
def evaluate_all(X_real, y_real, X_syn, y_syn):
    out = {}

    out.update(run_many_rf_trials(X_real, y_real, X_syn, y_syn))
    out.update(correlation_diff(X_real, X_syn))
    out.update(per_feature_tests(X_real, X_syn))

    return out