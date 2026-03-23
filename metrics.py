import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu



def rf_distinguish_auc(X_real, X_syn, seed=42, n_estimators=300):
    X_all = np.vstack([X_real, X_syn])
    s_all = np.concatenate([
        np.zeros(len(X_real), dtype=int),
        np.ones(len(X_syn), dtype=int)
    ])

    X_train, X_test, s_train, s_test = train_test_split(
        X_all, s_all,
        test_size=0.2,
        random_state=seed,
        stratify=s_all
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed
    )
    rf.fit(X_train, s_train)

    p = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(s_test, p)
    sep = max(auc, 1 - auc)

    return {
        "rf_auc": auc,
        "rf_sep": sep,
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
def evaluate_all(X_real, X_syn, seed=42):
    out = {}

    out.update(rf_distinguish_auc(X_real, X_syn, seed=seed))
    out.update(correlation_diff(X_real, X_syn))
    out.update(per_feature_tests(X_real, X_syn))

    return out