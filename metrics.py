import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def rf_distinguish_auc(X_real, X_syn, seed=0):
    X_all = np.vstack([X_real, X_syn])
    s_all = np.concatenate([
        np.zeros(len(X_real), dtype=int),
        np.ones(len(X_syn), dtype=int)
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X_all, s_all, test_size=0.2, random_state=seed, stratify=s_all
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    return auc