from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
import numpy as np


def ensure_binary_target(y):
    y = np.asarray(y)

    # already integer/bool
    if y.dtype.kind in {"i", "b"}:
        return y.astype(int)

    # convert string / object labels to 0/1
    unique = np.unique(y)

    if len(unique) != 2:
        raise ValueError(f"Expected binary target, got {unique}")

    return (y == unique[1]).astype(int)

def load_breast():
    data = load_breast_cancer()

    X = data.data
    y = ensure_binary_target(data.target)

    feature_names = list(data.feature_names)

    return {
        "dataset": "breast_cancer",
        "category": "clinical_tabular",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }

def load_diabetes():
    data = fetch_openml("diabetes", version=1, as_frame=False)

    X = data.data
    y = ensure_binary_target(data.target)

    feature_names = list(data.feature_names)

    return {
        "dataset": "diabetes",
        "category": "metabolic_tabular",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }