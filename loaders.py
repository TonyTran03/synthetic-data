from sklearn.datasets import load_breast_cancer

def load_breast():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    return {
        "dataset": "breast_cancer",
        "category": "clinical_tabular",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }