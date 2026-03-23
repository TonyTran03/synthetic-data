import numpy as np
from sklearn.mixture import GaussianMixture


def sample_gmm(X, y, n0, n1, seed=42, n_components=2):
    X0 = X[y == 0]
    X1 = X[y == 1]

    gmm0 = GaussianMixture(n_components=n_components, random_state=seed)
    gmm1 = GaussianMixture(n_components=n_components, random_state=seed)

    gmm0.fit(X0)
    gmm1.fit(X1)

    X_syn0, _ = gmm0.sample(n0)
    X_syn1, _ = gmm1.sample(n1)

    X_syn = np.vstack([X_syn0, X_syn1]).astype(np.float32)
    y_syn = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int),
    ])

    return X_syn, y_syn