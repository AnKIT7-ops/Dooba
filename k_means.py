import numpy as np
def kmeans(X, k, max_iter=100, tol=1e-4, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centers = X[init_idx].astype(float)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = X[labels == j]
            new_centers[j] = members.mean(axis=0) if len(members) > 0 else X[rng.integers(0, n_samples)]
        if np.allclose(centers, new_centers, atol=tol):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels
X = np.array([
    [2, 10], [2, 5], [8, 4], [5, 8],
    [7, 5], [6, 4], [1, 2], [4, 9]
])
centers, labels = kmeans(X, k=3, random_state=0)
print("Centers:\n", centers)
print("Labels:\n", labels)
