# Read an unsupervised dataset and group the dataset based on similarity based on k-meansclustering .
import numpy as np

# Colors
COLOR = "\033[34m"  # Ink blue
RESET = "\033[0m"   # Reset to default color

print(f"{COLOR}1. Initialization{RESET}")

# Number of clusters
k = 3

# Data points
X = np.array([
    [2, 10],
    [2, 5],
    [8, 4],
    [5, 8],
    [7, 5],
    [6, 4],
    [1, 2],
    [4, 9]
])

# Initial cluster centers μj
mu = np.array([
    [2, 10],  # cluster 1
    [5, 8],   # cluster 2
    [1, 2]    # cluster 3
])

print("Initial cluster centres (μj):")
for i, center in enumerate(mu, 1):
    print(f"μ{i}: {center}")

print(f"\n{COLOR}2. Learning (K-means iterations){RESET}")

iteration = 1
while True:
    print(f"\n------------------ Iteration {iteration} ------------------")

    # Step 1: Compute distance of each point to every cluster center
    distances = np.array([[np.linalg.norm(x - m) for m in mu] for x in X])

    # Print distances nicely
    print("\nDistances matrix:")
    for i, d in enumerate(distances):
        formatted = ['{:.2f}'.format(dist) for dist in d]
        print(f"Point {X[i]} -> {formatted}")

    # Step 2: Assign points to nearest cluster
    labels = np.argmin(distances, axis=1)

    # Print cluster assignment
    print("\nCluster assignments:")
    for i, x in enumerate(X):
        print(f"Point {x} → Cluster {labels[i] + 1}")

    # Step 3: Compute new cluster centres
    new_mu = np.array([
        X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else mu[j]
        for j in range(k)
    ])

    # Print clusters with new centers
    for j in range(k):
        points_in_cluster = X[labels == j]
        print(f"\nCluster {j + 1}: {points_in_cluster.tolist()}")
        print(f"New centre μ{j + 1}: [{', '.join(f'{coord:.2f}' for coord in new_mu[j])}]")

    # Step 4: Check for convergence
    if np.allclose(mu, new_mu):
        print(f"\n{COLOR}*** Converged! Final cluster centres: ***{RESET}")
        for i, center in enumerate(new_mu, 1):
            print(f"μ{i}: [{', '.join(f'{coord:.2f}' for coord in center)}]")
        break

    mu = new_mu
    iteration += 1
