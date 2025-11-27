# Read a dataset and perform unsupervised learning using SOM algorithm.
# Step 1: Import Libraries
BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

print(f"{BLUE}Step 1: Libraries imported successfully.{RESET}")

# Step 2: Load and Normalize Dataset
iris = load_iris()
X = iris.data  # All 4 features

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"{GREEN}Step 2: Dataset loaded and normalized.{RESET}")

# Step 3: Initialize SOM Parameters
grid_rows, grid_cols = 5, 5
input_dim = X_scaled.shape[1]
num_neurons = grid_rows * grid_cols

# Randomly initialize weights for each neuron
weights = np.random.rand(num_neurons, input_dim)

def learning_rate(t, max_iter):
    return 0.5 * (1 - t / max_iter)

print(f"{BLUE}Step 3: SOM initialized with {num_neurons} neurons.{RESET}")

# Step 4: Train SOM
max_iter = 100

for t in range(max_iter):
    eta = learning_rate(t, max_iter)
    for x in X_scaled:
        # Compute distance of x to all neuron weight vectors
        distances = np.linalg.norm(weights - x, axis=1)
        # Best Matching Unit (BMU)
        bmu_index = np.argmin(distances)
        # Update BMU weights
        weights[bmu_index] += eta * (x - weights[bmu_index])

print(f"{GREEN}Step 4: SOM training completed over {max_iter} iterations.{RESET}")

# Step 5: Assign Clusters (BMU index for each sample)
assignments = []
for x in X_scaled:
    distances = np.linalg.norm(weights - x, axis=1)
    bmu_index = np.argmin(distances)
    assignments.append(bmu_index)

df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = assignments

print(f"{BLUE}Step 5: Cluster assignments completed.{RESET}")
print(df.head())

# Step 6: PCA-Based Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = assignments

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'],
                      c=df_pca['Cluster'], cmap='tab10', s=50)
plt.title("SOM Clustering on PCA-Reduced Iris Data", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

print(f"{GREEN}Step 6: PCA-based visualization complete.{RESET}")
