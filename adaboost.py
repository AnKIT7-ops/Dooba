import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Weather':      ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature':  ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Windy':        ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'True'],
    'EnjoySports':  ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical features and labels
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Features and labels
X = df[['Weather', 'Temperature', 'Windy']].values
y = df['EnjoySports'].values

# Convert labels from {0,1} to {-1,1}
y = np.where(y == 1, 1, -1)

# Initialize weights
N = len(y)
w = np.ones(N) / N

# Number of boosting rounds
T = 10

alphas = []
classifiers = []

# AdaBoost training
for t in range(T):
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X, y, sample_weight=w)

    predictions = stump.predict(X)

    # Weighted error
    error = np.sum(w * (predictions != y)) / np.sum(w)

    # Stop if error is too small or too large
    if error <= 0 or error >= 0.5:
        break

    # Compute alpha
    alpha = np.log((1 - error) / error)

    # Update weights
    w = w * np.exp(alpha * (predictions != y))
    w = w / np.sum(w)

    alphas.append(alpha)
    classifiers.append(stump)

# Final prediction (ensemble)
final_pred = np.zeros(len(y))
for alpha, stump in zip(alphas, classifiers):
    final_pred += alpha * stump.predict(X)

final_pred = np.sign(final_pred)

# Accuracy
accuracy = accuracy_score(y, final_pred)

print("**Output:**")
print(f"Number of weak learners used: {len(classifiers)}")
print(f"Training Accuracy: {accuracy * 100:.2f}%")
print("\nActual Labels:", y)
print("Predicted Labels:", final_pred)
