import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# Sample data
data = {
    'Deadline': ['Urgent', 'Urgent', 'Near', 'None', 'None', 'None', 'Near', 'Near', 'Near', 'Urgent'],
    'Party':    ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Lazy':     ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No'],
    'Activity': ['Party', 'Study', 'Party', 'Party', 'Pub', 'Party', 'Study', 'TV', 'Party', 'Study']
}

df = pd.DataFrame(data)

# Label encoding for each column (separately)
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features and target
X = df[['Deadline', 'Party', 'Lazy']].values
y = df['Activity'].values

n_points = len(X)
n_samples = 5  # number of trees in the ensemble

classifiers = []

# Train bagging ensemble of decision stumps
for i in range(n_samples):
    # Bootstrap sample
    sample_indices = np.random.randint(0, n_points, n_points)
    sample_X = X[sample_indices]
    sample_y = y[sample_indices]

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(sample_X, sample_y)
    classifiers.append(tree)


def bagging_predict(X):
    # Predictions from all classifiers: shape (n_samples, n_points)
    predictions = np.array([clf.predict(X) for clf in classifiers])

    # Decode each stump's predictions for display
    decoded_stump_preds = []
    for i in range(n_samples):
        decoded_stump_preds.append(
            le_dict['Activity'].inverse_transform(predictions[i])
        )

    # Majority vote for final prediction
    final_preds = []
    for i in range(len(X)):
        votes = predictions[:, i]  # all classifiers' predictions for point i
        final_label = Counter(votes).most_common(1)[0][0]
        final_preds.append(final_label)

    final_preds_decoded = le_dict['Activity'].inverse_transform(final_preds)

    return decoded_stump_preds, final_preds_decoded


# Get predictions from the bagging ensemble
stump_predictions, final_predictions = bagging_predict(X)

# For metrics we compare encoded labels (y) with encoded bagged predictions
y_encoded = le_dict['Activity'].transform(final_predictions)

accuracy = accuracy_score(y, y_encoded)
precision = precision_score(y, y_encoded, average='macro', zero_division=0)
recall = recall_score(y, y_encoded, average='macro', zero_division=0)
f1 = f1_score(y, y_encoded, average='macro', zero_division=0)

# Decode actual labels for printing
actual_labels = le_dict['Activity'].inverse_transform(y)

print("***** Bagging Ensemble Results *****")
print(f"Number of weak learners: {len(classifiers)}")
print(f"Training Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}\n")

print("Tree Stump Predictions (per weak learner):")
for i, stump in enumerate(stump_predictions, 1):
    print(f"Stump {i}: {list(stump)}")

print("\nCorrect Classes:")
print(list(actual_labels))

print("\nBagged Results:")
print(list(final_predictions))
