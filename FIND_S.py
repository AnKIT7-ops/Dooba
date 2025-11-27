# ---------- FIND-S ALGORITHM ----------

import pandas as pd

# Change this to your file name
df = pd.read_csv("data.csv")

# Assume: last column is target, others are attributes
X = df.iloc[:, :-1].values  # attributes
y = df.iloc[:, -1].values   # class label (Yes/No)

# Initialize most specific hypothesis
hypothesis = ['Ø'] * X.shape[1]   # 'Ø' means no value yet

for attrs, label in zip(X, y):
    if label == "Yes":  # Only positive examples
        for i in range(len(attrs)):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = attrs[i]
            elif hypothesis[i] != attrs[i]:
                hypothesis[i] = '?'  # generalize

print("Final Find-S hypothesis:")
print(hypothesis)
