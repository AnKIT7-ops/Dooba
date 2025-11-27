# ------ CANDIDATE ELIMINATION ALGORITHM ------

import pandas as pd

# Load your dataset
df = pd.read_csv("data.csv")

# Assume: last column is target, others are attributes
X = df.iloc[:, :-1].values.tolist()  # attributes as list of lists
y = df.iloc[:, -1].values.tolist()   # labels (Yes/No)

num_attrs = len(X[0])

# Most specific hypothesis S and most general hypothesis G
S = [['Ø'] * num_attrs]
G = [['?'] * num_attrs]

# Get domains of each attribute from data
domains = []
for col in range(num_attrs):
    domains.append(set(row[col] for row in X))

def is_consistent(h, x):
    for hi, xi in zip(h, x):
        if hi != '?' and hi != xi:
            return False
    return True

def more_general_or_equal(h1, h2):
    more_general = False
    for a, b in zip(h1, h2):
        if a == '?' and b != '?':
            more_general = True
        elif a != '?' and a != b:
            return False
    return more_general or h1 == h2

for attrs, label in zip(X, y):
    if label == "Yes":
        # Remove from G those not consistent with positive example
        G = [g for g in G if is_consistent(g, attrs)]

        # Generalize S toward this positive example
        s = S[0][:]
        for i in range(num_attrs):
            if s[i] == 'Ø':
                s[i] = attrs[i]
            elif s[i] != attrs[i]:
                s[i] = '?'
        S = [s]

        # Remove g in G that are more specific than S
        G = [g for g in G if more_general_or_equal(g, S[0])]

    else:  # Negative example ("No")
        # Remove S that incorrectly cover this negative example
        S = [s for s in S if not is_consistent(s, attrs)]

        new_G = []
        for g in G:
            if is_consistent(g, attrs):
                # Specialize g
                for i in range(num_attrs):
                    if g[i] == '?':
                        for val in domains[i]:
                            if val != attrs[i]:
                                new_h = g[:]
                                new_h[i] = val
                                if not S or more_general_or_equal(new_h, S[0]):
                                    new_G.append(new_h)
                    elif g[i] == attrs[i]:
                        for val in domains[i]:
                            if val != attrs[i]:
                                new_h = g[:]
                                new_h[i] = val
                                if not S or more_general_or_equal(new_h, S[0]):
                                    new_G.append(new_h)
            else:
                new_G.append(g)

        # Remove duplicates
        G_clean = []
        for h in new_G:
            if h not in G_clean:
                G_clean.append(h)
        G = G_clean

print("Final S (most specific hypotheses):")
for s in S:
    print(s)

print("\nFinal G (most general hypotheses):")
for g in G:
    print(g)

print("\nVersion space is all hypotheses between S and G.")
