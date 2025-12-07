import numpy as np

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'No'],
    ['Rain', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Change', 'Yes'],
]

attributes = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast']


def more_general(h1, h2):
    return all(x == '?' or (x != '∅' and x == y) for x, y in zip(h1, h2))


def candidate_elimination(examples):
    S = ['∅'] * len(attributes)
    G = [['?'] * len(attributes)]

    for example in examples:
        x, target = example[:-1], example[-1].lower()

        # Positive Example
        if target == 'yes':
            # Update S
            for i in range(len(S)):
                if S[i] == '∅':
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = '?'

            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(len(S)))]

        # Negative Example
        else:
            G_new = []
            for g in G:
                # If g wrongly classifies x as positive → specialize it
                if all(g[i] == '?' or g[i] == x[i] for i in range(len(g))):
                    # Specialize
                    for i in range(len(g)):
                        if g[i] == '?':
                            # all possible domain values for this attribute except x[i]
                            values = set(ex[i] for ex in examples)
                            for val in values:
                                if val != x[i]:
                                    g_new = g.copy()
                                    g_new[i] = val
                                    if not any(more_general(h, g_new) for h in G_new):
                                        G_new.append(g_new)
                else:
                    G_new.append(g)

            G = G_new

    return S, G


S, G = candidate_elimination(data)

print("Final Specific Hypothesis:", S)
print("Final General Hypotheses:", G)
