import math

def foil_gain(p0, n0, p1, n1):
    if p1 == 0:
        return float('-inf')
    def log_ratio(x, y):
        if x == 0:
            return 0
        return math.log2(x / (x + y))
    return p1 * (log_ratio(p1, n1) - log_ratio(p0, n0))

def foil(dataset, target_attr):
    positives = [row for row in dataset if row[target_attr] == 'Yes']
    negatives = [row for row in dataset if row[target_attr] == 'No']

    rules = []

    while positives:
        best_rule = None
        best_score = float('-inf')

        p0 = len(positives)
        n0 = len(negatives)

        print(f"\nPositives remaining: {p0}, Negatives: {n0}")

        for col in dataset[0].keys():
            if col == target_attr:
                continue
            for val in set(row[col] for row in dataset):
                pos_covered = [p for p in positives if p[col] == val]
                neg_covered = [n for n in negatives if n[col] == val]

                p1 = len(pos_covered)
                n1 = len(neg_covered)

                if p1 == 0:
                    continue

                score = foil_gain(p0, n0, p1, n1)

                print(f"Testing: {col}={val}, p1={p1}, n1={n1}, score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_rule = (col, val, pos_covered)

        if not best_rule:
            print("No valid rule found, stopping.")
            break

        col, val, covered_pos = best_rule
        rules.append(f"If {col} = {val} THEN {target_attr} = Yes")

        positives = [p for p in positives if p not in covered_pos]

    return rules


dataset = [
    {'Outlook': 'Sunny', 'Temp': 'Hot', 'Humidity': 'High', 'Windy': 'False', 'Play': 'No'},
    {'Outlook': 'Sunny', 'Temp': 'Hot', 'Humidity': 'High', 'Windy': 'True', 'Play': 'No'},
    {'Outlook': 'Overcast', 'Temp': 'Hot', 'Humidity': 'High', 'Windy': 'False', 'Play': 'Yes'},
    {'Outlook': 'Rain', 'Temp': 'Mild', 'Humidity': 'High', 'Windy': 'False', 'Play': 'Yes'},
    {'Outlook': 'Rain', 'Temp': 'Cool', 'Humidity': 'Normal', 'Windy': 'False', 'Play': 'Yes'},
    {'Outlook': 'Rain', 'Temp': 'Cool', 'Humidity': 'Normal', 'Windy': 'True', 'Play': 'No'},
    {'Outlook': 'Overcast', 'Temp': 'Mild', 'Humidity': 'Normal', 'Windy': 'True', 'Play': 'Yes'},
    {'Outlook': 'Sunny', 'Temp': 'Mild', 'Humidity': 'High', 'Windy': 'False', 'Play': 'Yes'}
]

rules = foil(dataset, 'Play')
print("\nFOIL Rules learned:")
for r in rules:
    print(r)