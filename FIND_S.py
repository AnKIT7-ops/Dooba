data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'No'],
    ['Rain', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Change', 'Yes'],]
def find_s(data):
    n_attributes = len(data[0]) - 1
    hypothesis = ['ϕ'] * n_attributes
    for idx, row in enumerate(data):
        attributes, label = row[:-1], row[-1]
        if label == 'Yes':
            print(f"Instance {idx+1} (positive): {attributes}")
            for i in range(n_attributes):
                if hypothesis[i] == 'ϕ':
                    hypothesis[i] = attributes[i]
                elif hypothesis[i] != attributes[i]:
                    hypothesis[i] = '?'
            print(f"Hypothesis after instance {idx+1}: {hypothesis}\n")
    print("Final hypothesis:", hypothesis)
    return hypothesis
find_s(data)
