import random

# Step 1: Generate uniform random samples in [0, 1]
def generate_uniform():
    return random.uniform(0, 1)

samples = [generate_uniform() for _ in range(10)]

# Step 2: Define the function f(x) = x^2
def f(x):
    return x**2

# Step 3: Evaluate f(x) for each sample
function_values = [f(x) for x in samples]

# Step 4: Monte Carlo estimate of the integral
monte_carlo_estimate = sum(function_values) / len(function_values)

# Step 5: Analytical value of the integral
analytical_integral = 1/3  # ∫₀¹ x² dx = 1/3

# Step 6: Display results
print("Random samples:", samples)
print("Function values:", function_values)
print("Monte Carlo estimate:", monte_carlo_estimate)
print("Analytical integral:", analytical_integral)
print("Absolute error:", abs(monte_carlo_estimate - analytical_integral))