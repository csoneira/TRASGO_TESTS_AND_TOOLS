import numpy as np
from scipy.optimize import least_squares

# Gaussian function definition
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Define the residual function to minimize
def residuals(vars, x1, x2, h1, h2, A):
    mu, sigma = vars
    res1 = h1 - gaussian(x1, A, mu, sigma)
    res2 = h2 - gaussian(x2, A, mu, sigma)
    return [res1, res2]

# Given parameters
x1 = 6.3   # First x-point
x2 = 12.6   # Second x-point
h1 = 0.6   # Height at x1
h2 = 0.3   # Height at x2
A = 1.0    # Maximum height of the Gaussian
sigma_approx = 2 # Known approximate value of sigma

# Initial guess for mu and sigma
initial_guess = [0.5 * (x1 + x2), sigma_approx]

# Solve for mu and sigma, allowing some variation around the known sigma
result = least_squares(residuals, initial_guess, args=(x1, x2, h1, h2, A))

mu, sigma = result.x

# Output the results
print(f"Fitted parameters:")
print(f"mu (mean)    = {mu}")
print(f"sigma (std)  = {sigma}")

# Example usage of the fitted Gaussian
x_values = np.linspace(x1 - 1, x2 + 1, 100)
y_values = gaussian(x_values, A, mu, sigma)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(x_values, y_values, label="Fitted Gaussian")
plt.scatter([x1, x2], [h1, h2], color='red', label="Given Points")
plt.axhline(y=A, color='green', linestyle='--', label="Maximum A")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
