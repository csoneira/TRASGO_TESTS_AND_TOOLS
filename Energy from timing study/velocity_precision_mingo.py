#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
c = 299792458  # Speed of light in m/s
c_mm_ns = c * 1e-6  # Speed of light in mm/ns

# Parameters
positions = np.array([0, 1*400/3, 2*400/3, 400])  # Positions in mm
position_uncertainty = 5  # mm
time_uncertainty = 0.3  # ns

# Generate times of arrival (true times + random noise)
true_times = positions / c_mm_ns  # Ideal times (in ns)
measured_times = true_times + np.random.normal(0, time_uncertainty, len(true_times))

# Linear fit function
def linear_fit(x, a, b):
    return a * x + b

# Initialize storage for fitted velocities and uncertainties
num_simulations = 1000
results = np.zeros((num_simulations, 2))  # Columns: [velocity, uncertainty]

# Loop to generate data, fit, and store results
for i in range(num_simulations):
    measured_times = true_times + np.random.normal(0, time_uncertainty, len(true_times))
    params, covariance = curve_fit(
        linear_fit, positions, measured_times, sigma=1/weights, absolute_sigma=True
    )
    velocity = 1 / params[0]
    velocity_uncertainty = np.sqrt(covariance[0, 0]) / (params[0] ** 2)
    results[i, :] = [velocity, velocity_uncertainty]

# Extract data for histograms
fitted_velocities = results[:, 0]
uncertainties = results[:, 1]
relative_uncertainties = uncertainties / fitted_velocities

#%%

# Define criteria for removing outliers
velocity_mean = np.mean(fitted_velocities)
velocity_std = np.std(fitted_velocities)

# Filter data within 3 standard deviations for velocities and uncertainties
filtered_indices = np.abs(fitted_velocities - velocity_mean) < 2 * velocity_std
filtered_velocities = fitted_velocities[filtered_indices]
filtered_uncertainties = uncertainties[filtered_indices]
filtered_relative_uncertainties = relative_uncertainties[filtered_indices]

# Create histograms with filtered data
fig, axs = plt.subplots(3, 1, figsize=(8, 12), tight_layout=True)

# Histogram of filtered velocities
axs[0].hist(filtered_velocities, bins='auto', alpha=0.7, color='blue')
axs[0].set_title("Histogram of Filtered Fitted Velocities")
axs[0].set_xlabel("Velocity (mm/ns)")
axs[0].set_ylabel("Frequency")

# Histogram of filtered uncertainties
axs[1].hist(filtered_uncertainties, bins='auto', alpha=0.7, color='green')
axs[1].set_title("Histogram of Filtered Uncertainties")
axs[1].set_xlabel("Uncertainty (mm/ns)")
axs[1].set_ylabel("Frequency")

# Histogram of filtered relative uncertainties
axs[2].hist(filtered_relative_uncertainties*100, bins='auto', alpha=0.7, color='red')
axs[2].set_title("Histogram of Filtered Relative Uncertainties")
axs[2].set_xlabel("Relative Uncertainty / %")
axs[2].set_ylabel("Frequency")

plt.show()


# %%
