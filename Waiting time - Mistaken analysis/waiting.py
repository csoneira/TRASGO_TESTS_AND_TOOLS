import numpy as np
import scipy.io
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.stats import nbinom

# Set the maximum number of particles for waiting time calculations
max_particles = 10  # Change this to any number of particles you want

# Load the .mat file
filename = 'mi0324311141342.mat'
data = scipy.io.loadmat(filename)

# List of 'T' and 'Q' variables of interest for signal selection
T_vars = ['T1_F', 'T1_B', 'T2_F', 'T2_B', 'T3_F', 'T3_B', 'T4_F', 'T4_B']
Q_vars = ['Q1_F', 'Q1_B', 'Q2_F', 'Q2_B', 'Q3_F', 'Q3_B', 'Q4_F', 'Q4_B']

# Initialize an array to store all event times (across detectors)
event_times = []

# Loop over each detector and add valid event times directly to the array
for detector_idx in range(4):
    T_front = data[T_vars[detector_idx * 2]]
    T_back = data[T_vars[detector_idx * 2 + 1]]
    Q_front = data[Q_vars[detector_idx * 2]]
    Q_back = data[Q_vars[detector_idx * 2 + 1]]
    
    # Convert T and Q matrices to dense arrays if they are sparse matrices
    if scipy.sparse.issparse(T_front):
        T_front = T_front.toarray()
    if scipy.sparse.issparse(T_back):
        T_back = T_back.toarray()
    if scipy.sparse.issparse(Q_front):
        Q_front = Q_front.toarray()
    if scipy.sparse.issparse(Q_back):
        Q_back = Q_back.toarray()

    for i in range(T_front.shape[0]):
        # Check if there's a signal in both front and back
        if np.sum((Q_front[i, :] != 0) & (Q_back[i, :] != 0)) == 1:
            # Collect valid high values for accurate waiting times
            high_values = np.concatenate((T_front[i, T_front[i, :] > 1e4], T_back[i, T_back[i, :] > 1e4]))
            event_times.extend(high_values)

# Sort event times
event_times = np.sort(event_times)

# Initialize lists to store waiting times, labels, and counts for each particle count
waiting_times_list = []
labels = []
bulk_counts = []

# Calculate waiting times for the next `k` particles where `k` ranges from 1 to max_particles
for k in range(1, max_particles + 1):
    waiting_times = (event_times[k:] - event_times[:-k]) / 1000  # Convert to microseconds
    waiting_times_list.append(waiting_times)
    labels.append(f'{k} adjacent')

    bulk_count = np.sum(waiting_times > 0.5)
    bulk_counts.append(bulk_count)


# Define bins for waiting time histograms (exponential-like distribution)
waiting_time_bins = np.logspace(np.log10(np.min(waiting_times_list[0][waiting_times_list[0] > 0])), 
                                np.log10(np.max(waiting_times_list[0])), 50)


# Plot all waiting time histograms ---------------------------------------------------------------------------
plt.figure(figsize=(10, 7))

n = len(waiting_times_list)  # Number of datasets/plots
# Generate n colors from the 'turbo' colormap
colors = plt.cm.turbo(np.linspace(0, 1, n + 2)[1:-1])
for i, waiting_times in enumerate(waiting_times_list):
    waiting_counts, _ = np.histogram(waiting_times, bins=waiting_time_bins)
    waiting_counts = waiting_counts / np.diff(waiting_time_bins)  # Normalize counts by bin width
    plt.plot(waiting_time_bins[:-1], waiting_counts, '.-', color=colors[i], label=labels[i])

# -----------------------------------------------------------------------------------------------
# def exponential(x, lambd):
#     return lambd * np.exp(-lambd * x)

# # Select waiting times for i=0
# waiting_times = waiting_times_list[0]

# # Split waiting times into left and right subsets around 0.5
# waiting_times_left = waiting_times[waiting_times < 0.5]
# waiting_times_right = waiting_times[waiting_times >= 0.5]

# # Create histograms
# waiting_counts, _ = np.histogram(waiting_times, bins=waiting_time_bins)
# waiting_counts = waiting_counts / np.diff(waiting_time_bins)  # Normalize counts by bin width

# # Fit exponential distributions to the left and right subsets
# # Fit for waiting_times_left
# popt_left, _ = curve_fit(exponential, waiting_times_left, np.ones_like(waiting_times_left) / len(waiting_times_left), maxfev=100000)
# lambda_left = popt_left[0]

# # Fit for waiting_times_right
# popt_right, _ = curve_fit(exponential, waiting_times_right, np.ones_like(waiting_times_right) / len(waiting_times_right), maxfev=100000)
# lambda_right = popt_right[0]

# # Calculate fitted values for plotting
# x_fit_left = np.linspace(min(waiting_times), 0.5, 100)
# y_fit_left = exponential(x_fit_left, lambda_left)
# x_fit_right = np.linspace(0.5, max(waiting_times), 100)
# y_fit_right = exponential(x_fit_right, lambda_right)

# # Plot the fitted exponential distributions
# lambd_l = 50000000
# lambd_r = 5000000
# test_domain = np.linspace(min(waiting_times), max(waiting_times), 100)

# plt.plot(test_domain, exponential(test_domain, lambd_l), 'r-', label=f'λ={lambda_left:.2f}')
# plt.plot(test_domain, exponential(test_domain, lambd_r), 'r-', label=f'λ={lambda_left:.2f}')

# plt.plot(x_fit_left, y_fit_left, 'r-', label=f'Fit left (λ={lambda_left:.2f})')
# plt.plot(x_fit_right, y_fit_right, 'r--', label=f'Fit right (λ={lambda_right:.2f})')
# -----------------------------------------------------------------------------------------------


# Add a horizontal line at 10 counts per bin for reference
plt.plot(waiting_time_bins[:-1], 10 / waiting_time_bins[:-1], color='black', alpha=0.5, linestyle='--', linewidth=1, label='10 Counts/Bin')
plt.plot(waiting_time_bins[:-1], 1 / waiting_time_bins[:-1], color='green', alpha=0.5, linestyle='--', linewidth=1, label='1 Count/Bin')

# Add the blue semitransparent region using axvspan
plt.axvspan(0, 0.5, color='blue', alpha=0.3, label="Muon Bulk Sim. Bound.")

# Add the vertical dashed line at 0.5 microseconds
plt.axvline(0.5, color='blue', linestyle='--', alpha=0.7)
plt.text(0.5 + 0.5*0.2, plt.ylim()[1]*0.9, '0.5 µs', color='blue', ha='left', alpha=0.7)

uncertainty = 3*10**(-4)
plt.axvline(uncertainty, color='red', linestyle='--', alpha=0.7)
plt.text(uncertainty + uncertainty*0.2, plt.ylim()[1]*0.9, 'Time uncertainty\nthreshold (300 ps)', color='red', ha='left', alpha=0.7)


dead_time = 3*10**(0)
plt.axvline(dead_time, color='red', linestyle='--', alpha=0.7)
plt.text(dead_time + dead_time*0.2, plt.ylim()[1]*0.9, 'Dead time\nthreshold (3 µs)', color='red', ha='left', alpha=0.7)


coincidence_window = 2*10**(-1)
plt.axvline(coincidence_window, color='green', linestyle='--', alpha=0.7)
plt.text(coincidence_window + coincidence_window*0.2, plt.ylim()[1]*0.1, 'Coincidence window\nthreshold (200 ns)', color='green', ha='left', alpha=0.7)


# Set log scale and labels
plt.xscale('log')
plt.yscale('log')
# plt.xlim([np.min(waiting_times_list[0][waiting_times_list[0] > 0]), 1e6])
plt.ylim([1e-4, None])
plt.title(f'Waiting Time Histograms for 1 to {max_particles} adjacent events')
plt.xlabel('Waiting Time (µs)')
plt.ylabel('Frequency Density (Counts / Bin Width)')
plt.xlim([np.min(waiting_times_list[0][waiting_times_list[0] > 0]), 1e6])
plt.legend()

plt.tight_layout()
plt.savefig('waiting.png', format='png')
plt.show()


# Plot the histogram of bulk particle counts -----------------------------------------------------------------

plt.figure(figsize=(7, 6))

print(bulk_counts)

# Compute total counts and normalize to percentages
bulk_counts = np.diff(np.concatenate(([0], bulk_counts)))

print(bulk_counts)

total_count = np.sum(bulk_counts)
bulk_percentages = (bulk_counts / total_count)
max_particles = len(bulk_counts)

# Prepare the data for fitting
x_data = np.arange(1, max_particles + 1)
y_data = bulk_percentages  # Using normalized data for fitting

# Define the exponential with power-law function
def power_exponential(x, a, b, c):
    return a * x**b * np.exp(-c * x)

# Initial guesses for parameters [a, b, c]
initial_guesses = [bulk_percentages[0], -1, 0.1]

# Fit the model
params, covariance = curve_fit(power_exponential, x_data, y_data, p0=initial_guesses, maxfev=10000)

# Calculate fitted values
fitted_values = power_exponential(x_data, *params)
equation_label = r'$y = {:.2f} \times x^{{{:.2f}}} \times e^{{-{:.2f} x}}$'.format(params[0], params[1], params[2])

# Plot the data and fitted curve
plt.bar(x_data, y_data, color='blue', alpha = 0.3, edgecolor='None', label='Normalized Data')
plt.plot(x_data, fitted_values, color='red', label=equation_label, linewidth=2)
# Uncomment the next line if you want a logarithmic y-axis
# plt.yscale('log')

plt.xlabel('Number of Particles in the EAS bulk')
plt.ylabel('Frequency (Normalized)')
plt.title('Number of particles inside of the 0.5 $\mu$s waiting time window\n(fitted to Exponential with Power-Law)')
plt.legend()
plt.xticks(x_data)
plt.tight_layout()
plt.savefig('particle_count.png', format='png')
plt.show()

# Display fitted parameters
print(f"Fitted parameters:")
print(f"a = {params[0]:.2f}")
print(f"b = {params[1]:.4f}")
print(f"c = {params[2]:.4f}")

