import numpy as np
import scipy.io
import scipy.sparse
import matplotlib.pyplot as plt

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

# Sort event times and calculate waiting times between consecutive events
event_times = np.sort(event_times)
waiting_times = np.diff(event_times) / 1000  # Convert to microseconds

# Define bins for waiting time histogram (exponential-like distribution)
waiting_time_bins = np.logspace(np.log10(np.min(waiting_times[waiting_times > 0])), 
                                np.log10(np.max(waiting_times)), 50)
waiting_counts, _ = np.histogram(waiting_times, bins=waiting_time_bins)

# Calculate the bin widths for normalization
waiting_bin_widths = np.diff(waiting_time_bins)
waiting_counts = waiting_counts / waiting_bin_widths  # Normalize counts by bin width

# Calculate event rates in 30-second intervals (Poisson-like distribution)
time_interval = 30 * 1e6  # 30 seconds in microseconds
start_time = np.min(event_times)
end_time = np.max(event_times)
num_intervals = int(np.ceil((end_time - start_time) / time_interval))

# Initialize an array to store event rates per interval
rates = np.zeros(num_intervals)
for i in range(num_intervals):
    t_min = start_time + i * time_interval
    t_max = t_min + time_interval
    rates[i] = np.sum((event_times >= t_min) & (event_times < t_max))

# Define bins for rate histogram (frequency of rates)
rate_bins = np.arange(1, np.max(rates) + 1)  # Start from 1 to exclude zero values
rate_counts, _ = np.histogram(rates, bins=rate_bins)

# Plot both histograms in a single figure with two subplots
plt.figure(figsize=(10, 14))

# Plot waiting time histogram with counts normalized by bin width, as a line plot
plt.subplot(2, 1, 1)
plt.plot(waiting_time_bins[:-1], waiting_counts, 'o-', label='Counts/Bin Width')
plt.xscale('log')
plt.yscale('log')
plt.title('Waiting Time Histogram (Normalized)')
plt.xlabel('Waiting Time (µs)')
plt.axvspan(0, 0.5, color='blue', alpha=0.2, label="Muon Bulk Simulated Boundaries")
plt.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='0.5 µs')

plt.ylabel('Frequency Density (Counts / Bin Width)')
plt.xlim([np.min(waiting_times[waiting_times > 0]), 2e7])
# plt.xlim([0, 2e7])
plt.legend()

# Plot event rate histogram with lines joining the points, and exclude zero values
plt.subplot(2, 1, 2)
plt.plot(rate_bins[:-1], rate_counts, 'o-', label='Event Rate Frequency')
plt.yscale('log')
plt.title('Event Rate Histogram (30-second Intervals)')
plt.xlabel('Events per Interval')
plt.ylabel('Frequency')
plt.xticks(rate_bins[:-1])
plt.xlim([1, np.max(rate_bins)])

plt.tight_layout()
plt.show()
