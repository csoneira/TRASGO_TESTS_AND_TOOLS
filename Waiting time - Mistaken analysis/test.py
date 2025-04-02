import numpy as np
import matplotlib.pyplot as plt

# Poisson rate (events per second)
rate = 0.01  # events per second

# Generate a Poisson-distributed number of events in one second for 1000 seconds
events_per_second = np.random.poisson(rate, 10000)

# Calculate waiting times (time between events)
# The waiting time for a Poisson process is exponentially distributed with mean 1/rate
waiting_times = np.random.exponential(1/rate, size=10000)

# Plot histogram of waiting times
plt.hist(waiting_times, bins=1000, edgecolor='black', alpha=0.7)
plt.title("Histogram of Waiting Times")
plt.xlabel("Waiting Time (s)")
plt.ylabel("Frequency")
plt.xscale('log')
plt.yscale('log')
plt.show()
