% Load the .mat file
filename = 'mi0324311141342.mat';
data = load(filename);

% List of 'T' and 'Q' variables of interest for signal selection
T_vars = {'T1_F', 'T1_B', 'T2_F', 'T2_B', 'T3_F', 'T3_B', 'T4_F', 'T4_B'};
Q_vars = {'Q1_F', 'Q1_B', 'Q2_F', 'Q2_B', 'Q3_F', 'Q3_B', 'Q4_F', 'Q4_B'};

% Initialize an array to store all event times (across detectors)
event_times = [];

% Loop over each detector and add valid event times directly to the array
for detector_idx = 1:4
    T_front = data.(T_vars{(detector_idx - 1) * 2 + 1});
    T_back = data.(T_vars{detector_idx * 2});
    Q_front = data.(Q_vars{(detector_idx - 1) * 2 + 1});
    Q_back = data.(Q_vars{detector_idx * 2});

    for i = 1:size(T_front, 1)
        if sum(Q_front(i, :) != 0 & Q_back(i, :) != 0) == 1
            % Directly append individual high values for accurate waiting times
            high_values = [T_front(i, T_front(i, :) > 1e4), T_back(i, T_back(i, :) > 1e4)];
            event_times = [event_times; high_values(:)];
        end
    end
end

% Sort event times and calculate waiting times between consecutive events
event_times = sort(event_times);
waiting_times = diff(event_times) / 1000; % Convert to microseconds

% Define narrower bins near zero to better capture exponential decay
waiting_time_bins = logspace(log10(min(waiting_times(waiting_times > 0))), log10(max(waiting_times)), 50);
waiting_counts = hist(waiting_times, waiting_time_bins);

% Calculate event rates in 30-second intervals (Poisson-like distribution)
time_interval = 30 * 1e6; % 30 seconds in microseconds
start_time = min(event_times);
end_time = max(event_times);
num_intervals = ceil((end_time - start_time) / time_interval);

% Initialize an array to store event rates per interval
rates = zeros(1, num_intervals);
for i = 1:num_intervals
    t_min = start_time + (i - 1) * time_interval;
    t_max = t_min + time_interval;
    rates(i) = sum(event_times >= t_min & event_times < t_max);
end

% Define bins for rate histogram (frequency of rates)
rate_bins = 0:max(rates);
rate_counts = hist(rates, rate_bins);

% Plot both histograms in a single figure with two subplots
figure;

% Plot waiting time histogram
subplot(1, 2, 1);
bar(waiting_time_bins, waiting_counts, 'hist');
set(gca, 'XScale', 'log', 'YScale', 'log'); % Log-log scale for exponential decay
title('Waiting Time Histogram');
xlabel('Waiting Time (µs)');
ylabel('Frequency');

% Plot event rate histogram
subplot(1, 2, 2);
bar(rate_bins, rate_counts, 'hist');
title('Event Rate Histogram (30-second Intervals)');
xlabel('Events per Interval');
ylabel('Frequency');
set(gca, 'YScale', 'log'); % Log scale on Y-axis for Poisson distribution

% Keep the plot open
waitforbuttonpress;
