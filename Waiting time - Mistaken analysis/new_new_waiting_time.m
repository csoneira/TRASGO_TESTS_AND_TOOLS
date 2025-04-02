% Parameters
n = 4; % Number of layers that must have non-zero values per row to include it

% Load the .mat file
filename = 'mi0324311141342.mat';
data = load(filename);

% Create a logical array indicating rows where all four layers are non-zero
valid_rows = (sum([data.T1_F ~= 0, data.T2_F ~= 0, data.T3_F ~= 0, data.T4_F ~= 0], 2) >= n) | ...
             (sum([data.T1_B ~= 0, data.T2_B ~= 0, data.T3_B ~= 0, data.T4_B ~= 0], 2) >= n);

% Initialize an array for filtered differences
filtered_diffs = [];

% Process each T variable separately
T_vars = {'T1_F', 'T1_B', 'T2_F', 'T2_B', 'T3_F', 'T3_B', 'T4_F', 'T4_B'};
for k = 1:length(T_vars)
    % Access each T variable and apply the row filter
    T_matrix = data.(T_vars{k})(valid_rows, :);
    
    % Calculate the average of values > 1e6 for each row
    row_averages = zeros(size(T_matrix, 1), 1);
    for i = 1:size(T_matrix, 1)
        row_values = T_matrix(i, :);
        high_values = row_values(row_values > 1e6); % Filter values greater than 1e6
        if !isempty(high_values)
            row_averages(i) = mean(high_values); % Calculate mean for filtered values
        else
            row_averages(i) = NaN; % Assign NaN if no values > 1e6
        end
    end
    
    % Calculate the differences between consecutive row averages
    diff_values = diff(row_averages);
    diff_values = diff_values(~isnan(diff_values)); % Remove NaNs
    filtered_diffs = [filtered_diffs; diff_values]; % Append to overall differences
end

% Continue with plotting as before...
filtered_diffs = filtered_diffs / 1000; % Convert to microseconds
filtered_diffs = filtered_diffs(filtered_diffs > 0); % Filter out non-positive values

% Calculate mean for Poisson fit in microseconds
lambda = mean(filtered_diffs);

% Create custom bin edges for finer resolution at lower values
bin_edges = logspace(log10(min(filtered_diffs)), log10(max(filtered_diffs)), 50);

% Plot histogram of filtered data with log scales
hist(filtered_diffs, bin_edges);
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca, 'FontSize', 14);  % Increase font size for readability


% Add title and labels with larger font size
title(sprintf('Histogram with Poisson Fit (\\lambda = %.2f µs)', lambda), 'FontSize', 16);
xlabel('Difference (µs)', 'FontSize', 14);
ylabel('Frequency (log scale)', 'FontSize', 14);

% Adjust x-axis limits based on filtered data
xlim([min(filtered_diffs), max(filtered_diffs)]);

% Keep the plot open
waitfor(gcf);
