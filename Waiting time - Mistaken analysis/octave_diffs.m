% Load the .mat file
filename = 'mi0324311141342.mat';
data = load(filename);

% List of 'T' variables of interest
T_vars = {'T1_F', 'T1_B', 'T2_F', 'T2_B', 'T3_F', 'T3_B', 'T4_F', 'T4_B'};

% Initialize an array to store differences for histogram
all_diffs = [];

for k = 1:length(T_vars)
    % Access each 'T' variable dynamically
    T_matrix = data.(T_vars{k});

    % Initialize an array to store row averages for values > 1e6
    row_averages = zeros(size(T_matrix, 1), 1);

    % Calculate the average of values > 1e6 for each row
    for i = 1:size(T_matrix, 1)
        row_values = T_matrix(i, :);
        high_values = row_values(row_values > 1e6); % Filter values greater than 1e6
        if !isempty(high_values)
            row_averages(i) = mean(high_values); % Calculate mean for filtered values
        else
            row_averages(i) = NaN; % Assign NaN if no values > 1e6
        end
    end

    % Calculate the differences between consecutive averages
    % diff_values = row_averages;
    diff_values = diff(row_averages);

    diff_values = diff_values(diff_values < 1e15);

    % Remove NaN values from diff_values (if any)
    diff_values = diff_values(~isnan(diff_values));

    % Append to the overall differences array for histogram
    all_diffs = [all_diffs; diff_values];
end



% Specify the quantile range (e.g., 5th to 95th percentile)
lower_quantile = 0.0001;
upper_quantile = 0.999;

all_diffs = all_diffs/1000

% Calculate the quantiles
lower_bound = quantile(all_diffs, lower_quantile);
upper_bound = quantile(all_diffs, upper_quantile);

% Filter the data to include only values within the specified quantile range
filtered_diffs = all_diffs(all_diffs >= lower_bound & all_diffs <= upper_bound);

% Print the quantile range for debugging
printf("Quantile Range: %.2f to %.2f\n", lower_bound, upper_bound);

% Calculate bin width for the filtered data
bin_width = 0.5 * iqr(filtered_diffs) * length(filtered_diffs)^(-1/3);
bin_edges = min(filtered_diffs):bin_width:max(filtered_diffs);

% Plot histogram of filtered data
hist(filtered_diffs, bin_edges);
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca, 'FontSize', 14);  % Increase font size for readability
title('Histogram of Differences within Specified Quantile Range');
xlabel('Difference');
ylabel('Frequency');

% Adjust x-axis limits based on filtered data
xlim([min(filtered_diffs), max(filtered_diffs)]);

% Keep the plot open
waitfor(gcf);
