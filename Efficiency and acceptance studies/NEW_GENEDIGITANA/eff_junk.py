# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Ensure 'time' is a datetime type and set it as the index
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

# Step 1: Resample in minutes to count combinations of 'generated_type' and 'fitted_type'
df_resampled = (
    df
    .groupby([pd.Grouper(freq='1min'), 'crossing_type', 'fitted_type'])
    .size()
    .reset_index(name='count')  # Count occurrences of each combination per minute
)

# Step 4: Create a unique column name for each generated-fitted type combination
df_resampled['pair_name'] = (
    'g' + df_resampled['crossing_type'] + '_f' + df_resampled['fitted_type']
)

# Pivot data to create separate columns for each pair_name, indexed by 'time'
pair_counts_per_minute = df_resampled.pivot(
    index='time', columns='pair_name', values='count'
).fillna(0).reset_index()

# Step 5: Merge the pair counts with accumulated_data on 'time'
accumulated_data = accumulated_data.merge(
    pair_counts_per_minute, on='time', how='left'
)

# Fill NaN values with 0 to account for any missing intervals
accumulated_data.fillna(0, inplace=True)

#%%

# Calculate the total sum of each column
total_sums = accumulated_data.drop(columns=['time']).sum()

# Sort the total sums in descending order
sorted_total_sums = total_sums.sort_values(ascending=False)

# Print the sorted total sums
print(sorted_total_sums)

#%%

# Separate matched and mismatched types by examining the index names
matched_count = sorted_total_sums.filter(like='_f').loc[
    [name for name in sorted_total_sums.index if name.split('_')[0][1:] == name.split('_')[1][1:]]
].sum()

mismatched_count = sorted_total_sums.filter(like='_f').sum() - matched_count

# Total count for generated-fitted pairs
total_count = matched_count + mismatched_count

# Calculate percentages
matched_percentage = (matched_count / total_count) * 100 if total_count else 0
mismatched_percentage = (mismatched_count / total_count) * 100 if total_count else 0

# Display results
print(f"Matched types (gXX_fXX): {matched_count} times, {matched_percentage:.2f}% of total")
print(f"Mismatched types (gXX_fYY): {mismatched_count} times, {mismatched_percentage:.2f}% of total")

#%%

# Filter sorted_total_sums to include only entries with '_f' in the index and having three or more digits in both generated and fitted types
filtered_total_sums = sorted_total_sums.loc[
    [name for name in sorted_total_sums.index if '_f' in name and len(name.split('_')[0][1:]) > 2 and len(name.split('_')[1][1:]) > 2]
]

# Count matched types where generated_type and fitted_type match
matched_count = filtered_total_sums.loc[
    [name for name in filtered_total_sums.index if name.split('_')[0][1:] == name.split('_')[1][1:]]
].sum()

# Count mismatched types
mismatched_count = filtered_total_sums.sum() - matched_count

# Total count for filtered generated-fitted pairs
total_count = matched_count + mismatched_count

# Calculate percentages
matched_percentage = (matched_count / total_count) * 100 if total_count else 0
mismatched_percentage = (mismatched_count / total_count) * 100 if total_count else 0

# Display results
print(f"Matched types (gXX_fXX): {matched_count} times, {matched_percentage:.2f}% of total")
print(f"Mismatched types (gXX_fYY): {mismatched_count} times, {mismatched_percentage:.2f}% of total")

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------