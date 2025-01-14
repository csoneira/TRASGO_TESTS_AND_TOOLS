print(N_TRACKS)
print(len(df_measured))

df_filtered_gen_mea = df_measured[(df_measured['generated_type'] != df_measured['measured_type'])]
print(f"'generated_type' is different from 'measured_type': {len(df_filtered_gen_mea)}")

df_filtered_mea_fit = df_measured[(df_measured['measured_type'] != df_measured['fitted_type'])]
print(f"'measured_type' is different from 'fitted_type': {len(df_filtered_mea_fit)}")

df_filtered_gen_fit = df_measured[(df_measured['generated_type'] != df_measured['fitted_type'])]
print(f"'generated_type' is different from 'fitted_type': {len(df_filtered_gen_fit)}")


# Calculate the differences
diff_gen_mea = len(df_measured[(df_measured['generated_type'] != df_measured['measured_type'])])
diff_mea_fit = len(df_measured[(df_measured['measured_type'] != df_measured['fitted_type'])])
diff_gen_fit = len(df_measured[(df_measured['generated_type'] != df_measured['fitted_type'])])

# Fancy triangle display
print("\nDifferences between types:\n")
print("         generated")
print("             ^")
print("           /   \\")
print(f"  {diff_gen_mea:<6}  /     \\ {diff_gen_fit:<6}")
print("         /       \\")
print("        /_________\\")
print(f"  measured       fitted\n            {diff_mea_fit:<6}")