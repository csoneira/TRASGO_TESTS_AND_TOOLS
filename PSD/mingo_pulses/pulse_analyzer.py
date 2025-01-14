import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_and_calculate_integrals(data, gate1_length, gate2_length):
    results = []
    current_event = []
    
    total_points = len(data)
    current_point = 0
    for i in range(1, len(data)):
        current_point += 1
        progress = current_point / total_points * 100
        print(f"\rProgress: {progress:.2f}%", end='')
        current_event.append(data[i])
        if data[i][0] < data[i - 1][0]:
            results.append(calculate_integrals(np.array(current_event), gate1_length, gate2_length))
            current_event = []
    if current_event:
        results.append(calculate_integrals(np.array(current_event), gate1_length, gate2_length))
    
    return results

def calculate_integrals(event, gate1_length, gate2_length):
    event_t1, event_wave1, event_t2, event_wave2 = event[:, 0], event[:, 1], event[:, 2], event[:, 3]

    leading_edge1 = np.argmax(event_wave1 > np.mean(event_wave1[:10]) + 5 * np.std(event_wave1[:10]))
    gate1_end1 = leading_edge1 + gate1_length
    gate2_end1 = gate1_end1 + gate2_length

    leading_edge2 = np.argmax(event_wave2 > np.mean(event_wave2[:10]) + 2 * np.std(event_wave2[:10]))
    gate1_end2 = leading_edge2 + gate1_length
    gate2_end2 = gate1_end2 + gate2_length

    gate1_integral1 = np.trapz(event_wave1[leading_edge1:gate1_end1], event_t1[leading_edge1:gate1_end1])
    gate2_integral1 = np.trapz(event_wave1[gate1_end1:gate2_end1], event_t1[gate1_end1:gate2_end1])

    gate1_integral2 = np.trapz(event_wave2[leading_edge2:gate1_end2], event_t2[leading_edge2:gate1_end2])
    gate2_integral2 = np.trapz(event_wave2[gate1_end2:gate2_end2], event_t2[gate1_end2:gate2_end2])

    return (gate1_integral1, gate2_integral1, gate1_integral2, gate2_integral2)

def save_integrals_to_file(results, output_filename):
    df = pd.DataFrame(results, columns=['Gate1_Integral1', 'Gate2_Integral1', 'Gate1_Integral2', 'Gate2_Integral2'])
    df.to_csv(output_filename, index=False)

def load_integrals_from_file(filename):
    return pd.read_csv(filename)

# Load the raw data from files once
data1 = np.loadtxt('mingo_pulses_Cs137.txt')
data2 = np.loadtxt('mingo_pulses_cosmic.txt')

# Define the gate ranges (adjust these values as needed)
gate_ranges = [
    (5, 5),
    (5, 10),
    (10, 20),
    (5, 50),
]

# Process and save the integrals for each gate range
for gate1_start, gate1_end in gate_ranges:
    results1 = read_and_calculate_integrals(data1, gate1_start, gate1_end)
    results2 = read_and_calculate_integrals(data2, gate1_start, gate1_end)
    
    output_filename1 = f'results_Cs137_Gate1_{gate1_start}_Gate2_{gate1_end}.csv'
    output_filename2 = f'results_Cosmic_Gate1_{gate1_start}_Gate2_{gate1_end}.csv'
    
    save_integrals_to_file(results1, output_filename1)
    save_integrals_to_file(results2, output_filename2)

# Plotting
fig, axs = plt.subplots(len(gate_ranges), 2, figsize=(18, 5 * len(gate_ranges)))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Reload the integrals from files and plot
for idx, (gate1_start, gate1_end) in enumerate(gate_ranges):
    filename_cs137 = f'results_Cs137_Gate1_{gate1_start}_Gate2_{gate1_end}.csv'
    filename_cosmic = f'results_Cosmic_Gate1_{gate1_start}_Gate2_{gate1_end}.csv'

    df_cs137 = load_integrals_from_file(filename_cs137)
    df_cosmic = load_integrals_from_file(filename_cosmic)

    df_cs137['EnergySum'] = (df_cs137['Gate1_Integral1'] + df_cs137['Gate2_Integral1'])
    df_cs137['PSD_Ratio'] = (df_cs137['Gate1_Integral1']) / (df_cs137['Gate1_Integral1'] + df_cs137['Gate2_Integral1'])
    
    df_cosmic['EnergySum'] = (df_cosmic['Gate1_Integral1'] + df_cosmic['Gate2_Integral1'])
    df_cosmic['PSD_Ratio'] = (df_cosmic['Gate1_Integral1']) / (df_cosmic['Gate1_Integral1'] + df_cosmic['Gate2_Integral1'])

    axs[idx, 0].hist2d(df_cs137['EnergySum'], df_cs137['PSD_Ratio'], bins=[1000, 1000], cmap='viridis')
    axs[idx, 0].set_title(f'Cs137 (Gate1: {gate1_start}, Gate2: {gate1_end})')
    axs[idx, 0].set_xlabel('Energy Sum')
    axs[idx, 0].set_ylabel('Short Integral / Integral')
    axs[idx, 0].grid(True)
    
    axs[idx, 1].hist2d(df_cosmic['EnergySum'], df_cosmic['PSD_Ratio'], bins=[1000, 1000], cmap='viridis')
    axs[idx, 1].set_title(f'Cosmic (Gate1: {gate1_start}, Gate2: {gate1_end})')
    axs[idx, 1].set_xlabel('Energy Sum')
    axs[idx, 1].set_ylabel('Short Integral / Integral')
    axs[idx, 1].grid(True)

plt.show()
