# To put into main_analysis.py in the time_to_position_average function:

# # Append y and Q to binary file---------------
#     with open('y_and_Q', 'ab') as f:
#         f.write(struct.pack('f', thick_strip))
#         for value in Q:
#             f.write(struct.pack('f', value))
# # --------------------------------------------

# Clear all variables from the global scope
globals().clear()

import struct
import numpy as np
import matplotlib.pyplot as plt

output_order = 0

def hist_1d(vdat, bin_number, title, axes_label, name_of_file):
    global output_order
    # plt.close()
    v=(8,5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot histograms on the single axis
    ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", \
            label=f"All hits, {len(vdat)} events", density = False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axes_label)
    plt.ylabel("Counts")
    # plt.xlim([log_low_lim_fig, None])
    # plt.ylim([0, 800])
    # plt.xscale("log");
    # plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

def hist_1d_list(vdat_list, bin_number, title, axes_label, name_of_file, colors=None, labels=None):
    global output_order
    # plt.close()
    v = (8, 5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot histograms on the single axis
    for i, vdat in enumerate(vdat_list):
        if colors is not None:
            color = colors[i] if i < len(colors) else None
        else:
            color = None
        if labels is not None:
            label = labels[i] if i < len(labels) else None
        else:
            label = None
        ax.hist(vdat, bins=bin_number, alpha=0.5, color=color, label=label, density=False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axes_label)
    plt.ylabel("Counts")
    # plt.xlim([log_low_lim_fig, None])
    # plt.ylim([0, 800])
    # plt.xscale("log");
    # plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

def hist_1d_list_subplt(vdat_list, bin_number, title, axes_label, name_of_file, colors=None, labels=None):
    global output_order
    # Set the figure size and create subplots
    w = 5.5 * np.array([ 1, 3/5 ]) # 4.5 was alright
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(w[0], w[1]), sharex=True)  # Shared x-axis
    
    # Plot histograms on separate axes
    for i, (vdat, ax) in enumerate(zip(vdat_list, axes.flatten())):
        # if i == 0:
            # ax.set_title(f"{title[:21]}")
        if colors is not None:
            color = colors[i] if i < len(colors) else None
        else:
            color = None
        if labels is not None:
            label = labels[i] if i < len(labels) else None
        else:
            label = None
        ax.hist(vdat, bins=bin_number, alpha=0.5, color=color, label=label, density=False)
        ax.legend(fontsize="x-small")
        ax.set_ylabel("Counts")

    axes[-1].set_xlabel(axes_label)  # Only set x-label on the last subplot
    # plt.suptitle(f"{title}")
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order += 1
    plt.show()
    plt.close()

# Define the number of floats per row
num_floats_per_row = 5  # Assuming each row contains 4 floats (y and Q)

# Initialize empty lists to store the loaded data
loaded_strip = []
loaded_Q = []

# Open the binary file for reading
with open('y_and_Q', 'rb') as f:
    # Read the binary data until the end of the file
    while True:
        # Read the next chunk of data
        data_chunk = f.read(4 * num_floats_per_row)  # Each float is 4 bytes
        
        # If there's no more data, break out of the loop
        if not data_chunk:
            break
        
        # Unpack the data_chunk into floats
        unpacked_data = struct.unpack('f' * num_floats_per_row, data_chunk)
        
        # Append the unpacked data to the appropriate list
        loaded_strip.append(unpacked_data[0])
        loaded_Q.append(unpacked_data[1:])

# Convert the lists to NumPy arrays
loaded_strip = np.array(loaded_strip)
loaded_Q = np.array(loaded_Q)

# Print the loaded data (for verification)
print("Loaded detections:", len(loaded_strip))

long_Q = np.ravel(loaded_Q)
long_Q = long_Q[ long_Q > 0 ]
hist_1d(long_Q, "auto", "Original charge", "Q / ToT ns", "og_q")
hist_1d(long_Q**0.22, "auto", "Transformed charge", "Q / ToT ns", "og_q")
# hist_1d_list([long_Q[long_Q > 2.5], long_Q**0.22], "auto", "Original charge", "Q / ToT ns", "og_q", colors=["red", "blue"], labels=["Original", "Transformed"])

# -----------------------------------------------------------------------------
# Shared charge case ----------------------------------------------------------
# -----------------------------------------------------------------------------

transf_exp = 1 # 0.22 was nice
# pos_uncer_corr = 1/np.sqrt(12) * 0.6
# pos_uncer_corr = 0.2
pos_uncer_corr = 0.2

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

formatted_widths_13 = ", ".join([f"{width:.2f}" for width in y_width_T1_and_T3])
formatted_widths_24 = ", ".join([f"{width:.2f}" for width in y_width_T2_and_T4])
print("------------------------------------------------")
print("Uncertainties in widths, shared charge case:")
print("VERY RELATED WITH THE IONIZATION SECTION")
print(f"\tT1 and T3: {formatted_widths_13} mm")
print(f"\tT2 and T4: {formatted_widths_24} mm")
print("------------------------------------------------")

# y_width_T1_and_T3 = np.array([0,0,0,0])
# y_width_T2_and_T4 = np.array([0,0,0,0])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    
    # cond = ( np.min(np.nonzero(Q)) / np.max(np.nonzero(Q)) ) < 0.3
    cond = True
    if np.count_nonzero(Q) <= 1 and cond: continue

    if thick_strip == 1:
        y_pos = y_pos_T2_and_T4.copy()
        # for j in range(len(y_pos)):
        #     # y_pos[j] = 10 + np.random.normal(y_pos[j], y_width_T2_and_T4[j] * pos_uncer_corr)
        #     dev = y_width_T2_and_T4[j] * pos_uncer_corr
        #     random_number = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
        #     y_pos[j] = random_number
            
    elif thick_strip == 4:
        y_pos = y_pos_T1_and_T3.copy()
        # for j in range(len(y_pos)):
        #     # y_pos[j] = np.random.normal(y_pos[j], y_width_T1_and_T3[j] * pos_uncer_corr)
        #     dev = y_width_T1_and_T3[j] * pos_uncer_corr
        #     random_number = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
        #     y_pos[j] = random_number
    
    Q_og = Q
    Q_trans = transformation(Q, transf_exp)
    
    y_og = np.sum(y_pos * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    pos_corr = 1
    y_trans = pos_corr * np.sum(y_pos * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)
    
# y_long_T2_and_T4_og = np.array(y_long_T2_and_T4_og)
# y_long_T1_and_T3_og = np.array(y_long_T1_and_T3_og)
# y_long_T2_and_T4_trans = np.array(y_long_T2_and_T4_trans)
# y_long_T1_and_T3_trans = np.array(y_long_T1_and_T3_trans)

# Histogram
hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, "Shared charge case. Linear and Transformed Y, T1 and T3", "Y / mm", "shared_linear_trans_T1_T3", colors=["red", "blue"], labels=["Linear", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, "Shared charge case. Linear and Transformed Y, T2 and T4", "Y / mm", "shared_linear_trans_T2_T4", colors=["red", "blue"], labels=["Linear", "Transformed"])



# -----------------------------------------------------------------------------
# Shared charge case with more style ------------------------------------------
# -----------------------------------------------------------------------------

transf_exp = 0.22 # 0.22 was nice
# pos_uncer_corr = 1/np.sqrt(12) * 0.6
# pos_uncer_corr = 0.2
pos_uncer_corr = 0.2

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

formatted_widths_13 = ", ".join([f"{width:.2f}" for width in y_width_T1_and_T3])
formatted_widths_24 = ", ".join([f"{width:.2f}" for width in y_width_T2_and_T4])
print("------------------------------------------------")
print("Uncertainties in widths, shared charge case:")
print("VERY RELATED WITH THE IONIZATION SECTION")
print(f"\tT1 and T3: {formatted_widths_13} mm")
print(f"\tT2 and T4: {formatted_widths_24} mm")
print("------------------------------------------------")

# y_width_T1_and_T3 = np.array([0,0,0,0])
# y_width_T2_and_T4 = np.array([0,0,0,0])

# general_crosstalk_bound = 0.5

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    
    # cond = ( np.min(np.nonzero(Q)) / np.max(np.nonzero(Q)) ) < 0.3
    cond = True
    if np.count_nonzero(Q) <= 1 and cond: continue
    
    # Q_nz = Q[Q != 0]
    # Q_nz = sorted(Q_nz)
    # if abs(Q_nz[0] / Q_nz[-1]) < 0.05 and Q_nz[-2] < general_crosstalk_bound: continue

    if thick_strip == 1:
        y_pos = y_pos_T2_and_T4.copy()
        # for j in range(len(y_pos)):
        #     # y_pos[j] = 10 + np.random.normal(y_pos[j], y_width_T2_and_T4[j] * pos_uncer_corr)
        #     dev = y_width_T2_and_T4[j] * pos_uncer_corr
        #     random_number = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
        #     y_pos[j] = random_number
            
    elif thick_strip == 4:
        y_pos = y_pos_T1_and_T3.copy()
        # for j in range(len(y_pos)):
        #     # y_pos[j] = np.random.normal(y_pos[j], y_width_T1_and_T3[j] * pos_uncer_corr)
        #     dev = y_width_T1_and_T3[j] * pos_uncer_corr
        #     random_number = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
        #     y_pos[j] = random_number
    
    Q_og = Q
    Q_trans = transformation(Q, transf_exp)
    
    q_tr_nz = Q_trans[Q_trans != 0]
    q_tr_nz = sorted(q_tr_nz)
    # q_tr_nz = np.array(q_tr_nz)
    if q_tr_nz[0] < 1.25: continue
    
    y_og = np.sum(y_pos * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    pos_corr = 1
    y_trans = pos_corr * np.sum(y_pos * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)

# Histogram
hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, "Shared charge case. Linear and Transformed Y, T1 and T3", "Y / mm", "shared_linear_trans_T1_T3", colors=["red", "blue"], labels=["Linear", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, "Shared charge case. Linear and Transformed Y, T2 and T4", "Y / mm", "shared_linear_trans_T2_T4", colors=["red", "blue"], labels=["Linear", "Transformed"])



# -----------------------------------------------------------------------------
# Charge in only one strip case -----------------------------------------------
# -----------------------------------------------------------------------------

# pos_uncer_corr = 1/np.sqrt(12) * 0.6
uncertain_factor = 0.2 # 0.5 was fantastic
uncertain_factor_og = 0.4 # 0.7 was OK

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

formatted_widths_13 = ", ".join([f"{width:.2f}" for width in y_width_T1_and_T3])
formatted_widths_24 = ", ".join([f"{width:.2f}" for width in y_width_T2_and_T4])
print("------------------------------------------------")
print("Uncertainties in widths, one strip case:")
print("VERY RELATED WITH THE IONIZATION SECTION")
print(f"\tT1 and T3: {formatted_widths_13} mm")
print(f"\tT2 and T4: {formatted_widths_24} mm")
print("------------------------------------------------")

# y_width_T1_and_T3 = np.array([0,0,0,0])
# y_width_T2_and_T4 = np.array([0,0,0,0])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i].copy()
    Q = loaded_Q[i,:].copy()
    
    # cond = ( np.min(np.nonzero(Q)) / np.max(np.nonzero(Q)) ) < 0.3
    cond = True
    if np.count_nonzero(Q) > 1 and cond: continue

    Q_only = Q[ np.nonzero(Q)[0] ]
    # uncertain_factor_og = 1/np.sqrt(12) / 2
    uncertain_factor_tr = uncertain_factor

    if thick_strip == 1:
        y_pos_og = y_pos_T2_and_T4.copy()
        y_pos_tr = y_pos_T2_and_T4.copy()
        for j in range(len(y_pos_og)):
            dev = y_width_T2_and_T4[j] * uncertain_factor_og
            random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
            y_pos_og[j] = random_value
            # y_pos_og[j] = np.random.normal(y_pos_og[j], dev)
            
            dev = y_width_T2_and_T4[j] * uncertain_factor_tr
            random_value = np.random.uniform(y_pos_tr[j] - dev, y_pos_tr[j] + dev)
            y_pos_tr[j] = random_value
            # y_pos_tr[j] = np.random.normal(y_pos_og[j], dev)
    elif thick_strip == 4:
        y_pos_og = y_pos_T1_and_T3.copy()
        y_pos_tr = y_pos_T1_and_T3.copy()
        for k in range(len(y_pos_og)):
            dev = y_width_T1_and_T3[k] * uncertain_factor_og
            random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
            y_pos_og[k] = random_value
            # y_pos_og[k] = np.random.normal(y_pos_og[k], dev)
            
            dev = y_width_T1_and_T3[k] * uncertain_factor_tr
            random_value = np.random.uniform(y_pos_tr[k] - dev, y_pos_tr[k] + dev)
            y_pos_tr[k] = random_value
            # y_pos_tr[k] = np.random.normal(y_pos_tr[k], dev)
    
    Q_og = Q
    Q_trans = Q
    
    y_og = np.sum(y_pos_og * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    # y_trans = 0.7 * np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    y_trans = np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)

hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, "One strip. Original and Transformed Y, T1 and T3", "Y / mm", "one_strip_linear_trans_T1_T3", colors=["red", "blue"], labels=["Original", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, "One strip. Original and Transformed Y, T2 and T4", "Y / mm", "one_strip_linear_trans_T2_T4", colors=["red", "blue"], labels=["Original", "Transformed"])


# -----------------------------------------------------------------------------
# Both together ---------------------------------------------------------------
# -----------------------------------------------------------------------------

# Shared charge
transf_exp = 0.22 # 0.22 was nice
pos_uncer_corr = 0.2

# Single strips
uncertain_factor = 0.5 # 0.5 was fantastic

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

formatted_widths_13 = ", ".join([f"{width:.2f}" for width in y_width_T1_and_T3])
formatted_widths_24 = ", ".join([f"{width:.2f}" for width in y_width_T2_and_T4])
print("------------------------------------------------")
print("Uncertainties in widths, shared charge case:")
print("VERY RELATED WITH THE IONIZATION SECTION")
print(f"\tT1 and T3: {formatted_widths_13} mm")
print(f"\tT2 and T4: {formatted_widths_24} mm")
print("------------------------------------------------")

# y_width_T1_and_T3 = np.array([0,0,0,0])
# y_width_T2_and_T4 = np.array([0,0,0,0])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    Q_og = Q
    
    # Double strips or more
    if np.count_nonzero(Q) > 1:
        if thick_strip == 1:
            y_pos = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos)):
                y_pos[j] = 10 + np.random.normal(y_pos[j], y_width_T2_and_T4[j] * pos_uncer_corr)
        elif thick_strip == 4:
            y_pos = y_pos_T1_and_T3.copy()
            for j in range(len(y_pos)):
                y_pos[j] = np.random.normal(y_pos[j], y_width_T1_and_T3[j] * pos_uncer_corr)
                
        Q_trans = transformation(Q, transf_exp)
        pos_scale_factor = 1.3
            
    # Only one strip
    if np.count_nonzero(Q) <= 1:
        Q_only = Q[ np.nonzero(Q)[0] ]
        uncertain_factor_og = 1/np.sqrt(12) / 2
        uncertain_factor_tr = uncertain_factor

        if thick_strip == 1:
            y_pos_og = y_pos_T2_and_T4.copy()
            y_pos_tr = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos)):
                dev = y_width_T2_and_T4[j] * uncertain_factor_og
                y_pos_og[j] = np.random.normal(y_pos_og[j], dev)
                
                dev = y_width_T2_and_T4[j] * uncertain_factor_tr
                y_pos_tr[j] = np.random.normal(y_pos_tr[j], dev)
        elif thick_strip == 4:
            y_pos_og = y_pos_T1_and_T3.copy()
            y_pos_tr = y_pos_T1_and_T3.copy()
            for j in range(len(y_pos)):
                dev = y_width_T1_and_T3[j] * uncertain_factor_og
                y_pos_og[j] = np.random.normal(y_pos_og[j], dev)
                
                dev = y_width_T1_and_T3[j] * uncertain_factor_tr
                y_pos_tr[j] = np.random.normal(y_pos_tr[j], dev)
                
        Q_trans = Q
        pos_scale_factor = 0.7
        
    y_og = np.sum(y_pos_og * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    y_trans = pos_scale_factor * np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)
   
hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, f"All cases. Original and Transformed Y, T1 and T3, {len(y_long_T1_and_T3_og)} events", "Y / mm", "all_cases_linear_trans_T1_T3", colors=["red", "blue"], labels=["Original", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, f"All cases. Original and Transformed Y, T2 and T4, {len(y_long_T2_and_T4_og)} events", "Y / mm", "all_cases_linear_trans_T2_T4", colors=["red", "blue"], labels=["Original", "Transformed"])

y_conc_og = np.concatenate((y_long_T1_and_T3_og, y_long_T2_and_T4_og))
y_conc_trans = np.concatenate((y_long_T1_and_T3_trans, y_long_T2_and_T4_trans))
cond = (abs(y_conc_og) < 200) & (abs(y_conc_trans) < 200)
hist_1d_list([y_conc_og[cond], y_conc_trans[cond]], 400, f"All cases. Original and Transformed Y, {len(y_conc_trans)} events", "Y / mm", "all_cases_linear_trans", colors=["red", "blue"], labels=["Original", "Transformed"])




# -----------------------------------------------------------------------------
# Both together with uniform --------------------------------------------------
# -----------------------------------------------------------------------------

# Shared charge
transf_exp = 0.22 # 0.22 was nice
pos_uncer_corr = 0.2

# Single strips
uncertain_factor = 0.49 # 0.5 was fantastic

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

formatted_widths_13 = ", ".join([f"{width:.2f}" for width in y_width_T1_and_T3])
formatted_widths_24 = ", ".join([f"{width:.2f}" for width in y_width_T2_and_T4])
print("------------------------------------------------")
print("Uncertainties in widths, shared charge case:")
print("VERY RELATED WITH THE IONIZATION SECTION")
print(f"\tT1 and T3: {formatted_widths_13} mm")
print(f"\tT2 and T4: {formatted_widths_24} mm")
print("------------------------------------------------")

# y_width_T1_and_T3 = np.array([0,0,0,0])
# y_width_T2_and_T4 = np.array([0,0,0,0])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    Q_og = Q
    
    # Double strips or more
    if np.count_nonzero(Q) > 1:
        # continue
        if thick_strip == 1:
            y_pos = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos)):
                y_pos[j] = 10 + np.random.normal(y_pos[j], y_width_T2_and_T4[j] * pos_uncer_corr)
        elif thick_strip == 4:
            y_pos = y_pos_T1_and_T3.copy()
            for j in range(len(y_pos)):
                y_pos[j] = np.random.normal(y_pos[j], y_width_T1_and_T3[j] * pos_uncer_corr)
                
        Q_trans = transformation(Q, transf_exp)
        pos_scale_factor = 1.3
            
    # Only one strip
    if np.count_nonzero(Q) <= 1:
        Q_only = Q[ np.nonzero(Q)[0] ]
        uncertain_factor_og = 0.2
        uncertain_factor_tr = uncertain_factor

        if thick_strip == 1:
            y_pos_og = y_pos_T2_and_T4.copy()
            y_pos_tr = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos_og)):
                dev = y_width_T2_and_T4[j] * uncertain_factor_og
                random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
                y_pos_og[j] = random_value
                # y_pos_og[j] = np.random.normal(y_pos_og[j], dev)
                
                dev = y_width_T2_and_T4[j] * uncertain_factor_tr
                random_value = np.random.uniform(y_pos_tr[j] - dev, y_pos_tr[j] + dev)
                y_pos_tr[j] = random_value
                # y_pos_tr[j] = np.random.normal(y_pos_og[j], dev)
        elif thick_strip == 4:
            y_pos_og = y_pos_T1_and_T3.copy()
            y_pos_tr = y_pos_T1_and_T3.copy()
            for k in range(len(y_pos_og)):
                dev = y_width_T1_and_T3[k] * uncertain_factor_og
                random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
                y_pos_og[k] = random_value
                # y_pos_og[k] = np.random.normal(y_pos_og[k], dev)
                
                dev = y_width_T1_and_T3[k] * uncertain_factor_tr
                random_value = np.random.uniform(y_pos_tr[k] - dev, y_pos_tr[k] + dev)
                y_pos_tr[k] = random_value
                # y_pos_tr[k] = np.random.normal(y_pos_tr[k], dev)
                
        Q_trans = Q
        pos_scale_factor = 1
        
    y_og = np.sum(y_pos_og * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    y_trans = pos_scale_factor * np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)
   
hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 390, f"All cases. Original and Transformed Y, T1 and T3, {len(y_long_T1_and_T3_og)} events", "Y / mm", "all_cases_linear_trans_T1_T3", colors=["red", "blue"], labels=["Original", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 390, f"All cases. Original and Transformed Y, T2 and T4, {len(y_long_T2_and_T4_og)} events", "Y / mm", "all_cases_linear_trans_T2_T4", colors=["red", "blue"], labels=["Original", "Transformed"])

y_conc_og = np.concatenate((y_long_T1_and_T3_og, y_long_T2_and_T4_og))
y_conc_trans = np.concatenate((y_long_T1_and_T3_trans, y_long_T2_and_T4_trans))
cond = (abs(y_conc_og) < 200) & (abs(y_conc_trans) < 200)
hist_1d_list([y_conc_og[cond], y_conc_trans[cond]], 390, f"All cases. Original and Transformed Y, {len(y_conc_trans)} events", "Y / mm", "all_cases_linear_trans", colors=["red", "blue"], labels=["Original", "Transformed"])










# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The final test
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# One strip
# -----------------------------------------------------------------------------

# Parameters -------------------------------
uncertain_factor_og = 0.4 # 0.7 was OK
uncertain_factor_tr = 0.2
# -----------------------------------------

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i].copy()
    Q = loaded_Q[i,:].copy()
    
    # cond = ( np.min(np.nonzero(Q)) / np.max(np.nonzero(Q)) ) < 0.3
    cond = True
    if np.count_nonzero(Q) > 1 and cond: continue


    if thick_strip == 1:
        y_pos_og = y_pos_T2_and_T4.copy()
        y_pos_tr = y_pos_T2_and_T4.copy()
        for j in range(len(y_pos_og)):
            dev = y_width_T2_and_T4[j] * uncertain_factor_og
            random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
            y_pos_og[j] = random_value
            
            dev = y_width_T2_and_T4[j] * uncertain_factor_tr
            random_value = np.random.uniform(y_pos_tr[j] - dev, y_pos_tr[j] + dev)
            y_pos_tr[j] = random_value
    elif thick_strip == 4:
        y_pos_og = y_pos_T1_and_T3.copy()
        y_pos_tr = y_pos_T1_and_T3.copy()
        for k in range(len(y_pos_og)):
            dev = y_width_T1_and_T3[k] * uncertain_factor_og
            random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
            y_pos_og[k] = random_value
            
            dev = y_width_T1_and_T3[k] * uncertain_factor_tr
            random_value = np.random.uniform(y_pos_tr[k] - dev, y_pos_tr[k] + dev)
            y_pos_tr[k] = random_value
    
    Q_og = Q
    Q_trans = Q
    
    y_og = np.sum(y_pos_og * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    y_trans = np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)

hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, "One strip. Original and Transformed Y, T1 and T3", "Y / mm", "one_strip_linear_trans_T1_T3", colors=["red", "blue"], labels=["Original", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, "One strip. Original and Transformed Y, T2 and T4", "Y / mm", "one_strip_linear_trans_T2_T4", colors=["red", "blue"], labels=["Original", "Transformed"])



# -----------------------------------------------------------------------------
# Double but is SHARED
# -----------------------------------------------------------------------------

# Parameters -------------------------------
transf_exp = 0.22 # 0.22 was nice
transformed_charge_crosstalk_bound = 1.25
# -----------------------------------------

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    
    Q_og = Q
    Q_trans = transformation(Q, transf_exp)
    
    cond = True
    if np.count_nonzero(Q) <= 1 and cond: continue
    
    q_tr_nz = Q_trans[Q_trans != 0]
    q_tr_nz = sorted(q_tr_nz)
    if q_tr_nz[0] > transformed_charge_crosstalk_bound:
        if thick_strip == 1:
            y_pos = y_pos_T2_and_T4.copy()
            # for j in range(len(y_pos)):
            #     # y_pos[j] = 10 + np.random.normal(y_pos[j], y_width_T2_and_T4[j] * pos_uncer_corr)
            #     dev = y_width_T2_and_T4[j] * pos_uncer_corr
            #     random_number = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
            #     y_pos[j] = random_number
                
        elif thick_strip == 4:
            y_pos = y_pos_T1_and_T3.copy()
            # for j in range(len(y_pos)):
            #     # y_pos[j] = np.random.normal(y_pos[j], y_width_T1_and_T3[j] * pos_uncer_corr)
            #     dev = y_width_T1_and_T3[j] * pos_uncer_corr
            #     random_number = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
            #     y_pos[j] = random_number
    else:
        continue
    
    y_og = np.sum(y_pos * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    y_trans = np.sum(y_pos * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)

# Histogram
hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, "Shared charge case. Linear and Transformed Y, T1 and T3", "Y / mm", "shared_linear_trans_T1_T3", colors=["red", "blue"], labels=["Linear", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, "Shared charge case. Linear and Transformed Y, T2 and T4", "Y / mm", "shared_linear_trans_T2_T4", colors=["red", "blue"], labels=["Linear", "Transformed"])



# -----------------------------------------------------------------------------
# Double but is CROSSTALK
# -----------------------------------------------------------------------------

# Parameters -------------------------------
transf_exp = 0.22 # 0.22 was nice
transformed_charge_crosstalk_bound = 1.25
uncertain_factor_og = 0.4 # 0.7 was OK
uncertain_factor_tr = 0.2
# -----------------------------------------

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    
    Q_og = Q
    Q_trans = transformation(Q, transf_exp)
    
    cond = True
    if np.count_nonzero(Q) <= 1 and cond: continue
    
    q_tr_nz = Q_trans[Q_trans != 0]
    q_tr_nz = sorted(q_tr_nz)
    
    if q_tr_nz[0] > transformed_charge_crosstalk_bound:
        continue
    else:
        Q_og = (Q == np.max(Q)) * Q
        Q_trans = (Q == np.max(Q)) * Q
        
        if thick_strip == 1:
            y_pos_og = y_pos_T2_and_T4.copy()
            y_pos_tr = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos_og)):
                dev = y_width_T2_and_T4[j] * uncertain_factor_og
                random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
                y_pos_og[j] = random_value
                
                dev = y_width_T2_and_T4[j] * uncertain_factor_tr
                random_value = np.random.uniform(y_pos_tr[j] - dev, y_pos_tr[j] + dev)
                y_pos_tr[j] = random_value
        elif thick_strip == 4:
            y_pos_og = y_pos_T1_and_T3.copy()
            y_pos_tr = y_pos_T1_and_T3.copy()
            for k in range(len(y_pos_og)):
                dev = y_width_T1_and_T3[k] * uncertain_factor_og
                random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
                y_pos_og[k] = random_value
                
                dev = y_width_T1_and_T3[k] * uncertain_factor_tr
                random_value = np.random.uniform(y_pos_tr[k] - dev, y_pos_tr[k] + dev)
                y_pos_tr[k] = random_value
        
        Q_og = Q
        Q_trans = Q
        
        y_og = np.sum(y_pos_og * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
        y_trans = np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
        
        if thick_strip == 1:
            y_long_T2_and_T4_og.append(y_og)
            y_long_T2_and_T4_trans.append(y_trans)
        elif thick_strip == 4:
            y_long_T1_and_T3_og.append(y_og)
            y_long_T1_and_T3_trans.append(y_trans)

# Histogram
hist_1d_list([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 400, "Shared charge case. Linear and Transformed Y, T1 and T3", "Y / mm", "shared_linear_trans_T1_T3", colors=["red", "blue"], labels=["Linear", "Transformed"])
hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 400, "Shared charge case. Linear and Transformed Y, T2 and T4", "Y / mm", "shared_linear_trans_T2_T4", colors=["red", "blue"], labels=["Linear", "Transformed"])





# -----------------------------------------------------------------------------
# The three cases
# -----------------------------------------------------------------------------

# Parameters -------------------------------
# Red
uncertain_factor_og = 0.485 # 0.49 was OK

# Blue
uncertain_factor_tr = 0.39 # 0.305 is nice, but i was using 0.4
transf_exp = 0.1 # 0.225 was nice, but i was using 0.1
transformed_charge_crosstalk_bound = 1.15 # 1.25 was nice, but i was using 1.1
# -----------------------------------------

def transformation(Q, exp):
    value = Q**exp
    return value

long_Q = np.ravel(loaded_Q)
long_Q = long_Q[ long_Q > 0 ]
hist_1d(long_Q, "auto", "Original charge", "Q / ToT ns", "og_q")
hist_1d(transformation(long_Q,transf_exp), "auto", "Transformed charge", "Q / ToT ns", "og_q")

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([63, 63, 63, 86]) # 85
y_width_T2_and_T4 = np.array([86, 63, 63, 63])

Q = loaded_Q

y_long_T1_and_T3_og = []
y_long_T2_and_T4_og = []
y_long_T1_and_T3_trans = []
y_long_T2_and_T4_trans = []

ext = 0
for i in range(len(Q)):
    thick_strip = loaded_strip[i].copy()
    Q = loaded_Q[i,:].copy()
    Q_og = Q
    
    # Single strip
    if np.count_nonzero(Q) <= 1:
        Q_trans = Q
        
        if thick_strip == 1:
            y_pos_og = y_pos_T2_and_T4.copy()
            y_pos_tr = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos_og)):
                dev = y_width_T2_and_T4[j] * uncertain_factor_og
                random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
                y_pos_og[j] = random_value
                
                dev = y_width_T2_and_T4[j] * uncertain_factor_tr
                random_value = np.random.uniform(y_pos_tr[j] - dev, y_pos_tr[j] + dev)
                y_pos_tr[j] = random_value
        elif thick_strip == 4:
            y_pos_og = y_pos_T1_and_T3.copy()
            y_pos_tr = y_pos_T1_and_T3.copy()
            for k in range(len(y_pos_og)):
                dev = y_width_T1_and_T3[k] * uncertain_factor_og
                random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
                y_pos_og[k] = random_value
                
                dev = y_width_T1_and_T3[k] * uncertain_factor_tr
                random_value = np.random.uniform(y_pos_tr[k] - dev, y_pos_tr[k] + dev)
                y_pos_tr[k] = random_value
                
    # Double strip
    elif np.count_nonzero(Q) > 1:
        Q_trans = transformation(Q, transf_exp)
        
        q_tr_nz = Q_trans[Q_trans != 0]
        q_tr_nz = sorted(q_tr_nz)
        
        # Shared
        if q_tr_nz[0] > transformed_charge_crosstalk_bound:
            Q_og = (Q == np.max(Q)) * Q
            
            if thick_strip == 1:
                y_pos_og = y_pos_T2_and_T4.copy()
                y_pos_tr = y_pos_T2_and_T4.copy()
                
                for j in range(len(y_pos_og)):
                    dev = y_width_T2_and_T4[j] * uncertain_factor_og
                    random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
                    y_pos_og[j] = random_value
            elif thick_strip == 4:
                y_pos_og = y_pos_T1_and_T3.copy()
                y_pos_tr = y_pos_T1_and_T3.copy()
                
                for k in range(len(y_pos_og)):
                    dev = y_width_T1_and_T3[k] * uncertain_factor_og
                    random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
                    y_pos_og[k] = random_value
        
        # Crosstalk
        if q_tr_nz[0] <= transformed_charge_crosstalk_bound:
            Q_og = (Q == np.max(Q)) * Q
            Q_trans = (Q == np.max(Q)) * Q
            
            if thick_strip == 1:
                y_pos_og = y_pos_T2_and_T4.copy()
                y_pos_tr = y_pos_T2_and_T4.copy()
                for j in range(len(y_pos_og)):
                    dev = y_width_T2_and_T4[j] * uncertain_factor_og
                    random_value = np.random.uniform(y_pos_og[j] - dev, y_pos_og[j] + dev)
                    y_pos_og[j] = random_value
                    
                    dev = y_width_T2_and_T4[j] * uncertain_factor_tr
                    random_value = np.random.uniform(y_pos_tr[j] - dev, y_pos_tr[j] + dev)
                    y_pos_tr[j] = random_value
            elif thick_strip == 4:
                y_pos_og = y_pos_T1_and_T3.copy()
                y_pos_tr = y_pos_T1_and_T3.copy()
                for k in range(len(y_pos_og)):
                    dev = y_width_T1_and_T3[k] * uncertain_factor_og
                    random_value = np.random.uniform(y_pos_og[k] - dev, y_pos_og[k] + dev)
                    y_pos_og[k] = random_value
                    
                    dev = y_width_T1_and_T3[k] * uncertain_factor_tr
                    random_value = np.random.uniform(y_pos_tr[k] - dev, y_pos_tr[k] + dev)
                    y_pos_tr[k] = random_value
    else:
        ext += 1
        continue
    
    y_og = np.sum(y_pos_og * Q_og / np.sum(Q_og)) / np.sum(Q_og / np.sum(Q_og))
    y_trans = np.sum(y_pos_tr * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    
    if thick_strip == 1:
        y_long_T2_and_T4_og.append(y_og)
        y_long_T2_and_T4_trans.append(y_trans)
    elif thick_strip == 4:
        y_long_T1_and_T3_og.append(y_og)
        y_long_T1_and_T3_trans.append(y_trans)

print(f"{ext} cases not taken into account.")

hist_1d_list_subplt([y_long_T1_and_T3_og, y_long_T1_and_T3_trans], 350, f"Y position, M1 and M3, {len(y_conc_trans)} events", "Y / mm", "all_cases_linear_trans", colors=["red", "blue"], labels=["Uniform from strip center", "Using the charge sharing"])

# hist_1d_list([y_long_T2_and_T4_og, y_long_T2_and_T4_trans], 300, f"Y position, M2 and M4 , {len(y_conc_trans)} events", "Y / mm", "all_cases_linear_trans", colors=["red", "blue"], labels=["Uniform from strip center", "Using the charge sharing"])

# y_conc_og = np.concatenate((y_long_T1_and_T3_og, y_long_T2_and_T4_og))
# y_conc_trans = np.concatenate((y_long_T1_and_T3_trans, y_long_T2_and_T4_trans))
# cond = (abs(y_conc_og) < 200) & (abs(y_conc_trans) < 200)
# hist_1d_list([y_conc_og[cond], y_conc_trans[cond]], 300, f"Y position , {len(y_conc_trans)} events", "Y / mm", "all_cases_linear_trans", colors=["red", "blue"], labels=["Uniform from strip center", "Using the charge sharing"])
# hist_1d_list_subplt([y_conc_og[cond], y_conc_trans[cond]], 300, f"Y position , {len(y_conc_trans)} events", "Y / mm", "all_cases_linear_trans", colors=["red", "blue"], labels=["Uniform from strip center", "Using the charge sharing"])








# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Prepared to implement -------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# The old one -------------------------------------------------------
# -----------------------------------------------------------------------------

# This is ancillary only due to this script
Q = loaded_Q
y_long = []

# -----------------------------------------------------------------------------
# This should be on the script already
# -----------------------------------------------------------------------------
y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])
y_width_T1_and_T3 = np.array([64, 64, 64, 99])
y_width_T2_and_T4 = np.array([99, 64, 64, 64])

def transformation(Q, exp):
    value = Q**exp
    return value

# Shared charge
transf_exp = 0.22 # 0.22 was nice
pos_uncer_corr = 0.2
# Single strips
uncertain_factor = 0.5 # 0.5 was fantastic
# -----------------------------------------------------------------------------

for i in range(len(Q)):
    thick_strip = loaded_strip[i]
    Q = loaded_Q[i,:].copy()
    Q_og = Q
    
    # HERE STARTS THE IMPORTANT PART TO IMPLEMENT -----------------------------
    if np.count_nonzero(Q) > 1:
        if thick_strip == 1:
            y_pos = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos)):
                dev = y_width_T2_and_T4[j] * pos_uncer_corr
                y_pos[j] = 10 + np.random.normal(y_pos[j], dev)
        elif thick_strip == 4:
            y_pos = y_pos_T1_and_T3.copy()
            for j in range(len(y_pos)):
                dev = y_width_T1_and_T3[j] * pos_uncer_corr
                y_pos[j] = np.random.normal(y_pos[j], dev)
        Q_trans = transformation(Q, transf_exp)
        pos_scale_factor = 1.3
            
    if np.count_nonzero(Q) <= 1:
        if thick_strip == 1:
            y_pos = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos)):
                dev = y_width_T2_and_T4[j] * uncertain_factor
                y_pos[j] = np.random.normal(y_pos[j], dev)
        elif thick_strip == 4:
            y_pos = y_pos_T1_and_T3.copy()
            for j in range(len(y_pos)):
                dev = y_width_T1_and_T3[j] * uncertain_factor
                y_pos[j] = np.random.normal(y_pos[j], dev)
                
        Q_trans = Q
        pos_scale_factor = 0.7
        
    y = pos_scale_factor * np.sum(y_pos * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    # HERE STOPS THE IMPORTANT PART TO IMPLEMENT ------------------------------
    y_long.append(y)
    
hist_1d(y_long, 500, "OLD VERSION!!!!!!!! All cases, all RPCs, Y positions", "Y / mm", "all_cases_all_rpcs")




# -----------------------------------------------------------------------------
# The new one -------------------------------------------------------
# -----------------------------------------------------------------------------

# This is ancillary only due to this script
y_long = []

# -----------------------------------------------------------------------------
# This should be on the script already
# -----------------------------------------------------------------------------

# Parameters -------------------------------
uncertain_factor = 0.4 # 0.305 is nice
transf_exp = 0.1 # 0.225 was nice
transformed_charge_crosstalk_bound = 1.1 # 1.25 was nice

def transformation(Q, exp):
    value = Q**exp
    return value

y_pos_T1_and_T3 = np.array([ 105.5,   42.5,  -20.5, -101. ])
y_pos_T2_and_T4 = np.array([  88. ,    7.5,  -55.5, -118.5])

y_width_T1_and_T3 = np.array([64, 64, 64, 85])
y_width_T2_and_T4 = np.array([85, 64, 64, 64])

for i in range(len(loaded_Q)):
    thick_strip = loaded_strip[i].copy()
    Q = loaded_Q[i,:].copy()
    Q_og = Q
    
    # -----------------------------------------
    # Single strip
    if np.count_nonzero(Q) <= 1:
        Q_trans = Q
        
        if thick_strip == 1:
            y_pos = y_pos_T2_and_T4.copy()
            for j in range(len(y_pos)):
                dev = y_width_T2_and_T4[j] * uncertain_factor
                random_value = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
                y_pos[j] = random_value
        elif thick_strip == 4:
            y_pos = y_pos_T1_and_T3.copy()
            for k in range(len(y_pos)):
                dev = y_width_T1_and_T3[k] * uncertain_factor
                random_value = np.random.uniform(y_pos[k] - dev, y_pos[k] + dev)
                y_pos[k] = random_value
                
    # Double strip
    elif np.count_nonzero(Q) > 1:
        Q_trans = transformation(Q, transf_exp)
        
        q_tr_nz = Q_trans[Q_trans != 0]
        q_tr_nz = sorted(q_tr_nz)
        
        # Shared
        if q_tr_nz[0] > transformed_charge_crosstalk_bound:
            if thick_strip == 1:
                y_pos = y_pos_T2_and_T4.copy()
                for j in range(len(y_pos)):
                    dev = y_width_T2_and_T4[j] * uncertain_factor
                    random_value = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
                    y_pos[j] = random_value
            elif thick_strip == 4:
                y_pos = y_pos_T1_and_T3.copy()
                for k in range(len(y_pos)):
                    dev = y_width_T1_and_T3[k] * uncertain_factor
                    random_value = np.random.uniform(y_pos[k] - dev, y_pos[k] + dev)
                    y_pos[k] = random_value
        
        # Crosstalk
        if q_tr_nz[0] <= transformed_charge_crosstalk_bound:
            Q_trans = (Q == np.max(Q)) * Q
            
            if thick_strip == 1:
                y_pos = y_pos_T2_and_T4.copy()
                for j in range(len(y_pos)):
                    dev = y_width_T2_and_T4[j] * uncertain_factor
                    random_value = np.random.uniform(y_pos[j] - dev, y_pos[j] + dev)
                    y_pos[j] = random_value
            elif thick_strip == 4:
                y_pos = y_pos_T1_and_T3.copy()
                for k in range(len(y_pos)):
                    dev = y_width_T1_and_T3[k] * uncertain_factor
                    random_value = np.random.uniform(y_pos[k] - dev, y_pos[k] + dev)
                    y_pos[k] = random_value
    
    y = np.sum(y_pos * Q_trans / np.sum(Q_trans)) / np.sum(Q_trans / np.sum(Q_trans))
    # -----------------------------------------
    
    y_long.append(y)
    

hist_1d(y_long, 400, "All cases, all RPCs, Y positions", "Y / mm", "all_cases_all_rpcs")








