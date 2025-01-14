import struct
import numpy as np

# Convert to numpy arrays or Pandas DataFrame if needed
import pandas as pd

# Parameters from the Fortran code
nsipms_bot = 6
nsipms_top = 1
nsipms_lat = 2

# Calculate the total number of floats to be read for each event
floats_per_event = nsipms_bot * nsipms_bot + nsipms_top * nsipms_top + nsipms_lat + 1 + 3

# photon_index and cristal_index are integers, so 2 additional integers
total_values_per_event = floats_per_event + 2

# Total bytes to read per event (floats are 4 bytes, integers are 4 bytes)
bytes_per_event = total_values_per_event * 4

def read_binary_file(file_path):
    """Reads binary data from the file and returns a structured list of events."""
    
    event_data = []  # This will store all the events

    with open(file_path, 'rb') as f:
        while True:
            # Read the number of bytes corresponding to one event
            event_bytes = f.read(bytes_per_event)
            
            if not event_bytes:
                break  # Exit loop when no more data
            
            # Read and unpack the floating point and integer values
            floats_format = f'{floats_per_event}f'  # Format for float values
            int_format = '2I'  # Format for two unsigned integers
            
            # Unpack the floats (SiPM_bot, SiPM_top, SiPM_lat, en_hit, xyz_hit)
            unpacked_floats = struct.unpack(floats_format, event_bytes[:floats_per_event * 4])
            
            # Unpack the photon_index and cristal_index as integers
            photon_index, cristal_index = struct.unpack(int_format, event_bytes[floats_per_event * 4:])
            
            # Organize the unpacked data into a list for each event
            signal_bot = unpacked_floats[:nsipms_bot * nsipms_bot]
            signal_top = unpacked_floats[nsipms_bot * nsipms_bot:nsipms_bot * nsipms_bot + nsipms_top * nsipms_top]
            signal_lat = unpacked_floats[nsipms_bot * nsipms_bot + nsipms_top * nsipms_top:nsipms_bot * nsipms_bot + nsipms_top * nsipms_top + nsipms_lat]
            en_hit = unpacked_floats[nsipms_bot * nsipms_bot + nsipms_top * nsipms_top + nsipms_lat]
            xyz_hit = unpacked_floats[nsipms_bot * nsipms_bot + nsipms_top * nsipms_top + nsipms_lat + 1:]
            
            # Append all data of the current event into the list
            event_data.append({
                'signal_bot': signal_bot,
                'signal_top': signal_top,
                'signal_lat': signal_lat,
                'en_hit': en_hit,
                'xyz_hit': xyz_hit,
                'photon_index': photon_index,
                'cristal_index': cristal_index
            })
    
    return event_data

# Example usage:
file_path = 'SiPM_hit_Poi_6x6.raw'  # Replace with the actual binary file path
events = read_binary_file(file_path)


# Flatten each event dictionary and convert to a DataFrame
event_df = pd.DataFrame(events)
# Print the first few rows of the DataFrame
print(event_df.head())





def flatten_event(row):
    # Each part corresponds to a fixed size component of your row
    sipms_signal = row['signal_bot']   # (nsipms_bot * nsipms_bot) length
    sipm_top = row['signal_top']       # single value
    sipm_lat = row['signal_lat']       # 2 values
    en_hit = [row['en_hit']]           # single value
    xyz_hit = row['xyz_hit']           # 3 values
    photon_index = [row['photon_index']]  # single integer
    cristal_index = [row['cristal_index']] # single integer
    
    # Concatenate all the components into a single flat array
    return np.hstack([sipms_signal, sipm_top, sipm_lat, en_hit, xyz_hit, photon_index, cristal_index])

# Apply this flattening function to all rows in event_df
flattened_data = np.array([flatten_event(row) for _, row in event_df.iterrows()])

# Print the resulting flattened matrix
print(flattened_data)