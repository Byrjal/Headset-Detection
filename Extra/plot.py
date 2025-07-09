import numpy as np
import matplotlib.pyplot as plt

# Choose DUT1 or DUT2
directory = "DUT1"


# Load data from datafiles (HeadsetOn)
file_path_on_h1 = "DUT1/HeadsetOn/DUT1_H1_1.csv"
file_path_on_h2 = "DUT1/HeadsetOn/DUT1_H2_2.csv"
file_path_on_jw = "DUT2/HeadsetOn/DUT2_JW_3.csv"
file_path_on_sb = "DUT2/HeadsetOn/DUT2_SB_4.csv"
file_path_on_syn = "DUT1/SyntheticData/HeadsetOn/SyntheticSample_HeadsetOn_5.csv"
data_on_h1 = np.genfromtxt(file_path_on_h1, delimiter=';', skip_header=3)
data_on_h2 = np.genfromtxt(file_path_on_h2, delimiter=';', skip_header=3)
data_on_jw = np.genfromtxt(file_path_on_jw, delimiter=';', skip_header=3)
data_on_sb = np.genfromtxt(file_path_on_sb, delimiter=';', skip_header=3)
data_on_syn = np.genfromtxt(file_path_on_syn, delimiter=',', skip_header=1)


# Extract columns
column_1_on_h1 = data_on_h1[:, 0]
column_2_on_h1 = data_on_h1[:, 1]
column_1_on_h2 = data_on_h2[:, 0]
column_2_on_h2 = data_on_h2[:, 1]
column_1_on_jw = data_on_jw[:, 0]
column_2_on_jw = data_on_jw[:, 1]
column_1_on_sb = data_on_sb[:, 0]
column_2_on_sb = data_on_sb[:, 1]
column_1_on_syn = data_on_syn[:, 0]
#column_2_on_syn = np.linspace(1e9, 3e9, 51)
full_frequency = np.linspace(1e9, 3e9, 2001)
column_2_on_syn = full_frequency[:]


# Load data from a datafile (HeadsetOff)
file_path_off_h1 = "DUT1/HeadsetOff/DUT1_H1_Table_1.csv"
file_path_off_h2 = "DUT1/HeadsetOff/DUT1_H2_Table_2.csv"
file_path_off_jw = "DUT2/HeadsetOff/DUT2_JW_Table_3.csv"
file_path_off_sb = "DUT2/HeadsetOff/DUT2_SB_Table_4.csv"
file_path_off_syn = "DUT1/SyntheticData/HeadsetOff/SyntheticSample_HeadsetOff_5.csv"
data_off_h1 = np.genfromtxt(file_path_off_h1, delimiter=';', skip_header=3)
data_off_h2 = np.genfromtxt(file_path_off_h2, delimiter=';', skip_header=3)
data_off_jw = np.genfromtxt(file_path_off_jw, delimiter=';', skip_header=3)
data_off_sb = np.genfromtxt(file_path_off_sb, delimiter=';', skip_header=3)
data_off_syn = np.genfromtxt(file_path_off_syn, delimiter=',', skip_header=1)

# Extract columns
column_1_off_h1 = data_off_h1[:, 0]
column_2_off_h1 = data_off_h1[:, 1]
column_1_off_h2 = data_off_h2[:, 0]
column_2_off_h2 = data_off_h2[:, 1]
column_1_off_jw = data_off_jw[:, 0]
column_2_off_jw = data_off_jw[:, 1]
column_1_off_sb = data_off_sb[:, 0]
column_2_off_sb = data_off_sb[:, 1]
column_1_off_syn = data_off_syn[:, 0]
column_2_off_syn = full_frequency[:]


def find_min_amplitude_and_frequency(data_dict):
    """
    This function calculates and prints the minimum amplitude and corresponding frequency
    for each dataset in the input dictionary.

    Parameters:
    - data_dict (dict): A dictionary where the keys are the dataset names (e.g. 'h1', 'h2', 'jw', 'sb', 'syn')
      and the values are tuples of (frequency, amplitude) data arrays.

    For synthetic data, the amplitude and frequency columns are reversed.
    """

    for label, (freq_data, amplitude_data) in data_dict.items():
        # For synthetic data, swap frequency and amplitude columns
        amplitude = amplitude_data
        frequency = freq_data

        # Find min amplitude and corresponding frequency
        min_amplitude = np.min(amplitude)
        min_freq = frequency[np.argmin(amplitude)]

        # Print the result
        print(f"Minimum Amplitude for {label}: {min_amplitude} at {min_freq / 1e9:.3f} GHz")


data_dict = {
    'h1': (data_on_h1[880:-1070, 0], data_on_h1[880:-1070, 1]),
    'h2': (data_on_h2[880:-1070, 0], data_on_h2[880:-1070, 1]),
    'jw': (data_on_jw[880:-1070, 0], data_on_jw[880:-1070, 1]),
    'sb': (data_on_sb[880:-1070, 0], data_on_sb[880:-1070, 1]),
    'syn': ((np.linspace(1e9, 3e9, 2001)[880:-1070]), data_on_syn[880:-1070, 0])
}

data_dict_2 = {
    'h1': (data_off_h1[880:-1070, 0], data_off_h1[880:-1070, 1]),
    'h2': (data_off_h2[880:-1070, 0], data_off_h2[880:-1070, 1]),
    'jw': (data_off_jw[880:-1070, 0], data_off_jw[880:-1070, 1]),
    'sb': (data_off_sb[880:-1070, 0], data_off_sb[880:-1070, 1]),
    'syn': ((np.linspace(1e9, 3e9, 2001)[880:-1070]), data_on_syn[880:-1070, 0])
}

find_min_amplitude_and_frequency(data_dict)



# Plot
plt.figure(figsize=(10, 6))
#plt.plot(column_1_on_h1, column_2_on_h1, label='Headset On', color='g')
#plt.plot(column_1_on_h2, column_2_on_h2, color='g')
#plt.plot(column_1_on_jw, column_2_on_jw, label='Headset On', color='g')
plt.plot(column_1_on_sb, column_2_on_sb, label='Headset On', color='g')
#plt.plot( column_2_on_syn, column_1_on_syn, color='y')
#plt.plot(column_1_off_h1, column_2_off_h1, label='Headset off', color='b')
#plt.plot(column_1_off_h2, column_2_off_h2, color='b')
#plt.plot(column_1_off_jw, column_2_off_jw, label='Headset off', color='b')
plt.plot(column_1_off_sb, column_2_off_sb, label='Headset off', color='b')
#plt.plot( column_2_off_syn, column_1_off_syn, label='Synthetic', color='y')

# Vertical lines
#plt.axvline(x=1.884e9, color='r')
#plt.axvline(x=1.904e9, color='r')
#plt.axvline(x=1.924e9, color='r')

plt.xlabel('Frequency [GHz]')
plt.ylabel('Amplitude [dB]')
plt.title('Amplitude Plot for DUT2')
plt.legend()
plt.savefig(f'DUT2_AP_oneInstance.pdf')
plt.show()

