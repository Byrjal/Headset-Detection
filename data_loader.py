import numpy as np
import os

def read_csv_files(directory, folder_type, use_step, step, use_freq_range, row_start, row_end, col_slice):
    """
    Reads .csv files from a specific subdirectory.

    Parameters:
        directory (str): Base path to the dataset.
        folder_type (str): Subfolder name ('HeadsetOn' or 'HeadsetOff').
        use_step (bool): Whether to apply downsampling or use full sample rate.
        step (int): Step size for downsampling.
        use_freq_range (bool): Whether to use a specific frequency range.
        row_start (int): Starting row index for specific freq range.
        row_end (int): Ending row index for specific freq range.
        col_slice (slice): Slice object for selecting which column(s) from the dataset to use.

    Returns:
        list of tuples: Each tuple contains (flattened_data_vector, file_name).
    """
    subdirectory = os.path.join(directory, folder_type)
    data_list = []
    for file_name in os.listdir(subdirectory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(subdirectory, file_name)
            data = np.genfromtxt(file_path, delimiter=';', skip_header=3)

            # Apply downsampling and specific freq interval based on configuration
            if use_freq_range:
                data = data[row_start:row_end:step if use_step else 1, col_slice]
            else:
                data = data[::step if use_step else 1, col_slice]

            # Flatten the data
            stacked_columns = data.T.flatten()
            data_list.append((stacked_columns, file_name))
    return data_list


def load_dataset(dataset_name, use_step, step, use_freq_range, row_start, row_end, col_slice):
    """
    Loads and combines data from 'HeadsetOn' and 'HeadsetOff' folders.

    Returns:
        X (np.ndarray): Feature vectors.
        y (np.ndarray): Labels (0 for HeadsetOff, 1 for HeadsetOn).
        file_names (np.ndarray): Corresponding file names.
    """
    base_path = dataset_name

    # Load data
    data_headset_off = read_csv_files(base_path, folder_type='HeadsetOff', use_step=use_step, step=step, use_freq_range=use_freq_range, row_start=row_start, row_end=row_end, col_slice=col_slice)
    data_headset_on = read_csv_files(base_path, folder_type='HeadsetOn', use_step=use_step, step=step, use_freq_range=use_freq_range, row_start=row_start, row_end=row_end, col_slice=col_slice)

    # Combine datasets and assign labels
    combined_data = [(vec, 0, file_name) for vec, file_name in data_headset_off] + \
                    [(vec, 1, file_name) for vec, file_name in data_headset_on]

    # Split combined data into features, labels, and filenames
    X = np.array([x for x, _, _ in combined_data])
    y = np.array([y for _, y, _ in combined_data])
    file_names = np.array([f for _, _, f in combined_data])
    return X, y, file_names