"""
Main script to load dataset, train and evaluate multiple machine learning models,
with options for frequency range, downsampling, cross-validation, and misclassifications.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import read_csv_files, load_dataset
from models import get_models
from training_evaluation import train_and_evaluate, cross_validation

# ======================= Configuration Parameters =======================
DATASET_NAME = "Data/OldData/DUT1"
dataset_name = DATASET_NAME.split('/')[-1]  # Extract 'DUT1' for logging

USE_STEP = False            # True = use downsampled, False = full sample rate
STEP = 2                    # Downsampling step (only used if USE_STEP is True)
USE_FREQ_RANGE = False      # True = use smaller freq range, False = use full freq range
ROW_START = 889             # Start index for freq range (only used if USE_FREQ_RANGE is True)
ROW_END = -1071             # End index (only used if USE_FREQ_RANGE is True; negative means count from end)
COLUMNS = slice(1, 2)       # Which column(s) to use (0=freq, 1=amplitude, 2=phase)
USE_CV = False              # True = run cross-validation
FOLDS = 9                   # Number of folds for CV (only used if USE_CV = True)
PRINT_MISCLASSIFIED = True  # True = prints the misclassified filenames

np.random.seed(42)

# ======================= Load and Prepare Dataset =======================
X, y, file_names = load_dataset(DATASET_NAME,  use_step=USE_STEP, step=STEP, use_freq_range=USE_FREQ_RANGE, row_start=ROW_START, row_end=ROW_END, col_slice=COLUMNS)

# Shuffle data
shuffled_indices = np.random.permutation(len(X))
X, y, file_names = X[shuffled_indices], y[shuffled_indices], file_names[shuffled_indices]

# Split into train/test
X_train, X_test, y_train, y_test, file_names_train, file_names_test = train_test_split(
    X, y, file_names, test_size=0.3, stratify=y, random_state=42
)

# ======================= Getting Frequency Range (for display only) =======================
freq_data = read_csv_files(directory=DATASET_NAME,
                            folder_type='HeadsetOff',
                            use_step=False,
                            step=1,
                            use_freq_range=False,
                            row_start=0,
                            row_end=None,
                            col_slice=slice(0, 1))
freq_vector = freq_data[0][0] # Get frequency values from first file

# Compute frequency bounds
freq_start = freq_vector[ROW_START]
end_index = ROW_END if ROW_END is not None else len(freq_vector)
if end_index < 0:
    end_index = len(freq_vector) + end_index
freq_end = freq_vector[end_index]

# ======================= Display Configuration Info =======================
print(f"{'=' * 20} Configurations {'=' * 20}")
print(f"Dataset: {dataset_name}")
print(f"Step Size: {STEP if USE_STEP else 'Full'}")
print(f"Freq Range: {freq_start/10e8:.3f} - {freq_end/10e8:.3f} GHz"
      if USE_FREQ_RANGE
      else f"Freq Range: {freq_vector[0]/10e8} - {freq_vector[-1]/10e8} GHz")
if USE_CV:
    print(f"Number of Folds: {FOLDS}")
print(f" ")

# ======================= Train and Evaluate Models =======================
models = get_models()
for name, model in models.items():
    if USE_CV:
        cross_validation(model, name, X, y, FOLDS)
    else:
        train_and_evaluate(model, name,
            X_train, y_train, X_test, y_test,
            file_name_test=file_names_test, dataset_name=DATASET_NAME,
            use_step=USE_STEP, step=STEP,
            use_freq_range=USE_FREQ_RANGE,
            save_plot=True,
            plot_dir="Plots/",
            print_misclassified=PRINT_MISCLASSIFIED)