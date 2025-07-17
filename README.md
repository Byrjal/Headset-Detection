# Headset-Detection

This project takes `.csv` data files from scattering parameters measured from the headset (using a network analyser) and utilises different machine learning models to detect or differentiate between the two states: the user wearing the headset or not (e.g. placed on a table).

The current software is configured to use S11 amplitude, focus on a narrowed frequency range (1.880-1.930 GHz), and perform binary classification. During development, different configurations were tested, including the full frequency range, various downsampling steps, and the use of additional S-parameters and phase data. However, the main goal of this software was to evaluate whether just the S11 amplitude in a narrowed frequency band would be sufficient for accurate classification - and results indicate that it is. All machine learning models achieved high accuracy, with the only exception of the Naive Bayes classifier (which is the most simple of them all). Downsampling was also tested and did not significantly degrade performance.

The software is not limited to the current configuration, as parameter configurations also supports full or partial frequency ranges, downsampling at custom steps, different machine learning models, cross-validation, and work with other columns from the data (e.g. phase or both amplitude and phase). However, generalisation to other headsets or data measurements can be a limitation, but the software is expected to work well as long as S11 remains the primary input.


## Getting Started
The project consists of:
- `main.py`: Loads data, trains, and evaluates models. Calls the following three files.
- `data_loader.py`: Two functions to read and preprocess `.csv` data.
- `models.py`: Defines and returns the machine learning models.
- `training_evaluation.py`: Training, evaluation, and cross-validation.
- Data
    - DUT1
      - HeadsetOff
      - HeadsetOn
- Plots: Folder mostly for confusion matrix plots.

Make sure to have the following packages installed:
```Python
pip install numpy scikit-learn matplotlib
```

To run the project, run the `main.py`.


### main.py
This script serves as the main entry point for the project. It performs the following tasks:
1. Optionally settings:
    1. Downsample with a step rate.
    2. Limit the frequency range with start and end indices.
    3. Choose which S-parameter columns to use (freq, amp, phase).
    4. The use of cross-validation and the number of folds.
    5. Show filenames of misclassified samples.
3. Load data using `data_loader.py`.
4. Shuffle and split the data into training and testing sets (depending on whether cross-validation is enabled).
5. Train and evaluate multiple machine learning models.
6. Model performance metrics printed to the console.
7. Optionally plot and save visual results.
8. Misclassified examples (if enabled) printed for further analysis.

Key Considerations:
- If `USE_STEP = True`, the data is downsampled. The `STEP` tells how many samples.
  - `STEP = 1`: No change, every sample are retained.
  - `STEP = 2`: Every second sample is kept, and so on.
- If `USE_FREQ_RANGE = True`, the `ROW_START` and `ROW_END` specify the index. For the DUT1 dataset, [889, -1071] corresponds to the range 1.880-1.930 GHZ.
- .csv files can vary in structure, so be aware the difference when loading data using `data_loader.py` and the Configuration Parameters (especially `COLUMNS`).
- The `np.random.seed(42)` ensures deterministic behavior and is essential for reproducibility. If it is removed, randomness is introduced into results.
- The `train_test_split()` uses a 70/30 split (standard) with `stratify=y` to maintain class balance. However, with imbalanced datasets, stratification is not a complete solution (try e.g. also data augmentation).
- The `cross-validation` and `train_test_split` are mutually exclusive in this implementation.
  - When `USE_CV = False`, a single train/test split is run. When `USE_CV = True`, the data is split into multiple folds and each one are tested. The `FOLDS` sets how many folds to use. If `FOLDS = 9`, the data is split into 9 parts: each fold is used once as a test set while the remaining 8 folds are used for training, and this process is repeated 9 times.


### data_loader.py
This file will:
1. Read `.csv` files from the subdirectories `HeadsetOff` and `HeadsetOn` wihtin the chosen directory.
2. Optionally:
   1. Downsample the data using a step size.
   2. Limit the frequency range.
   3. Select specific columns from the `.csv` data.
3. Flatten and store each file's data as a 1D feature vector.
4. Return combined feature vectors, labels, and filenames for classification tasks.

Key Considerations for `read_csv_files()`:
- The first three header rows in the `.csv` files are skipped using `skip_header = 3`, as these rows do not contain any data. Be aware of the possible differences between datasets (DUT1 and future new data).
- The data is flatten using `data.T.flatten()` to convert the 2D matrix into a 1D feature vector. The data is loaded with the shape `(number_samples, number_columns)`. So for the DUT1 dataset when using both the amp and phase columns, the shape is `(2001, 2)`. After flattening, the shape becomes `(4002,)`, as the machine learning models in `scikit-learn` expect each sample to be a 1D feature vector.
- Each file is returned as a tuple: `(flattened_data_vector, file_name)`.

Key Considerations for `load_dataset()`:
- It combines the data and assigns the labels, 0 for `HeadsetOff` and 1 for `HeadsetOn`.
- It returns the feature matrix `X` of shape `[n_samples, n_features]` (fx for DUT1 `(96, 2001)` with 96 files and 2001 amplitude values per file), a list of the labels `y` (0 and 1), and a list of the corresponding filenames.

 
### models.py
This file returns a dictionary of the used machine learning classification models. Each model is initialised with default or explicitly chosen parameters. The machine learning models used:
1. [Naive Bayes](https://scikit-learn.org/stable/api/sklearn.naive_bayes.html): The most simple. Good as a baseline, but limited when feature interactions matter.
2. [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): `make_pipeline` is used to ensure consistent feature scaling, which is crucial for models sensitive to scale. `StandardScaler` is used to normalise features. The `max_iter = 200` ensures the model converges during training. Commonly used due to its efficiency.
3. [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html): `make_pipeline` is crucial for distance-based models. Also `StandardScaler` is used, and `n_neighbors = 3` (standard to choose odd numbers, often 3, max 9). 
4. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html): Easy to interpret and does not require feature scaling. Can overfit if not regularised (e.g. `max_depth`).
5. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): More robust and generalisable than a single decision tree. Number of trees `n_estimators = 100` and limits of tree size to reduce overfitting `max_depth = 10`.

All models can be tuned using `GridSearchCV` , `RandomisedSearchCV`, or sometimes manually changing the parameters.


### training_evaluation.py
This file contains two functions used for evaluating machine learning models:
1. `train_and_evaluate()`: for standard train/test split evaluation.<br> This function trains a given classifier and evaluates it using:
   - Train/test accuracy metrics
   - Confusion matrix visualisation
   - Optional export of plots to file
   - Optional listing of misclassified samples
2. `cross_validation()`: for cross-validation evaluation.<br>
This function performs k-fold cross-validation on the given model and prints the accuracy statistics. This is useful for estimating the model's generalisation performance across different subsets of the dataset.


## Used Python Libraries
### NumPy
[NumPy](https://numpy.org/doc/stable/) is the fundamental package for scientific computing in Python. It provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
```python
import numpy as np
```

In this project, NumPy is used for:
- `np.random.seed()`: Sets a seed for random number generators.
- `np.genfromtxt()`: Load data from a text file.
- `np.array()`: Creates arrays.
- `np.random.permutation()`: Randomly permute an array.
- `np.where()`: Finds indices where a condition is true.


### os
[os](https://docs.python.org/3/library/os.html) is a built-in Python module that provides a portable way of using operating system dependent functionality. 
```python
import os
```

In this project, os are used for:
- `os.path.join()`: Constructs a full path by concatenating one or more path components while automatically inserting the appropriate path separator.
- `os.listdir()`: Returns a list of the names of the entries in a directory.


### Matplotlib (.pyplot)
[Matplotlib](https://matplotlib.org/) is Python's core plotting library, used to create static, animated, and interactive visualisations in Python. Most of the matplotlib utilities lies under the `.pyplot` submodule, and are usually imported under the `plt` alias.
```python
import matplotlib.pyplot as plt
```

In this project, matplotlib is used for:
- `plt.cm.Blues`: Builtin colormap (used for confusion matrix).
- `plt.title()`: Sets plot title.
- `plt.savefig()`: Saves the current figure (default `.png`).
- `plt.show()`: Displays plot on screen.


### scikit-learn
[Scikit-learn](https://scikit-learn.org/dev/index.html) is an open-source machine learning library built on Numpy, Scipy and matplotlib. Often also called sklearn.
```python
import sklearn
```

In this project, scikit-learn is used for:
- `GausianNB()`: The Gaussian Naive Bayes model. Imported from sklearn.naive_bayes.
- `LogisticRegression`: The logistic regression model. Imported from sklearn.linear_model.
- `KNeighborsClassifier`: The KNN model. Imported from sklearn.neighbors.
- `DecisionTreeClassifier`: The decision tree model. Imported from sklearn.tree.
- `RandomForestClassifier`: The random forest model. Imported from sklearn.ensemble.
- `train_test_split`: Split arrays/matrices into random train and test subsets. Imported from sklearn.model_selection.
- `cross_val_score`: Evaluates a score by cross-validation. Imported from sklearn.model_selection.
- `accuracy_score`: Accuracy classification score. Imported from sklearn.metrics.
- `confusion_matrix`: Compute confusion matrix to evaluate the accuracy. Imported from sklearn.metrics.
- `ConfusionMatrixDisplay`: Visualise the confusion matrix. Imported from sklearn.metrics.
- `StandardScaler`: Standardise features by removing the mean and scaling to unit variance. Imported from sklearn.preprocessing.
- `make_pipeline`: Constructs a pipeline (a sequence of data transformers with an optional final predictor) from the given estimators. Imported from sklearn.pipeline.


## Contributions
This project originated as part of a 9th semester internship by Stine Byrjalsen and Julie Timmermann Werge, in collaboration with RTX A/S. Since then, improvements have been made by Stine Byrjalsen as part of a student assistant position at RTX A/S. 
