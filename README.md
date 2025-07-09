# Headset-Detection

This project takes .csv data files from scattering parameters measured from the headset and utilises different machine learning models to detect or differentiate between the two states: the user wearing the headset or not (e.g. placed on a table).
Real-world use cases for implementing the detection in a headset can be:
- Pause/resume calls depending on whether the headset is on the head.
- It can reduce call misses.
- Auto-play/pause music or videos when user wears or removes the headset.
- Trigger auto power-off to conserve battery.

## Getting Started
The project consists of:
- `main.py`: Loads data, trains, and evaluates models. Calls the following three files.
- `data_loader.py`: Two functions to read and preprocess .csv data.
- `models.py`: Defines and returns the machine learning models.
- `training_evaluation.py`: Training, evaluation, and cross-validation.
- Data
  - OldData: Data collected during the 9th semester project at AAU.
    - DUT1
      - HeadsetOff
      - HeadsetOn
  - NewData: New data collected at RTX A/S in Summer.
    - ...
- Extra
  - `plot.py`: Extra visualisations.
  - `SMOTE.py`: Synthetic augmenting for imbalanced data.
- Plots: Folder mostly for confusion matrix plots.

Make sure to have the following packages installed:
```Python
pip install numpy scikit-learn matplotlib
```

To run the project, run the `main.py`.

### main.py
This file will:
1. Optionally:
    1. Downsample with a step rate.
    2. Limit the frequency range with start and end indices.
    3. Which S-parameter columns to use (freq, amp, phase).
    4. The use of cross-validation and the number of folds.
    5. Show filenames of misclassified samples.
3. Load data.
4. Shuffle and split the data into training and testing sets (depending on whether cross-validation is enabled).
5. Train and evaluate multiple machine learning models.
6. Model performance metrics printed to the console.
7. Optionally plot and save visual results.
8. Misclassified examples (if enabled) printed for further analysis.

Key considerations:
- If `USE_STEP = True`, the data is downsampled. The `STEP` tells how many samples.
  - `STEP = 1`: No change, every sample.
  - `STEP = 2`: Every second sample are kept, and so on.
- If `USE_FREQ_RANGE = True`, the `ROW_START` and `ROW_END` specify the index. For the DUT1 dataset, [889, -1071] corresponds to the range 1.880-1.930 GHZ.
- .csv files can vary in structure, so be aware the difference when loading data using `data_loader.py`.
- The `np.random.seed(42)` ensures deterministic behavior and is essential for reproducibility. If it is removed, randomness are introduced in production.
- The `train_test_split()` uses a 70/30 split (standard) with `stratify=y` to maintain class balance. However, with imbalanced datasets, stratification is not a complete solution (try e.g. also data augmentation).
- The `cross-validation` and `train_test_split` are mutually exclusive in this implementation.
  - When `USE_CV = False`, a single train/test split is run. When `USE_CV = True`, the data is split into multiple folds and each one are tested. The `FOLDS` sets how many folds to use. If `FOLDS = 9`, 8 parts are trained on and 1 are tested on, and it is repeated 9 times.



<!--
### data_loader.py
NumPy and os packages are imprted.
Seed-values are chosen.
The .csv files in the chosen directory are read appended to a list called data_list.

### models.py
Sklearn packages are imported.
The machine learning models and there hyperparameters are defined.

### training_evaluation.py
NumPy, matplotlib and sklearn packages are imported.
The data are used to train and test the machine learning models.
The train_and_evaluate function are defined.


## Used Packages
### NumPy
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
```python
import numpy as np
```
NumPy documentation link:
https://numpy.org/doc/stable/

In this project, NumPy are used for:
- `random.seed`
- `genfromtxt`
- `shape`
- `array`
- `random.permutation`
- `where`


### os
```python
import os
```

In this project, os are used for:
- `path.join`
- `listdir`


### Matplotlib (.pyplot)
```python
import matplotlib.pyplot as plt
```
In this project, matplotlib.pyplot is used for:
- plotting the confusion matrix
  - `cm.Blues`
  - `title`
  - `savefig`
  - `show`
    

### scikit-learn
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
```

In this project, scikit-learn or sklearn is used for:
- ...


## Contributions & Help
This project has been conducted by Stine Byrjalsen in collaboration with RTX A/S.

Any questions or help, don't hesistate to contact byrjal99@gmail.com
-->
