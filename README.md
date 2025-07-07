# Headset-Detection

This project takes .csv data files from scattering parameters measured from the headset and utilises different machine learning models to detect or differentiate between the two states: the user wearing the headset or not (e.g. placed on a table).
Real-world use cases for implementing the detection in a headset can be vast. Some are:
- Pause/resume calls depending on whether the headset is on the head.
- It can reduce call misses.
- Auto-play/pause music or videos when user wears or removes the headset.
- Trigger auto power-off to conserve battery.

## Getting Started
The repository consists of:
- main.py
- plot.py
- Data
  - OldData
  - NewData\n
You have to run ...

### main.py
Multiple packages are imported.
Seed-values are chosen.
Directory are chosen.
The .csv files in the chosen directory are read appended to a list called data_list.
Then the data are combined and shuffled.
This data are used to train and test the machine learning models.
The train_and_evaluate function are defined and called.

### plot.py


## Used Packages
### NumPy
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
```python
import numpy as np
```
NumPy documentation link:
https://numpy.org/doc/stable/

In this project, NumPy are used for:
- random.seed
- genfromtxt
- shape
- array
- random.permutation
- where


### os
```python
import os
```

In this project, os are used for:
- path.join
- listdir


### Matplotlib (.pyplot)
```python
import matplotlib.pyplot as plt
```
In this project, matplotlib.pyplot is used for:
- plotting the confusion matrix
  - cm.Blues
  - title
  - savefig
  - show
    

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
- 


<!--
## Contributions & Help
This project has been conducted by Stine Byrjalsen in collaboration with RTX A/S.

Any questions or help, don't hesistate to contact byrjal99@gmail.com
-->
