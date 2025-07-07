# Headset-Detection

This project takes .csv data files from scattering parameters measured from the headset and utilises different machine learning models to detect or differentiate between the two states: the user wearing the headset or not (e.g. placed on a table).
Real-world use cases for implementing the detection in a headset can be vast. Some are:
- Pause/resume calls depending on whether the headset is on the head.
- It can reduce call misses.
- Auto-play/pause music or videos when user wears or removes the headset.
- Trigger auto power-off to conserve battery.

## Get Started
The repository consists of:
- main.py
- plot.py
- Data
  - OldData
  - NewData

### main.py
Multiple packages are imported.
Seed-values are chosen.
Directory are chosen.
The .csv files in the chosen directory are read appended to a list called data_list.
Then the data are combined and shuffled.
This data are used to train and test the machine learning models.
The train_and_evaluate function are defined and called.

### plot.py
<!--
## Contributions & Help
This project has been conducted by Stine Byrjalsen in collaboration with RTX A/S.

Any questions or help, don't hesistate to contact byrjal99@gmail.com
-->
