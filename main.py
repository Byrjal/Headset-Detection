import numpy as np
import os
import matplotlib.pyplot as plt

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

from sklearn.naive_bayes import GaussianNB

seed_value = 42
np.random.seed(seed_value)

# Define the directory paths
dir_path_DUT1 = 'DUT1'
dir_path_DUT2 = 'DUT2'


# Function to read all CSV files in a directory
def read_csv_files(directory, folder_type='HeadsetOff', step=2):
    subdirectory = os.path.join(directory, folder_type)
    data_list = []
    for file_name in os.listdir(subdirectory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(subdirectory, file_name)
            data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
            #data = data[:, 1:-2]
            # data = data[889:-1071, 1:-2] #881:-1110 #[:, 1:-1] #880:-1070
            #data = data[::step, 1:-2]
            data = data[889:-1071:step, 1:-2]
            #print(np.shape(data))
            # Reshape to get each column stacked vertically
            stacked_columns = data.T.flatten()
            data_list.append((stacked_columns, file_name))
    return data_list


# The choice of the directories; it can take multiple directories
directories = [dir_path_DUT2]
combined_data = []
for directory in directories:
    data_no_head = read_csv_files(directory, folder_type='HeadsetOff', step=5)
    data_head = read_csv_files(directory, folder_type='HeadsetOn', step=5)
    # Putting labels on (no head = 0, head = 1):
    combined_data.extend([(vector, 0, file_name) for vector, file_name in data_no_head])
    combined_data.extend([(vector, 1, file_name) for vector, file_name in data_head])

# Extract features and labels
X = np.array([item[0] for item in combined_data])
y = np.array([item[1] for item in combined_data])
file_names = np.array([item[2] for item in combined_data])  # Store file names
print(X.shape)
# Shuffle the combined data and labels
shuffled_indices = np.random.permutation(X.shape[0])
X_shuffled = X[shuffled_indices]
y_shuffled = y[shuffled_indices]

# Split the shuffled data into training and testing sets (also, remembering the file names)
X_train, X_test, y_train, y_test, file_names_train, file_names_test = train_test_split(
    X_shuffled, y_shuffled, file_names[shuffled_indices], test_size=0.2, random_state=42, stratify=y_shuffled
)


# Function that train, predict, and evaluate the different models
def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, file_names_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracy_test = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'{model_name} Test Accuracy: {accuracy_test * 100:.2f}%')
    print(f'{model_name} Train Accuracy: {acc_train * 100:.2f}%')
    #print(f'{model_name} Confusion Matrix:')
    print(conf_matrix)
    #print(f'{model_name} Classification Report:')
    #print(class_report)

    # Identify misclassifications
    #misclassified_indices = np.where(y_pred != y_test)[0]
    #misclassified_file_names = file_names_test[misclassified_indices]

    #if len(misclassified_file_names) > 0:
    #    print(f'Misclassified files for {model_name}:')
    #    for file_name in misclassified_file_names:
    #        print(file_name)
    #    print(f'\n')
    #else:
    #    print(f'All files classified correctly for {model_name}. \n')

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} CM - DUT2 (SFI - Less Samples - Step 5)')
    safe_name = model_name.replace(" ", "")
    #plt.savefig(f'DUT2_CM_{safe_name}_SFI_LessSamples_step5.pdf')
    plt.show()

def train_and_evaluate_cv(model, model_name, X, y):
    scores = cross_val_score(model, X, y, cv=9, scoring='accuracy')

    print(f'{model_name} Cross-Validation Accuracy Scores: {scores}')
    print(f'{model_name} Mean Accuracy: {scores.mean() * 100:.2f}%')
    print(f'{model_name} Standard Deviation: {scores.std() * 100:.2f}%')

    model.fit(X, y)
    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    print(f'{model_name} Confusion Matrix:\n{conf_matrix}')

NaiveBayes_model = GaussianNB()
train_and_evaluate_cv(NaiveBayes_model, 'Naive Bayes', X_shuffled, y_shuffled)

# Logistic Regression
logistic_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
train_and_evaluate_cv(logistic_model, 'Logistic Regression', X_shuffled, y_shuffled)

# K-Nearest Neighbor (KNN)
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
train_and_evaluate_cv(knn_model, 'KNN', X_shuffled, y_shuffled)

# Decision Tree
decision_tree_model = DecisionTreeClassifier()
train_and_evaluate_cv(decision_tree_model, 'Decision Tree', X_shuffled, y_shuffled)

# Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
train_and_evaluate_cv(random_forest_model, 'Random Forest', X_shuffled, y_shuffled)



NaiveBayes_model = GaussianNB()
train_and_evaluate(NaiveBayes_model, 'Naive Bayes', X_train, y_train, X_test, y_test, file_names_test)

# Logistic Regression
logistic_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
train_and_evaluate(logistic_model, 'Logistic Regression', X_train, y_train, X_test, y_test, file_names_test)

# K-Nearest Neighbor (KNN)
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
train_and_evaluate(knn_model, 'KNN', X_train, y_train, X_test, y_test, file_names_test)

# Decision Tree
decision_tree_model = DecisionTreeClassifier()
train_and_evaluate(decision_tree_model, 'Decision Tree', X_train, y_train, X_test, y_test, file_names_test)

# Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
train_and_evaluate(random_forest_model, 'Random Forest', X_train, y_train, X_test, y_test, file_names_test)
