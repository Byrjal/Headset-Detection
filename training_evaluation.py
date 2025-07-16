import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, file_name_test, dataset_name, use_step, step, use_freq_range, save_plot, plot_dir, print_misclassified):
    """
    Trains the model on train data, evaluates on test data, prints metrics, shows confusion matrix,
    optionally saves the plot and prints misclassified file names.

    Parameters:
        model: The classifier from models.py.
        model_name (str): Name of the model.
        X_train, y_train: Training features and labels.
        X_test, y_test: Testing features and labels.
        file_name_test: Corresponding file names for test samples.
        dataset_name (str): Dataset path name.
        use_step (bool): Whether downsampling was used.
        step (int): Step size if downsampling was applied.
        use_freq_range (bool): Whether a frequency range subset was used.
        save_plot (bool): Whether to save the confusion matrix plot.
        plot_dir (str): Directory where plots are saved.
        print_misclassified (bool): Whether to list misclassified files.
    """
    # Train the model and make predictions
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calculate accuracy scores and confusion matrix
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Get the dataset name of the final directory only
    dataset_name = dataset_name.split('/')[-1]

    # Print performance metrics
    print(f"{'=' * 15} {model_name} Evaluation Results {'=' * 15}")
    print(f'Test Accuracy: {acc_test * 100:.2f}%')
    print(f'Train Accuracy: {acc_train * 100:.2f}%')
    print(f'Confusion Matrix:')
    print(conf_matrix)

    # Display the confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)

    # Create a title and filename for the plot based on configuration settings
    ex_parts = []
    if use_freq_range:
        ex_parts.append("SFI")
    if use_step:
        ex_parts.append(f"Less Samples - Step {step}")
    ex_name = " - ".join(ex_parts) if ex_parts else "Full"
    plt.title(f"{model_name} - {dataset_name} ({ex_name})")
    ex_file = ex_name.replace(" ", "").replace("-", "_")

    # Save the confusion matrix plot if enabled
    if save_plot:
        filename = f"{dataset_name}_CM_{model_name.replace(' ', '')}_{ex_file}.pdf"
        filepath = plot_dir + filename
        plt.savefig(filepath)
        print(f"Saved confusion matrix to: {filepath}")
    plt.show()

    # Optionally print misclassified filenames
    if print_misclassified:
        misclassified_indices = np.where(y_pred_test != y_test)[0]
        misclassified_file_names = file_name_test[misclassified_indices]
        if len(misclassified_file_names) > 0:
            print(f" ")
            print(f'Misclassified files for {model_name}:')
            for file_name in misclassified_file_names:
                print(file_name)
            print('\n')
        else:
            print(f'All files classified correctly for {model_name}.\n')


def cross_validation(model, model_name, X, y, cv):
    """
    Performs cross-validation on the given model and prints results.

    Parameters:
        model: Classifier from models.py.
        model_name (str): Name of the model.
        X, y: Feature matrix and labels.
        cv (int): Number of folds for cross-validation.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{'=' * 15} {model_name} Cross-Validation Results {'=' * 15}")
    print(f"Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: {scores.std() * 100:.2f}%")