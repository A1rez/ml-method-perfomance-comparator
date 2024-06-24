# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importing machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Importing classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets # Example dataset

# Define function for K-Nearest Neighbors
def KNNeighbors (xTrain, xTest, yTrain, Ytest):
    knn = KNeighborsClassifier(n_neighbors=3)  # Initialize KNN with 3 neighbors
    knn.fit(xTrain, yTrain)  # Fit the model on training data

    # Make predictions on training and test data
    train_predictions = knn.predict(xTrain)
    test_predictions = knn.predict(xTest)
    
    # Calculate confusion matrices
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    # Generate classification report
    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    # Return performance metrics and confusion matrices
    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

# Define function for Random Forest
def RandomForest(xTrain, xTest, yTrain, Ytest):
    rf = RandomForestClassifier(n_estimators= 300, random_state = 38)  # Initialize Random Forest with 300 trees
    rf.fit(xTrain, yTrain)  # Fit the model on training data

    # Make predictions on training and test data
    train_predictions = rf.predict(xTrain)
    test_predictions = rf.predict(xTest)
    
    # Calculate confusion matrices
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    # Generate classification report
    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    # Return performance metrics and confusion matrices
    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

# Define function for Support Vector Machine
def SvM(xTrain, xTest, yTrain, Ytest):
    svm = SVC(kernel='linear', probability=True)  # Initialize SVM with a linear kernel
    svm.fit(xTrain, yTrain)  # Fit the model on training data

    # Make predictions on training and test data
    train_predictions = svm.predict(xTrain)
    test_predictions = svm.predict(xTest)
    
    # Calculate confusion matrices
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    # Generate classification report
    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    # Return performance metrics and confusion matrices
    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

# Define function for Multi-layer Perceptron
def MlP(xTrain, xTest, yTrain, Ytest):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)  # Initialize MLP
    mlp.fit(xTrain, yTrain)  # Fit the model on training data

    # Make predictions on training and test data
    train_predictions = mlp.predict(xTrain)
    test_predictions = mlp.predict(xTest)
    
    # Calculate confusion matrices
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    # Generate classification report
    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    # Return performance metrics and confusion matrices
    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

# Function to plot performance metrics
def plot_performance(Perf):
    labels_x = ['accuracy', 'precision', 'recall', 'f1-score']  # Metrics
    labels_y = ['KNN', 'RANDOM FOREST', 'SVM', 'MLP']  # Model names

    plt.figure(figsize=(8, 6))
    plt.imshow(Perf, cmap='YlGn', aspect='auto', vmin=0, vmax=1)  # Heatmap of performance metrics

    # Set x and y axis labels
    plt.xticks(np.arange(len(labels_x)), labels_x)
    plt.yticks(np.arange(len(labels_y)), labels_y)

    # Annotate the heatmap with metric values
    for i in range(len(labels_y)):
        for j in range(len(labels_x)):
            if Perf[i, j] > 0.8:
                plt.text(j, i, format(Perf[i, j], '.2f'), ha="center", va="center", color="white", fontsize=14)
            else:
                plt.text(j, i, format(Perf[i, j], '.2f'), ha="center", va="center", color="black", fontsize=14)

    # Draw grid lines
    for y in range(Perf.shape[0] + 1):
        plt.axhline(y - 0.5, color='gray', linewidth=0.5)
    for x in range(Perf.shape[1] + 1):
        plt.axvline(x - 0.5, color='gray', linewidth=0.5)

    # Color bar for heatmap
    cbar = plt.colorbar()
    cbar.set_label('Values')

    plt.title('Models Performance')
    plt.show()
    return

# Function to plot confusion matrices
def confu_plot(Conf_data):
    labels = iris.target_names  # Adjust to dataset labels
    plt.figure(figsize=(18, 8))
    
    # Plot confusion matrix for training data
    plt.subplot(1, 2, 1)
    sns.heatmap(Conf_data[0], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Training Data')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Plot confusion matrix for test data
    plt.subplot(1, 2, 2)
    sns.heatmap(Conf_data[1], annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    plt.show()

# Load the Iris dataset
iris = datasets.load_iris()  # Adjust to your dataset

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=46)

# Initialize lists to store performance metrics and confusion matrices
both_data = []
desempenho = []
confusion_data = []

# Evaluate K-Nearest Neighbors
both_data = (KNNeighbors(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

# Evaluate Random Forest
both_data = (RandomForest(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

# Evaluate Support Vector Machine
both_data = (SvM(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

# Evaluate Multi-layer Perceptron
both_data = (MlP(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

# Convert performance metrics to numpy array for plotting
desempenho=np.array(desempenho)

# Plot performance metrics of all models
plot_performance(desempenho)

# Ask user if they want to see confusion matrix for any model
option = (input('Do you want to see the confusion matrix of some method?\n y- yes\tn-no\n'))
if option != 'y' and option != 'n':
    while option != 'y' and option != 'n':
        option =(input('Invalid option! Please choosa a valid option.\nDo you want to see the confusion matrix of some method?\n y- yes\tn-no\n'))

# If yes, ask user to choose the model
if option == 'y':
    chosen_method = int(input('Please, enter the number corresponding to method you want to plot the confusio matrix:\n1-KNN\n2-Random Forest\n3-SVM\n4-MLP\n'))
    if chosen_method != 1 and chosen_method != 2 and chosen_method != 3 and chosen_method != 4:
        while chosen_method != 1 and chosen_method != 2 and chosen_method != 3 and chosen_method != 4:
            chosen_method = int(input('Invalid option! Please choosa a valid option.\nPlease, enter the number corresponding to method you want to plot the confusio matrix:\n1-KNN\n2-Random Forest\n3-SVM\n4-MLP\n'))
    else:
        # Plot the chosen model's confusion matrix
        confu_plot(confusion_data[chosen_method-1])
