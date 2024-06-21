import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets #example dataset

def KNNeighbors (xTrain, xTest, yTrain, Ytest):

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(xTrain, yTrain)

    train_predictions = knn.predict(xTrain)
    test_predictions = knn.predict(xTest)
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

def RandomForest(xTrain, xTest, yTrain, Ytest):

    rf = RandomForestClassifier(n_estimators= 300, random_state = 38)

    rf.fit(xTrain, yTrain)

    train_predictions = rf.predict(xTrain)
    test_predictions = rf.predict(xTest)
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)


    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

def SvM(xTrain, xTest, yTrain, Ytest):

    svm = SVC(kernel='linear', probability=True)

    svm.fit(xTrain, yTrain)

    train_predictions = svm.predict(xTrain)
    test_predictions = svm.predict(xTest)
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

def MlP(xTrain, xTest, yTrain, Ytest):

    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)

    mlp.fit(xTrain, yTrain)

    train_predictions = mlp.predict(xTrain)
    test_predictions = mlp.predict(xTest)
    train_confu_matrix = confusion_matrix(yTrain, train_predictions)
    test_confu_matrix = confusion_matrix(Ytest, test_predictions)

    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']], [train_confu_matrix,test_confu_matrix]

def plot_performance(Perf):
    labels_x = ['accuracy', 'precision', 'recall', 'f1-score']
    labels_y = ['KNN', 'RANDOM FOREST', 'SVM', 'MLP']

    plt.figure(figsize=(8, 6))
    plt.imshow(Perf, cmap='YlGn', aspect='auto', vmin=0, vmax=1)

    plt.xticks(np.arange(len(labels_x)), labels_x)
    plt.yticks(np.arange(len(labels_y)), labels_y)

    for i in range(len(labels_y)):
        for j in range(len(labels_x)):
            if Perf[i, j] > 0.8:
                plt.text(j, i, format(Perf[i, j], '.2f'), ha="center", va="center", color="white", fontsize=14)
            else:
                plt.text(j, i, format(Perf[i, j], '.2f'), ha="center", va="center", color="black", fontsize=14)

    for y in range(Perf.shape[0] + 1):
        plt.axhline(y - 0.5, color='gray', linewidth=0.5)

    for x in range(Perf.shape[1] + 1):
        plt.axvline(x - 0.5, color='gray', linewidth=0.5)

    cbar = plt.colorbar()
    cbar.set_label('Values')

    plt.title('Models Performance')
    plt.show()

    return

def confu_plot(Conf_data):
    labels = iris.target_names          #adjust to your dataset
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(Conf_data[0], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Training Data')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 2, 2)
    sns.heatmap(Conf_data[1], annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    plt.show()


iris = datasets.load_iris()         #adjust to your dataset

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=46)

both_data = []
desempenho = []
confusion_data = []

both_data = (KNNeighbors(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

both_data = (RandomForest(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

both_data = (SvM(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

both_data = (MlP(X_train, X_test, y_train, y_test))
desempenho.append(both_data[0])
confusion_data.append(both_data[1])

desempenho=np.array(desempenho)

plot_performance(desempenho)

option = (input('Do you want to see the confusion matrix of some method?\n y- yes\tn-no\n'))
if option != 'y' and option != 'n':
    while option != 'y' and option != 'n':
        option =(input('Invalid option! Please choosa a valid option.\nDo you want to see the confusion matrix of some method?\n y- yes\tn-no\n'))
if option == 'y':
    chosen_method = int(input('Please, enter the number corresponding to method you want to plot the confusio matrix:\n1-KNN\n2-Random Forest\n3-SVM\n4-MLP\n'))
    if chosen_method != 1 and chosen_method != 2 and chosen_method != 3 and chosen_method != 4:
        while chosen_method != 1 and chosen_method != 2 and chosen_method != 3 and chosen_method != 4:
            chosen_method = int(input('Invalid option! Please choosa a valid option.\nPlease, enter the number corresponding to method you want to plot the confusio matrix:\n1-KNN\n2-Random Forest\n3-SVM\n4-MLP\n'))
    else:
        confu_plot(confusion_data[chosen_method-1])
