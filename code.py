import pandas as pd
import numpy as pd
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
