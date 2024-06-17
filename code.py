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

    train_predictions = knn.predict(X_train)
    test_predictions = knn.predict(X_test)

    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']]

def RandomForest(xTrain, xTest, yTrain, Ytest):

    rf = RandomForestClassifier(n_estimators= 300, random_state = 38)

    rf.fit(xTrain, yTrain)

    train_predictions = rf.predict(X_train)
    test_predictions = rf.predict(X_test)


    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']]

def SvM(xTrain, xTest, yTrain, Ytest):

    svm = SVC(kernel='linear', probability=True)

    svm.fit(xTrain, yTrain)

    train_predictions = svm.predict(X_train)
    test_predictions = svm.predict(X_test)


    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall'],class_report['macro avg']['f1-score']]

def MlP(xTrain, xTest, yTrain, Ytest):

    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)

    mlp.fit(xTrain, yTrain)

    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)


    class_report = classification_report(y_pred=test_predictions, y_true=y_test, output_dict=True)

    return [class_report['accuracy'],class_report['macro avg']['precision'],class_report['macro avg']['recall']
        ,class_report['macro avg']['f1-score']]


iris = datasets.load_iris()         #adjust to your dataset

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=46)

desempenho = []

desempenho.append(KNNeighbors(X_train, X_test, y_train, y_test))
desempenho.append(RandomForest(X_train, X_test, y_train, y_test))
desempenho.append(SvM(X_train, X_test, y_train, y_test))
desempenho.append(MlP(X_train, X_test, y_train, y_test))
