import cv2
import os

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm

import seaborn as sns

import numpy as np

def main():

    data = pd.read_csv('hog.csv')

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #print(X.shape)
    #print(y.shape)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)

    X_test_std = scaler.transform(X_test)



    #clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 150), random_state=3, activation='relu', max_iter=60, verbose=10, learning_rate_init=.1)
    #clf = DecisionTreeClassifier()
    clf = svm.SVC(gamma='auto', C=2)

    #clf = svm.LinearSVC(C=2, max_iter=1000 )


    print("starting fit proccess, it may take a while")
    clf.fit(X_train_std, y_train)

    res = clf.predict(X_test_std)
    acc_score = accuracy_score(y_test, res)
    cm = confusion_matrix(y_test, res)

    print("Training set score: %f" % clf.score(X_train_std, y_train))
    print("Test set score: %f" % clf.score(X_test_std, y_test))

    #print(res)
    min = np.amin(cm)
    max = np.amax(cm)

    print(max, ', ', min)
    print(classification_report(y_test, res))

    plt.figure(figsize = (10,7))
    cmap = sns.cm.rocket_r
    sns.set(font_scale=1.4)
    sns.heatmap(cm, center=True, cmap='magma', fmt=".1f", vmin=min, vmax=max)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()
