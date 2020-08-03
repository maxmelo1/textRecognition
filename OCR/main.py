import cv2
import os

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm

import string

import pickle
import itertools

import seaborn as sns

import numpy as np

def main():

    data = pd.read_csv('hog.csv', index_col=False)
    #data = pd.read_csv('hu.csv', index_col=False)

    X = data.iloc[:,:-1].values
    y = data['label'].values#.iloc[:,-1]

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #print(X_train[0])
    #input()
    #print(data.shape)
    #print(data.tail)
    #print(X_test.shape)
    #print(data['label'])
    #print(y)
    #print(y_train.tail)
    #print(y_train.shape)

    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    #scaler.fit(X_train)
    #X_train_std = scaler.transform(X_train)

    #X_test_std = scaler.transform(X_test)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)


    #eps = 1e-4
    #X_train_log = np.log(X_train+eps )
    #X_test_log  = np.log(X_test+eps)

    #print(X_train_log[0])

    #clf = MLPClassifier(solver='sgd', alpha=1e-5, power_t=0.25, hidden_layer_sizes=(140, 160, 63), random_state=3, activation='relu', max_iter=500, verbose=10, learning_rate_init=.1, learning_rate='adaptive', n_iter_no_change=30)
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(324, 162, 62),  
        activation='relu', max_iter=1000, verbose=10, learning_rate_init=.001, 
        learning_rate='adaptive', n_iter_no_change=300, early_stopping=True)
    #clf = DecisionTreeClassifier()
    #clf = svm.SVC(gamma='auto', C=2)
    #clf = KNeighborsClassifier(n_neighbors=1)

    #clf = svm.LinearSVC(C=2, max_iter=1000 )


    print("starting fit proccess, it may take a while")
    #clf.fit(X_train_std, y_train)
    clf.fit(X_train_std, y_train)

    #res = clf.predict(X_test_std)
    res = clf.predict(X_test_std)

    filename = 'model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    print('model saved!')

    acc_score = accuracy_score(y_test, res)
    cm = confusion_matrix(y_test, res)
    

    print("Training set score: %f" % clf.score(X_train_std, y_train))
    print("Test set score: %f" % clf.score(X_test_std, y_test))

    #print(res)
    min = np.amin(cm)
    max = np.amax(cm)

    print(max, ', ', min)
    #print(classification_report(y_test, res))

    nros = list(range(0,10))
    letras_mai = list(string.ascii_uppercase)
    letras_min = list(string.ascii_lowercase)
    

    labels = nros + letras_mai + letras_min
    tick_marks = np.arange(len(labels))
    

    plt.figure(figsize = (24,24), dpi=100)
    #cmap = plt.cm.Greens
    #sns.set(font_scale=1.4)
    #sns.heatmap(cm, center=True, cmap='Greens', fmt=".1f", vmin=0, vmax=max)
    plt.imshow(cm, interpolation='nearest', cmap='Greens' )
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    plt.colorbar()
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.savefig('cm.png', dpi=100)
    #plt.show()


if __name__ == "__main__":
    main()
