import cv2
import os
from math import copysign, log10

from skimage.feature import hog
from skimage.feature import daisy


from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

import seaborn as sns

import numpy as np

def main():
    CAMINHO01 = 'English/Img/GoodImg/Bmp/Sample00'
    CAMINHO02 = 'English/Img/GoodImg/Bmp/Sample0'
    path = []

    dataSet = []
    
    train_vector = []
    label_train = []
    test_vector = []
    label_test  = []

    hogFeatures = []
    
    train = []
    test = []
    
    dim = (64,64)
    for n in range(1,63):
        if(n < 10):
            train_ids = next(os.walk('English/Img/GoodImg/Bmp/Sample00'+str(n)))[2]
            caminho = CAMINHO01
        if(n >=10):
            train_ids = next(os.walk('English/Img/GoodImg/Bmp/Sample0'+str(n)))[2]
            caminho = CAMINHO02
            
        lenTrain = int(len(train_ids) * 0.9)
        train.append([ caminho + str(n)+ '/' + train_ids[i] for i in range(0, int(len(train_ids)*0.9)) ])
        #train.append(train_ids[0:int(len(train_ids)*0.9)])
        lenTest = int(len(train_ids) * 0.1)
        test.append([ caminho + str(n)+ '/' + train_ids[i] for i in range(int(len(train_ids)*0.9), len(train_ids)) ])
        #test.append(train_ids[int(len(train_ids)*0.9):int(len(train_ids))])

        #print(test)

        for m in train_ids:
            way = caminho + str(n)+ '/' + m
            path.append(way)
    cont = 0

    #train images
    for el in train:
        for im in el:
            #print(im)
            img = cv2.imread(im,0)
            resized_img = cv2.resize(img,dim)
            fd  = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), multichannel=False)
            #fd2 = daisy(resized_img, step=180, radius=10,  histograms=6, orientations=8)
            moments = cv2.moments(resized_img)
            fd3 = cv2.HuMoments(moments)

            for i in range(0,7):
                fd3[i] = -1* copysign(1.0, fd3[i]) * log10(abs(fd3[i]))
            #print(fd2[0,0].shape)
            #print(fd3.shape)
            hogFeatures.append(fd)
            dataSet.append(resized_img)

            #fds = np.concatenate((fd,fd2[0,0]),axis=0)   
            fds = np.concatenate((fd, fd3[0]), axis=0)
            #print(fds.shape)

            label = int(im.split("/")[4][-3:])
            train_vector.append(fds)
            label_train.append(label)
            #print(train_vector[-1][0])

    #test images
    #train images
    for el in test:
        for im in el:
            #print(im)
            img = cv2.imread(im,0)
            resized_img = cv2.resize(img,dim)
            fd = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), multichannel=False)
            #fd2 = daisy(resized_img, step=180, radius=10,  histograms=6, orientations=8)
            moments = cv2.moments(resized_img)
            fd3 = cv2.HuMoments(moments)
            for i in range(0,7):
                fd3[i] = -1* copysign(1.0, fd3[i]) * log10(abs(fd3[i]))
            #print(np.shape(fd))

            fds = np.concatenate((fd, fd3[0]), axis=0)

            hogFeatures.append(fd3[0])
            dataSet.append(resized_img)

            label = int(im.split("/")[4][-3:])
            #test_vector.append(fd)
            test_vector.append(fds)
            label_test.append(label)
            #print(np.shape(test_vector[0][0]))


    print(np.shape(test_vector))
    

    scaler = StandardScaler()
    scaler.fit(train_vector)
    train_vector = scaler.transform(train_vector)

    test_vector = scaler.transform(test_vector)

    #clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 150), random_state=3, activation='relu', max_iter=60, verbose=10, learning_rate_init=.1)
    #clf = DecisionTreeClassifier()
    clf = svm.SVC(gamma='auto', C=2)

    #clf = svm.LinearSVC(C=2, max_iter=1000 )


    print("starting fit proccess, it may take a while")
    clf.fit(train_vector, label_train)

    res = clf.predict(test_vector)
    acc_score = accuracy_score(label_test, res)
    cm = confusion_matrix(label_test, res)

    print("Training set score: %f" % clf.score(train_vector, label_train))
    print("Test set score: %f" % clf.score(test_vector, label_test))

    #print(res)
    count = 0
    for i in range(len(res)):
        if label_test[i] == res[i]:
            count += 1
        #print("expected: %d, found %d" %(label_test[i], res[i]))
    print("acc: %f" %( count / len(res) ))
    print("acc score: %f " % acc_score)
    #print(cm)
    print(classification_report(label_test, res))

    cmap = sns.cm.rocket_r
    sns.heatmap(cm, center=True, cmap='coolwarm', fmt=".1f", vmin=-1)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()    

    '''
    for n in dataSet:
        cv2.imshow('',n)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

if __name__ == "__main__":
    main()
