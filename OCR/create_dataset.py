import os

from math import copysign, log10

import numpy as np

import pandas as pd

import cv2

from skimage.feature import hog
from skimage.feature import daisy


PATH1 = 'English/Img/GoodImg/Bmp/Sample00'
PATH2 = 'English/Img/GoodImg/Bmp/Sample0'

def extract_features(dim = (64,64)):

    train_ids = []

    train_fds    = []
    train_labels = []

    fds = []

    for n in range(1,63):

        if(n < 10):
            aux = next(os.walk('English/Img/GoodImg/Bmp/Sample00'+str(n)))[2]
            caminho = PATH1
        if(n >=10):
            aux = next(os.walk('English/Img/GoodImg/Bmp/Sample0'+str(n)))[2]
            caminho = PATH2

        aux_lbl = [n]*len(aux)
        #print(aux_lbl)

        train_labels += aux_lbl
        train_ids += [ caminho + str(n)+ '/' + ids for ids in aux ]

    #print("first: ", len(label_ids))
    #print("seccond: ", len(train_ids))


    for im in train_ids:
        img = cv2.imread(im,0)
        resized_img = cv2.resize(img,dim)
        fd  = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), multichannel=False)
        fd2 = daisy(resized_img, step=180, radius=10,  histograms=6, orientations=8)

        moments = cv2.moments(resized_img)
        fd3 = cv2.HuMoments(moments)

        for i in range(0,7):
            fd3[i] = -1* copysign(1.0, fd3[i]) * log10(abs(fd3[i]))

        #print(fd2[0,0].shape)
        #print(fd3.shape)


        fds = np.concatenate((fd,fd2[0,0]),axis=0)
        fds = np.concatenate((fds, fd3[0]), axis=0)

        train_fds.append(fds)
        #print(train_vector[-1][0])

        #print(np.shape(train_fds))
        #input()


    #print(len(train_fds[7704]))
    #print(len(train_labels))

    raw_data = pd.DataFrame(train_fds)
    raw_data['label'] = train_labels

    print(raw_data.tail())

    raw_data.to_csv('features.csv')


extract_features()
