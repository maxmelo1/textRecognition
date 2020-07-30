import os

from math import copysign, log10

import numpy as np

import pandas as pd

import imutils
import cv2

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage.feature import daisy


PATH1 = 'English/Img/GoodImg/Bmp/Sample00'
PATH2 = 'English/Img/GoodImg/Bmp/Sample0'

def extract_features(dim = (64,64)):

    train_ids = []

    train_fds    = []
    train_labels = []

    fds = []

    fds_hog  = []
    fds_daiy = []
    fds_hu   = []

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

    #n=0
    for im in train_ids:
        img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)



        edged = imutils.auto_canny(img)

        resized_img = cv2.resize(edged,dim)

        #print(resized_img)
        #cv2.imshow('image',resized_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        fd  = hog(resized_img, orientations=8, pixels_per_cell=(8, 8),
          cells_per_block=(2, 2), multichannel=False, transform_sqrt=True, block_norm="L1")
        fd2 = daisy(resized_img, step=180, radius=10, rings=2, histograms=5, orientations=8)

        _,thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.resize(thresh,dim)

        moments = cv2.moments(thresh)
        fd3 = cv2.HuMoments(moments).flatten()

        for i in range(0,7):
            if fd3[i] != 0:
                fd3[i] = -1* copysign(1.0, fd3[i]) * log10(abs(fd3[i]))
            else:
                fd3[i] = 0


        #print(fd.shape)
        #print(fd2[0,0].shape)
        #print(fd3.shape)
        #print(n)
        #n += 1

        #fds_hog  = np.concatenate((fds_hog,fd),axis=0)
        #fds_daiy = np.concatenate((fds_daisy,fd2[0,0]),axis=0)
        #fds_hu   = np.concatenate((fds_hu,fd3),axis=0)


        fds_hog.append(fd)
        fds_daiy.append(fd2[0,0])
        fds_hu.append(fd3)

        #print(len(fds_hog[-1]))
        #input()



        #fds = np.concatenate((fd,fd2[0,0]),axis=0)
        #fds = np.concatenate((fds, fd3[0]), axis=0)

        #train_fds.append(fds)
        #print(train_vector[-1][0])

        #print(np.shape(train_fds))
        #input()


    #print(len(train_fds[7704]))
    #print(len(train_labels))

    hog_data = pd.DataFrame(fds_hog)
    hog_data['label'] = train_labels

    daisy_data = pd.DataFrame(fds_daiy)
    daisy_data['label'] = train_labels

    hu_data = pd.DataFrame(fds_hu)
    hu_data['label'] = train_labels

    print(hog_data.tail())

    hog_data.to_csv('hog.csv', index=False)
    daisy_data.to_csv('daisy.csv', index=False)
    hu_data.to_csv('hu.csv', index=False)


extract_features(dim=(32,32))
