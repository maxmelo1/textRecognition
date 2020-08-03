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

PATH1_MSK = 'English/Img/GoodImg/Msk/Sample00'
PATH2_MSK = 'English/Img/GoodImg/Msk/Sample0'

def extract_features(dim = (64,64)):

    train_ids = []
    #train_fds    = []
    train_labels = []
    train_msks = []

    fds = []

    fds_hog  = []
    fds_daiy = []
    fds_hu   = []

    for n in range(1,63):

        if(n < 10):
            aux = next(os.walk('English/Img/GoodImg/Bmp/Sample00'+str(n)))[2]
            aux_msk     = next(os.walk(PATH1_MSK+str(n)))[2]
            caminho     = PATH1
            caminho_msk = PATH1_MSK
        if(n >=10):
            aux = next(os.walk('English/Img/GoodImg/Bmp/Sample0'+str(n)))[2]
            aux_msk     = next(os.walk(PATH2_MSK+str(n)))[2]
            caminho     = PATH2
            caminho_msk = PATH2_MSK

        aux_lbl = [n]*len(aux)
        #print(aux_lbl)

        train_labels    += aux_lbl
        train_ids       += [ caminho + str(n)+ '/' + ids for ids in aux ]
        train_msks      += [ caminho_msk + str(n)+ '/' + ids for ids in aux_msk ]

    #print("first: ", len(label_ids))
    #print("seccond: ", len(train_ids))

    #n=0
    idx = 0
    for im, msk_id in zip(train_ids, train_msks):
        img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        _,aux = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        msk     = cv2.imread(msk_id, cv2.IMREAD_GRAYSCALE)
        _, msk  = cv2.threshold(msk,127,255,cv2.THRESH_BINARY)

        #print(type(aux))

        img_cropped = cv2.bitwise_and(aux, msk)

        n_white = np.sum(img_cropped == 255)
        n_black = np.sum(img_cropped == 0)

        #if n_black < n_white:
        #    img_cropped = cv2.bitwise_not(img_cropped)
        if idx == 4:
            print(im)

        img_cropped = cv2.bitwise_and(img_cropped, msk)

        #contours, hierarchy = cv2.findContours(img_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #cnt = contours[0]
        #print(cnt)

        #mask = np.zeros(img_cropped.shape, np.uint8)
        #cv2.drawContours(mask, contours, -1, 255, 2)
        #x,y,w,h = cv2.boundingRect(cnt)
        #roi = img_cropped[y:y+h, x:x+w]
        #cv2.drawContours(mask, cnt, -1, 255, 1)

        #plt.imshow(img_cropped, 'gray')
        #plt.show()

        #for i,cnt in enumerate(contours):
        #for h in hierarchy:
            
            #if hierarchy[0][i][3] != -1:
            #    mask = np.zeros(img_cropped.shape, np.uint8)
            #    cv2.drawContours(mask, contours, i, 255, cv2.FILLED)

            #mask = np.zeros(img_cropped.shape, np.uint8)
            #cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
            #print(h[1])
            #if h[2] > 0:
            #    cv2.drawContours(mask, [contours[h[2]]], 0, 255, cv2.FILLED )

            #print(cnt)
            #x,y,w,h = cv2.boundingRect(cnt)
            #rect = cv2.minAreaRect(cnt)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)

            #cv2.drawContours(mask, [box], i, 255, cv2.FILLED)

            #print(box)
            #roi = img_cropped[y:y+h, x:x+w]


            #plt.imshow(mask, 'gray')
            #plt.show()

        


        #print(im)
        #print(msk_id)

        #plt.imshow(img3, 'gray')
        #plt.show()

        #input()

        #edged = imutils.auto_canny(img)

        
        #resized_img = cv2.resize(img,dim)
        resized_img = cv2.resize(img_cropped,dim)
        cv2.imwrite("outimgs/"+str(idx)+".png", resized_img);
        idx +=1;

        #print(resized_img)
        #cv2.imshow('image',resized_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        fd  = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
          cells_per_block=(2, 2), multichannel=False, transform_sqrt=True, block_norm="L1")

        winSize = dim
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        winStride = (8,8)
        
        padding = (8,8)
        locations = ((10,20),)

        #hd = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
        #                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

        #fd = hd.compute(resized_img,winStride,padding,locations).flatten()

        #fd  = hog(resized_img, orientations=8, pixels_per_cell=(8, 8),
        #  cells_per_block=(2, 2), multichannel=False, transform_sqrt=True, block_norm="L1")

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
        #input()

        #fds_hog  = np.concatenate((fds_hog,fd),axis=0)
        #fds_daiy = np.concatenate((fds_daisy,fd2[0,0]),axis=0)
        #fds_hu   = np.concatenate((fds_hu,fd3),axis=0)


        fds_hog.append(fd)
        fds_daiy.append(fd2[0,0])
        fds_hu.append(fd3)

    

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
