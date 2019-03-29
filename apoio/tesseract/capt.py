# -*- coding: utf-8 -*-


import pytesseract as ocr
import numpy as np
import cv2
from PIL import Image
import os

imagem = Image.open('bemvindo.png').convert('RGB')


npimagem = np.asarray(imagem).astype(np.uint8)  



im = cv2.cvtColor(npimagem, cv2.COLOR_RGB2GRAY) 


ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 


binimagem = Image.fromarray(thresh) 


phrase = ocr.image_to_string(imagem, lang='por')

a = 'espeak -vpt+f5 "{0}"'.format(phrase) 
print a
os.system(a)

