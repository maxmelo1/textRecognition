from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm

from skimage.feature import hog

import pickle

import imutils
import cv2

import numpy as np

#supondo que treinou o clf com hog
def extract_features_hog(filename, dim=(32,32)):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	edged = imutils.auto_canny(img)
	resized_img = cv2.resize(edged,dim)

	return hog(resized_img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False, transform_sqrt=True, block_norm="L1")


def main():
	filename = 'model.sav'
	
	clf = pickle.load(open(filename, 'rb'))

	img_name = 'test2.png'

	#img = np.zeros(1)
	#img = np.concatenate((img, extract_features_hog(img_name)), axis=0)
	img = extract_features_hog(img_name)
	

	#print(img.shape)

	#é o nro da pasta das amostras de m, no dataset
	gt = 49

	#se tiver várias amostras pra testar
	#result = clf.score(X_test, Y_test)

	#se tiver uma só
	result = clf.predict([img])
	print(result)

if __name__ == "__main__":
	main()