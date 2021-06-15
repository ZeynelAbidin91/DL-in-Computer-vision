# import necessary packages
from numpy.lib.function_base import append
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import argparse
import os
import cv2
from imutils.paths import list_images
import mahotas
import numpy as np

# build argument parser and parse the input arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--trainData', required=True, 
                help='Path to input image dataset')
ap.add_argument('-t', '--testData', required=True, 
                help='Path to test image dataset')
args = vars(ap.parse_args())

# build 'describe' function to extract 'Haralick Features' from both dataset
data = []
labels = []

trainImagePaths = list_images(args['trainData'])
for path in trainImagePaths:
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # extract label and haralick texture features
    label = os.path.split(path)[1].split('_')[0]
    features = mahotas.features.haralick(gray).mean(axis=0)

        
    data.append(features)
    labels.append(label)
print(data)
'''
    

testImagePaths = list_images(args['testData'])

clf = LinearSVC(C=10.0, random_state=42)
clf.fit(data, labels)
print('SVM training...')

#predictions = clf.predict(test_X)
#print(predictions)
#print('Accuracy of texture classifier = {}'
#                    .format(accuracy_score(test_y, predictions)))

for imagePath in testImagePaths:
    print(imagePath)
    image2 = cv2.imread(imagePath)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # extract label and haralick texture features
    features = mahotas.features.haralick(gray2).mean(axis=0)

    # 
    pred = clf.predict(features.reshape(1, -1))[0]
    cv2.putText(image2, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 3)

    cv2.imshow('Predicted Image', image2)
    cv2.waitKey(0)'''