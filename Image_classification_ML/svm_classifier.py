from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# generate the XOR data
tl = np.random.uniform(size=(100, 2)) + np.array([-2.0, 2.0])
tr = np.random.uniform(size=(100, 2)) + np.array([2.0, 2.0])
br = np.random.uniform(size=(100, 2)) + np.array([2.0, -2.0])
bl = np.random.uniform(size=(100, 2)) + np.array([-2.0, -2.0])
X = np.vstack([tl, tr, br, bl])
y = np.hstack([[1] * len(tl), [-1] * len(tr), [1] * len(br), [-1] * len(bl)])

# construct the training and testing split by taking 75% of the data for training
# and 25% for testing 
(trainX, testX, trainY, testY ) = train_test_split(X, y, test_size = 0.25, random_state=42)
"""
# train the linear SVM model, evaluate it, and show the results
print('[RESULTS] SVM w/ Linear Kernel')
model = SVC(kernel='linear')
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX)))
"""
# train the SVM + polynomaila kernel model, evaluateit, show the results
print('[RESULTS] SVM w/ Polynomial Kernel')
model = SVC(kernel='poly', degree=2, coef0=1)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX)))