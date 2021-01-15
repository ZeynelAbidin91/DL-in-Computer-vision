from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from preprocessing import SimplePreprocessor
#from datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import numpy as np
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
ap.add_argument('-k', '--neighbors',type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1,
                help='# of jobs for k-NN distance (-1 uses available cores)')
args = vars(ap.parse_args())

# gran the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the image width, height and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

class SimpleDatasetLoader:
    def __init__(self, preprocessor=None):
        # store the image preprocessor
        self.preprocessor = preprocessor
        
        # if the preprocessors are None, initialize them as an
        # empty list

        if self.preprocessor is None:
            self.preprocessor = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            #check to see if our preprocessors are not None
            if self.preprocessor is not None:
                    
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessor:
                    image = p.preprocess(image)
            # treat our processed image as a 'feature vector'
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
        

            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,  
                      len(imagePaths)))

            # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

# initialize the image preprocessor, load the dataset from disk,
# and reshpe the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessor=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumptionof the images
print("[INFO] features matrix: {:.1f}MB".format(
        data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using %75 of 
# the data for training and testing and the remaining %25 for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                test_size=0.25, random_state = 42)

# train and evaluate a K-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                            n_jobs=args['jobs'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
