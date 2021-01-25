from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from pyimagesearch.preprocessing import ImageToArrayPreprocessor 
#from pyimagesearch.preprocessing import SimpleProcessor 
#from pyimagesearch.datasets import SimpleDatasetLoader 
#from pyimagesearch.nn.conv import ShallowNet 
from keras.optimizers import SGD 
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 
import cv2
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the image width, height and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.width, self.height),
                         interpolation=self.inter)

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        # apply the Keras utility function that correctly 
        # rearranges the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape
        # to be "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV =>RELU layer
        model.add(Conv2D(32, (3, 3), padding="same",
                    input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

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



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
                help='path to input dataset')
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel 
# intensities to the range[0, 1]
sdl = SimpleDatasetLoader(preprocessor=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model ...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3,
                        classes=3)
model.compile(loss='categorical_crossentropy', optimizers=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),
target_names=["cat", "dog", "panda"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()