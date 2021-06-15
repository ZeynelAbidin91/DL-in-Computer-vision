# import the necessary packages
from numpy.lib.type_check import imag
from skimage._shared.version_requirements import require
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure, feature
from imutils import paths
import os
import imutils
import argparse
import cv2

# construct the argument parser and parse the aruments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input image dataset')
ap.add_argument('-t', '--testData', required=True,
                help='Path to test image dataset')
args = vars(ap.parse_args())

data = []
labels = []

# load and store training image dataset
for imagePath in paths.list_images(args['dataset']):
    # get the label
    label = os.path.split(imagePath)[1].split('_')[0]
    
    # get the features
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    # find contours in the edege map, keeping only the largest
    # one which is presumed to be the car logo
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # extract the logo of the car and resize it to a canonical width
	# and height
    (x, y, w, h) = cv2.boundingRect(c)
    logo = gray[y:y + h, x:x + w]
    logo = cv2.resize(logo, (200, 100))

    # extract Histogram of Oriented Gradients from the logo
    H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    
    # update the lists of data
    data.append(H)
    labels.append(label)

# train ML algorithm
print("[INFO] training classifier...")
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(data, labels)

# loop over the test dataset
for (i, imagePath) in enumerate(paths.list_images(args["testData"])):

	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logo = cv2.resize(gray, (200, 100))

	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
	pred = clf.predict(H.reshape(1, -1))[0]

	# visualize the HOG image
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

	# draw the prediction on the test image and display it
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
	cv2.imshow("Test Image #{}".format(i + 1), image)
	cv2.waitKey(0)