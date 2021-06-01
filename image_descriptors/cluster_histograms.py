# import the necessary packages
import sys
sys.path.append('C:/Users/zeyne/Downloads/DL-in-Computer-vision/pyimagesearch')
from descriptors.labhistogram import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
from matplotlib import pyplot as plt
import cv2

# construct the argument parse and paerse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, 
                help='Path to input image dataset directory')
ap.add_argument('-k', '--clusters', default=2,
                help='Integer for # of cluster to generate')
args = vars(ap.parse_args())

# initialize the image descriptor along with the image matrix
desc = LabHistogram([8, 8, 8])
data = []

# grab the image paths from the the dataset directory
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.array(sorted(imagePaths))

# loop over the input dataset of images
for imagePath in imagePaths:
    # load the image, describe the image, then updatethe list of data
    image = cv2.imread(imagePath)
    hist = desc.describe(image)
    data.append(hist)

print(data[4])

# cluster the color histograms
clt = KMeans(n_clusters=args["clusters"])
labels = clt.fit_predict(data)

# loop over the unique labels
for label in np.unique(labels):
    # grab all images paths that are assigned to the current label
    labelPaths = imagePaths[np.where(labels == label)]

    # loop over the image paths that belong to the current label
    for (i, path) in enumerate(labelPaths):
        # load the image and display it 
        image = cv2.imread(path)
        cv2.imshow("Cluster {}, Image #{}".format(label+1, i+1), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 