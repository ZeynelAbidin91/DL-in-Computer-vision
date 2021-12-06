from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import random
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to the directory that contains the images to be indexed')
ap.add_argument('-f', '--feature-db', required=True, 
                help='Path to where the features database will be stored')
ap.add_argument('a', '--approx-images', type=int, default=500,
                help='Approximate # of images in the dataset')
ap.add_argument('-b', '--max-buffer-size', type=int, default=50000,
                help='Maximum buffer size for # of features to be stored in memory')
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the
# descriptor pipeline
detector =FeatureDetector_create('SURF')
# detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create('RootSIFT')
dad = DetectAndDescribe(detector, descriptor)
"""
Lines 25-27 set up our keypoint detection and local invariant descriptor 
pipeline. In our CBIR lessons, we used the Fast Hessian (i.e. SURF) 
keypoint detector, but here we’ll use the GFTT detector instead. It’s 
very common to use either the GFTT or Harris detector when using the BOVW 
model for classification; however, you should perform experiments 
evaluating each keypoint detector and go with the detector that obtained
the best accuracy. In order to describe the region surrounding each 
keypoint, we’ll use the RootSIFT descriptor, which will produce a 
128-dim feature vector for each keypoint region.
"""

# initialize the feature indexer
fi = FeatureIndexer(args['features_db'], estNumImages=args['approx_images'],
                    maxBufferSize=args['max_buffer_size'], verbose=True)

# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)



# loop over the images in the dataset
for (i, imagePath) in enumerate (imagePaths):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        fi.debug("processed {} images".format(i), msgType='[PROGRESS]')
        
    # extract the image filename(i.e. the unique image ID) from the image 
    # path, then load the image itself
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(320, image.shape[1])) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # describe the image
    (kps, descs) = dad.describe(image)

    # if either the keypoints or descriptor are None, then ignore the image
    if kps is None or descs is None:
        continue
    # index the features
    fi.add(filename, kps, descs)

# finish the indexing process
fi.finish()

    