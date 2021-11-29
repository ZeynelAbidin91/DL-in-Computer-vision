# import the necessary packages
import argparse
import imutils
import cv2


# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help="Path to input image")
args = vars(ap.parse_args())

# load the image and convert to grayscale
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect keypoints, and then extract the local invariant descriptors
detector = cv2.xfeatures2d.SIFT_create()
(kps, descs) = detector.detectAndCompute(gray, None)

# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))