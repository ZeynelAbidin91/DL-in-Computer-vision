# import the necessary packages
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# apply histogram equalization to stretcv the contrast of our image
eq = cv2.equalizeHist(image)
print(eq[272, 146])

# show our images -- notice how the contrast of the second image
# has been stretched
cv2.imshow('Original', image)
cv2.imshow('Histogram Equaliztion', eq)
cv2.waitKey(0)