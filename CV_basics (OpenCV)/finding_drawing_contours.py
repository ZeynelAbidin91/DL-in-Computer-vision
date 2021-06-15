# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# show the original image
cv2.imshow("Original", image)

# find contours in the image and draw ALL contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print('Found {} contours'.format(len(cnts)))

# output image
cv2.imshow('All contours', clone)
cv2.waitKey(0)

cv2.destroyAllWindows()
clone = image.copy()

print(len(cnts))

for (i,c) in enumerate(cnts):
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Single Contour", clone)
    cv2.waitKey(0)




