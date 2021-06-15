import argparse
import cv2
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Image', image)

# cropping an image is accomplished using simple NumPy array slices --
# let's crop the face from the image
face = image[85:250, 85:250]
cv2.imshow('Face', face)
cv2.waitKey(0)