from __future__ import print_function
import sys
sys.path.append('C:/Users/zeyne/Downloads/DL-in-Computer-vision/pyimagesearch')
from license_plate.licenseplate import LicensePlateDetector
from imutils import paths 
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images' ,required=True, 
                help="Path to input images")
args = vars(ap.parse_args())

# loop over the images
for imagePath in sorted(list(paths.list_images(args["images"]))):
    # load the image
    image = cv2.imread(imagePath)
    print(imagePath)

    #if the width is greater than 640 pixels, then resize the image
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    # initialize the license plate detector and detect the license plates and charactors    # the license plate
    lpd = LicensePlateDetector(image)
    plates = lpd.detect()

    # loop over the license plate regions and draw the bounding box surrounding the 
    # license plate
    for lpbox in plates:
        lpbox = np.array(lpbox).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [lpbox], -1, (0, 255, 0), 2)
    
    # display the output image
    cv2.imshow("image", image)
    cv2.waitKey(0)



