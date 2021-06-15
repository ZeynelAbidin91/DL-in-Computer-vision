
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Image', image)

kernelSizes = [(3, 3), (9, 9), (15, 15)]
'''for (kx, ky) in kernelSizes:
    blurred = cv2.blur(image, (kx, ky))
    cv2.imshow('Blurred', blurred)
    cv2.waitKey(0)'''

for (kx, ky) in kernelSizes:
    blurred = cv2.GaussianBlur(image, (kx, ky), 0)
    cv2.imshow('Blurred', blurred)
    cv2.waitKey(0)

for k in (3, 9, 15):
	blurred = cv2.medianBlur(image, k)
	cv2.imshow("Median {}".format(k), blurred)
	cv2.waitKey(0)

params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

for (diameter, sigmaColor, sigmaSpace) in params:
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor,
                                    sigmaSpace)
    cv2.imshow('Blurred bilateral', blurred)
    cv2.waitKey(0)
