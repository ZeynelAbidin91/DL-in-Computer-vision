
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

for i in range(0,3):
    eroded = cv2.erode(gray.copy(), None, iterations=i + 1)
    cv2.imshow('Eroded {} times'.format(i+1),eroded)
    cv2.waitKey(0)

cv2.destroyAllWindows()

for i in range(0, 3):
    dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
    cv2.imshow('Dilated {} times'.format(i+1), dilated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
kernelSizes = [(3,3), (5,5), (7,7)]

for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening: ({}, {})".format(kernelSize[0],
                kernelSize[1]), opening)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    cv2.imshow('Original', image)

for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing: ({}, {})".format(kernelSize[0], kernelSize[1]), closing)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)
 
# loop over the kernels and apply a "morphological gradient" operation
# to the image
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	cv2.imshow("Gradient: ({}, {})".format(kernelSize[0], kernelSize[1]), gradient)
	cv2.waitKey(0)

# To test out the top hat operator
# construct a rectangular kernel and apply a blackhat operation which
# enables us to find dark regions on a light background
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

# # similarly, a tophat (also called a "whitehat") operation will enable
# us to find light regions on a dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

cv2.imshow('Original', gray)
cv2.imshow('Blackhat', blackhat)
cv2.imshow('Tophat', tophat)
cv2.waitKey(0)