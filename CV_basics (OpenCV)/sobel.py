import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', gray)

gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

cv2.imshow('gX', gX)
cv2.imshow('gY', gY)

# combine the sobel X and Y representations into a single image
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
cv2.imshow('Sobel Combined', sobelCombined)
cv2.waitKey(0)

print(np.arctan2(186.0, -7.0) * (180.0 / np.pi))

x = np.array([[44, 67, 96], [231, 184, 224], [51, 253, 36]], np.float64)
gX = cv2.Sobel(x, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(x, ddepth=cv2.CV_64F, dx=0, dy=1)
print(gY, gX)
print(np.arctan2(319, 23) * (180.0 / np.pi))