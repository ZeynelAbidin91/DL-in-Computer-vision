import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
#cv2.imshow('Image', image)

flipped = cv2.flip(image, 1)
cv2.imshow('Flipped', flipped)

rotated = imutils.rotate(image, 45)

flipped = cv2.flip(rotated, -1)
cv2.imshow('Flipped', flipped)

#flipped = cv2.flip(image, -1)
#cv2.imshow('Flipped', flipped)
cv2.waitKey(0)