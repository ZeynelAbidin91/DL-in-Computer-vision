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
print(flipped[235, 259])

rotated = imutils.rotate(flipped, 45)

flipped = cv2.flip(rotated, 0)
cv2.imshow('Flipped', flipped)
print(flipped[189, 441])

#flipped = cv2.flip(image, -1)

#cv2.imshow('F
# lipped', flipped)
cv2.waitKey(0)