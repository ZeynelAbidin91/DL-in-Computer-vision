import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to the input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
(h, w) = image.shape[:2]
#cv2.imshow('Image', image)
#print(h, w)
(b, g, r) = image[0, 0]

(cX, cY) = (w // 2, h // 2)

top_left = image[0:cY, 0:cX]
cv2.imshow('Top-Left Corner', top_left)
cv2.waitKey(0)