import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
#cv2.imshow('Image', image)

# we need to keep in mind aspect ratio so the image does not look skewed
# or distorted -- therefore, we calculate the ratio of the new image to
# the old image. Let's make our new image have a width of 150 pixels
r = 2
dim = (int(image.shape[1] * r), int(image.shape[0] * r))

# perform the actual resizing of the image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized (WIDTH)', resized)
print(resized[20, 74])
#resized = imutils.resize(image, width=100)
#cv2.imshow('Resized (WIDTH)', resized)

cv2.waitKey(0)