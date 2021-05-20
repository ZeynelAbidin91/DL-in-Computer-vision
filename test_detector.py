# import the necessary packages
from imutils import paths
import argparse
import dlib
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
ap.add_argument("-t", "--testing", required=True, help="Path to directory of testing images")
args = vars(ap.parse_args())
# load the detector
detector = dlib.simple_object_detector(args["detector"])
# loop over the testing images
image = cv2.imread(args['testing'])
boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	# loop over the bounding boxes and draw them
for b in boxes:
	(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
	cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 4)
	# show the image
print(boxes)
cv2.imshow("Image", image)
cv2.waitKey(0)