import argparse
import cv2
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find all contours
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for (i, c) in enumerate(cnts):
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)


    # compute the convex hull of the contour, then use the area of the
	# original contour and the area of the convex hull to compute the
	# solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)

    # initialize the character text
    char = '?'


    # if the solidity is high, the we are examining an '0'
    if solidity > 0.9:
        char = '0'
    # otherwise, if the solidity it still reasonably high, 
    # are examining an 'X'
    elif solidity > 0.5:
        char = 'X'
    
    # if the character is not unknown, draw it
    if char != '?':
        cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        cv2.putText(image, char, (x, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
    
    # show the contour properties
    print('{} (Contour #{}) -- solidityd={:.2f}'.format(char, (i + 1),
          solidity))

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0) 