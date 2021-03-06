# import the necessary packages
from pyimagesearch.object_detection.helpers import pyramid
from pyimagesearch.object_detection.helpers import sliding_window
import argparse
import cv2
import time
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", required=True, type=int, help="width of sliding window")
ap.add_argument("-t", "--height", required=True, type=int, help="height of sliding window")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

# load the input image and unpack the comman line arguments
image = cv2.imread(args['image'])
(winW, winH) = (args['width'], args['height'])

# loop over the image pyramid
count_window = 0 
for layer in pyramid(image, scale=args['scale']):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(layer, stepSize=10, windowSize=(winW, winH)):
        # if the current window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE WE WOULD PROCESS THE WINDOW, EXTRACT HOG FEATURES, AND
		# APPLY A MACHINE LEARNING CLASSIFIER TO PERFORM OBJECT DETECTION

        # since we do not have a classifier yet, let's just draw the window
        clone = layer.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        count_window += 1
    
        # normally we would leave out this line, but let's pause execution
		# of our script so we can visualize the window
        cv2.waitKey(1)
        time.sleep(0.025)
print(count_window)