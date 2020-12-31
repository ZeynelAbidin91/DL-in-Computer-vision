import cv2
import argparse
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Blurred', blurred)

# instead of manually specifying the threshold value, we can use adaptive
# thresholding to examine neighborhoods of pixels and adaptively threshold
# each neighborhood -- in this example, we'll calculate the mean value
# of the neighborhood area of 25 pixels and threshold based on that value;
# finally, our constant C is subtracted from the mean calculation (in this
# case 15)
thresh = cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 15)
cv2.imshow('OPENCV Mean thresh', thresh)

# scikit-image adaptive thresholding, it just feels a lot more "Pythonic"
T = threshold_local(blurred, 25, offset=5, method='gaussian')
thresh = (blurred < T).astype('uint8') * 255
cv2.imshow("scikit-image Mean Thresh", thresh)
cv2.waitKey(0)
    