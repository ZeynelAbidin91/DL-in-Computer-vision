import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7,7), 0)
cv2.imshow('Blurred',blurred)


# aplly basic thresholding -- the first parameter is the image
# we want to threshold, the second value is our threshold check
# if a pixel value is greater than threshold, we set it to be black
# otherwise it is white
'''(T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inverse', threshInv)

'''
(T, thresh) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh)
cv2.waitKey(0)
'''

# finally, we can visualize only the masked regions in the image
cv2.imshow('Output', cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)'''

'''# to apply the otsu's threshold method
(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | 
    cv2.THRESH_OTSU) 
cv2.imshow("Threshold", threshInv)
print("Otsu's thresholding value: {}".format(T))
cv2.waitKey(0)

# finally, we can visualize only the masked regions in the image
cv2.imshow("Output", cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)'''


