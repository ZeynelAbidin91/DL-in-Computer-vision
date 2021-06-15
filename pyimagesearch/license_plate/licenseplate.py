# import the necessary packages
import numpy as np
import cv2
import imutils

class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20):
        # store the image to detect license plates in and the min width 
        # and min height of the license plate region
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH

    def detect(self):
        # detect and return the license plate regions in the image
        return self.detectPlates()
    
    def detectPlates(self):
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []

        # convert the image to grayscale, an apply the blackhat operation
        gray = cv2.cvtColor(np.float32(self.image), cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # find regions in the image that are light
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        # compute the Scharr gradient representaation of the blackhat image in the  
        # x-direction and sclae the resulting image into the range[0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv.CV_32F if imutils.is_cv2() else
                            cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    
        '''Here, we compute the Sobel gradient along the x-axis of the blackhat image,
         revealing regions of the image that are not only dark against a light 
         background, but also contain vertical changes in gradient, such as the 
         license plate characters themselves. We take this gradient image and then 
         scale it back into the range [0, 255] by using min/max scaling:'''

        # blur the gradient representation, apply a closing operation, and
        # threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosions and dilations on the image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # take the bitwise 'and' between the 'light' regions of the image, then perform
		# another series of erosions and dilations
        #thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        # find the contours in the thresholdedimage
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute 
            # the area and aspect ratio
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

            # ensure the aspect ratio, width, and height of the bounding box
            # fall within tolerable limits, then update the list of license 
            # plate regions
            if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
                regions.append(box)
            
        # return the list of license plate regions
        return regions