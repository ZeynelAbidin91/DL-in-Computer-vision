import cv2
import numpy as np
import imutils


def detect_object(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c1 = max(cnts, key = cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
'''
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(x+w)
'''
    

img = "shapes_example.png"
#cv2.imshow('IMage,', image)
#cv2.waitKey(0)
detect_object(img)