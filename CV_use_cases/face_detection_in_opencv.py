# import the necessary packages
import argparse
import cv2
import imutils

# construcy the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True, 
                help='Path to face cascade')
ap.add_argument('-i', '--image', required=True, 
                help='Path to input image')
args = vars(ap.parse_args())

# load the image, and convert it to grayscale
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier(args['face'])
faceRects = detector.detectMultiScale(gray, scaleFactor=1.8, 
            minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
print("I found {} face(s)".format(len(faceRects)))

# loop over found faces and draw a rectangle around each
for face in faceRects:
    (x, y, w, h) = face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)

# show detected faces
cv2.imshow('Detected faces', image)
cv2.waitKey(0) 