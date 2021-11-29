# import the necessary package
import argparse
import cv2
import imutils

# contruct the argument parser, and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True,
                help='Path to the haar cascade')
ap.add_argument('-v', '--video', help='Path to the video file or webcam')
args = vars(ap.parse_args())

# load the face detector
detector = cv2.CascadeClassifier(args['face'])

# handle if it is video file of webcam stream
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

# looping over
while True:
    # get current frame and check if it is reached to the end
    (grabbed, frame) = camera.read()

    if args.get('Video') and not grabbed:
        break
    
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = detector.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faceRects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    # show the frame to our screeen
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
