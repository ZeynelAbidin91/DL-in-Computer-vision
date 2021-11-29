# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='Path to the (optional) video file')
args = vars(ap.parse_args())

# if a video path was nor supplied, grab the reference to the webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args['video'])


# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # a frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    
    # show the frame to our screen
    cv2.imshow('Frame', imutils.resize(frame, width=600))
    key = cv2.waitKey(1) & 0xFF

    if key ==ord("q"):
        break
# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows()