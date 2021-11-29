# import the necessary packages
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--first', required=True, 
                help='Path to first input image to compare matches')
ap.add_argument('-s', '--second', required=True, 
                help='Path to second input image to compare matches')
ap.add_argument('-d', '--detector', default = 'SURF', type=str,
                help="Keypoint detector to use. "
		 "Options ['BRISK', 'DENSE', 'DOG', 'SIFT', 'FAST', 'FASTHESSIAN', 'SURF', 'GFTT', 'HARRIS', 'MSER', 'ORB', 'STAR']")
ap.add_argument('-e', '--extractor', default='SIFT', type=str,
                help= "Feature extractor to use: ['RootSIFT', 'SIFT', 'SURF']")
ap.add_argument('-m', '--matcher', default = "BruteForce", type=str,
                help="Feature matcher to use. Options ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']")
ap.add_argument('-v', '--visualize', type=str, default='Yes',
                help="Whether the visualiztion image should be shown. Options ['Yes', 'No', 'Each']")
args = vars(ap.parse_args())

# determine the detector, the extractor, and the matcher
if args["detector"] == "DOG":
	detector = FeatureDetector_create("SIFT")
elif args["detector"] == "FASTHESSIAN":
	detector = FeatureDetector_create("SURF")
else:
    detector = FeatureDetector_create(args['detector'])

extractor = DescriptorExtractor_create(args['extractor'])

matcher = DescriptorMatcher_create(args['matcher'])

# load images
imageA = cv2.imread(args['first'])
imageB = cv2.imread(args['second'])

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Detect keypoints and extract features
kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

(kpsA, featuresA) = extractor.compute(grayA, kpsA)
(kpsB, featuresB) = extractor.compute(grayB, kpsB)

# Show Feature matches
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
matches = []
if rawMatches is not None:
    for rm in rawMatches:
        if len(rm) == 2 and rm[0] < rm[1] * 0.8:
            matches.append((rm[0].trainIdx, rm[1].queryIdx))

    # show some diagnostic information
    print("# of keypoints from first image: {}".format(len(kpsA)))
    print("# of keypoints from second image: {}".format(len(kpsB)))
    print("# of matched keypoints: {}".format(len(matches)))

# initialize the output visualization image
(hA, wA) = imageA.shape[:2]
(hB, wB) = imageB.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = imageA
vis[0:hB, wA:] = imageB
# loop over the matches
for (trainIdx, queryIdx) in matches:
	# generate a random color and draw the match
	color = np.random.randint(0, high=255, size=(3,))
	color = tuple(map(int, color))
	ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
	ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
	cv2.line(vis, ptA, ptB, color, 2)
	# check to see if each match should be visualized individually
	if args["visualize"] == "Each":
		cv2.imshow("Matched", vis)
		cv2.waitKey(0)
# show the visualization
if args["visualize"] == "Yes":
	cv2.imshow("Matched", vis)
	cv2.waitKey(0)

