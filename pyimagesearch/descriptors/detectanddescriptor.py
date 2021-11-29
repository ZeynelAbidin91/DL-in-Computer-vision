import numpy as np

class DetectAndDescribe:
    def __init__(self, detector, descriptor):
        # store the keypoint detector and local invariant descriptor
        self.detector = detector
        self.descriptor = descriptor

        pass

    def describe(self, image, useKplist=True):
        # detect keypoints in the image and extract local
        # invariant descriptors
        kps = self.detector.detect(image)
        (kps, descs) = self.descriptor.compute(image, kps)

        # if there are ne keypoints or descriptors, return None
        if len(kps) == 0:
            return (None, None)

        # check to see if the keypoints should be 
        # converted to a Numpy array
        if useKplist:
            kps = np.int0([kp.pt for kp in kps])
        
        # return a tuple of the keypoints and descriptors

        