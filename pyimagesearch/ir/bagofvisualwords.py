# import the necessary packages

from scipy import sparse
from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import numpy as np

class BagOfVisualWords:
    def __init__(self, codebook, sparse=True):
        # store the codebook used to compute the bag of visual words reprentation
        # for each image along with the flag used to control whether sparse or dense
        # histograms arew constructed
        self.codebook = codebook
        self.sparse = sparse
    
    def describe(self, features):
        # compute euclidian distance between the features and cluster centers,
        # grab the indexes of the smallest distances for each cluster, and construct
        #a bag of visual words representation 
        D = pairwise.euclidean_distances(features, Y=self.codebook)
        (words, counts) = np.unique(np.argmin(D, axis=1), return_counts=True)

        # check to see if a sparse histogram should be constructed
        if self.sparse:
            hist = csr_matrix((counts, (np.zeros((len(words),)), words)),
                shape=(1, len(self.codebook)), dtype="float")  
        
        # otherwise, construct a dense histogram of visual word counts
        else:
            hist = np.zeros((len(self.codebook),), dtype="float")
            hist[words] = counts
        
        return hist