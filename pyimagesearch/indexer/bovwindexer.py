# import the necessary packages
from .baseindexer import BaseIndexer
from scipy import sparse
import numpy as np
import h5py

class BOVWIndexer(BaseIndexer):
    def __init__(self, fvectorSize, dbPath, estNumImages=500, maxBufferSize=500,
                dbResizeFactor=2, verbose=True):
                # call the parent constructor
                super((BOVWIndexer, self).__init__(dbPath, estNumImages=estNumImages,
			                maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
			                verbose=verbose))
                
                # open the HDF5 database for writing, initializing the database within
                # the group, the BOVW buffer list, and the BOVW index into dataset
                self.db = h5py.File(self.dbPath, mode="w")
                self.bovwDB = None
                self.bovwBuffer = None
                self.idxs = {"bovw" : 0}

                # store the feature vector size of the bag-of-visual-words, then
                # initialize the document frequency counts to be accumulated and
                # actual total number of images in the database
                self.fvectorSize = fvectorSize
                self._df = np.zeros((fvectorSize,), dtype="float")
                self.totalImages = 0
                
