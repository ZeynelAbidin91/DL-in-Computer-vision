from __future__ import print_function
import sys
sys.path.insert(0,"..")
from pyimagesearch.ir import BagOfVisualWords
import numpy as np

np.random.seed(42)
vocab = np.random.uniform(size=(5, 36))
features = np.random.uniform(size=(500, 36))
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))
