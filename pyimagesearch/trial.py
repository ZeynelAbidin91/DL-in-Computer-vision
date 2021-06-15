import numpy as np
#from scipy.spatial import distance as dists

A = [0.16, 0.04, 0.12, 0.03, 0.14, 0.17, 0.06, 0.27]
B = [0.08, 0.12, 0.07, 0.13, 0.04, 0.22, 0.27, 0.08]
'''
def chi2_distance(histA, histB):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB))

print(chi2_distance(A, B))
'''
#print(((A - B) ** 2 )/(A + B))
dists