import numpy as np

def getidx(sys, block, k):
    idx = np.where(sys.kranges[block] == k)[0][0]
    return idx