import numpy as np

def getk(sys, block, tau):
    wfn = sys.wfns[block][:, tau - sys.idxs[block]]
    idx = np.where(wfn == 1)[0][0]
    return sys.kranges[block][idx]