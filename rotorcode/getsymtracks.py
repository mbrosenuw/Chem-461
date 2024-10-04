from tqdm import tqdm
from getblock import getblock
import numpy as np
from getk import getk
def gettracks(jsys, ujsys, jmax):
    tracks = []
    for jpp in tqdm(range(jmax + 1)):
        if jpp == 0:
            jmin = 0
        else:
            jmin = jpp - 1

        if jpp == jmax:
            topj = jmax
        else:
            topj = jpp + 1

        for tpp in range(0, 2 * jpp + 1):
            blockpp, phasepp, eopp = getblock(jsys[jpp].j, tpp, jsys[jpp].idxs)
            kpp = getk(jsys[jpp], blockpp, tpp)
            E1 = jsys[jpp].energies[blockpp][tpp - jsys[jpp].idxs[blockpp]]
            for jp in range(jmin, topj + 1):
                for tp in range(0, 2 * jp + 1):
                    blockp, phasep, eop = getblock(ujsys[jp].j, tp,ujsys[jp].idxs)
                    kp = getk(ujsys[jp], blockp, tp)
                    if np.abs(kp-kpp) <= 1:
                        E2 = ujsys[jp].energies[blockp][tp - ujsys[jp].idxs[blockp]]
                        tracks.append([jp, kp, phasep, jpp, kpp, phasepp, E1, E2])
    return np.array(tracks)