import numpy as np
from hamiltonian import getH
from eigenblock import eigenblock
from symmetrize import Htransform

class Rotor:
    def __init__(self, j, consts):
        self.j = j
        self.a = consts[0]
        self.b = consts[1]
        self.c = consts[2]
        self.oldH = getH(self)
        self.H, self.ks = Htransform(getH(self))
        if not np.any(np.round(np.imag(self.H),5)):
            self.H = np.real(self.H)
        else:
            print('imaginary component of Hamiltonian present')
        [self.fullenergies, self.energies, self.wfns, self.idxs, self.kranges] = eigenblock(self.H, j)
