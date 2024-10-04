import numpy as np
from scipy.constants import physical_constants as pc

Kb = pc['Boltzmann constant in inverse meter per kelvin'][0]/100

def getdenom(jmax, syslist,T):
    def calcelem(energy):
        return np.exp(-energy / (Kb * T))
    vc = np.vectorize(calcelem)
    denom = 0
    for j in range(jmax+1):
        denom += np.sum(vc(syslist[j].fullenergies))
    return denom

def boltzmann(E, T, denom):
    num = np.exp(-E/(Kb*T))
    return num/denom
