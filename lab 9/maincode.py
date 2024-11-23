import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import scipy as sc

p = 747 #torr
T = 23.0

vAlc = np.linspace(0,1,11) #in ml
vTol = vAlc[::-1]

vfAlc = vAlc/(vAlc+vTol)
vfTol = vTol/(vAlc+vTol)

mmtol = 92.14 #g/mol
dtol = 0.8669 #g/ml
mdtol = dtol/mmtol

mmalc = 60.0952
dalc = 0.803
mdalc = dalc/mmalc

molAlc = mdalc*vAlc
molTol = mdtol*vTol

mfAlc = molAlc/(molAlc+molTol)
mfTol = molTol/(molAlc+molTol)

ri = np.array([1.4962, 1.4849, 1.4741, 1.4624, 1.4510, 1.4401, 1.4292, 1.4181, 1.4065, 1.3954, 1.3850])

plt.figure()
plt.scatter(vfAlc, ri)
plt.show()

plt.figure()
plt.scatter(mfAlc,ri)
plt.show()