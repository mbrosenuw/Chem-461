import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants as pc
import scipy

br = pc['Bohr radius'][0]
me = pc['electron mass'][0]
He = pc['Hartree energy'][0]
c = scipy.constants.c * 100
h = scipy.constants.h
mtrans = pc['atomic mass constant'][0] / me
amkg = pc['atomic mass constant'][0]
rtrans = 10 ** -10 / br
m1 = 1.007825
m2 = 34.968852
args = {
    "mass1": m1 * mtrans,  # mass electron
    "mass2": m2 * mtrans,  # Hartree
    "re": 1.279 * rtrans,  # bohr
    "alpha": np.sqrt(8 * np.pi ** 2 * c * amkg * (m1 * m2) / (m1 + m2) * 52.5 / h) * br,  # 1/bohr
    "De": 2992.0 / (4 * 0.01755) * h * c / He,  # Hartree
    'EStrength': 0.05,
}

m1 = 2.0140
m2 = 34.968852
args2 = {
    "mass1": m1 * mtrans,  # mass electron
    "mass2": m2 * mtrans,  # Hartree
    "re": 1.279 * rtrans,  # bohr
    "alpha": np.sqrt(8 * np.pi ** 2 * c * amkg * (m1 * m2) / (m1 + m2) * 26.5 / h) * br,  # 1/bohr
    "De": 2144.0 / (4 * 0.01236) * h * c / He,  # Hartree
    'EStrength': 0.05,
}


def dvr_grid(domain, NumGP):
    grid = np.linspace(domain[0], domain[1], NumGP)
    return grid


def CM_kinE(grid, mu):
    T = np.zeros((len(grid), len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid)):
            coeff = 1 / (2 * mu * np.diff(grid)[0] ** 2) * (-1) ** (i - j)
            if i == j:
                val = np.pi ** 2 / 3
            else:
                val = 2 / (i - j) ** 2
            T[i][j] = coeff * val
    return T


def MO_potE(grid, re, De, alpha):
    V = np.zeros((len(grid), len(grid)))
    for i in range(len(grid)):
        V[i][i] = De * (1 - np.exp(-alpha * (grid[i] - re))) ** 2
    return V


def H(T, V):
    H = T + V
    energies, wfns = np.linalg.eigh(H)
    return energies, wfns


def runDVR(domain, N, args):
    # mu = (args['mass1'] * args['mass2'])/(args['mass1'] + args['mass2'])
    mu = args['mass1'] * args['mass2'] / (args['mass1'] + args['mass2'])
    grid = dvr_grid(domain, N)
    T = CM_kinE(grid, mu)
    V = MO_potE(grid, args['re'], args['De'], args['alpha'])
    energies, wfns = H(T, V)
    return energies, wfns, grid, V


energies, wfns, grid, V = runDVR((1, 8), 500, args)
escale = He / (h * c)
grid = grid/rtrans
xmat = np.diag(grid)

plt.figure(figsize=(8,9))
plt.plot(grid, np.diag(V) * escale, label = 'Potential')
plt.plot(grid, (wfns.T[0]**2 + energies[0]) * escale, color = 'orange', label = 'n = 0')
plt.axvline(wfns.T[0] @ xmat @ wfns.T[0].T, linestyle=':', color = 'orange', linewidth=2, label = 'n = 0')

plt.plot(grid, (wfns.T[1]**2 + energies[1]) * escale, color = 'green', label = 'n = 1')
plt.axvline(wfns.T[1] @ xmat @ wfns.T[1].T, linestyle=':', color = 'green', linewidth=2, label = 'n = 1')
plt.plot(grid, (wfns.T[2]**2 + energies[2]) * escale, color = 'red', label = 'n = 2')
plt.axvline(wfns.T[2] @ xmat @ wfns.T[2].T, linestyle=':', color = 'red', linewidth=2, label = 'n = 2')
plt.xlim([0.75,2.0])
plt.ylim([-500,60000])
plt.title('Plot of $\\langle x \\rangle$ for HCl', fontsize=18)
plt.xlabel('Bond Length [Angstroms]', fontsize=14)
plt.ylabel('Energy [(cm$^{-1}$)]', fontsize=14)
plt.legend(loc='best')
plt.show()

energies, wfns, grid, V = runDVR((1, 8), 500, args2)
escale = He / (h * c)
grid = grid/rtrans
xmat = np.diag(grid)
plt.figure(figsize=(8,9))
plt.plot(grid, np.diag(V) * escale, label = 'Potential')
plt.plot(grid, (wfns.T[0]**2 + energies[0]) * escale, color = 'orange', label = 'n = 0')
plt.axvline(wfns.T[0] @ xmat @ wfns.T[0].T, linestyle=':', color = 'orange', linewidth=2, label = 'n = 0')

plt.plot(grid, (wfns.T[1]**2 + energies[1]) * escale, color = 'green', label = 'n = 1')
plt.axvline(wfns.T[1] @ xmat @ wfns.T[1].T, linestyle=':', color = 'green', linewidth=2, label = 'n = 1')
plt.plot(grid, (wfns.T[2]**2 + energies[2]) * escale, color = 'red', label = 'n = 2')
plt.axvline(wfns.T[2] @ xmat @ wfns.T[2].T, linestyle=':', color = 'red', linewidth=2, label = 'n = 2')
plt.xlim([0.75,2.0])
plt.ylim([-500,60000])
plt.title('Plot of $\\langle x \\rangle$ for DCl', fontsize=18)
plt.xlabel('Bond Length [Angstroms]', fontsize=14)
plt.ylabel('Energy [(cm$^{-1}$)]', fontsize=14)
plt.legend(loc='best')
plt.show()
