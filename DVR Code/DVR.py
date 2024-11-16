import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants as pc
import scipy

#behold, the constants required to work in atomic units
br = pc['Bohr radius'][0]
me = pc['electron mass'][0]
He = pc['Hartree energy'][0]
c = scipy.constants.c * 100
h = scipy.constants.h
mtrans = pc['atomic mass constant'][0] / me
amkg = pc['atomic mass constant'][0]
rtrans = 10 ** -10 / br


class DVR():
    def __init__(self, nu, chi, re, mu, name):
        """
                Numerically solve the Schrodinger equation for a morse potential surface, and calculate the intensities
                of the fundamental and overtone transition
                Input:
                nu - the vibrational frequency for a transition (fundamental or overtone) [cm-1]
                chi - the anharmonicity parameter [unitless]
                re - bond length [angstroms]
                Output:
                DVR object
                """
        self.De = nu / (4 * chi) * h * c / He # Hartree
        self.alpha = np.sqrt(8 * np.pi ** 2 * c * amkg * mu * nu*chi / h) * br # 1/ bohr
        self.re = re * rtrans #angstrom -> bohr
        self.mu = mu*mtrans
        self.name = name

    def runDVR(self, N = 500):
        """
        Run the DVR
        :param N: number of grid points to consider in colbert-miller grid
        :return:
                energies - ordered list of energy levels [cm-1]
                wavefunctions - energy ordered matrix of position-space wavefunctions (columns of matrix)
                grid - the positions for plotting (x eigenvalues)
        """
        #get DVR grid
        grid = dvr_grid([0,3*self.re], N)
        #calculate colbert-miller T
        T = CM_kinE(grid, self.mu)
        #calculate morse potential
        V = MO_potE(grid, self.re, self.De, self.alpha)
        #construct hamiltonian and diagonalize
        energies, wfns = H(T, V)
        #UNITS
        escale = He / (h * c)
        xmatog = np.diag(grid)
        grid = grid / rtrans
        xmat = np.diag(grid)
        plt.figure(figsize=(8, 9))
        #plot potential, wavefunctions, and expected value of <x>
        plt.plot(grid, np.diag(V) * escale, label='Potential')
        plt.plot(grid, (wfns.T[0] ** 2 + energies[0]) * escale, color='orange', label='n = 0')
        plt.axvline(wfns.T[0] @ xmat @ wfns.T[0].T, linestyle=':', color='orange', linewidth=2, label='n = 0')
        plt.plot(grid, (wfns.T[1] ** 2 + energies[1]) * escale, color='green', label='n = 1')
        plt.axvline(wfns.T[1] @ xmat @ wfns.T[1].T, linestyle=':', color='green', linewidth=2, label='n = 1')
        plt.plot(grid, (wfns.T[2] ** 2 + energies[2]) * escale, color='red', label='n = 2')
        plt.axvline(wfns.T[2] @ xmat @ wfns.T[2].T, linestyle=':', color='red', linewidth=2, label='n = 2')
        plt.xlim([0.75, 2.0])
        plt.ylim([-500, 60000])

        #calculate intensities!
        f = (wfns.T[1] @ xmatog @ wfns.T[0].T)**2
        print(f)
        o = (wfns.T[2] @ xmatog @ wfns.T[0].T)**2

        y_min, y_max = plt.ylim()
        middle_y = (y_min + y_max) / 2
        plt.text(1.4, middle_y, f'Fundamental intensity = {r4(f)}\nOvertone intensity = {r4(o)}',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

        plt.title('Plot of $\\langle x \\rangle$ for ' + self.name, fontsize=18)
        plt.xlabel('Bond Length [Angstroms]', fontsize=14)
        plt.ylabel('Energy [(cm$^{-1}$)]', fontsize=14)
        plt.legend(loc='best')
        plt.show()

        #turn on this line if you would like output
        return energies, wfns, grid




def r4(arr):
    #get 4 significant figures for displaying on plot
    def round_single_value(val):
        if val == 0:
            return 0
        else:
            return np.round(val, decimals=3 - int(np.floor(np.log10(np.abs(val)))))

    vectorized_round = np.vectorize(round_single_value)
    return vectorized_round(arr)

def dvr_grid(domain, NumGP):
    #get evenly spaced colbert-miller type grid
    grid = np.linspace(domain[0], domain[1], NumGP)
    return grid


def CM_kinE(grid, mu):
    #populate the kinetic energy matrix according to colbert-miller equation A7
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
    #evaluate the Morse potential value at each position eigenvalue
    V = np.zeros((len(grid), len(grid)))
    for i in range(len(grid)):
        V[i][i] = De * (1 - np.exp(-alpha * (grid[i] - re))) ** 2
    return V


def H(T, V):
    #add together the potential and kinetic energy, and then diagonalize in the position basis
    H = T + V
    energies, wfns = np.linalg.eigh(H)
    return energies, wfns


#using my experimental values - example usage
nu = 2992
chi = 0.01755
m1 = 1.007825
m2 = 34.968852
mu = (m1 * m2) / (m1 + m2)
t = DVR(nu, chi,1.279, mu, 'HCl')
t.runDVR()


nu = 2144
chi = 0.01236
m1 = 2.0140
m2 = 34.968852
mu = (m1 * m2) / (m1 + m2)
t = DVR(nu, chi,1.279, mu, 'DCl')
t.runDVR()

