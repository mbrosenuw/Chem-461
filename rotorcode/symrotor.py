import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from rotor import Rotor
from boltzmann import getdenom, boltzmann
from getsymtracks import gettracks
from symLS import linestrength as ls
from tqdm import tqdm
def spectra(coeffs, ucoeffs, mu, jmax, T,name, lims, width,showE):
    freq = np.arange(lims[0], lims[1], width/5)
    spacing = width / np.diff(freq)[0]
    window = np.round(np.array([-3 * spacing, 3 * spacing]), 0).astype('int')
    jsys = []
    ujsys = []

    lines1 = []
    lines2 = []
    for j in range(jmax + 1):
        sys = Rotor(j, coeffs)
        jsys.append(sys)
        sys2 = Rotor(j, ucoeffs)
        ujsys.append(sys2)
        lines1.append(sys.fullenergies)
        lines2.append(sys2.fullenergies)

    if showE:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax1 = axes[0]
        ax2 = axes[1]
        legend = []
        color = iter(cm.rainbow(np.linspace(0, 1, jmax + 1)))
        for j in range(jmax + 1):
            c = next(color)
            legend.append('J = ' + str(j))
            ax1.hlines(jsys[j].fullenergies, xmin=j, xmax=j + 1, colors=c)
            ax2.hlines(ujsys[j].fullenergies, xmin=j, xmax=j + 1, colors=c)
        ax1.legend(legend, loc='upper left')
        ax1.set_title('Energy levels for ' + name)
        ax1.set_xlabel('J Value')
        ax1.set_ylabel('Energy (cm-1)')
        ax2.legend(legend, loc='upper left')
        ax2.set_title('Upper Vibrational Level Energy levels for ' + name)
        ax2.set_xlabel('J Value')
        ax2.set_ylabel('Energy (cm-1)')
        plt.show()

    denom = getdenom(jmax, jsys, T)
    tracks = gettracks(jsys, ujsys, jmax)

    def trans(track):
        E1 = track[6]
        E2 = track[7]
        spectra = np.zeros((freq.shape[0]))
        dE = E2 - E1
        idx = np.abs(freq - dE).argmin()
        samplespace = idx + window
        if samplespace[1] > freq.shape[0]:
            samplespace[1] = freq.shape[0] - 1
        elif samplespace[0] < 0:
            samplespace[0] = 0
        sample = freq[samplespace[0]:samplespace[1]]

        S = ls(int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4]), int(track[5]), mu)
        B = boltzmann(E1, T, denom)
        spectra[samplespace[0]:samplespace[1]] = (S * B * np.exp(-((sample - dE) ** 2) / (2 * width ** 2)))
        return spectra

    spectra = np.zeros((tracks.shape[0], freq.shape[0]))
    with tqdm(total=tracks.shape[0], desc="calculating spectra") as pbar:
        for i, track in enumerate(tracks):
            spectra[i][:]= trans(track)
            pbar.update(1)
    spectra = np.sum(spectra, axis = 0)
    plt.figure()
    plt.plot(freq, spectra, color='red', linewidth=0.5)
    plt.xlim(lims)
    plt.title('Rotational spectrum of ' + name)
    plt.xlabel('Energy (cm-1)')
    plt.ylabel('Intensity')
    plt.show()

def kspectra(coeffs, ucoeffs, mu, jmax, T,name, lims, width,showE):
    freq = np.arange(lims[0], lims[1], width/5)
    spacing = width / np.diff(freq)[0]
    window = np.round(np.array([-3 * spacing, 3 * spacing]), 0).astype('int')
    jsys = []
    ujsys = []

    lines1 = []
    lines2 = []
    for j in range(jmax + 1):
        sys = Rotor(j, coeffs)
        jsys.append(sys)
        sys2 = Rotor(j, ucoeffs)
        ujsys.append(sys2)
        lines1.append(sys.fullenergies)
        lines2.append(sys2.fullenergies)

    if showE:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax1 = axes[0]
        ax2 = axes[1]
        legend = []
        color = iter(cm.rainbow(np.linspace(0, 1, jmax + 1)))
        for j in range(jmax + 1):
            c = next(color)
            legend.append('J = ' + str(j))
            ax1.hlines(sys.energies, xmin=j, xmax=j + 1, colors=c)
            ax2.hlines(sys2.energies, xmin=j, xmax=j + 1, colors=c)
        ax1.legend(legend, loc='upper left')
        ax1.set_title('Energy levels for ' + name)
        ax1.set_xlabel('J Value')
        ax1.set_ylabel('Energy (cm-1)')
        ax2.legend(legend, loc='upper left')
        ax2.set_title('Upper Vibrational Level Energy levels for ' + name)
        ax2.set_xlabel('J Value')
        ax2.set_ylabel('Energy (cm-1)')
        plt.show()

    denom = getdenom(jmax, jsys, T)
    tracks = gettracks(jsys, ujsys, jmax)

    def trans(track):
        E1 = track[6]
        E2 = track[7]
        spectra = np.zeros((freq.shape[0]))
        dE = E2 - E1
        dk = track[0]-track[3]
        idx = np.abs(freq - dE).argmin()
        samplespace = idx + window
        if samplespace[1] > freq.shape[0]:
            samplespace[1] = freq.shape[0] - 1
        elif samplespace[0] < 0:
            samplespace[0] = 0
        sample = freq[samplespace[0]:samplespace[1]]

        S = ls(int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4]), int(track[5]), mu)
        B = boltzmann(E1, T, denom)
        spectra[samplespace[0]:samplespace[1]] = (S * B * np.exp(-((sample - dE) ** 2) / (2 * width ** 2)))
        return spectra,dk

    spectrap = np.zeros((freq.shape[0]))
    spectra = np.zeros((freq.shape[0]))
    spectram = np.zeros((freq.shape[0]))
    for track in tracks:
        line, dk = trans(track)
        if dk == -1:
            spectram = spectram + line
        elif dk == 0:
            spectra = spectra + line
        elif dk == 1:
            spectrap = spectrap + line

    f, axes = plt.subplots(3, 1)
    axes[0].plot(freq, spectram, color='red')
    axes[0].set_xlim(lims)
    axes[1].plot(freq, spectra, color='red')
    axes[1].set_xlim(lims)
    axes[2].plot(freq, spectrap, color='red')
    axes[2].set_xlim(lims)
    plt.suptitle('Rotational spectrum of ' + name)
    plt.show()