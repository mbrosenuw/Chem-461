import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from rotor import Rotor
from boltzmann import getdenom, boltzmann
from getasymtracks import gettracks
from asymLS import fastLS as ls
import tjsyms
from tqdm import tqdm
import time

def spectra(coeffs, ucoeffs, mu, jmax, T,name, lims, width,showE, stats = [1,1,1,1]):
    tic = time.time()
    freq = np.arange(lims[0], lims[1], width/5)
    spacing = width / np.diff(freq)[0]
    window = np.round(np.array([-5 * spacing, 5 * spacing]), 0).astype('int')
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
    tjs = tjsyms.loadtjs('./tjs.csv')
    tjsh = tjs.copy()
    def trans(track):
        E1 = track[10]
        E2 = track[11]
        spectra = np.zeros((freq.shape[0]))
        dE = E2 - E1
        idx = np.abs(freq - dE).argmin()
        samplespace = idx + window
        if samplespace[1] > freq.shape[0]:
            samplespace[1] = freq.shape[0] - 1
        elif samplespace[0] < 0:
            samplespace[0] = 0
        sample = freq[samplespace[0]:samplespace[1]]

        S = ls(ujsys[int(track[0])], int(track[1]), int(track[2]), int(track[3]), int(track[4]), jsys[int(track[5])],  int(track[6]), int(track[7]), int(track[8]), int(track[9]),mu, tjs, stats)
        B = boltzmann(E1, T, denom)
        spectra[samplespace[0]:samplespace[1]] = (S * B * np.exp(-((sample - dE) ** 2) / (2 * width ** 2)))
        return spectra

    spectra = np.zeros((tracks.shape[0], freq.shape[0]))
    with tqdm(total=tracks.shape[0], desc="calculating spectra") as pbar:
        for i, track in enumerate(tracks):
            spectra[i][:]= trans(track)
            pbar.update(1)
    spectra = np.sum(spectra, axis = 0)
    toc = time.time()
    elapsed_time = toc - tic
    print(f"Elapsed time: {elapsed_time} seconds")

    # if tjsh != tjs:
    #     tjsyms.writetjs('./tjs.csv', tjs)
    return freq, spectra