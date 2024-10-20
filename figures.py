
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

filename = 'Data Set 1_SPR2020.csv'
specT = np.array([])
freq2 = np.array([])
if os.path.exists(filename):
    with open(filename, "r", newline="",encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            freq2 = np.append(freq2, float(row[0]))
            specT = np.append(specT, float(row[1]))

specA = -np.log10((specT+14)/np.max((specT+14)))
spec2 = specA/np.max(specA)


# s1 = [1,1,20]
# freq2, spec2 = asymrotor.spectra(s1, s1, [0,0,1], jmax=20, T = 100, name="prolate", lims = [-100,100], width = 0.1, showE=False)
fig, axs = plt.subplots(nrows=2, ncols=1, figsize = (6,5))

# plt.plot( freq + shift, -spec, color='red', linewidth=0.5)
ax = axs[1]
ax.plot(freq2, spec2, color='blue', linewidth=0.5)
ax.set_title('Rotational spectrum of DCl', fontsize=18)
ax.set_xlabel('Energy (cm-1)', fontsize=14)
ax.set_ylabel('Intensity', fontsize=14)
ax.set_xlim([1900, 2250])

# s2 = [20,20,1]
# freq, spec = asymrotor.spectra(s2, s2, [0,0,1], jmax=20, T = 100, name="oblate", lims = [-300, 300], width = 0.1, showE=False)
ax = axs[0]
ax.plot(freq2, spec2, color='blue', linewidth=0.5)
ax.set_title('Vibrational spectrum of DCl', fontsize= 18)
ax.set_xlabel('Energy (cm-1)', fontsize= 14)
ax.set_ylabel('Intensity', fontsize= 14)
# Highlight a vertical region between x=2 and x=4
ax.axvspan(1900, 2250, color='green', alpha=0.3)

plt.subplots_adjust(top=0.88, hspace=0.75)
plt.show()

