import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import symrotor

filename = 'Data Set 1_SPR2020.csv'
spec2 = np.array([])
freq2 = np.array([])
if os.path.exists(filename):
    with open(filename, "r", newline="", encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            freq2 = np.append(freq2, float(row[0]))
            spec2 = np.append(spec2, float(row[1]))

spec2 = np.log10(np.reciprocal(spec2 + 100 - 85))
spec2 = spec2 - np.min(spec2)
spec2 = spec2 / np.max(spec2)

hcl = [10.52] * 3
lims = [-250, 250]
hclshift = 2887
hclfreq, hclspec = symrotor.spectra(hcl, hcl, [1, 1, 1], 30, 300, 'hcl', lims, 0.5, False)
hclspec = hclspec / np.max(hclspec)

condition = (hclfreq < -5) | (hclfreq > 5)

# Apply the condition to x and y
hclfreq = hclfreq[condition]
hclspec = hclspec[condition]
hclspec = hclspec - np.min(hclspec)
hclspec = hclspec / np.max(hclspec)

hclo = [10.56] * 3
lims = [-250, 250]
hcloshift = 5669
hclofreq, hclospec = symrotor.spectra(hclo, hclo, [1, 1, 1], 30, 300, 'hclo', lims, 0.5, False)
hclospec = hclospec / np.max(hclospec)

condition = (hclofreq < -5) | (hclofreq > 5)

# Apply the condition to x and y
hclofreq = hclofreq[condition]
hclospec = hclospec[condition]
hclospec = hclospec - np.min(hclospec)
hclospec = hclospec / np.max(hclospec) * 0.013

dcl = [5.419] * 3
lims = [-250, 250]
dclshift = 2091
dclfreq, dclspec = symrotor.spectra(dcl, dcl, [1, 1, 1], 30, 300, 'dcl', lims, 0.5, False)
dclspec = dclspec / np.max(dclspec)

condition = (dclfreq < -5) | (dclfreq > 5)

# Apply the condition to x and y
dclfreq = dclfreq[condition]
dclspec = dclspec[condition]
dclspec = dclspec - np.min(dclspec)
dclspec = dclspec / np.max(dclspec) * 0.47

dclo = [5.436] * 3
lims = [-250, 250]
dcloshift = 4129
dclofreq, dclospec = symrotor.spectra(dclo, dclo, [1, 1, 1], 30, 300, 'dclo', lims, 0.5, False)
dclospec = dclospec / np.max(dclospec)

condition = (dclofreq < -5) | (dclofreq > 5)

# Apply the condition to x and y
dclofreq = dclofreq[condition]
dclospec = dclospec[condition]
dclospec = dclospec - np.min(dclospec)
dclospec = dclospec / np.max(dclospec) * 0.005

plt.figure(figsize=(5, 8))

spec2 = spec2 - np.min(spec2)
spec2 = spec2 / np.max(spec2)
plt.plot(freq2, spec2, label='Experimental Spectrum')
plt.plot(hclfreq + hclshift, -hclspec, label='HCl Fundamental (theory)')
plt.plot(hclofreq + hcloshift, -hclospec, label='HCl Overtone (theory)')
plt.plot(dclfreq + dclshift, -dclspec, label='DCl Fundamental (theory)')
plt.plot(dclofreq + dcloshift, -dclospec, label='DCl Overtone (theory)')
plt.ylabel('Relative Intensity', fontsize=14)
plt.xlabel('E/hc (cm-1)', fontsize=14)
plt.legend(loc='best')
plt.title('Theoretical fit for HCl/DCl Rotation-Vibration spectrum', fontsize=16)
plt.show()
