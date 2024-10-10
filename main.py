import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import scipy.signal as sc


filename = 'Data Set 1_SPR2020.csv'
specT = np.array([])
freq = np.array([])
if os.path.exists(filename):
    with open(filename, "r", newline="",encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            freq = np.append(freq, float(row[0]))
            specT = np.append(specT, float(row[1]))

specA = -np.log10((specT+14)/np.max((specT+14)))
# specA = specA/np.max(specA)
# plt.figure()
# plt.plot(freq,specA)
# plt.show()

from uwpchem461 import Analyse
analyse = Analyse()
# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

# # Plotting the dataset for specT vs wavenumber
# axs[0].plot(freq, specT, linestyle='solid',
#          marker='None', color='black', linewidth=1)
# axs[0].set_xlabel('Energy [1/cm]')
# axs[0].set_ylabel('Transmission [%]')
#
# # Plotting the dataset for absorbance vs wavenumber
# axs[1].plot(freq, specA, linestyle='solid',
#          marker='None', color='black', linewidth=1)
# axs[1].set_xlabel('Energy [1/cm]')
# axs[1].set_ylabel('Absorbance [Au]')
# fig.suptitle('Rotation-Vibration Spectrum for HCl/DCl mixture')
# plt.show()

cDClfundp = (freq > 1900) & (freq < 2250)
cDCloverp = (freq > 3950) & (freq < 4250)
cHClfundp = (freq >2550) & (freq <3150)
cHCloverp = (freq > 5400) & (freq < 5850)

DClfundp = np.vstack((freq[cDClfundp], specA[cDClfundp]/np.max(specA[cDClfundp]))).T
DCloverp = np.vstack((freq[cDCloverp], specA[cDCloverp]/np.max(specA[cDCloverp]))).T
HClfundp = np.vstack((freq[cHClfundp], specA[cHClfundp]/np.max(specA[cHClfundp]))).T
HCloverp = np.vstack((freq[cHCloverp], specA[cHCloverp]/np.max(specA[cHCloverp]))).T

cDClfund = (freq > 1900) & (freq < 2261)
cDClover = (freq > 3989) & (freq < 4225)
cHClfund = (freq >2550) & (freq <3150)
cHClover = (freq > 5484) & (freq < 5808)

DClfund = np.vstack((freq[cDClfund], specA[cDClfund]/np.max(specA[cDClfund]))).T
DClover = np.vstack((freq[cDClover], specA[cDClover]/np.max(specA[cDClover]))).T
HClfund = np.vstack((freq[cHClfund], specA[cHClfund]/np.max(specA[cHClfund]))).T
HClover = np.vstack((freq[cHClover], specA[cHClover]/np.max(specA[cHClover]))).T

bDCLf = analyse.getbase(DClfund.tolist(), 0.07)
bDCLo = analyse.getbase(DClover.tolist(), 0.33)
bHCLf = analyse.getbase(HClfund.tolist(), 0.075)
bHCLo = analyse.getbase(HClover.tolist(), 0.1)


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
axs[0,0].plot(DClfundp[:,0], DClfundp[:,1], label = 'DCl Fundamental')
axs[0,0].plot(DClfund[:,0], bDCLf, label = 'baseline')
axs[0,0].set_title('DCl Fundamental')
axs[0,0].set_xlabel('Energy [cm^-1]')
axs[0,0].set_ylabel('Abs [Au]')
axs[1,0].plot(DCloverp[:,0], DCloverp[:,1], label = 'DCl Overtone')
axs[1,0].plot(DClover[:,0], bDCLo, label = 'baseline')
axs[1,0].set_title('DCl Overtone')
axs[1,0].set_xlabel('Energy [cm^-1]')
axs[1,0].set_ylabel('Abs [Au]')
axs[0,1].plot(HClfundp[:,0], HClfundp[:,1], label = 'HCL Fundamental')
axs[0,1].plot(HClfund[:,0], bHCLf, label = 'baseline')
axs[0,1].set_title('HCl Fundamental')
axs[0,1].set_xlabel('Energy [cm^-1]')
axs[0,1].set_ylabel('Abs [Au]')
axs[1,1].plot(HCloverp[:,0], HCloverp[:,1], label='HCl Overtone')
axs[1,1].plot(HClover[:,0], bHCLo, label ='baseline')
axs[1,1].set_title('HCl Overtone')
axs[1,1].set_xlabel('Energy [cm^-1]')
axs[1,1].set_ylabel('Abs [Au]')
fig.suptitle('HCl and DCl Bands')
plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
plt.show()

DClfund[:,1] = DClfund[:,1] -bDCLf
DClover[:,1] = DClover[:,1] -bDCLo
HClfund[:,1] = HClfund[:,1] -bHCLf
HClover[:,1] = HClover[:,1] -bHCLo
DClfund[(DClfund < 0)] = 0
DClover[(DClover < 0)] = 0
HClfund[(HClfund < 0)] = 0
HClover[(HClover < 0)] = 0

DClft = DClfund[(DClfund[:,0] > 1926) & (DClfund[:,0] < 2212.7),:]
DClot = DClover[((DClover[:,0] > 3989) & (DClover[:,0] < 4115)) | ((DClover[:,0] > 4117) & (DClover[:,0] < 4196)),:]
HClft = HClfund[(HClfund[:,0] > 2626.7) & (HClfund[:,0] <3085),:]
HClot = HClover[(HClover[:,0] > 5484) & (HClover[:,0] < 5796),:]

DClfpeaks, _ = sc.find_peaks(DClft[:,1], height=0.05)
DClopeaks, _ = sc.find_peaks(DClot[:,1], height=0.05)
HClfpeaks, _ = sc.find_peaks(HClft[:,1], height=0.05)
HClopeaks, _ = sc.find_peaks(HClot[:,1], height=0.05)
DClfpf = DClft[DClfpeaks,:]
DClopf = DClot[DClopeaks,:]
HClfpf = HClft[HClfpeaks,:]
HClopf = HClot[HClopeaks,:]

d37f = DClfpf[1::2, :]
d35f = DClfpf[::2, :]
d37o = DClopf[::2, :]
d35o = DClopf[1::2, :]
h35f = HClfpf[::2, :]
h37f = HClfpf[1::2, :]
h35o = HClopf[::2, :]
h37o = HClopf[1::2, :]

d35fp = d35f[d35f[:,0] < 2090, :]
d35fr = d35f[d35f[:,0] > 2090, :]
d37fp = d37f[d37f[:,0] < 2090, :]
d37fr = d37f[d37f[:,0] > 2090, :]
h35fp = h35f[h35f[:,0] < 2880, :]
h35fr = h35f[h35f[:,0] > 2880, :]
h37fp = h37f[h37f[:,0] < 2880, :]
h37fr = h37f[h37f[:,0] > 2880, :]

d35op = d35o[d35o[:,0] < 4125, :]
d35or = d35o[d35o[:,0] > 4125, :]
d37op = d37o[d37o[:,0] < 4125, :]
d37or = d37o[d37o[:,0] > 4125, :]
h35op = h35o[h35o[:,0] < 5667, :]
h35or = h35o[h35o[:,0] > 5667, :]
h37op = h37o[h37o[:,0] < 5667, :]
h37or = h37o[h37o[:,0] > 5667, :]

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
# axs[0,0].plot(DClfund[:,0], DClfund[:,1])
# axs[0,0].scatter(d35fp[:,0], d35fp[:,1], label = 'DCl(35) P', color = 'red', marker = 's')
# axs[0,0].scatter(d35fr[:,0], d35fr[:,1], label = 'DCl(35) R', color = 'orange', marker = 'o')
# axs[0,0].scatter(d37fp[:,0], d37fp[:,1], label = 'DCl(37) P', color = 'green', marker = '^')
# axs[0,0].scatter(d37fr[:,0], d37fr[:,1], label = 'DCl(37) R', color = 'purple', marker = '*')
# axs[0,0].set_title('DCl Fundamental')
# axs[0,0].set_xlabel('Energy [cm^-1]')
# axs[0,0].set_ylabel('Abs [Au]')
# axs[0,0].legend(loc = 'best')
# axs[1,0].plot(DClover[:,0], DClover[:,1])
# axs[1,0].scatter(d35op[:,0], d35op[:,1], label = 'DCl(35) P', color = 'red', marker = 's')
# axs[1,0].scatter(d35or[:,0], d35or[:,1], label = 'DCl(35) R', color = 'orange', marker = 'o')
# axs[1,0].scatter(d37op[:,0], d37op[:,1], label = 'DCl(37) P', color = 'green', marker = '^')
# axs[1,0].scatter(d37or[:,0], d37or[:,1], label = 'DCl(37) R', color = 'purple', marker = '*')
# axs[1,0].set_title('DCl Overtone')
# axs[1,0].set_xlabel('Energy [cm^-1]')
# axs[1,0].set_ylabel('Abs [Au]')
# axs[1,0].legend(loc = 'best')
# axs[0,1].plot(HClfund[:,0], HClfund[:,1])
# axs[0,1].scatter(h35fp[:,0], h35fp[:,1], label = 'HCl(35) P', color = 'red', marker = 's')
# axs[0,1].scatter(h35fr[:,0], h35fr[:,1], label = 'HCl(35) R', color = 'orange', marker = 'o')
# axs[0,1].scatter(h37fp[:,0], h37fp[:,1], label = 'HCl(37) P', color = 'green', marker = '^')
# axs[0,1].scatter(h37fr[:,0], h37fr[:,1], label = 'HCl(37) R', color = 'purple', marker = '*')
# axs[0,1].set_title('HCl Fundamental')
# axs[0,1].set_xlabel('Energy [cm^-1]')
# axs[0,1].set_ylabel('Abs [Au]')
# axs[0,1].legend(loc = 'best')
# axs[1,1].plot(HClover[:,0], HClover[:,1])
# axs[1,1].scatter(h35op[:,0], h35op[:,1], label = 'HCl(35) P', color = 'red', marker = 's')
# axs[1,1].scatter(h35or[:,0], h35or[:,1], label = 'HCl(35) R', color = 'orange', marker = 'o')
# axs[1,1].scatter(h37op[:,0], h37op[:,1], label = 'HCl(37) P', color = 'green', marker = '^')
# axs[1,1].scatter(h37or[:,0], h37or[:,1], label = 'HCl(37) R', color = 'purple', marker = '*')
# axs[1,1].set_title('HCl Overtone')
# axs[1,1].set_xlabel('Energy [cm^-1]')
# axs[1,1].set_ylabel('Abs [Au]')
# axs[1,1].legend(loc = 'best')
# fig.suptitle('Corrected HCl and DCl Bands')
# plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
# plt.show()

def label(p, r):
    p = p[p[:,0].argsort()[::-1]]
    parr = np.hstack((p, -np.arange(1, p.shape[0] + 1).reshape(-1, 1)))
    r = r[r[:, 0].argsort()]
    rarr = np.hstack((r, np.arange(1, r.shape[0] + 1).reshape(-1, 1)))
    out = np.vstack((parr, rarr))
    return out[out[:, 2].argsort()]

h35fl = label(h35fp, h35fr)
h35fpoly, residuals, _, _, _ = np.polyfit(h35fl[:,2], h35fl[:,0], deg = 2, full= True)
h35fpolyerr = (residuals[0] if len(residuals) > 0 else 0)/h35fl.shape[0]

h37fl = label(h37fp, h37fr)
h37fpoly, residuals, _, _, _ = np.polyfit(h37fl[:,2], h37fl[:,0], deg = 2, full= True)
h37fpolyerr = (residuals[0] if len(residuals) > 0 else 0)/h37fl.shape[0]

d35fl = label(d35fp, d35fr)
d35fpoly, residuals, _, _, _ = np.polyfit(d35fl[:,2], d35fl[:,0], deg = 2, full= True)
d35fpolyerr = (residuals[0] if len(residuals) > 0 else 0)/d35fl.shape[0]

d37fl = label(d37fp, d37fr)
d37fpoly, residuals, _, _, _ = np.polyfit(d37fl[:,2], d37fl[:,0], deg = 2, full= True)
d37fpolyerr = (residuals[0] if len(residuals) > 0 else 0)/d37fl.shape[0]

h35ol = label(h35op, h35or)
h35opoly, residuals, _, _, _ = np.polyfit(h35ol[:,2], h35ol[:,0], deg = 2, full= True)
h35opolyerr = (residuals[0] if len(residuals) > 0 else 0)/h35ol.shape[0]

h37ol = label(h37op, h37or)
h37opoly, residuals, _, _, _ = np.polyfit(h37ol[:,2], h37ol[:,0], deg = 2, full= True)
h37opolyerr = (residuals[0] if len(residuals) > 0 else 0)/h37ol.shape[0]

d35ol = label(d35op, d35or)
d35opoly, residuals, _, _, _ = np.polyfit(d35ol[:,2], d35ol[:,0], deg = 2, full= True)
d35opolyerr = (residuals[0] if len(residuals) > 0 else 0)/d35ol.shape[0]

d37ol = label(d37op, d37or)
d37opoly, residuals, _, _, _ = np.polyfit(d37ol[:,2], d37ol[:,0], deg = 2, full= True)
d37opolyerr = (residuals[0] if len(residuals) > 0 else 0)/d37ol.shape[0]

mval = np.linspace(-10,10,100)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
axs[0,0].scatter(h35fl[:,2], h35fl[:,0], label = 'HCl(35) data', color='red', marker='o')
axs[0,0].scatter(d35fl[:,2], d35fl[:,0], label = 'DCl(35) data', color='blue', marker='s')
axs[0,0].plot(mval, np.polyval(h35fpoly, mval), label = 'HCl(35) fit', color='red')
axs[0,0].plot(mval, np.polyval(d35fpoly, mval), label = 'DCl(35) fit', color='blue')
axs[0,0].set_title('Cl(35) Fundamentals')
axs[0,0].set_xlim([-10,10])
axs[0,0].set_xlabel('m')
axs[0,0].set_ylabel('Energy [cm^-1]')
axs[0,0].legend(loc = 'best')
axs[1,0].scatter(h37fl[:,2], h37fl[:,0], label = 'HCl(37) data', color='red', marker='o')
axs[1,0].scatter(d37fl[:,2], d37fl[:,0], label = 'DCl(37) data', color='blue', marker='s')
axs[1,0].plot(mval, np.polyval(h37fpoly, mval), label = 'HCl(37) fit', color='red')
axs[1,0].plot(mval, np.polyval(d37fpoly, mval), label = 'DCl(37) fit', color='blue')
axs[1,0].set_title('Cl(37) Fundamentals')
axs[1,0].set_xlabel('m')
axs[1,0].set_ylabel('Energy [cm^-1]')
axs[1,0].legend(loc = 'best')
axs[1,0].set_xlim([-10,10])
axs[0,1].scatter(h35ol[:,2], h35ol[:,0], label = 'HCl(35) data', color='red', marker='o')
axs[0,1].scatter(d35ol[:,2], d35ol[:,0], label = 'DCl(35) data', color='blue', marker='s')
axs[0,1].plot(mval, np.polyval(h35opoly, mval), label = 'HCl(35) fit', color='red')
axs[0,1].plot(mval, np.polyval(d35opoly, mval), label = 'DCl(35) fit', color='blue')
axs[0,1].set_title('Cl(37) Overtones')
axs[0,1].set_xlabel('m')
axs[0,1].set_ylabel('Energy [cm^-1]')
axs[0,1].legend(loc = 'best')
axs[0,1].set_xlim([-10,10])
axs[1,1].scatter(h37ol[:,2], h37ol[:,0], label = 'HCl(37) data', color='red', marker='o')
axs[1,1].scatter(d37ol[:,2], d37ol[:,0], label = 'DCl(37) data', color='blue', marker='s')
axs[1,1].plot(mval, np.polyval(h37opoly, mval), label = 'HCl(37) fit', color='red')
axs[1,1].plot(mval, np.polyval(d37opoly, mval), label = 'DCl(37) fit', color='blue')
axs[1,1].set_title('Cl(37) Overtones')
axs[1,1].set_xlabel('m')
axs[1,1].set_ylabel('Energy [cm^-1]')
axs[1,1].legend(loc = 'best')
axs[1,1].set_xlim([-10,10])
fig.suptitle('Energy vs m')
plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
plt.show()
