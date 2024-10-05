import numpy as np
from matplotlib import pyplot as plt
import os
import csv


filename = 'Rosen_Au24.csv'
specT = np.array([])
# filename = 'I-HOD_10K_lowres.csv'
freq = np.array([])
if os.path.exists(filename):
    with open(filename, "r", newline="",encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            freq = np.append(freq, float(row[0]))
            specT = np.append(specT, float(row[1]))

specA = -np.log10((specT+14)/np.max((specT+14)))
specA = specA/np.max(specA)


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

cDClfund = (freq > 1900) & (freq < 2250)
cDClover = (freq > 3967) & (freq < 4236)
cHClfund = (freq >2500) & (freq <3150)
cHClover = (freq > 5487) & (freq < 5826)

DClfund = np.vstack((freq[cDClfund], specA[cDClfund]/np.max(specA[cDClfund]))).T

DClover = np.vstack((freq[cDClover], specA[cDClover]/np.max(specA[cDClover]))).T

HClfund = np.vstack((freq[cHClfund], specA[cHClfund]/np.max(specA[cHClfund]))).T

HClover = np.vstack((freq[cHClover], specA[cHClover]/np.max(specA[cHClover]))).T

bDCLf = analyse.getbase(DClfund.tolist(), 0.14)
bDCLo = analyse.getbase(DClover.tolist(), 0.70)
bHCLf = analyse.getbase(HClfund.tolist(), 0.152)
bHCLo = analyse.getbase(HClover.tolist(), 0.172)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
axs[0,0].plot(DClfund[:,0], DClfund[:,1], label = 'DCl Fundamental')
axs[0,0].plot(DClfund[:,0], bDCLf, label = 'baseline')
axs[0,0].set_title('DCl Fundamental')
axs[0,0].set_xlabel('Energy [cm^-1]')
axs[0,0].set_ylabel('Abs [Au]')
axs[1,0].plot(DClover[:,0], DClover[:,1], label = 'DCl Overtone')
axs[1,0].plot(DClover[:,0], bDCLo, label = 'baseline')
axs[1,0].set_title('DCl Overtone')
axs[1,0].set_xlabel('Energy [cm^-1]')
axs[1,0].set_ylabel('Abs [Au]')
axs[0,1].plot(HClfund[:,0], HClfund[:,1], label = 'HCL Fundamental')
axs[0,1].plot(HClfund[:,0], bHCLf, label = 'baseline')
axs[0,1].set_title('HCl Fundamental')
axs[0,1].set_xlabel('Energy [cm^-1]')
axs[0,1].set_ylabel('Abs [Au]')
axs[1,1].plot(HClover[:,0], HClover[:,1], label='HCl Overtone')
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


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
axs[0,0].plot(DClfund[:,0], DClfund[:,1], label = 'DCl Fundamental')
axs[0,0].set_title('DCl Fundamental')
axs[0,0].set_xlabel('Energy [cm^-1]')
axs[0,0].set_ylabel('Abs [Au]')
axs[1,0].plot(DClover[:,0], DClover[:,1], label = 'DCl Overtone')
axs[1,0].set_title('DCl Overtone')
axs[1,0].set_xlabel('Energy [cm^-1]')
axs[1,0].set_ylabel('Abs [Au]')
axs[0,1].plot(HClfund[:,0], HClfund[:,1], label = 'HCL Fundamental')
axs[0,1].set_title('HCl Fundamental')
axs[0,1].set_xlabel('Energy [cm^-1]')
axs[0,1].set_ylabel('Abs [Au]')
axs[1,1].plot(HClover[:,0], HClover[:,1], label='HCl Overtone')
axs[1,1].set_title('HCl Overtone')
axs[1,1].set_xlabel('Energy [cm^-1]')
axs[1,1].set_ylabel('Abs [Au]')
fig.suptitle('Corrected HCl and DCl Bands')
plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
plt.show()
