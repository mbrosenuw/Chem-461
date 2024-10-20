import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import scipy as sc


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
specA = specA/np.max(specA)
# plt.figure()
# plt.plot(freq,specA)
# plt.show()

from uwpchem461 import Analyse
analyse = Analyse()
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,5))

# # Plotting the dataset for specT vs wavenumber
# axs[0].plot(freq, specT, linestyle='solid',
#          marker='None', color='black', linewidth=1)
# axs[0].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[0].set_ylabel('Transmission [%]')

# Plotting the dataset for absorbance vs wavenumber
# axs.plot(freq, specA, linestyle='solid',
#          marker='None', color='black', linewidth=1)
# axs.set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs.set_ylabel('Absorbance [Au]', fontsize=14)
# fig.suptitle('Rotation-Vibration Spectrum for HCl/DCl mixture', fontsize=18)
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


# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,9))
# axs[0,0].plot(DClfundp[:,0], DClfundp[:,1], label = 'Signal')
# axs[0,0].plot(DClfund[:,0], bDCLf, label = 'Baseline')
# axs[0,0].set_title('DCl Fundamental', fontsize=16)
# axs[0,0].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[0,0].set_ylabel('Abs [Au]', fontsize=14)
# axs[0,0].legend()
# axs[1,0].plot(DCloverp[:,0], DCloverp[:,1], label = 'Signal')
# axs[1,0].plot(DClover[:,0], bDCLo, label = 'Baseline')
# axs[1,0].set_title('DCl Overtone', fontsize=16)
# axs[1,0].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[1,0].set_ylabel('Abs [Au]', fontsize=14)
# axs[1,0].legend()
# axs[0,1].plot(HClfundp[:,0], HClfundp[:,1], label = 'Signal')
# axs[0,1].plot(HClfund[:,0], bHCLf, label = 'Baseline')
# axs[0,1].set_title('HCl Fundamental', fontsize=16)
# axs[0,1].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[0,1].set_ylabel('Abs [Au]', fontsize=14)
# axs[0,1].legend()
# axs[1,1].plot(HCloverp[:,0], HCloverp[:,1], label = 'Signal')
# axs[1,1].plot(HClover[:,0], bHCLo, label ='Baseline')
# axs[1,1].set_title('HCl Overtone', fontsize=16)
# axs[1,1].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[1,1].set_ylabel('Abs [Au]', fontsize=14)
# axs[1,1].legend()
# fig.suptitle('HCl and DCl Bands', fontsize=18)
# plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
# plt.show()

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

DClfpeaks, _ = sc.signal.find_peaks(DClft[:,1], height=0.05)
DClopeaks, _ = sc.signal.find_peaks(DClot[:,1], height=0.05)
HClfpeaks, _ = sc.signal.find_peaks(HClft[:,1], height=0.05)
HClopeaks, _ = sc.signal.find_peaks(HClot[:,1], height=0.05)
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

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,9))
# axs[0,0].plot(DClfund[:,0], DClfund[:,1], zorder = 1)
# axs[0,0].scatter(d35fp[:,0], d35fp[:,1], label = 'DCl(35) P', color = 'red', marker = 's')
# axs[0,0].scatter(d35fr[:,0], d35fr[:,1], label = 'DCl(35) R', color = 'orange', marker = 'o')
# axs[0,0].scatter(d37fp[:,0], d37fp[:,1], label = 'DCl(37) P', color = 'green', marker = '^')
# axs[0,0].scatter(d37fr[:,0], d37fr[:,1], label = 'DCl(37) R', color = 'purple', marker = '*')
# axs[0,0].set_title('DCl Fundamental', fontsize=16)
# axs[0,0].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[0,0].set_ylabel('Abs [Au]', fontsize=14)
# axs[0,0].legend(loc = 'best')
# axs[1,0].plot(DClover[:,0], DClover[:,1], zorder = 1)
# axs[1,0].scatter(d35op[:,0], d35op[:,1], label = 'DCl(35) P', color = 'red', marker = 's')
# axs[1,0].scatter(d35or[:,0], d35or[:,1], label = 'DCl(35) R', color = 'orange', marker = 'o')
# axs[1,0].scatter(d37op[:,0], d37op[:,1], label = 'DCl(37) P', color = 'green', marker = '^')
# axs[1,0].scatter(d37or[:,0], d37or[:,1], label = 'DCl(37) R', color = 'purple', marker = '*')
# axs[1,0].set_title('DCl Overtone', fontsize=16)
# axs[1,0].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[1,0].set_ylabel('Abs [Au]', fontsize=14)
# axs[1,0].legend(loc = 'best')
# axs[0,1].plot(HClfund[:,0], HClfund[:,1], zorder = 1)
# axs[0,1].scatter(h35fp[:,0], h35fp[:,1], label = 'HCl(35) P', color = 'red', marker = 's')
# axs[0,1].scatter(h35fr[:,0], h35fr[:,1], label = 'HCl(35) R', color = 'orange', marker = 'o')
# axs[0,1].scatter(h37fp[:,0], h37fp[:,1], label = 'HCl(37) P', color = 'green', marker = '^')
# axs[0,1].scatter(h37fr[:,0], h37fr[:,1], label = 'HCl(37) R', color = 'purple', marker = '*')
# axs[0,1].set_title('HCl Fundamental', fontsize=16)
# axs[0,1].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[0,1].set_ylabel('Abs [Au]', fontsize=14)
# axs[0,1].legend(loc = 'best')
# axs[1,1].plot(HClover[:,0], HClover[:,1], zorder = 1)
# axs[1,1].scatter(h35op[:,0], h35op[:,1], label = 'HCl(35) P', color = 'red', marker = 's')
# axs[1,1].scatter(h35or[:,0], h35or[:,1], label = 'HCl(35) R', color = 'orange', marker = 'o')
# axs[1,1].scatter(h37op[:,0], h37op[:,1], label = 'HCl(37) P', color = 'green', marker = '^')
# axs[1,1].scatter(h37or[:,0], h37or[:,1], label = 'HCl(37) R', color = 'purple', marker = '*')
# axs[1,1].set_title('HCl Overtone', fontsize=16)
# axs[1,1].set_xlabel('E/hc [cm^-1]', fontsize=14)
# axs[1,1].set_ylabel('Abs [Au]', fontsize=14)
# axs[1,1].legend(loc = 'best')
# fig.suptitle('Corrected HCl and DCl Bands', fontsize=18)
# plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
# plt.show()

def label(p, r):
    p = p[p[:,0].argsort()[::-1]]
    parr = np.hstack((p, -np.arange(1, p.shape[0] + 1).reshape(-1, 1)))
    r = r[r[:, 0].argsort()]
    rarr = np.hstack((r, np.arange(1, r.shape[0] + 1).reshape(-1, 1)))
    out = np.vstack((parr, rarr))
    out = out[out[:, 2].argsort()]
    poly, cov = np.polyfit(out[:, 2], out[:, 0], deg=2, cov=True)
    err = np.sqrt(np.diag(cov))
    return out, poly, err

def r4(arr):
    def round_single_value(val):
        if val == 0:
            return 0
        else:
            return np.round(val, decimals=3 - int(np.floor(np.log10(np.abs(val)))))

    vectorized_round = np.vectorize(round_single_value)
    return vectorized_round(arr)
def getfconsts(poly, errs, name):
    consts = np.zeros(3)
    consts[0] = -poly[0]
    consts[1] = 0.5 * (poly[1] - 2*poly[0])
    consts[2] = poly[2]
    fiterr = errs
    fiterr[1] = np.sqrt((0.5*errs[1])**2 + errs[2]**2)
    consts = r4(consts)
    fiterr = r4(fiterr)
    # print(name, ' & ', consts[0], ' & ', consts[1], ' & ', consts[2], ' & ', fiterr[0], ' & ', fiterr[1], ' & ', fiterr[2], '\\\\')
    return consts, fiterr

def getoconsts(poly, errs, name, nuf, nuferr):
    consts = np.zeros(3)
    consts[0] = -0.5 * poly[0]
    consts[1] = 0.25 * (2*poly[1] - 3*poly[0])
    consts[2] = poly[2]
    fiterr = errs
    fiterr[0] = 0.5*fiterr[0]
    fiterr[1] = np.sqrt((0.5*errs[1])**2 + (0.75*errs[2])**2)
    consts = r4(consts)
    fiterr = r4(fiterr)
    nuu = consts[2]
    nuuerr = fiterr[2]
    nu0 = 3*nuf - nuu
    X = (nuu - 2*nuf)/(2*nuu - 6*nuf)
    Xerr = np.sqrt(((nuf*nuuerr)/(2 * (nuu-3*nuf)**2))**2 + ((nuu*nuferr)/(2 * (-nuu+3*nuf)**2))**2)
    nu0err = np.sqrt((3*nuferr)**2 + nuuerr**2)
    # print(name, ' & ', consts[0], ' & ', consts[1], ' & ', consts[2],  ' & ', fiterr[0], ' & ', fiterr[1], ' & ', fiterr[2],  '\\\\')
    return consts, fiterr

def getHparams(consts, errs, mu, name):
    hbar = sc.constants.hbar
    amkg = sc.constants.atomic_mass
    mu = amkg*mu
    c = sc.constants.c * 100 #in cm/s
    re = np.sqrt(hbar/ (4 * np.pi*mu * c * consts[1]))*10**10
    reerr = np.sqrt(hbar/ (4 * np.pi*mu * c * consts[1]**2))*10**10
    I = hbar / (4*np.pi*c * consts[1]) * errs[1] / amkg * 10**20
    Ierr = hbar / (4*np.pi*c * consts[1]**2) * errs[1] / amkg * 10**20
    k = (2 * np.pi * c * consts[2])**2 * mu
    kerr = 2 * (2*np.pi * c)**2 * consts[2] * errs[2] * mu
    print(name, ' & ', r4(k), ' & ', r4(I), ' & ', r4(re),  ' & ', r4(kerr), ' & ', r4(Ierr), ' & ', r4(reerr),  '\\\\')
    return np.array([k, I, re]), np.array([kerr,Ierr,reerr])

def getratios(consts1, consts2, err1, err2, mu1, mu2, name1, name2):
    vibratio = np.sqrt(mu1/mu2)
    expvibratio = consts2[2]/consts1[2]
    expvrerr = np.sqrt((err2[2]/consts1[2])**2 + (consts2[2]/consts1[2]**2 * err1[2])**2)
    rotratio = mu1/mu2
    exprotratio = consts2[1]/consts1[1]
    exprrerr = np.sqrt((err2[1] / consts1[1]) ** 2 + (consts2[1] / consts1[1] ** 2 * err1[1]) ** 2)
    print('calc: & ' + name1 + ':' + name2, ' & ', r4(vibratio), ' & ', r4(rotratio), '\\\\')
    print('exp : & ' + name1 + ':' + name2, ' & ', r4(expvibratio), ' & ', r4(exprotratio), ' & ', r4(expvrerr), ' & ', r4(exprrerr),'\\\\')



def getthermo(consts, errs, mass, name):
    from scipy.constants import physical_constants as pc
    kb = pc['Boltzmann constant in inverse meter per kelvin'][0] / 100
    N = pc['N_A']
    h = sc.constants.h
    c = sc.constants.c
    T = 300 #kelvin
    P =  101325 #Pa
    l = h / np.sqrt(2 * np.pi *kb *T)
    errl = 0
    Trot = h * c * consts[1]/ kb
    Troterr = h * c * errs[1]/ kb
    Tvib = h * c * consts[2]/ kb
    Tviberr= h * c * errs[2]/ kb
    Ztr = N * kb * T / (P * l**3)
    Ztrerr= 0
    Zrot = T/Trot
    Zroterr = T/Trot**2 * Troterr
    Zvib = 1/(1-np.exp(-Tvib/T))
    Zviberr = np.exp(Tvib/T)/ (T* (np.exp(Tvib/T) - 1)**2) * Tviberr
    U = 5/2 * N*kb*T + N*kb*Tvib/(np.exp(Tvib/T) - 1)
    F = -N*kb*T(np.log(np.exp(1)*Ztr/N) + np.log(Zrot*Zvib))
    H = 7/2 * N*kb*T + N*kb*Tvib/(np.exp(Tvib/T) - 1)
    C = (N * kb * Tvib**2 * np.exp(Tvib/T))/ (T**2 * (np.exp(Tvib/T) - 1)**2) + 5*N*kb/2
    S = (U - F)/T
    Uerr = ((N*kb * ((Tvib - T)*np.exp(Tvib/T) + T)/ ()))

h35fl, h35fpoly, h35ferrs = label(h35fp, h35fr)
h35fc, h35ferr = getfconsts(h35fpoly, h35ferrs, 'H$^{35}$Cl')
# getHparams(h35fc, h35ferr, 0.9797012, 'H$^{35}$Cl Fundamental')

h37fl, h37fpoly, h37ferrs = label(h37fp, h37fr)
h37fc, h37ferr = getfconsts(h37fpoly, h37ferrs, 'H$^{37}$Cl')
# getHparams(h37fc, h37ferr, 0.98118624, 'H$^{37}$Cl Fundamental')

d35fl, d35fpoly, d35ferrs = label(d35fp, d35fr)
d35fc, d35ferr = getfconsts(d35fpoly, d35ferrs, 'D$^{35}$Cl')
# getHparams(d35fc, d35ferr, 1.9044132, 'D$^{35}$Cl Fundamental')

d37fl, d37fpoly, d37ferrs = label(d37fp, d37fr)
d37fc, d37ferr = getfconsts(d37fpoly, d37ferrs, 'D$^{37}$Cl')
# getHparams(d37fc, d37ferr, 1.91003288, 'D$^{37}$Cl Fundamental')

h35ol, h35opoly, h35oerrs = label(h35op, h35or)
h35oc, h35oerr = getoconsts(h35opoly, h35oerrs, 'H$^{35}$Cl', h35fc[2], h35ferr[2])
# getHparams(h35oc, h35oerr, 0.9797012, 'H$^{35}$Cl Overtone')

h37ol, h37opoly, h37oerrs = label(h37op, h37or)
h37oc, h37oerr = getoconsts(h37opoly, h37oerrs, 'H$^{37}$Cl', h37fc[2], h37ferr[2])
# getHparams(h37oc, h37oerr, 0.98118624, 'H$^{37}$Cl Overtone')

d35ol, d35opoly, d35oerrs = label(d35op, d35or)
d35oc, d35oerr = getoconsts(d35opoly, d35oerrs, 'D$^{35}$Cl', d35fc[2], d35ferr[2])
# getHparams(d35oc, d35oerr, 1.9044132, 'D$^{35}$Cl Overtone')

d37ol, d37opoly, d37oerrs = label(d37op, d37or)
d37oc, d37oerr = getoconsts(d37opoly, d37oerrs, 'D$^{37}$Cl', d37fc, d37ferr)
# getHparams(d37oc, d37oerr, 1.91003288, 'D$^{37}$Cl Overtone')


# getratios( h37fc, h35fc, h37ferr, h35ferr, 0.98118624,0.9797012,'F: H$^{37}$Cl', 'H$^{35}$Cl')
# getratios( d35fc, h35fc, d35ferr, h35ferr, 1.9044132,0.9797012,'F: D$^{35}$Cl', 'H$^{35}$Cl')
# getratios( d37fc, h35fc, d37ferr, h35ferr, 1.91003288,0.9797012,'F: D$^{37}$Cl', 'H$^{35}$Cl')
# getratios( d35fc, h37fc, d35ferr, h37ferr, 1.9044132,0.98118624,'F: D$^{35}$Cl', 'H$^{37}$Cl')
# getratios( d37fc, h37fc, d37ferr, h37ferr, 1.91003288,0.98118624,'F: D$^{37}$Cl', 'H$^{37}$Cl')
# getratios( d37fc, h37fc, d37ferr, h37ferr, 1.91003288,0.98118624,'F: D$^{37}$Cl', 'H$^{37}$Cl')
#
#
# getratios( h37oc, h35oc, h37oerr, h35oerr, 0.98118624,0.9797012,'O: H$^{37}$Cl', 'H$^{35}$Cl')
# getratios( d35oc, h35oc, d35oerr, h35oerr, 1.9044132,0.9797012,'O: D$^{35}$Cl', 'H$^{35}$Cl')
# getratios( d37oc, h35oc, d37oerr, h35oerr, 1.91003288,0.9797012,'O: D$^{37}$Cl', 'H$^{35}$Cl')
# getratios( d35oc, h37oc, d35oerr, h37oerr, 1.9044132,0.98118624,'O: D$^{35}$Cl', 'H$^{37}$Cl')
# getratios( d37oc, h37oc, d37oerr, h37oerr, 1.91003288,0.98118624,'O: D$^{37}$Cl', 'H$^{37}$Cl')
# getratios( d37oc, h37oc, d37oerr, h37oerr, 1.91003288,0.98118624,'O: D$^{37}$Cl', 'H$^{37}$Cl')



def geterr(errs, m):
    return np.sqrt(errs[0]**2 + m**2*errs[1]**2 + (m**4)*errs[2]**2)



# mval = np.linspace(-10,10,21)
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
# axs[0,0].scatter(h35fl[:,2], h35fl[:,0], label = 'HCl(35) data', color='red', marker='o')
# axs[0,0].scatter(d35fl[:,2], d35fl[:,0], label = 'DCl(35) data', color='blue', marker='s')
# axs[0,0].plot(mval, np.polyval(h35fpoly, mval), label = 'HCl(35) fit', color='red')
# axs[0,0].plot(mval, np.polyval(d35fpoly, mval), label = 'DCl(35) fit', color='blue')
# axs[0,0].errorbar(mval, np.polyval(h35fpoly, mval), yerr=5*geterr(h35ferrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[0,0].errorbar(mval, np.polyval(d35fpoly, mval), yerr=5*geterr(d35ferrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[0,0].set_title('Cl(35) Fundamentals', fontsize=16)
# axs[0,0].set_xlim([-10,10])
# axs[0,0].set_xlabel('m', fontsize=14)
# axs[0,0].set_ylabel('E/hc [cm^-1]', fontsize=14)
# axs[0,0].legend(loc = 'best')
# axs[1,0].scatter(h37fl[:,2], h37fl[:,0], label = 'HCl(37) data', color='red', marker='o')
# axs[1,0].scatter(d37fl[:,2], d37fl[:,0], label = 'DCl(37) data', color='blue', marker='s')
# axs[1,0].plot(mval, np.polyval(h37fpoly, mval), label = 'HCl(37) fit', color='red')
# axs[1,0].plot(mval, np.polyval(d37fpoly, mval), label = 'DCl(37) fit', color='blue')
# axs[1,0].errorbar(mval, np.polyval(h35fpoly, mval), yerr=5*geterr(h37ferrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[1,0].errorbar(mval, np.polyval(d35fpoly, mval), yerr=5*geterr(d37ferrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[1,0].set_title('Cl(37) Fundamentals', fontsize=16)
# axs[1,0].set_xlabel('m', fontsize=14)
# axs[1,0].set_ylabel('E/hc [cm^-1]', fontsize=14)
# axs[1,0].legend(loc = 'best')
# axs[1,0].set_xlim([-10,10])
# axs[0,1].scatter(h35ol[:,2], h35ol[:,0], label = 'HCl(35) data', color='red', marker='o')
# axs[0,1].scatter(d35ol[:,2], d35ol[:,0], label = 'DCl(35) data', color='blue', marker='s')
# axs[0,1].plot(mval, np.polyval(h35opoly, mval), label = 'HCl(35) fit', color='red')
# axs[0,1].plot(mval, np.polyval(d35opoly, mval), label = 'DCl(35) fit', color='blue')
# axs[0,1].errorbar(mval, np.polyval(h35opoly, mval), yerr=5*geterr(h35oerrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[0,1].errorbar(mval, np.polyval(d35opoly, mval), yerr=5*geterr(d35oerrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[0,1].set_title('Cl(37) Overtones', fontsize=16)
# axs[0,1].set_xlabel('m', fontsize=14)
# axs[0,1].set_ylabel('E/hc [cm^-1]', fontsize=14)
# axs[0,1].legend(loc = 'best')
# axs[0,1].set_xlim([-10,10])
# axs[1,1].scatter(h37ol[:,2], h37ol[:,0], label = 'HCl(37) data', color='red', marker='o')
# axs[1,1].scatter(d37ol[:,2], d37ol[:,0], label = 'DCl(37) data', color='blue', marker='s')
# axs[1,1].plot(mval, np.polyval(h37opoly, mval), label = 'HCl(37) fit', color='red')
# axs[1,1].plot(mval, np.polyval(d37opoly, mval), label = 'DCl(37) fit', color='blue')
# axs[1,1].errorbar(mval, np.polyval(h37opoly, mval), yerr=5*geterr(h37oerrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[1,1].errorbar(mval, np.polyval(d37opoly, mval), yerr=5*geterr(d37oerrs, mval), fmt='none', color='black',
#                   ecolor='gray', elinewidth=2, capsize=0, label= '10x error')
# axs[1,1].set_title('Cl(37) Overtones', fontsize=16)
# axs[1,1].set_xlabel('m', fontsize=14)
# axs[1,1].set_ylabel('E/hc [cm^-1]', fontsize=14)
# axs[1,1].legend(loc = 'best')
# axs[1,1].set_xlim([-10,10])
# fig.suptitle('Energy vs m', fontsize=18)
# plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
# plt.show()


