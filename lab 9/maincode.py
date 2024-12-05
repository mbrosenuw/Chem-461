import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import scipy as sc
from scipy.stats import linregress

p = 747  # torr
T = 23.0

vAlc = np.linspace(0, 1, 11)  # in ml
vTol = vAlc[::-1]

vfAlc = vAlc / (vAlc + vTol)
vfTol = vTol / (vAlc + vTol)

mmtol = 92.14  # g/mol
dtol = 0.8669  # g/ml
mdtol = dtol / mmtol

mmalc = 60.0952
dalc = 0.803
mdalc = dalc / mmalc

molAlc = mdalc * vAlc
molTol = mdtol * vTol

mfAlc = molAlc / (molAlc + molTol)
mfTol = molTol / (molAlc + molTol)

ri = np.array([1.4962, 1.4849, 1.4741, 1.4624, 1.4510, 1.4401, 1.4292, 1.4181, 1.4065, 1.3954, 1.3850])


# Define a lookup function for a given y
def lookup_x(y, m, b):
    return (y - b) / m


def fitandplot(x, y, xlabel, ylabel, title):
    slope, intercept, r, _, std_err = linregress(x, y)
    rsq = r ** 2

    # Predicted y values based on the linear fit
    y_fit = slope * x + intercept

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data Points', zorder=5)
    plt.plot(x, y_fit, color='red', label='Fitted Line', zorder=4)

    # Add equation and R^2 to the plot
    equation_text = f"$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {rsq:.2f}$"
    plt.text(0.5, 0.85, equation_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Customize the plot
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    return slope, intercept


vfs, vfi = fitandplot(vfAlc, ri, 'Volume Fraction of Alcohol [unitless]', 'Refractive index [unitless]',
                      'Refractive index vs Volume Fraction')
mfs, mfi = fitandplot(mfAlc, ri, 'Mole Fraction of Alcohol [unitless]', 'Refractive index [unitless]',
                      'Refractive index vs Mole Fraction')

vTol = 30
vAlc = np.array([0, 1, 2, 3, 5, 8, 13])
mTol1 = mdtol * vTol
mAlc1 = mdalc * vAlc
mTot1 = mTol1 + mAlc1
mfalc1 = np.array([ma / mt for ma, mt in zip(mAlc1, mTot1)])
T1 = np.array([109, 107.5, 106.5, 103.2, 100.0, 97.8, 93.6]) + 273.15
rif1 = np.array([1.496, 1.4938, 1.4908, 1.4820, 1.4804, 1.4680, 1.4595])
rig1 = np.array([1.496, 1.4776, 1.4555, 1.4465, 1.4433, 1.4380, 1.4280])

vAlc = 30
vTol = np.array([0, 1, 3, 6, 9, 14, 19])
mTol2 = mdtol * vTol
mAlc2 = mdalc * vAlc
mTot2 = mTol1 + mAlc1
mfalc2 = np.array([mAlc2 / mt for mt in mTot2])
T2 = np.array([97, 96.8, 96.3, 95, 93.8, 92.1, 93.4]) + 273.15
rif2 = np.array([1.387, 1.3889, 1.3948, 1.4040, 1.4105, 1.4225, 1.4310])
rig2 = np.array([1.387, 1.3924, 1.4050, 1.4118, 1.4206, 1.4290, 1.4335])

Ttot = np.hstack((T1, T2))
rif = np.hstack((rif1, rif2))
rig = np.hstack((rig1, rig2))

plt.figure()
plt.scatter(rif, Ttot, color='blue', label='Liquid Phase')
plt.scatter(rig, Ttot, color='red', label='Gas Phase')
plt.legend(loc='best')
plt.title('Temperature vs Refractive Index', fontsize=14)
plt.xlabel('Refractive Index [unitless]', fontsize=12)
plt.ylabel('Temperature [K]', fontsize=12)
plt.show()

mff = lookup_x(rif, mfs, mfi)
mfg = lookup_x(rig, mfs, mfi)
plt.figure()
plt.scatter(mff, Ttot, color='blue', label='Liquid Phase')
plt.scatter(mfg, Ttot, color='red', label='Gas Phase')
plt.legend(loc='best')
plt.title('Temperature vs Mole Fraction', fontsize=14)
plt.xlabel('Mole Fraction [unitless]', fontsize=12)
plt.ylabel('Temperature [K]', fontsize=12)
plt.show()
