import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from os.path import join as j
from scipy.constants import physical_constants as pc
from matplotlib.ticker import ScalarFormatter

class Run():
    def __init__(self, path, n240, n290, n322, start,stop, temp,cut = False,c322 = 0, r322 = 10):
        self.path = path
        self.n240 = n240
        self.n290 = n290
        self.n322 = n322
        self.start = start
        self.stop = stop
        self.d240 = opentime(j(self.path, self.n240), ravg = True)
        self.d290 = opentime(j(self.path, self.n290), ravg=True)
        self.d322 = opentime(j(self.path, self.n322), ravg=True, w = r322)
        self.temp = temp
        self.cut = cut
        self.c322 = c322


    def plotData(self):
        fig, axs = plt.subplots(figsize=(12, 4), ncols=3, nrows=1)
        fig.subplots_adjust(wspace=0.3)
        axs[0].plot(self.d240[:, 0], self.d240[:, 1])
        axs[0].set_title(self.n240)
        axs[0].set_title('240 nm Abs vs time ')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Abs')
        axs[1].plot(self.d290[:, 0], self.d290[:, 1])
        axs[1].set_title(self.n290)
        axs[1].set_title('290 nm Abs vs time ')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Abs')
        axs[2].plot(self.d322[:, 0], self.d322[:, 1])
        axs[2].set_title(self.n322)
        axs[2].set_title('322 nm Abs vs time ')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Abs')
        fig.suptitle(f'Plots of Abs vs time for {self.temp}K', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def fitdata(self):
        fig, axs = plt.subplots(figsize=(12, 4), ncols=3, nrows=1)
        fig.subplots_adjust(wspace=0.3)
        c = (self.d240[:, 0] > self.start) & (self.d240[:, 0] < self.stop)
        self.Ai240, self.kTot240, self.sAi240, self.skTot240 = fitabs(self.d240[c, :], axs[0], name=self.n240, runname = self.temp + '_240')
        axs[0].set_title(self.n240)
        axs[0].set_title('240 nm Abs vs time ')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Abs')
        c = (self.d290[:, 0] > self.start) & (self.d290[:, 0] < self.stop)
        self.Ai290, self.kTot290, self.sAi290, self.skTot290 = fitabs(self.d290[c, :], axs[1], name=self.n290, runname = self.temp+ '_290')
        axs[1].set_title(self.n290)
        axs[1].set_title('290 nm Abs vs time ')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Abs')
        if self.cut:
            c = (self.d322[:,0] > self.c322)
            self.d322[c,1]= 0
        c = (self.d322[:, 0] > self.start) & (self.d322[:, 0] < self.stop)
        self.k3, self.sk3 = fit322(self.d322[c, :], axs[2], name=self.n322, runname = self.temp + '_322')
        axs[2].set_title(self.n322)
        axs[2].set_title('322 nm Abs vs time ')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Abs')
        fig.suptitle(f'Fits of Abs vs time for {self.temp}K', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def getrates(self):
        bp0 = 8 * 10**(-4) * 28/35
        l = 1
        params = np.array([[self.Ai240],[self.Ai290],[self.kTot240]])
        mat = np.array([[l* bp0/self.kTot240 * 5400 , l* bp0/self.kTot240 * 6500, l* bp0/self.kTot240 * 800],
                        [l* bp0/self.kTot290 * 15800,0,0],[1,1,1]])
        k1,k2,k4 = np.linalg.solve(mat,params).flatten()
        # print(self.temp, ' & ', r4(k1), ' & ', r4(k2), ' & ', r4(self.k3), ' & ', r4(k4), ' & ', r4(self.kTot240), '\\\\')

        Sy = np.array([[self.sAi240**2, 0,0],[0, self.sAi290**2, 0],[0,0, self.skTot240]])
        Sk = np.linalg.inv(mat) @ Sy @ np.linalg.inv(mat).T
        sk1, sk2, sk4 = np.sqrt(np.diag(Sk))
        print(self.temp, ' & ', r4(sk1), ' & ', r4(sk2), ' & ', r4(self.sk3), ' & ', r4(sk4), ' & ', r4(self.skTot240),
              '\\\\')
        return np.array([float(self.temp), k1,k2,self.k3,k4])
def openspec(filename, ravg=False, w = 5):
    #open a file and return the contents of x-y data, with the option to do a 'nearest neighbors' average for y.
    #filename - the file you want to open
    #ravg - yes no to averaging
    #w window width for averaging

    # Initialize lists to store x and y data
    x_data = []
    y_data = []

    # Open the file and read line by line
    with open(filename, 'r') as file:
        start_reading = False

        for line in file:
            # Check for the beginning of data section
            if '>>>>>Begin Spectral Data<<<<<' in line:
                start_reading = True
                continue

            # If in data section, split and store x and y values
            if start_reading:
                # Split line into x and y values
                parts = line.split()
                if len(parts) == 2:  # Ensure it's a data line
                    x, y = map(float, parts[1:])
                    x_data.append(x)
                    y_data.append(y)
        if ravg:
            y_data = np.convolve(y_data, np.ones(w) / w, mode='same')
    x_data = x_data - np.min(x_data)
    return np.vstack((x_data, y_data)).T

def opentime(filename, ravg=False, w = 5):
    #open a file and return the contents of x-y data, with the option to do a 'nearest neighbors' average for y.
    #filename - the file you want to open
    #ravg - yes no to averaging
    #w window width for averaging

    # Initialize lists to store x and y data
    x_data = []
    y_data = []

    # Open the file and read line by line
    with open(filename, 'r') as file:
        start_reading = False

        for line in file:
            # Check for the beginning of data section
            if '>>>>>Begin Spectral Data<<<<<' in line:
                start_reading = True
                continue

            # If in data section, split and store x and y values
            if start_reading:
                # Split line into x and y values
                parts = line.split()
                if len(parts) == 3:  # Ensure it's a data line
                    x, y = map(float, parts[1:])
                    x_data.append(x)
                    y_data.append(y)

        y_data = np.array(y_data)
        y_data[y_data < 0 ] = 0
        if ravg:
            y_data = np.convolve(y_data, np.ones(w) / w, mode='same')

    x_data = (x_data - np.min(x_data))/1000
    return np.vstack((x_data, y_data)).T


def fitabs(run,ax, name, runname):

    run[:,0] = run[:,0] - np.min(run[:,0])
    Ainf = np.max(run[:, 1])+0.01
    def b15(run, Ainf):
        yprime = np.log(Ainf-run[:,1])+0.01
        slope, intercept = np.polyfit(run[:,0], yprime, 1)
        dA = np.exp(intercept)
        k = -slope
        return dA,k

    dA, k = b15(run, Ainf)

    def model_func(t, Ainf, dA, k):
        return Ainf - dA * np.exp(-k * t)
    p0 = [Ainf, dA, k]
    params, covariance = curve_fit(model_func, run[:,0], run[:,1], p0=p0)
    param_errors = np.sqrt(np.diag(covariance))
    # Extract the fitting parameters
    Ainf, dA, k = params
    sAinf, sdA, sk = param_errors

    # Generate y values based on the fitted model
    y_fit = model_func(run[:,0], Ainf, dA, k)
    resid = run[:, 1] - y_fit
    std = np.std(resid)
    num_points = 7
    indices = np.linspace(0, len(run[:, 0]) - 1, num_points, dtype=int)
    # Plot the original data and the fitted curve
    ax.plot(run[:, 0], run[:, 1], label='Data', color='gray', linewidth=0.5)
    ax.plot(run[:, 0], y_fit, label='Fit', color='blue', linewidth=2)
    ax.errorbar(run[indices, 0][1:-1], y_fit[indices][1:-1], yerr=std, fmt='none', color='red', label='Error bars',
                capsize=3)
    equation_text = f'$y = {Ainf:.2f} - {dA:.2f} e^{{-{k:.4f} t}}$'
    ax.text(0.05, -0.25, equation_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', color='black')
    ax.set_title(name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abs')
    ax.legend()
    # print(runname, ' & ', r6(Ainf), ' & ', r6(dA), ' & ', r6(k), ' & ', r6(sAinf), ' & ', r6(sdA), ' & ', r6(sk), '\\\\')
    return Ainf, k, sAinf, sk

def fit322(run,ax, name, runname):
    run[:, 0] = run[:, 0] - np.min(run[:, 0])
    run[:, 1] = run[:, 1] - np.min(run[:, 1])
    def model_func2(t, B,D,k3,kT):
        return B*(1-np.exp(-kT * t)) - D*(np.exp(-k3*t) - np.exp(-kT*t))


    p0 = [0.05,0.05,0.3,0.03]
    bounds = (0, [1, 1, 0.5, 0.1])
    model_func2(run[:,0], 0.05,0.05,0.3,0.03)
    params, covariance = curve_fit(model_func2, run[:,0], run[:,1], p0=p0, bounds = bounds)
    param_errors = np.sqrt(np.diag(covariance))
    # Extract the fitting parameters
    B,D,k3,kT = params
    sB, sD, sk3, skT = param_errors

    # Generate y values based on the fitted model
    y_fit = model_func2(run[:,0], B,D,k3,kT)
    resid = run[:,1] - y_fit
    std = np.std(resid)
    num_points = 7
    indices = np.linspace(0, len(run[:, 0]) - 1, num_points, dtype=int)
    ax.plot(run[:, 0], run[:, 1], label='Data', color='gray', linewidth=0.5)
    ax.plot(run[:, 0], y_fit, label='Fit', color='blue', linewidth=2)
    ax.errorbar(run[indices, 0][1:-1], y_fit[indices][1:-1], yerr=std, fmt='none', color='red', label='Error bars',
                capsize=3)
    equation_text = f'$y = {B:.2f}(1-e^{{-{kT:.4f} t}}) - {D:.2f}(e^{{-{k3:.4f} t}}-e^{{-{kT:.4f} t}})$'
    ax.text(0.05, -0.25, equation_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', color='black')
    ax.set_title(name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abs')
    ax.legend()
    # print(runname, ' & ', r6(B), ' & ', r6(D), ' & ', r6(k3), ' & ', r6(kT), ' & ', r6(sB), ' & ', r6(sD),
    #       ' & ', r6(sk3), ' & ', r6(skT), '\\\\')
    return k3, sk3

def r6(arr):
    def round_single_value(val):
        if val == 0:
            return 0
        else:
            return np.round(val, decimals=5 - int(np.floor(np.log10(np.abs(val)))))

    vectorized_round = np.vectorize(round_single_value)
    return vectorized_round(arr)

def r4(arr):
    def round_single_value(val):
        if val == 0:
            return 0
        else:
            return np.round(val, decimals=3 - int(np.floor(np.log10(np.abs(val)))))

    vectorized_round = np.vectorize(round_single_value)
    return vectorized_round(arr)

r28_1_1 = Run('lab1 data/timeresolved/28_1', '28_1_r1_240nm.txt', '28_1_r1_290nm.txt','28_1_r1_322nm.txt',
              start = 180, stop = 700, temp='28.1')
r28_1_2 = Run('lab1 data/timeresolved/28_1', '28_1_r2_240nm.txt', '28_1_r2_290nm.txt','28_1_r2_322nm.txt',
              start = 123, stop = 500, temp='28.1')
r28_1_3 = Run('lab1 data/timeresolved/28_1', '28_1_r3_240nm.txt', '28_1_r3_290nm.txt','28_1_r3_322nm.txt',
              start = 72, stop = 750, temp='28.1', cut = False, c322 = 181, r322 = 100)
r18_6_r1 = Run('lab1 data/timeresolved/18_6', '18_6_r1_240nm.txt', '18_6_r1_290nm.txt', '18_6_r1_322nm.txt',
               start = 120, stop =750, temp='18.6')
# r18_6_r2 = Run('lab1 data/timeresolved/18_6', '18_6_r2_240nm.txt', '18_6_r2_290nm.txt', '18_6_r2_322nm.txt',
#                start = 120, stop =750, temp='18.6')
r23_0_r1 = Run('lab1 data/timeresolved/23_0', '23_0_r1_240nm.txt', '23_0_r1_290nm.txt', '23_0_r1_322nm.txt',
               start = 140, stop =1110, temp='23.0')
r33_4_r1 = Run('lab1 data/timeresolved/33_4', '33_4_r1_240nm.txt', '33_4_r1_290nm.txt', '33_4_r1_322nm.txt',
               start = 90, stop = 630, temp='33.4')

data = [r18_6_r1,r23_0_r1, r28_1_1, r28_1_2,r28_1_3, r33_4_r1]

# runrates = []
# for r in data:
#     r.plotData()
#     r.fitdata()
#
#     runrates.append(r.getrates())
#
# runrates = np.array(runrates)
# np.save('runrates.npy', runrates)

runrates = np.load('runrates.npy')
def arrhenius(rates):
    tvec = rates[:,0]
    N = pc['Avogadro constant'][0]
    k = pc['Boltzmann constant'][0]
    xvec = 1/(N*k*(tvec+273.15))
    k1 = rates[:, 1]
    k2 = rates[:, 2]
    k3 = rates[:, 3]
    k4 = rates[:, 4]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

    # Subplot 1
    axs[0, 0].scatter(xvec, np.log(k1), c='k')
    axs[0, 0].set_title('$\\ln{k_1}$ vs $\\frac{1}{N_Ak_BT}$')
    axs[0, 0].set_ylabel('$\\ln{k_1}$')
    axs[0, 0].set_xlabel('$\\frac{1}{N_Ak_BT}$')
    coeffs, cov1 = np.polyfit(xvec, np.log(k1), 1, cov=True)
    slope, intercept = coeffs
    axs[0, 0].plot(xvec, np.polyval([slope, intercept], xvec), 'r-', label='Fitted line')
    equation_text = '$\\ln{k} = ' + str(np.round(intercept, 2)) + ' + ' + str(
        np.round(slope, 2)) + '\\frac{1}{N_Ak_BT}$'
    axs[0, 0].text(0.05, -0.25, equation_text, transform=axs[0, 0].transAxes, fontsize=9, verticalalignment='top')

    # Subplot 2
    axs[0, 1].scatter(xvec, np.log(k2), c='k')
    axs[0, 1].set_title('$\\ln{k_2}$ vs $\\frac{1}{N_Ak_BT}$')
    axs[0, 1].set_ylabel('$\\ln{k_2}$')
    axs[0, 1].set_xlabel('$\\frac{1}{N_Ak_BT}$')
    coeffs, cov2 = np.polyfit(xvec, np.log(k2), 1, cov=True)
    slope2, intercept2 = coeffs
    axs[0, 1].plot(xvec, np.polyval([slope2, intercept2], xvec), 'r-', label='Fitted line')
    equation_text = '$\\ln{k} = ' + str(np.round(intercept2, 2)) + ' + ' + str(
        np.round(slope2, 2)) + '\\frac{1}{N_Ak_BT}$'
    axs[0, 1].text(0.05, -0.25, equation_text, transform=axs[0, 1].transAxes, fontsize=9, verticalalignment='top')

    # Subplot 3
    axs[1, 0].scatter(xvec, np.log(k3), c='k')
    axs[1, 0].set_title('$\\ln{k_3}$ vs $\\frac{1}{N_Ak_BT}$')
    axs[1, 0].set_ylabel('$\\ln{k_3}$')
    axs[1, 0].set_xlabel('$\\frac{1}{N_Ak_BT}$')
    coeffs, cov3 = np.polyfit(xvec, np.log(k3), 1, cov=True)
    slope3, intercept3 = coeffs
    axs[1, 0].plot(xvec, np.polyval([slope3, intercept3], xvec), 'r-', label='Fitted line')
    equation_text = '$\\ln{k} = ' + str(np.round(intercept3, 2)) + ' + ' + str(
        np.round(slope3, 2)) + '\\frac{1}{N_Ak_BT}$'
    axs[1, 0].text(0.05, -0.25, equation_text, transform=axs[1, 0].transAxes, fontsize=9, verticalalignment='top')

    # Subplot 4
    axs[1, 1].scatter(xvec, np.log(k4), c='k')
    axs[1, 1].set_title('$\\ln{k_4}$ vs $\\frac{1}{N_Ak_BT}$')
    axs[1, 1].set_ylabel('$\\ln{k_4}$')
    axs[1, 1].set_xlabel('$\\frac{1}{N_Ak_BT}$')
    coeffs, cov4 = np.polyfit(xvec, np.log(k4), 1, cov=True)
    slope4, intercept4 = coeffs
    axs[1, 1].plot(xvec, np.polyval([slope4, intercept4], xvec), 'r-', label='Fitted line')
    equation_text = '$\\ln{k} = ' + str(np.round(intercept4, 2)) + ' + ' + str(
        np.round(slope4, 2)) + '\\frac{1}{N_Ak_BT}$'
    axs[1, 1].text(0.05, -0.25, equation_text, transform=axs[1, 1].transAxes, fontsize=9, verticalalignment='top')

    # Apply scientific notation to all axes
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')

    fig.suptitle('k vs $\\frac{1}{N_Ak_BT}$', fontsize=18)
    plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4)
    plt.show()
    print('k_1' , ' & ' , r4(np.exp(intercept)) , ' & ' , r4(-1 * slope) , '\\\\')
    print('k_2' , ' & ' , r4(np.exp(intercept2)) , ' & ' , r4(-1 * slope2) , '\\\\')
    print('k_3' , ' & ' , r4(np.exp(intercept3)) , ' & ' , r4(-1 * slope3) , '\\\\')

    covs = [cov1, cov2, cov3, cov4]
    ic = [intercept, intercept2, intercept3]

    for i, cov in enumerate(covs):
        berr, merr = np.sqrt((np.diag(cov)).flatten())
        Aerr = np.exp(ic[i])*berr
        print('k_',i+1, ' & ', r4(Aerr), ' & ', r4(merr), '\\\\')
arrhenius(runrates)