import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from os.path import join as j

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
        fitabs(self.d240[c, :], axs[0], name=self.n240, runname = self.temp + '_240')
        axs[0].set_title(self.n240)
        axs[0].set_title('240 nm Abs vs time ')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Abs')
        c = (self.d290[:, 0] > self.start) & (self.d290[:, 0] < self.stop)
        fitabs(self.d290[c, :], axs[1], name=self.n290, runname = self.temp+ '_290')
        axs[1].set_title(self.n290)
        axs[1].set_title('290 nm Abs vs time ')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Abs')
        if self.cut:
            c = (self.d322[:,0] > self.c322)
            self.d322[c,1]= 0
        c = (self.d322[:, 0] > self.start) & (self.d322[:, 0] < self.stop)
        fit322(self.d322[c, :], axs[2], name=self.n322, runname = self.temp + '_322')
        axs[2].set_title(self.n322)
        axs[2].set_title('322 nm Abs vs time ')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Abs')
        fig.suptitle(f'Fits of Abs vs time for {self.temp}K', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

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
    print(runname, ' & ', r6(B), ' & ', r6(D), ' & ', r6(k3), ' & ', r6(kT), ' & ', r6(sB), ' & ', r6(sD), ' & ', r6(sk3), ' & ', r6(skT), '\\\\')

def r6(arr):
    def round_single_value(val):
        if val == 0:
            return 0
        else:
            return np.round(val, decimals=5 - int(np.floor(np.log10(np.abs(val)))))

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

for r in data:
    r.plotData()
    r.fitdata()