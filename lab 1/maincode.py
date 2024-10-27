import numpy as np
import matplotlib.pyplot as plt

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
        if ravg:
            y_data = np.convolve(y_data, np.ones(w) / w, mode='same')
    x_data = x_data - np.min(x_data)
    return np.vstack((x_data, y_data)).T

data = opentime('lab1 data/timeresolved/28_1/28_1_r3_240nm.txt', ravg=False)
plt.plot(data[:,0], data[:,1])
data = opentime('lab1 data/timeresolved/28_1/28_1_r3_240nm.txt', ravg=True, w=10)
plt.plot(data[:,0], data[:,1])
plt.show()