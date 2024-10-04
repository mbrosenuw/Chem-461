import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import asymrotor

filename = 'Rosen_Au24.csv'
spec2 = np.array([])
freq2 = np.array([])
if os.path.exists(filename):
    with open(filename, "r", newline="",encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            freq2 = np.append(freq2, float(row[0]))
            spec2 = np.append(spec2, float(row[1]))

spec2 = np.log10(np.reciprocal(spec2+100-85))
spec2 = spec2 - np.min(spec2)
spec2 = spec2/np.max(spec2)

l = 10.577
u = 10.06
hcl = [l]*3
hcl = [u]*3
lims =[-250,250]
shift  = 5667.7
freq, spec = asymrotor.spectra(hcl, hcl, [1,1,1],15,300,'hcl', lims, 0.5, False)
spec = spec/np.max(spec)
np.save('spec.npy', spec)
np.save('freq.npy', freq)

# # Stack the x and y arrays horizontally
# data = np.column_stack((freq2, spec2))
#
# # Save the data as a CSV file
# np.savetxt('xy_pairs.csv', data, delimiter=',', header='x,y', comments='', fmt='%.10f')
#

plt.figure(figsize=(5,8))

condition = (freq < -5) | (freq > 5)
c2 = (freq2 > 5400) & (freq2 < 5900)

# Apply the condition to x and y
freq = freq[condition]
spec = spec[condition]
spec = spec- np.min(spec)
spec = spec/np.max(spec)
freq2 = freq2[c2]
spec2 = spec2[c2]
spec2 = spec2- np.min(spec2)
spec2 = spec2/np.max(spec2)
plt.plot(freq2, spec2)
plt.plot(freq + shift, spec)
plt.ylabel('Intensity')
plt.xlabel('Energy (cm-1)')
plt.show()

