import numpy as np
import asymrotor
import matplotlib.pyplot as plt
from rotor import Rotor
from matplotlib.pyplot import cm

jmax=2
jlist = list(range(jmax+1))
brange = np.linspace(1,7,100)
arange = np.linspace(10,7,100)
print(arange)
energylist = []
k = []
for (b,a) in zip(brange,arange):
    energies = []
    k.append((2*b-a-1)/(a-1))
    for j in jlist:
        r = Rotor(j, [a, b, 1])
        energies.append(np.sort(r.fullenergies))
    if not energylist:
        energylist = energies
    else:
        energylist = [np.vstack((matrix, new_row)) for matrix, new_row in zip(energylist, energies)]


colors = cm.get_cmap('tab10', jmax+1)
color = iter([colors(i) for i in range(colors.N)])
plt.figure(figsize=(8,5))
for (matrix, j) in zip(energylist,jlist):
    c = next(color)
    for row in matrix.T:
        plt.plot(k,row, color=c, label='j = ' +str(j))

plt.title('Energy levels vs Asymmetry Parameter')
plt.xlabel('κ')
plt.ylabel('Energy (cm-1)')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='right')
plt.show()