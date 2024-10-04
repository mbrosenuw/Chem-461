import numpy as np

def eigenblock(H, j):
    numo = j + j%2
    numop = int(numo // 2)
    numom = int(numo // 2)
    nume = H.shape[0]-numo
    if j % 2 == 0:
        numep = int(nume//2 + 1)
        numem = int(nume//2)
    else:
        numep = int(nume // 2)
        numem = int(nume // 2 + 1)

    idxs = np.cumsum([0,numop, numep, numem, numom])
    eig = []
    subwfns = []
    if j % 2 == 0:
        eprange = np.arange(0,j + 1,2)
        emrange = np.arange(2,j + 1,2)
    else:
        eprange = np.arange(2, j + 1, 2)
        emrange = np.arange(0, j + 1, 2)
    kranges = [np.arange(1,j + 1,2).astype('int'), eprange.astype('int'), emrange.astype('int'), np.arange(1,j + 1,2).astype('int')]
    vec = np.zeros(H.shape)
    for i in range(len(idxs)-1):
        b = H[idxs[i]:idxs[i+1], idxs[i]:idxs[i+1]]
        beig, bwfn = np.linalg.eigh(b)
        vec[idxs[i]:idxs[i+1], idxs[i]:idxs[i+1]] = bwfn
        eig.append(beig)
        if i == 0 or i == 1:
            bwfn = bwfn[::-1]
        subwfns.append(bwfn)
    eiglist = np.hstack(eig)
    return eiglist, eig, subwfns, idxs, kranges