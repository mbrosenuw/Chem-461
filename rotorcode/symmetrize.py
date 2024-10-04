import numpy as np

def Htransform(H):
    n = H.shape[0]
    j = (n-1)/2
    u = np.zeros((n, n), dtype=complex)
    if j %2 == 0:
        u[n // 2, n // 2] = 1
    else:
        u[n // 2, n // 2] = 1j
    for k in range(n // 2):
        u[k, k] = (-1)**j *np.sqrt(2) / 2
        u[k, n - k - 1] = np.sqrt(2) / 2
        u[n - k - 1, k] = (-1)**j *-1j * np.sqrt(2) / 2
        u[n - k - 1, n - k - 1] = 1j * np.sqrt(2) / 2
    kidxs = np.arange(0, n, 1)
    evenkidx = np.arange(j%2, n,2)
    oddkidx = kidxs[~np.isin(kidxs, evenkidx)]
    mp = len(oddkidx)//2
    newidxs = np.concatenate((oddkidx[:mp], evenkidx, oddkidx[mp:])).astype('int')
    A = np.matrix(np.zeros((n,n)))
    ks = genks(int(j))
    ks2 = np.zeros((int(2*j+1),2))
    for i in range(n):
        A[i,newidxs[i]] = 1
        ks2[i] = ks[newidxs[i]][:]
    u = A @ u
    a = np.matrix(u) @ np.matrix(H)
    b = a @ np.matrix(u).H
    return b, ks2

def genks(j):
    ks = np.zeros((2*j+1, 2))
    if j %2 ==0:
        ks[j][1] = 0
    else:
        ks[j][1] = 1
    for i in range(1,j+1):
        ks[j-i] = [i,0]
        ks[j + i] = [i, 1]
    return ks