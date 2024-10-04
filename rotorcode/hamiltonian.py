import numpy as np

def jsquare(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b:
                mat[a,b] = j * (j + 1)
    return mat

def jcsquare(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b:
                mat[a,b] = kvec[b]*kvec[a]
    return mat

def jbodyplus(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b-1:
                mat[a,b] = (j * (j+1) - kvec[b]*(kvec[b]-1))**(1/2)
    return mat

def jbodyminus(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b+1:
                mat[a,b] = (j * (j+1) - kvec[b]*(kvec[b]+1))**(1/2)
    return mat

def getH(rotor):
    kvec = np.arange(-rotor.j,rotor.j+1,1)
    t1 = (rotor.a + rotor.b)/2 * jsquare(rotor.j, kvec)
    t2 = (rotor.c - (rotor.a + rotor.b)/2) * jcsquare(rotor.j, kvec)
    t3 = (rotor.a - rotor.b)/4 * (np.linalg.matrix_power(jbodyplus(rotor.j, kvec),2) + np.linalg.matrix_power(jbodyminus(rotor.j, kvec),2))
    return t1 + t2 + t3
