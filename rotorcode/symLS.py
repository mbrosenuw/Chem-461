import numpy as np
from LScoeffs import beta
from sympy.physics.wigner import wigner_3j as tj

def preterms(j1, k1, s1, j2, k2, s2):
    b1 = beta(s1, k1)
    b2 = beta(s2, k2)
    coeff1 = 2 * j2 + 1
    coeff2 = 2 * j1 + 1
    return np.abs(b2*b1)**2 * coeff1 * coeff2/3

def linestrength(j2, k2, s2, j1, k1, s1, mu):
    mux = mu[0]
    muy = mu[1]
    muz = mu[2]
    if k1 == k2:
        if k1 == 0 and k2 == 0:
            if s1 == s2:
                elem = 0
            else:
                elem = 4 * muz ** 2 * tj(j1, 1, j2, 0,0,0) ** 2 * (1- (-1)**(j1 + s2))**2
        else:
            if s1 == s2:
                elem = 0
            else:
                elem = 4 * muz ** 2 * tj(j1, 1, j2, -k1, 0, k2)**2
    elif k2 == k1 + 1:
        if k1 != 0:
            if s1 == s2:
                elem = 2 * mux ** 2 * tj(j1, 1, j2, -k1, -1, k1+1) ** 2
            else:
                elem = 2 * muy ** 2 * tj(j1, 1, j2, -k1, -1, k1 + 1)** 2
        else:
            if s1 == s2:
                elem = 2 * mux ** 2 * tj(j1, 1, j2, 0, -1, 1)** 2 * (1 + (-1)**(j1 + s2))**2
            else:
                elem = 2 * muy ** 2 * tj(j1, 1, j2, 0, -1, 1)** 2 * (1 + (-1)**(j1 + s2))**2
    elif k2 == k1-1:
        if k1 != 1:
            if s1 == s2:
                elem = 2 * mux ** 2 * tj(j1, 1, j2, -k1, 1, k1-1) ** 2
            else:
                elem = 2 * muy ** 2 * tj(j1, 1, j2, -k1, 1, k1 - 1) ** 2
        else:
            if s1 == s2:
                elem = 2 * mux ** 2 * tj(j1, 1, j2, -1, 1, 0)** 2 * (1 + (-1)**(j2 + s2))**2
            else:
                elem = 2 * muy ** 2 * tj(j1, 1, j2, -1, 1, 0)** 2 * (1 + (-1)**(j2 + s2))**2
    return preterms(j1, k1, s1, j2, k2, s2) * elem

def fulllinestrength(j2, k2, s2, j1, k1,s1, mu):
    mx = mu[0]
    my = mu[1]
    mz = mu[2]
    pre = preterms(j1, k1,s1,j2, k2, s2)
    t1 = -np.sqrt(2)/2 * tj(j1, 1 , j2, -k1, 1, k2) * ((mx - 1j * my) + (-1)**(s1 + s2) * (mx + 1j * my))
    t2 = np.sqrt(2) / 2 * tj(j1, 1, j2, -k1, -1, k2) * ((mx + 1j * my) + (-1) ** (s1 + s2) * (mx - 1j * my))
    t3 = mz * tj(j1, 1, j2, -k1, 0, k2) * (1- (-1)**(s1 + s2))
    t4 = np.sqrt(2) / 2 * tj(j1, 1, j2, k1, -1, k2) * ((-1)**(j1 + s2) * (mx - 1j * my) + (-1)**(s1 + j1) * (mx + 1j * my))
    t5 = mz * tj(j1, 1, j2, k1, 0, k2) * ((-1)**(j1 + s2) - (-1)**(s1 + j1))
    elem = np.abs(t1 + t2 + t3 + t4 + t5)**2
    return pre * elem
