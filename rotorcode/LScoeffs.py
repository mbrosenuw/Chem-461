import numpy as np

def delta(n1, n2):
    if n1 == n2:
        return 1
    else:
        return 0

def beta(phase,k):
    num = (-1)**phase
    denom = (2 * (1 + delta(k, 0)))
    return np.emath.sqrt(num / denom)
