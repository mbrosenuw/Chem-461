from sympy.physics.wigner import wigner_3j as tj
import numpy as np
import os
import csv

def tj2(j1,j2, k1,q,k2,tjs):
    if np.abs(k1) == 0:
        k1 = 0
    if np.abs(k2) == 0:
        k2 = 0
    coeff = 1
    if j1 > j2:
        key = ",".join(map(str, [j2, j1, q, k2, k1]))
        if key in tjs:
            sym= tjs[key]
        else:
            key = ",".join(map(str, [j2, j1, -1*q, -1*k2, -1*k1]))
            if key in tjs:
                sym = tjs[key]
            else:
                tjs[key] = tj(1, j2, j1, q, k2, k1).evalf(50)
                print('getting 3j')
                sym =  tjs[key]
    elif j1 < j2:
        key = ",".join(map(str, [j1,j2,q,k1,k2]))
        if key in tjs:
            sym= tjs[key]
        else:
            key = ",".join(map(str, [j1,j2,-1*q,-1*k1,-1*k2]))
            if key in tjs:
                sym = tjs[key]
            else:
                tjs[key] = tj(1, j1, j2, q, k1, k2).evalf(50)
                print('getting 3j')
                sym =  tjs[key]
    elif j1 == j2:
        if k1 ==k2:
            key = ",".join(map(str, [j2, j1, q, k2, k1]))
            if key in tjs:
                sym = tjs[key]
            else:
                key = ",".join(map(str, [j2, j1, -1 * q, -1 * k2, -1 * k1]))
                if key in tjs:
                    coeff = -1
                    sym = tjs[key]
                else:
                    tjs[key] = tj(1, j2, j1, q, k2, k1).evalf(50)
                    print('getting 3j')
                    sym = tjs[key]
        elif k1 > k2:
            key = ",".join(map(str, [j2, j1, q, k2, k1]))
            if key in tjs:
                sym = tjs[key]
            else:
                key = ",".join(map(str, [j2, j1, -1 * q, -1 * k2, -1 * k1]))
                if key in tjs:
                    coeff = -1
                    sym = tjs[key]
                else:
                    tjs[key] = tj(1, j2, j1, q, k2, k1).evalf(50)
                    print('getting 3j')
                    sym = tjs[key]
        elif k2 > k1:
            key = ",".join(map(str, [j1, j2, q, k1, k2]))
            coeff = -1
            if key in tjs:
                sym = tjs[key]
            else:
                key = ",".join(map(str, [j1, j2, -1 * q, -1 * k1, -1 * k2]))
                if key in tjs:
                    coeff = 1
                    sym = tjs[key]
                else:
                    tjs[key] = tj(1, j1, j2, q, k1, k2).evalf(50)
                    print('getting 3j')
                    sym = tjs[key]
    else:
        print('problem')
    return coeff * sym

def loadtjs(filename):
    tjs = {}
    if os.path.exists(filename):
        with open(filename, "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                tjs[row[0]] = float(row[1])
    return tjs

def writetjs(filename, tjs):
    with open(filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in tjs.items():
            writer.writerow([key,value])