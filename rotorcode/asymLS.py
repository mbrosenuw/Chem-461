import numpy as np
from LScoeffs import beta
from getidx import getidx
from tjsyms import tj2

def preterms(j1,j2):
    coeff1 = 2 * j2 + 1
    coeff2 = 2 * j1 + 1
    return  coeff1 * coeff2 /3

def semifastLS(sys2, tau2, block2, phase2, eo2, sys1, tau1, block1, phase1, eo1, mu, tjs, stats):
    mx = mu[0]
    my = mu[1]
    mz = mu[2]
    wfn1 = sys1.wfns[block1][:, tau1 - sys1.idxs[block1]]
    krange1 = sys1.kranges[block1]
    wfn2 = sys2.wfns[block2][:, tau2 - sys2.idxs[block2]]
    krange2 = sys2.kranges[block2]
    maxk2 = krange2[-1]
    mink2 = krange2[0]
    
    sum = 0
    c2 = 0
    if phase1 == phase2 and eo1 == eo2:
        c2 = 0
        sum = 0
    elif phase1 == phase2 and eo1 != eo2:
        c2 = 2 * mx ** 2
        if c2 != 0:
            for k1 in krange1:
                alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                b1 = beta(phase1, k1)
                if k1-1 >= mink2 and k1-1 <= maxk2:
                    k2 = k1 - 1
                    alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                    b2 = np.conjugate(beta(phase2, k2))
                    term = -1 * tj2(sys1.j, sys2.j, -k1, 1, k2, tjs)
                    if k1 == 1:
                        term += (-1) **(phase1 + sys1.j) * tj2(sys1.j, sys2.j, 1, -1, 0, tjs)
                    sum += alpha1 * alpha2 * b1 * b2 * term
                if k1+1 >= mink2 and k1+1 <= maxk2:
                    k2 = k1 + 1
                    alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                    b2 = np.conjugate(beta(phase2, k2))
                    term = tj2(sys1.j, sys2.j, -k1, -1, k2, tjs)
                    if k1 == 0:
                        term += (-1) ** (phase1 + sys1.j) * tj2(sys1.j, sys2.j, 0, -1, 1, tjs)
                    sum += alpha1 * alpha2 * b1 * b2 * term
    elif phase1 != phase2 and eo1 == eo2:
        c2 = 4 * mz **2
        if c2 != 0:
            for k1 in krange1:
                alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                b1 = beta(phase1, k1)
                if k1 >= mink2 and k1 <= maxk2:
                    k2 = k1
                    alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                    b2 = np.conjugate(beta(phase2, k2))
                    term = tj2(sys1.j, sys2.j, -k1, 0, k2, tjs)
                    if k1 == 0:
                        term += (-1) **(phase2 + sys1.j+1) * tj2(sys1.j, sys2.j, 0, 0, 0, tjs)
                    sum += alpha1 * alpha2 * b1 * b2 * term
    elif phase1 != phase2 and eo1 != eo2:
        c2 = 2 * my ** 2
        if c2 != 0:
            for k1 in krange1:
                alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                b1 = beta(phase1, k1)
                if k1-1 >= mink2 and k1-1 <= maxk2:
                    k2 = k1 - 1
                    alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                    b2 = np.conjugate(beta(phase2, k2))
                    term = tj2(sys1.j, sys2.j, -k1, 1, k2, tjs)
                    if k1 == 1:
                        term += (-1) **(phase1 + sys1.j) * tj2(sys1.j, sys2.j, 1, -1, 0, tjs)
                    sum += alpha1 * alpha2 * b1 * b2 * term
                if k1+1 >= mink2 and k1+1 <= maxk2:
                    k2 = k1 + 1
                    alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                    b2 = np.conjugate(beta(phase2, k2))
                    term = tj2(sys1.j, sys2.j, -k1, -1, k2, tjs)
                    if k1 == 0:
                        term += (-1) ** (phase1 + sys1.j) * tj2(sys1.j, sys2.j, 0, -1, 1, tjs)
                    sum += alpha1 * alpha2 * b1 * b2 * term
    return preterms(sys1.j, sys2.j) * c2 * np.abs(sum)**2


def LS(sys2, tau2, block2, phase2, eo2, sys1, tau1, block1, phase1, eo1, mu, tjs, stats):
    mx = mu[0]
    my = mu[1]
    mz = mu[2]
    wfn1 = sys1.wfns[block1][:, tau1 - sys1.idxs[block1]]
    krange1 = sys1.kranges[block1]
    wfn2 = sys2.wfns[block2][:, tau2 - sys2.idxs[block2]]
    krange2 = sys2.kranges[block2]

    sum = 0
    for k1 in krange1:
        alpha1 = wfn1[getidx(sys1, block1, k1), 0]
        b1 = beta(phase1, k1)
        sign = (-1) ** k1
        for k2 in krange2:
            alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
            b2 = np.conjugate(beta(phase2, k2))
            if alpha1 != 0 and alpha2 != 0:
                elem = 0
                if k1 == k2:
                    if k1 == 0:
                        sym = tj2(sys1.j, sys2.j, k1, 0, k2, tjs)
                        elem += mz * sym * ((-1) ** (sys1.j + phase2) - (-1) ** (phase1 + sys1.j))
                    sym = tj2(sys1.j, sys2.j, -k1, 0, k2, tjs)
                    elem += mz * sym * (1 - (-1) ** (phase1 + phase2))
                elif k2 == (k1 + 1):
                    if k1 == 0:
                        sym = tj2(sys1.j, sys2.j, k1, -1, k2, tjs)
                        elem += np.sqrt(2) / 2 * sym * (
                                (-1) ** (sys1.j + phase2) * (mx - 1j * my) + (-1) ** (phase1 + sys1.j) * (
                                mx + 1j * my))
                    sym = tj2(sys1.j, sys2.j, -k1, -1, k2, tjs)
                    elem += np.sqrt(2) / 2 * sym * ((mx + 1j * my) + (-1) ** (phase1 + phase2) * (mx - 1j * my))
                elif k2 == (k1 - 1):
                    if k1 == 1:
                        sym = tj2(sys1.j, sys2.j, k1, -1, k2, tjs)
                        elem += np.sqrt(2) / 2 * sym * (
                                (-1) ** (sys1.j + phase2) * (mx - 1j * my) + (-1) ** (phase1 + sys1.j) * (
                                mx + 1j * my))
                    sym = tj2(sys1.j, sys2.j, -k1, 1, k2, tjs)
                    elem += -np.sqrt(2) / 2 * sym * ((mx - 1j * my) + (-1) ** (phase1 + phase2) * (mx + 1j * my))
                sum += alpha1 * alpha2 * b1 * b2 * sign * elem

    return preterms(sys1.j, sys2.j) * np.abs(sum) ** 2


def fastLS(sys2, tau2, block2, phase2, eo2, sys1, tau1, block1, phase1, eo1, mu, tjs, stats):
    mx = mu[0]
    my = mu[1]
    mz = mu[2]
    wfn1 = sys1.wfns[block1][:, tau1 - sys1.idxs[block1]]
    krange1 = sys1.kranges[block1]
    wfn2 = sys2.wfns[block2][:, tau2 - sys2.idxs[block2]]
    krange2 = sys2.kranges[block2]
    maxk2 = krange2[-1]
    mink2 = krange2[0]

    #stats is given in ee eo oe oo
    if eo1 == eo2:
        if eo1 == 0:
            stat = stats[0]
        else:
            stat = stats[3]
    else:
        if eo1 == 0:
            stat = stats[1]
        else:
            stat = stats[2]
    sum = 0
    c2 = 0
    if phase1 == phase2:
        if eo1 == eo2:
            c2 = 0
            sum = 0
        else:
            c2 = 2 * mx ** 2
            if c2 != 0:
                if krange1[0]==0:
                    k1 = 0
                    k2 = 1
                    if k2 >= mink2 and k2 <= maxk2:
                        alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                        b1 = beta(phase1, k1)
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, 0, -1, 1, tjs) * (1 + (-1)**(phase1+sys1.j))
                        sum += (-1) ** k1 * alpha1 *alpha2 * b1 * b2 * term
                    krange1 = krange1[1:]
                elif krange1[0] == 1:
                    k1 = 1
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)

                    k2 = 0
                    if k2 >= mink2 and k2 <= maxk2:
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = -1 * tj2(sys1.j, sys2.j, -1, 1, 0, tjs) * (1 + (-1) ** (phase1 + sys2.j))
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term

                    k2 = 2
                    if k2 >= mink2 and k2 <= maxk2:
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -1, -1, 2, tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
                    krange1 = krange1[1:]

                for k1 in krange1:
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)
                    if k1 - 1 >= mink2 and k1 - 1 <= maxk2:
                        k2 = k1 - 1
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -k1, 1, k2, tjs)
                        sum += (-1) ** k1 * -1 * alpha1 * alpha2 * b1 * b2 * term
                    if k1 + 1 >= mink2 and k1 + 1 <= maxk2:
                        k2 = k1 + 1
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -k1, -1, k2, tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
    elif phase1 != phase2:
        if eo1 == eo2:
            c2 = 4 * mz**2
            if c2 != 0:
                if krange1[0] == 0:
                    k1 = 0
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)

                    k2 = 0
                    if k2 >= mink2 and k2 <= maxk2:
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, 0, 0, 0, tjs) * (1 - (-1) ** (phase2 + sys1.j))
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
                    krange1 = krange1[1:]

                for k1 in krange1:
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)
                    if k1 >= mink2 and k1 <= maxk2:
                        k2 = k1
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -k1, 0, k2, tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
        else:
            c2 = 2 * my ** 2
            if c2 != 0:
                if krange1[0] == 0:
                    k1 = 0
                    k2 = 1
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)
                    alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                    b2 = np.conjugate(beta(phase2, k2))
                    term = tj2(sys1.j, sys2.j, 0, -1, 1, tjs) * (1 + (-1) ** (phase1 + sys1.j))
                    sum += (-1) ** k1 *  alpha1 * alpha2 * b1 * b2 * term
                    krange1 = krange1[1:]
                elif krange1[0] == 1:
                    k1 = 1
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)

                    k2 = 0
                    if k2 >= mink2 and k2 <= maxk2:
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -1, 1, 0, tjs)+ (-1) ** (phase1 + sys1.j) * tj2(sys1.j, sys2.j,1,-1,0,tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term

                    k2 = 2
                    if k2 >= mink2 and k2 <= maxk2:
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -1, -1, 2, tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
                        krange1 = krange1[1:]

                for k1 in krange1:
                    alpha1 = wfn1[getidx(sys1, block1, k1), 0]
                    b1 = beta(phase1, k1)
                    if k1 - 1 >= mink2 and k1 - 1 <= maxk2:
                        k2 = k1 - 1
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -k1, 1, k2, tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
                    if k1 + 1 >= mink2 and k1 + 1 <= maxk2:
                        k2 = k1 + 1
                        alpha2 = np.conjugate(wfn2[getidx(sys2, block2, k2), 0])
                        b2 = np.conjugate(beta(phase2, k2))
                        term = tj2(sys1.j, sys2.j, -k1, -1, k2, tjs)
                        sum += (-1) ** k1 * alpha1 * alpha2 * b1 * b2 * term
    else:
        print('not valid combo of states')
    return stat * preterms(sys1.j, sys2.j) * c2 * np.abs(sum) ** 2


