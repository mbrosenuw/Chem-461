def getblock(j, tau, idxs):
    if j ==0:
        block = 1
        phase = 0
        eo = 0
    elif j == 1:
        if tau ==0:
            block = 0
            phase = 0
            eo = 1
        elif tau ==1:
            block = 2
            phase = 1
            eo = 0
        elif tau ==2:
            block = 3
            phase = 1
            eo = 1
    else:
        if idxs[0] <= tau < idxs[1]:
            block = 0
            phase = 0
            eo = 1
        elif idxs[1] <= tau < idxs[2]:
            block = 1
            phase = 0
            eo = 0
        elif idxs[2] <= tau < idxs[3]:
            block = 2
            phase = 1
            eo = 0
        elif idxs[3] <= tau < idxs[4]:
            block = 3
            phase = 1
            eo = 1
        else:
            print('what have you done')
    return block, phase, eo