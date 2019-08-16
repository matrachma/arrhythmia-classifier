from math import sqrt

# Baseline Wander Removal
# Daubechies 4 Constant
c0 = (1+sqrt(3))/(4*sqrt(2))
c1 = (3+sqrt(3))/(4*sqrt(2))
c2 = (3-sqrt(3))/(4*sqrt(2))
c3 = (1-sqrt(3))/(4*sqrt(2))


def conv(x, h):
    length = len(x) + len(h) - 1
    y = [0]*length

    for i in range(len(y)):
        for j in range(len(h)):
            if (i-j >= 0) and (i-j < len(x)):
                y[i] += h[j] * x[i-j]

    return y


def db4_dec(x, level):
    lpk = [c0, c1, c2, c3]
    hpk = [c3, -c2, c1, -c0]
    lp_ds, hp_ds = None, None

    result = [[]]*(level+1)
    x_temp = x[:]
    for i in range(level):
        lp = conv(x_temp, lpk)
        hp = conv(x_temp, hpk)

        lp_ds = [0]*int(len(lp)/2)
        hp_ds = [0]*int(len(hp)/2)
        for j in range(len(lp_ds)):
            lp_ds[j] = lp[2*j+1]
            hp_ds[j] = hp[2*j+1]

        result[level-i] = hp_ds
        x_temp = lp_ds[:]

    result[0] = lp_ds
    return result


def db4_rec(signals, level):
    lpk = [c3, c2, c1, c0]
    hpk = [-c0, c1, -c2, c3]

    cp_sig = signals[:]
    for i in range(level):
        lp = cp_sig[0]
        hp = cp_sig[1]

        if len(lp) > len(hp):
            length = 2*len(hp)
        else:
            length = 2*len(lp)

        lpu = [0]*(length+1)
        hpu = [0]*(length+1)
        index = 0
        for j in range(length+1):
            if j % 2 != 0:
                lpu[j] = lp[index]
                hpu[j] = hp[index]
                index += 1

        lpc = conv(lpu, lpk)
        hpc = conv(hpu, hpk)

        lpt = lpc[3:-3]
        hpt = hpc[3:-3]

        org = [0]*len(lpt)
        for j in range(len(org)):
            org[j] = lpt[j] + hpt[j]

        if len(cp_sig) > 2:
            cp_sig = [org]+cp_sig[2:]
        else:
            cp_sig = [org]

    return cp_sig[0]


def calc_energy(x):
    total = 0
    for i in x:
        total += i*i
    return total


def bwr(raw):
    en1 = 0
    en2 = 0

    curlp = raw[:]
    num_dec = 0
    while True:
        [lp, hp] = db4_dec(curlp, 1)

        en0 = en1
        en1 = en2
        en2 = calc_energy(hp)
        if en0 > en1 and en1 < en2:
            last_lp = curlp
            break

        curlp = lp[:]
        num_dec = num_dec+1

    base = last_lp[:]
    for i in range(num_dec):
        base = db4_rec([base, [0]*len(base)], 1)

    ecg_out = [0]*len(raw)
    for i in range(len(raw)):
        ecg_out[i] = raw[i] - base[i]

    return base, ecg_out
