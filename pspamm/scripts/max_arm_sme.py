def getBlocksize(m, n, k, v_size=2):
    # v_size default is 2, however for SVE that parameter will always be larger
    bm = v_size
    bn = v_size
    bk = 0

    for bki in range(1, k + 1, 1):
        #if tileable(k, bki) and ARM_condition(bm, bn, bki, v_size):
        if ARM_condition(bm, bn, bki, v_size):
            bk = bki

    if bk == 0:
        raise RuntimeError("Could not find an appropriate block size. We suggest padding the matrix dimensions")
    
    return (bm, bn, bk)


def ARM_condition(bm, bn, bk, v_size):
    # ceiling division
    vm = -(bm // -v_size)
    vn = -(bn // -v_size)
    return (vn + vm) * bk <= 32


def tileable(m, bm):
    return m % bm == 0

