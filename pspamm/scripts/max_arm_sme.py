def getBlocksize(m, n, bk, v_size=2):
    # v_size default is 2, however for SVE that parameter will always be larger
    bm = 2
    bn = 1
    maxval = 0

    for i in range(1, m + 1, 1):
        next_multiple = i
        while next_multiple % v_size != 0:
            next_multiple += 1
        for j in range(1, n + 1):
            if ARM_condition(next_multiple, j, bk, v_size) and tileable(m, i):
                if i * j >= maxval:
                    maxval = i * j
                    bm = i
                    bn = j

    if maxval == 0:
        raise RuntimeError("Could not find an appropriate block size. We suggest padding the matrix dimensions")

    return (bm, bn)


def get_blocksize(m, n, k, v_size=2):
    bm_ovh = m % v_size
    bn_ovh = n % v_size
    bk_ovh = k % v_size
    bm = 2
    bn = 1
    bk = 2
    maxval = 0

    # TODO: bk > 1 ?
    for h in range(2, k + 1):
        for i in range(1, m + 1, 1):
# TODO: next_multiple also for j param, similar to i!
            next_multiple = i
            while next_multiple % v_size != 0:
                next_multiple += 1
            for j in range(1, n + 1):
#                if ARM_condition(next_multiple, j, h, v_size) and tileable(m, i):
#                   (bn*0 + bk) * vm + bn * bk <= 32
                if (j*0 + h) * (-(i // -v_size)) + j * h <= 32:
                    if i * j * h >= maxval:
                        maxval = i * j * h
                        bm = i
                        bn = j
                        bk = h

    return (bm, bn, bk)
      


def ARM_condition(bm, bn, bk, v_size):
    # ceiling division
    vm = -(bm // -v_size)  
    return (bn + bk) * vm + bn <= 32 and vm >= bk # and bn >= bk


def tileable(m, bm):
    return m % bm == 0



def main():
    m = 80
    n = 80
    k = 80
    bk = 1
    v_size = 8
    bm, bn = getBlocksize(m, n, bk, v_size)
    vm = -(bm // -v_size)


    a1 = [[vm * c + r for c in range(bn)] for r in range(vm)]
    a = [[vm * c + r for c in range(bk)] for r in range(vm)]
    b = [[vm * bk + bn * r + c for c in range(bn)] for r in range(bk)]
    c = [[32 - vm * bn + vm * c + r for c in range(bn)] for r in range(vm)]

#    for bm in 
#    print(f"a1={a1}")
#    print(f"bm={bm}, bn={bn}")
#    print(f"a={a}")
#    print(f"b={b}")
#    print(f"c={c}")

    bm, bn, bk = get_blocksize(m, n, k, v_size)
    vm = -(bm // -v_size)
    a = [[vm * c + r for c in range(bk)] for r in range(vm)]
    b = [[vm * bk + bn * r + c for c in range(bn)] for r in range(bk)]
#    c = [[32 - vm * bn + vm * c + r for c in range(bn)] for r in range(vm)]

#    for bm in
    print(f"bm={bm}, bn={bn}, bk={bk}")
    print(f"a={a}")
    print(f"b={b}")
#    print(f"c={c}")



if __name__ == '__main__':
    main()
