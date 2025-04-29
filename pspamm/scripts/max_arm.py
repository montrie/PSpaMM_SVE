def getBlocksize(m , n, bk, v_size=2):

	bm = v_size
	bn = 1
	maxval = 0

	for i in range(v_size, m+1, v_size):
		for j in range(1, n+1):
			if ARM_condition(i, j, bk, v_size):
				if i*j > maxval:
					maxval = i*j
					bm = i
					bn = j

	return (bm, bn)


def ARM_condition(bm, bn, bk, v_size):
#  v_size = 2
  # ceiling division
  vm = -(bm // -v_size)
  return (bn+bk) * vm + bn <= 32
