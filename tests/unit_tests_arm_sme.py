#!/usr/bin/env python3

import sve_testsuite_generator as generator

import pspamm.scripts.max_arm_sme as max_sme

import sys

v_len = 4

if len(sys.argv) == 2:
    v_len = int(sys.argv[1]) // 128

blocksize_algs = [max_sme]
v_size = 2 * v_len
v_size_s = 4 * v_len
bitlen = v_len * 128
kernels = []

# define the maximum allowed difference between elements of our solution and the reference solution for
# double and single precision
delta_sp = 1e-4 # epsilon is around e-7 => /2 ... For most cases, 1e-6 is enough
delta_dp = 1e-7 # epsilon is around e-15 => /2

# batches of 5 unit tests following the same alpha/beta parameter configuration:
# 1. alpha = 1.0, beta = 0.0 -> default config
# 2. alpha = 1.0, beta != 0.0 and beta != 1.0
# 3. alpha != 1.0, beta = 0.0
# 4. alpha != 1.0, beta = 1.0
# 5. alpha != 1.0, beta != 0.0 and beta != 1.0

# kernels.append(generator.DenseKernel("sme_base_test1", 8, 8, 8, 8, 8, 8, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test2", 8, 8, 8, 8, 8, 8, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test3", 8, 8, 8, 8, 8, 8, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test4", 8, 8, 8, 8, 8, 8, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test5", 8, 8, 8, 8, 8, 8, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sme_base_test6", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test7", 16, 16, 16, 16, 16, 16, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test8", 16, 16, 16, 16, 16, 16, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test9", 16, 16, 16, 16, 16, 16, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test10", 16, 16, 16, 16, 16, 16, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))

# kernels.append(generator.DenseKernel("sme_base_test11", 64, 64, 64, 64, 64, 64, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test12", 64, 64, 64, 64, 64, 64, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test13", 64, 64, 64, 64, 64, 64, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test14", 64, 64, 64, 64, 64, 64, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test15", 64, 64, 64, 64, 64, 64, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))

# kernels.append(generator.DenseKernel("sme_base_test16", 64, 64, 16, 64, 64, 64, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test16_5", 16, 16, 8, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test17", 64, 64, 16, 64, 64, 64, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test18", 64, 64, 16, 64, 64, 64, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test19", 64, 64, 16, 64, 64, 64, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test20", 64, 64, 16, 64, 64, 64, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))

# kernels.append(generator.DenseKernel("sme_base_test21", 192, 192, 192, 192, 192, 192, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test22", 192, 192, 192, 192, 192, 192, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test23", 192, 192, 192, 192, 192, 192, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test24", 192, 192, 192, 192, 192, 192, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test25", 192, 192, 192, 192, 192, 192, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))

# kernels.append(generator.SparseKernel("sme_sparse_test1", 8, 8, 8, 8, 0, 8, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 1), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test2", 8, 8, 8, 8, 0, 8, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 3), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test3", 8, 8, 8, 8, 0, 8, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 8), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test4", 8, 8, 8, 8, 0, 8, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 8), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test5", 8, 8, 8, 8, 0, 8, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 37), delta_dp))

# kernels.append(generator.SparseKernel("sme_sparse_test6", 16, 16, 16, 16, 0, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test6_5", 16, 16, 8, 16, 0, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(8, 16, 12), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test7", 16, 16, 16, 16, 0, 16, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test8", 16, 16, 16, 16, 0, 16, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test9", 16, 16, 16, 16, 0, 16, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test10", 16, 16, 16, 16, 0, 16, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))

# kernels.append(generator.SparseKernel("sme_sparse_test11", 64, 64, 64, 64, 0, 64, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test12", 64, 64, 64, 64, 0, 64, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test13", 64, 64, 64, 64, 0, 64, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test14", 64, 64, 64, 64, 0, 64, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test15", 64, 64, 64, 64, 0, 64, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))

# kernels.append(generator.SparseKernel("sme_sparse_test16", 64, 64, 16, 64, 0, 64, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test17", 64, 64, 16, 64, 0, 64, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test18", 64, 64, 16, 64, 0, 64, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test19", 64, 64, 16, 64, 0, 64, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test20", 64, 64, 16, 64, 0, 64, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))

# kernels.append(generator.SparseKernel("sme_sparse_test21_5", 128, 128, 128, 128, 0, 128, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(128, 128, 1680), delta_dp))

# kernels.append(generator.SparseKernel("sme_sparse_test21", 192, 192, 192, 192, 0, 192, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 1), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test22", 192, 192, 192, 192, 0, 192, 1.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test23", 192, 192, 192, 192, 0, 192, 2.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test24", 192, 192, 192, 192, 0, 192, 2.0, 1.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test25", 192, 192, 192, 192, 0, 192, 2.0, 2.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))

# kernels.append(generator.DenseKernel("sme_comp_test1", 8, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test2", 16, 8, 8, 16, 16, 8, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test3", 8, 16, 16, 8, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test4", 8, 16, 16, 8, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test5", 8, 16, 16, 8, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test16_5", 16, 16, 8, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(8, 8, v_size, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernelS("sme_single_base_test1", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(16, 16, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))
# kernels.append(generator.DenseKernelS("sme_single_base_test2", 16, 16, 16, 16, 16, 16, 1.0, 2.0, [x.getBlocksize(16, 16, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))
# kernels.append(generator.DenseKernelS("sme_single_base_test3", 16, 16, 16, 16, 16, 16, 2.0, 0.0, [x.getBlocksize(16, 16, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))
# kernels.append(generator.DenseKernelS("sme_single_base_test4", 16, 16, 16, 16, 16, 16, 2.0, 1.0, [x.getBlocksize(16, 16, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))
# kernels.append(generator.DenseKernelS("sme_single_base_test5", 16, 16, 16, 16, 16, 16, 2.0, 2.0, [x.getBlocksize(16, 16, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))

kernels.append(generator.DenseKernelS("sme_single_base_test6", 32, 32, 32, 32, 32, 32, 1.0, 0.0, [x.getBlocksize(32, 32, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))

kernels.append(generator.DenseKernelS("sme_single_base_test11", 64, 64, 64, 64, 64, 64, 1.0, 0.0, [x.getBlocksize(64, 64, v_size_s, v_size_s) for x in blocksize_algs], delta_sp))



print(kernels)

#kernels.append(generator.DenseKernel("sme_base_test2", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(16, 16, 1, v_size) for x in blocksize_algs], delta_dp))
#kernels.append(generator.DenseKernel("sme_base_test3", 8, 8, 8, 8, 8, 8, 2.1, 2.1, [x.getBlocksize(8, 8, 1, v_size) for x in blocksize_algs], delta_dp))


"""
# test cases for double precision multiplication
kernels.append(generator.DenseKernel("sve_mixed_test1", 9, 9, 9, 9, 9, 9, 1.0, 0.0, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test2", 9, 9, 9, 9, 0, 9, 4.0, 2.5, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size) for x in blocksize_algs], generator.generateMTX(9, 9, 20), delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test3", 18, 18, 18, 18, 0, 18, 3.4, -2.5, [(1, 1), (3, 3), (6, 6)] + [x.getBlocksize(18, 18, 1, v_size) for x in blocksize_algs], generator.generateMTX(18, 18, 59), delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test4", 80, 80, 80, 80, 0, 80, 0.0, -2.5, [(4, 4), (8, 8)] + [x.getBlocksize(80, 80, 1, v_size) for x in blocksize_algs], generator.generateMTX(80, 80, 312), delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test5", 8, 8, 8, 10, 0, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 6), delta_dp))
kernels.append(generator.DenseKernel("sve_mixed_test6", 8, 8, 8, 10, 8, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sve_test3", 4, 4, 4, 4, 4, 4, 2.0, 2.0, [(4, 4)], delta_dp))

kernels.append(generator.SparseKernel("sve_test1", 8, 56, 56, 8, 0, 8, 1.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 1, v_size) for x in blocksize_algs], generator.generateMTX(56, 56, 30), delta_dp))
kernels.append(generator.DenseKernel("sve_test2", 8, 40, 40, 8, 40, 8, 3.0, 2.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_test3", 8, 56, 56, 8, 56, 8, 0.0, 0.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 1, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.SparseKernel("sve_arm_only_test1", 2, 3, 4, 2, 0, 2, 1.1233, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], generator.generateMTX(4, 3, 5), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test2", 2, 3, 4, 20, 0, 14, 1.0, 1.0, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], generator.generateMTX(4, 3, 5), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test3", 32, 80, 50, 32, 0, 32, 1.0, 3.0, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size) for x in blocksize_algs], generator.generateMTX(50, 80, 294), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test4", 32, 32, 32, 34, 0, 32, 1.0, 0.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size) for x in blocksize_algs], generator.generateMTX(32, 32, 24), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test5", 2, 1, 1, 2, 0, 8, 1.0, -1.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size) for x in blocksize_algs], generator.generateMTX(1, 1, 1), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test6", 2, 2, 2, 2, 0, 2, 2.0, 234234.123, [(2, 1)] + [x.getBlocksize(2, 2, 1, v_size) for x in blocksize_algs], generator.generateMTX(2, 2, 1), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test7", 16, 5, 7, 16, 0, 16, 0.0, -1.123, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size) for x in blocksize_algs], generator.generateMTX(7, 5, 35), delta_dp))

kernels.append(generator.DenseKernel("sve_arm_only_test8", 2, 3, 4, 2, 4, 2, 1.0, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test9", 2, 3, 4, 20, 12, 14, 2.0, 1.123, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test10", 32, 80, 50, 32, 50, 32, 0.0, 0.2, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test11", 32, 32, 32, 33, 68, 32, 1231.0, 14443.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test12", 2, 1, 1, 2, 1, 8, 1.0, 3.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test13", 2, 3, 3, 2, 3, 2, 1.0, 0.0, [(2, 1)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test14", 16, 5, 7, 16, 7, 16, 1.0, 1.0, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sve_arm_only_test15", 23, 29, 31, 23, 31, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test16", 23, 29, 31, 23, 0, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size) for x in blocksize_algs], generator.generateMTX(31, 29, 61), delta_dp))

# test cases for single precision multiplication
kernels.append(generator.DenseKernelS("sve_single_prec_test_S1", 9, 9, 9, 9, 9, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sve_single_prec_test_S2", 15, 15, 15, 15, 15, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sve_single_prec_test_S3", 23, 23, 23, 23, 23, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sve_single_prec_test_S4", 23, 31, 13, 23, 13, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.SparseKernelS("sve_single_prec_test_S5", 9, 9, 9, 9, 0, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(9, 9, 8), delta_sp))
kernels.append(generator.SparseKernelS("sve_single_prec_test_S6", 15, 15, 15, 15, 0, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(15, 15, 22), delta_sp))
kernels.append(generator.SparseKernelS("sve_single_prec_test_S7", 23, 23, 23, 23, 0, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(23, 23, 52), delta_sp))
kernels.append(generator.SparseKernelS("sve_single_prec_test_S8", 23, 31, 13, 23, 0, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(13, 31, 40), delta_sp))
"""

generator.make(kernels, f"arm_sme{bitlen}")
