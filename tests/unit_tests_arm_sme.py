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

kernels.append(generator.DenseKernel("sme_base_test1", 8, 8, 8, 8, 8, 8, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test2", 8, 8, 8, 8, 8, 8, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test3", 8, 8, 8, 8, 8, 8, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test4", 8, 8, 8, 8, 8, 8, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test5", 8, 8, 8, 8, 8, 8, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sme_base_test6", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test6_5", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test7", 16, 16, 16, 16, 16, 16, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test7_5", 16, 16, 16, 16, 16, 16, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test8", 16, 16, 16, 16, 16, 16, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test8_5", 16, 16, 16, 16, 16, 16, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test9", 16, 16, 16, 16, 16, 16, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test9_5", 16, 16, 16, 16, 16, 16, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test10", 16, 16, 16, 16, 16, 16, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test10_5", 16, 16, 16, 16, 16, 16, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sme_base_test11", 64, 64, 64, 64, 64, 64, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test12", 64, 64, 64, 64, 64, 64, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test13", 64, 64, 64, 64, 64, 64, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test14", 64, 64, 64, 64, 64, 64, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test15", 64, 64, 64, 64, 64, 64, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sme_base_test16", 64, 64, 16, 64, 64, 64, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test16_5", 16, 16, 8, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test17", 64, 64, 16, 64, 64, 64, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test18", 64, 64, 16, 64, 64, 64, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test19", 64, 64, 16, 64, 64, 64, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test20", 64, 64, 16, 64, 64, 64, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sme_base_test21", 192, 192, 192, 192, 192, 192, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test22", 192, 192, 192, 192, 192, 192, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test23", 192, 192, 192, 192, 192, 192, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test24", 192, 192, 192, 192, 192, 192, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sme_base_test25", 192, 192, 192, 192, 192, 192, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.SparseKernel("sme_sparse_test1", 8, 8, 8, 8, 0, 8, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 1), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test2", 8, 8, 8, 8, 0, 8, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 3), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test3", 8, 8, 8, 8, 0, 8, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 8), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test4", 8, 8, 8, 8, 0, 8, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 8), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test5", 8, 8, 8, 8, 0, 8, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 37), delta_dp))

kernels.append(generator.SparseKernel("sme_sparse_test6", 16, 16, 16, 16, 0, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
# kernels.append(generator.SparseKernel("sme_sparse_test6_5", 16, 16, 8, 16, 0, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], generator.generateMTX(8, 16, 12), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test7", 16, 16, 16, 16, 0, 16, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test8", 16, 16, 16, 16, 0, 16, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test9", 16, 16, 16, 16, 0, 16, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test10", 16, 16, 16, 16, 0, 16, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_dp))

kernels.append(generator.SparseKernel("sme_sparse_test11", 64, 64, 64, 64, 0, 64, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test12", 64, 64, 64, 64, 0, 64, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test13", 64, 64, 64, 64, 0, 64, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test14", 64, 64, 64, 64, 0, 64, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test15", 64, 64, 64, 64, 0, 64, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 64, v_size) for x in blocksize_algs], generator.generateMTX(64, 64, 400), delta_dp))

kernels.append(generator.SparseKernel("sme_sparse_test16", 64, 64, 16, 64, 0, 64, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test17", 64, 64, 16, 64, 0, 64, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test18", 64, 64, 16, 64, 0, 64, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test19", 64, 64, 16, 64, 0, 64, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test20", 64, 64, 16, 64, 0, 64, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], generator.generateMTX(16, 64, 100), delta_dp))

kernels.append(generator.SparseKernel("sme_sparse_test21_5", 128, 128, 128, 128, 0, 128, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 128, v_size) for x in blocksize_algs], generator.generateMTX(128, 128, 1680), delta_dp))

kernels.append(generator.SparseKernel("sme_sparse_test21", 192, 192, 192, 192, 0, 192, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 1), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test22", 192, 192, 192, 192, 0, 192, 1.0, 2.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test23", 192, 192, 192, 192, 0, 192, 2.0, 0.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test24", 192, 192, 192, 192, 0, 192, 2.0, 1.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))
kernels.append(generator.SparseKernel("sme_sparse_test25", 192, 192, 192, 192, 0, 192, 2.0, 2.0, [x.getBlocksize(v_size, v_size, 192, v_size) for x in blocksize_algs], generator.generateMTX(192, 192, 3680), delta_dp))

# kernels.append(generator.DenseKernel("sme_comp_test1", 8, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test2", 16, 8, 8, 16, 16, 8, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test3", 8, 16, 16, 8, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test4", 8, 16, 16, 8, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_comp_test5", 8, 16, 16, 8, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 16, v_size) for x in blocksize_algs], delta_dp))
# kernels.append(generator.DenseKernel("sme_base_test16_5", 16, 16, 8, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size, v_size, 8, v_size) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernelS("sme_single_base_test1", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test2", 16, 16, 16, 16, 16, 16, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test3", 16, 16, 16, 16, 16, 16, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test4", 16, 16, 16, 16, 16, 16, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test5", 16, 16, 16, 16, 16, 16, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], delta_sp))

kernels.append(generator.DenseKernelS("sme_single_base_test6", 32, 32, 32, 32, 32, 32, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test7", 32, 32, 32, 32, 32, 32, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test8", 32, 32, 32, 32, 32, 32, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test9", 32, 32, 32, 32, 32, 32, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test10", 32, 32, 32, 32, 32, 32, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], delta_sp))

kernels.append(generator.DenseKernelS("sme_single_base_test11", 64, 64, 64, 64, 64, 64, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test12", 64, 64, 64, 64, 64, 64, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test13", 64, 64, 64, 64, 64, 64, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test14", 64, 64, 64, 64, 64, 64, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test15", 64, 64, 64, 64, 64, 64, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], delta_sp))

kernels.append(generator.DenseKernelS("sme_single_base_test16", 192, 192, 192, 192, 192, 192, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test17", 192, 192, 192, 192, 192, 192, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test18", 192, 192, 192, 192, 192, 192, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test19", 192, 192, 192, 192, 192, 192, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernelS("sme_single_base_test20", 192, 192, 192, 192, 192, 192, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], delta_sp))

kernels.append(generator.SparseKernelS("sme_sparse_single_base_test1", 16, 16, 16, 16, 0, 16, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test2", 16, 16, 16, 16, 0, 16, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test3", 16, 16, 16, 16, 0, 16, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test4", 16, 16, 16, 16, 0, 16, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test5", 16, 16, 16, 16, 0, 16, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 16, v_size_s) for x in blocksize_algs], generator.generateMTX(16, 16, 25), delta_sp))

kernels.append(generator.SparseKernelS("sme_sparse_single_base_test6", 32, 32, 32, 32, 0, 32, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], generator.generateMTX(32, 32, 102), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test7", 32, 32, 32, 32, 0, 32, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], generator.generateMTX(32, 32, 102), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test8", 32, 32, 32, 32, 0, 32, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], generator.generateMTX(32, 32, 102), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test9", 32, 32, 32, 32, 0, 32, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], generator.generateMTX(32, 32, 102), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test10", 32, 32, 32, 32, 0, 32, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 32, v_size_s) for x in blocksize_algs], generator.generateMTX(32, 32, 102), delta_sp))

kernels.append(generator.SparseKernelS("sme_sparse_single_base_test11", 64, 64, 64, 64, 0, 64, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], generator.generateMTX(64, 64, 409), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test12", 64, 64, 64, 64, 0, 64, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], generator.generateMTX(64, 64, 409), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test13", 64, 64, 64, 64, 0, 64, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], generator.generateMTX(64, 64, 409), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test14", 64, 64, 64, 64, 0, 64, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], generator.generateMTX(64, 64, 409), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test15", 64, 64, 64, 64, 0, 64, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 64, v_size_s) for x in blocksize_algs], generator.generateMTX(64, 64, 409), delta_sp))

kernels.append(generator.SparseKernelS("sme_sparse_single_base_test16", 192, 192, 192, 192, 0, 192, 1.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], generator.generateMTX(192, 192, 3686), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test17", 192, 192, 192, 192, 0, 192, 1.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], generator.generateMTX(192, 192, 3686), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test18", 192, 192, 192, 192, 0, 192, 2.0, 0.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], generator.generateMTX(192, 192, 3686), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test19", 192, 192, 192, 192, 0, 192, 2.0, 1.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], generator.generateMTX(192, 192, 3686), delta_sp))
kernels.append(generator.SparseKernelS("sme_sparse_single_base_test20", 192, 192, 192, 192, 0, 192, 2.0, 2.0, [x.getBlocksize(v_size_s, v_size_s, 192, v_size_s) for x in blocksize_algs], generator.generateMTX(192, 192, 3686), delta_sp))


#kernels.append(generator.DenseKernel("sme_base_test2", 16, 16, 16, 16, 16, 16, 1.0, 0.0, [x.getBlocksize(16, 16, 1, v_size) for x in blocksize_algs], delta_dp))
#kernels.append(generator.DenseKernel("sme_base_test3", 8, 8, 8, 8, 8, 8, 2.1, 2.1, [x.getBlocksize(8, 8, 1, v_size) for x in blocksize_algs], delta_dp))

generator.make(kernels, f"arm_sme{bitlen}")
