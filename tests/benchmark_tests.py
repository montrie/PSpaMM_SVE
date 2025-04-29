import benchmark_generator as generator
import testsuite_generator as neon_generator

import pspamm.scripts.max_arm as max_neon
import pspamm.scripts.max_arm_sme as max_sme

import sys

v_len = 4

if len(sys.argv) == 2:
    v_len = int(sys.argv[1]) // 128

is_K_fixed = False 
fixed_K = 80

sme_v_size = v_len * 2
sme_v_size_s = v_len * 4
neon_v_size = 2
neon_v_size_s = 4

kernels = []

interval_size = 20  #30
sme_step = 8
sme_start = 8
sme_end = interval_size * sme_start

neon_step = 2
neon_start = 2
neon_end = interval_size * sme_start # ensures that sme and neon benchmarks end at same m=n dimension

alpha = 1.0
beta = 0.0

delta_dp = 1e-7
delta_sp = 1e-4


for i in range(sme_start, sme_end, sme_step):
    #generate dense kernels
    k = fixed_K if is_K_fixed else i
    bm, bn, bk = max_sme.getBlocksize(i, i, k, sme_v_size) # TODO: should be i, i instead of 8, 8
    kernels.append(generator.DenseKernel(f"sme_benchmark_fixed_k_{i}_double", i, i, k, i, k, i, alpha, beta, [(bm, bn, bk)], delta_dp))
"""
for i in range(sme_start*2, sme_end*2, sme_step*2):
#for i in range(368, 389, 1000):
    k = fixed_K if is_K_fixed else i
    bm, bn, bk = max_sme.getBlocksize(i, i, k, sme_v_size_s) # TODO: should be i, i instead of 8, 8
    kernels.append(generator.DenseKernelS(f"sme_benchmark_fixed_k_{i}_single", i, i, k, i, k, i, alpha, beta, [(bm, bn, bk)], delta_sp))
#    if i >= 16:
#        break

for i in range(sme_start, sme_end, sme_step):
    #generate dense kernels
    k = fixed_K if is_K_fixed else i
    bm, bn, bk = max_sme.getBlocksize(i, i, k, sme_v_size) # TODO: should be i, i instead of 8, 8
    kernels.append(generator.SparseKernel(f"sme_benchmark_fixed_k_{i}_double_sparse", i, i, k, i, 0, i, alpha, beta, [(bm, bn, bk)], generator.generateMTX(k, i, max(1, int(0.05 * k * i))), delta_dp))

for i in range(sme_start*2, sme_end*2, sme_step*2):
#for i in range(240, sme_end*2, sme_step*2):
    k = fixed_K if is_K_fixed else i
    bm, bn, bk = max_sme.getBlocksize(i, i, k, sme_v_size_s) # TODO: should be i, i instead of 8, 8
    kernels.append(generator.SparseKernelS(f"sme_benchmark_fixed_k_{i}_single_sparse", i, i, k, i, 0, i, alpha, beta, [(bm, bn, bk)], generator.generateMTX(k, i, max(1, int(0.05 * k * i))), delta_sp))

"""
generator.make(kernels, "arm_sme512")


kernels = []
"""
for i in range(neon_start, neon_end, neon_step):
    bk = 1
    k = fixed_K if is_K_fixed else i
    bm, bn = max_neon.getBlocksize(i, i, bk, neon_v_size)
    kernels.append(generator.DenseKernel(f"neon_benchmark_fixed_k_{i}_double", i, i, k, i, k, i, alpha, beta, [(bm, bn)], delta_dp))
"""
for i in range(neon_start*2, 180, neon_step*2):
    bk = 1
    k = fixed_K if is_K_fixed else i
    bm, bn = max_neon.getBlocksize(i, i, bk, neon_v_size_s)
    kernels.append(generator.DenseKernelS(f"neon_benchmark_fixed_k_{i}_single", i, i, k, i, k, i, alpha, beta, [(bm, bn)], delta_sp))

"""
for i in range(neon_start, neon_end, neon_step):
    bk = 1
    k = fixed_K if is_K_fixed else i
    bm, bn = max_neon.getBlocksize(i, i, bk, neon_v_size)
    kernels.append(generator.SparseKernel(f"neon_benchmark_fixed_k_{i}_double_sparse", i, i, k, i, 0, i, alpha, beta, [(bm, bn)], generator.generateMTX(k, i, max(1, int(0.05 * k * i))), delta_dp))

for i in range(neon_start*2, 180, neon_step*2):
    bk = 1
    k = fixed_K if is_K_fixed else i
    bm, bn = max_neon.getBlocksize(i, i, bk, neon_v_size_s)
    kernels.append(generator.SparseKernelS(f"neon_benchmark_fixed_k_{i}_single_sparse", i, i, k, i, 0, i, alpha, beta, [(bm, bn)], generator.generateMTX(k, i, max(1, int(0.05 * k * i))), delta_sp))
"""
generator.make(kernels, "arm")

