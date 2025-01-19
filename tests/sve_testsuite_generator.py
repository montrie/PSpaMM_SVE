#!/usr/bin/python3
from collections import namedtuple
import subprocess
import numpy as np
import random
import sys
import os
import testsuite_generator as test_generator

BASEDIR = 'build'

SparseKernel = namedtuple('SparseKernel', 'name m n k lda ldb ldc alpha beta block_sizes mtx delta')
DenseKernel = namedtuple('DenseKernel', 'name m n k lda ldb ldc alpha beta block_sizes delta')

SparseKernelS = namedtuple('SparseKernelS', 'name m n k lda ldb ldc alpha beta block_sizes mtx delta')
DenseKernelS = namedtuple('DenseKernelS', 'name m n k lda ldb ldc alpha beta block_sizes delta')

setup_prefetching = """
template <typename T>
void setup_prefetch(T*& prefetch, T* matrix, unsigned n, unsigned ldc) {
 posix_memalign(reinterpret_cast<void **>(&prefetch), 64, ldc*n*sizeof(T));
 std::memcpy(prefetch, matrix, ldc*n*sizeof(T));
}
"""

def generateMTX(k, n, nnz):
    return test_generator.generateMTX(k, n, nnz)

def make(kernels, arch):
    os.makedirs(os.path.join(BASEDIR, arch), exist_ok=True)

    f = open(os.path.join(BASEDIR, f'{arch}_testsuite.cpp'), 'w')

    f.write(test_generator.head_of_testsuite)

    include_single_prec = False

    for kern in kernels:
        arguments = ['pspamm-generator', str(kern.m), str(kern.n), str(kern.k), str(kern.lda),
                     str(kern.ldb), str(kern.ldc), str(kern.alpha), str(kern.beta)]

        if isinstance(kern, SparseKernel) or isinstance(kern, SparseKernelS):
            arguments += ['--mtx_filename', kern.mtx]

        prec = 's' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else 'd'
        arguments += ['--precision', prec]
        if prec == 's':
            include_single_prec = True

        block_sizes = list(set(kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]

            if arch == "knl":
                assert (bm % 8 == 0 and (bn + 1) * (bm / 8) <= 32)
            elif arch == "arm":
                assert (bm % 2 == 0 and (bn + 1) * (bm / 2) + bn <= 32)
            elif arch.startswith("arm_"):
                veclen = int(arch[7:])
                assert veclen % 128 == 0 and veclen <= 2048
                reglen = veclen // 128
                v_len = 2 * reglen if prec == 'd' else 4 * reglen
                # this should be the same assertion as in ../scripts/max_arm_sve.py
                bk = v_len# 1
                # ceiling division
                vm = -(bm // -v_len)  
                vk = -(bk // -v_len) # should come out to 1 for bk = 1, meaning no changes for sve
                if not ((bn + vk) * vm + bn * vk <= 32):
                    print(f'Skipping block size {bm}x{bn} for {arch}')
                    continue

            name = kern.name + '_' + str(bm) + '_' + str(bn)

            additional_args = ['--output_funcname', name, '--output_filename', os.path.join(BASEDIR, arch, name + '.h'),
                               '--output_overwrite']
            additional_args += ['--bm', str(bm), '--bn', str(bn), '--bk', str(bk), '--arch', arch, '--prefetching', 'BL2viaC']
            """
            subprocess.STDOUT for stderr
            """
            try:
                subprocess.check_output(arguments + additional_args, stderr=sys.stderr, env={**os.environ, "delDEBUG_SUBPROCESS": "1"})
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            f.write('#include "' + arch + '/' + kern.name + '_' + str(bm) + '_' + str(bn) + '.h"\n')

    f.write('\n')
    # necessary functions are defined in testsuite_generator.py
    f.write(test_generator.function_definitions)
    f.write(setup_prefetching)
    f.write(test_generator.setup_main)
    # add variable declarations for single precision test cases
    f.write("""  std::tuple<float*, float*, float*, float*, float*> fpointers;
  float falpha; float fbeta;
  double* prefetch;
  float* fprefetch;
  double* Atrans;
  float* fAtrans;
  """)
  
    for kern in kernels:

        block_sizes = list(set(kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]

            prec = 's' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else 'd'

            if arch.startswith("arm_"):
                veclen = int(arch[7:])
                assert veclen % 128 == 0 and veclen <= 2048
                reglen = veclen // 128
                v_len = 2 * reglen if prec == 'd' else 4 * reglen
                # this should be the same assertion as in ../scripts/max_arm_sve.py
                bk = v_len #1
                # ceiling division
                vm = -( bm // -v_len)
                vk = -( bk // -v_len) # should come out to 1 for bk = 1, meaning no changes for sve
                if not ((bn + vk) * vm + bn * vk <= 32):
                    # print(f'Skipping block size {bm}x{bn} for {arch}')
                    continue

            name = kern.name + '_' + str(bm) + '_' + str(bn)

            if isinstance(kern, SparseKernel) or isinstance(kern, SparseKernelS):
                mtx = kern.mtx
            else:
                mtx = ""
            # for double precision: set prec to '' to conform to test_generator.function_definitions
            prec = 'f' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else ''

            f.write("""
  {p}alpha = {alpha}; {p}beta = {beta}; ldb = {ldb};
  {p}pointers = pre<{T}>({m}, {n}, {k}, {lda}, ldb, {ldc}, "{mtx}");
  posix_memalign(reinterpret_cast<void **>(&Atrans), 64, {lda}*{ldb}*sizeof({T}));
  transpose_matrix(std::get<0>({p}pointers), {p}Atrans, {lda}, {ldb});
  printf("\\n");
  pretty_print({m}, {k}, {lda}, std::get<0>{p}(pointers));
  printf("\\n");
  pretty_print({k},{m}, {lda}, Atrans);
  printf("\\n");
  setup_prefetch({p}prefetch, std::get<3>({p}pointers), {n}, {ldc});
  {name}({A}, std::get<{sparse}>({p}pointers), std::get<3>({p}pointers), {p}alpha, {p}beta, {p}prefetch);
  result = post<{T}>({m}, {n}, {k}, {lda}, &ldb, {ldc}, &{p}alpha, &{p}beta, std::get<0>({p}pointers), std::get<1>({p}pointers), std::get<3>({p}pointers), std::get<4>({p}pointers), {delta:.7f});
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>({p}pointers)); free(std::get<1>({p}pointers)); free(std::get<2>({p}pointers)); free(std::get<3>({p}pointers)); free(std::get<4>({p}pointers)); free({p}prefetch); free({p}Atrans);
""".format(m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldc=kern.ldc, alpha=kern.alpha, beta=kern.beta,
           mtx=mtx, delta=kern.delta, name=name, sparse=2 if kern.ldb == 0 else 1, A="Atrans" if arch.startswith("arm_sme") else "std::get<0>({p}pointers)".format(p=prec), 
           p=prec, T="float" if prec == 'f' else "double"))

    f.write(test_generator.end_of_testsuite)
