#!/home/hpc/pr63so/ga96voz2/bin/python3.6

from collections import namedtuple
import subprocess
import numpy as np
import random
import sys

def generateMTX(k, n, nnz):

	assert(nnz <= k * n)

	filename = str(k) + 'x' + str(n) + '_' + str(nnz) + '.mtx'

	f = open('mtx/' + filename, 'w')

	f.write('%%MatrixMarket matrix coordinate real general\n%\n' + str(k) + ' ' + str(n) + ' ' + str(nnz))

	zeros = set()

	for i in range(1,k+1):
		for j in range(1,n+1):
			zeros.add((i,j))

	nonzeros = random.sample(zeros, nnz)

	for entry in nonzeros:
		f.write('\n' + str(entry[0]) + ' ' + str(entry[1]) + ' ' + str(random.uniform(0.00001, 1000)))

	f.close()

	return filename

if len(sys.argv) > 1:
	arch = sys.argv[1]
else:
	arch = 'knl'

DenseKernel = namedtuple('DenseKernel', 'name m n k lda ldb ldc beta block_sizes mtx delta')
SparseKernel = namedtuple('SparseKernel', 'name m n k lda ldb ldc beta block_sizes mtx delta')

kernels = []

kernels.append(SparseKernel("test1", 8, 56, 56, 8, 0, 8, 0, [(8, 4), (8,1)], generateMTX(56, 56, 30), 0.0000001))
kernels.append(DenseKernel("test2", 8, 40, 40, 8, 40, 8, 1, [(8, 5), (8,2)], generateMTX(40, 40, 40*40), 0.0000001))
kernels.append(DenseKernel("test3", 8, 56, 56, 8, 56, 8, 0, [(8, 3), (8, 5)], generateMTX(56, 56, 56*56), 0.0000001))

if arch == "knl":

	knl_kernels = []

	knl_kernels.append(SparseKernel("knl_only_test1", 8, 2, 1, 8, 0, 8, 0, [(8,1)], generateMTX(1, 2, 1), 0.0000001))
	knl_kernels.append(SparseKernel("knl_only_test2", 24, 40, 40, 32, 0, 24, 1, [(8, 2), (16,7)], generateMTX(40, 40, 20), 0.0000001))
	knl_kernels.append(SparseKernel("knl_only_test3", 8, 2, 1, 8, 0, 16, 0, [(8, 1)], generateMTX(1, 2, 2), 0.0000001))
	knl_kernels.append(SparseKernel("knl_only_test4", 8, 20, 10, 40, 0, 8, 0, [(8, 20), (8,3)], generateMTX(10, 20, 1), 0.0000001))
	knl_kernels.append(SparseKernel("knl_only_test5", 64, 5, 10, 64, 0, 64, 1, [(32, 2), (8,14)], generateMTX(10, 5, 1), 0.0000001))
	knl_kernels.append(SparseKernel("knl_only_test6", 8, 1, 1, 16, 0, 56, 0, [(8, 1)], generateMTX(1, 1, 1), 0.0000001))
	knl_kernels.append(SparseKernel("knl_only_test7", 8, 24, 40, 8, 0, 8, 1, [(8, 24), (8,1)], generateMTX(40, 24, 1), 0.0000001))

	knl_kernels.append(DenseKernel("knl_only_test8", 8, 2, 1, 8, 1, 8, 0, [(8,1)], generateMTX(1, 2, 1*2), 0.0000001))
	knl_kernels.append(DenseKernel("knl_only_test9", 24, 40, 40, 32, 60, 32, 1, [(8, 2), (16,7)], generateMTX(40, 40, 40*40), 0.0000001))
	knl_kernels.append(DenseKernel("knl_only_test10", 56, 56, 56, 64, 59, 64, 0, [(8, 28), (8,27), (40, 5)], generateMTX(56, 56, 56*56), 0.0000001))
	knl_kernels.append(DenseKernel("knl_only_test11", 8, 20, 10, 40, 10, 8, 0, [(8, 20), (8,3)], generateMTX(10, 20, 10*20), 0.0000001))
	knl_kernels.append(DenseKernel("knl_only_test12", 64, 5, 10, 64, 12, 64, 1, [(32, 2), (8,14)], generateMTX(10, 5, 10*5), 0.0000001))
	knl_kernels.append(DenseKernel("knl_only_test13", 8, 1, 1, 16, 1, 56, 0, [(8, 1)], generateMTX(1, 1, 1), 0.0000001))
	knl_kernels.append(DenseKernel("knl_only_test14", 8, 24, 40, 8, 41, 8, 1, [(8, 24), (8,1)], generateMTX(40, 24, 40*24), 0.0000001))

	kernels += knl_kernels

elif arch == "arm":

	arm_kernels = []

	arm_kernels.append(SparseKernel("arm_only_test1", 2, 3, 4, 2, 0, 2, 0, [(2, 1), (2,3)], generateMTX(4, 3, 5), 0.0000001))
	arm_kernels.append(SparseKernel("arm_only_test2", 2, 3, 4, 20, 0, 14, 1, [(2, 2), (2,3)], generateMTX(4, 3, 5), 0.0000001))
	arm_kernels.append(SparseKernel("arm_only_test3", 32, 80, 50, 32, 0, 32, 0, [(8, 5), (6,7)], generateMTX(50, 80, 294), 0.0000001))
	arm_kernels.append(SparseKernel("arm_only_test4", 32, 32, 32, 33, 0, 32, 0, [(4, 4), (4,3)], generateMTX(32, 32, 24), 0.0000001))
	arm_kernels.append(SparseKernel("arm_only_test5", 2, 1, 1, 2, 0, 8, 1, [(2, 1)], generateMTX(1, 1, 1), 0.0000001))
	arm_kernels.append(SparseKernel("arm_only_test6", 2, 2, 2, 2, 0, 2, 0, [(2, 1)], generateMTX(2, 2, 1), 0.0000001))
	arm_kernels.append(SparseKernel("arm_only_test7", 16, 5, 7, 16, 0, 16, 1, [(8, 1), (8,2)], generateMTX(7, 5, 35), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test8", 2, 3, 4, 2, 4, 2, 0, [(2, 1), (2,3)], generateMTX(4, 3, 4*3), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test9", 2, 3, 4, 20, 12, 14, 1, [(2, 2), (2,3)], generateMTX(4, 3, 4*3), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test10", 32, 80, 50, 32, 50, 32, 0, [(8, 5), (6,7)], generateMTX(50, 80, 50*80), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test11", 32, 32, 32, 33, 68, 32, 1, [(4, 4), (4,3)], generateMTX(32, 32, 32*32), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test12", 2, 1, 1, 2, 1, 8, 0, [(2, 1)], generateMTX(1, 1, 1), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test13", 2, 3, 3, 2, 3, 2, 0, [(2, 1)], generateMTX(3, 3, 3*3), 0.0000001))
	arm_kernels.append(DenseKernel("arm_only_test14", 16, 5, 7, 16, 7, 16, 1, [(8, 1), (8,2)], generateMTX(7, 5, 7*5), 0.0000001))

	kernels += arm_kernels


f = open('testsuite.cpp', 'w')
	
f.write("""#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <tuple>

""")

for kern in kernels:

	arguments = ['./../sparsemmgen.py', str(kern.m), str(kern.n), str(kern.k), str(kern.lda), str(kern.ldb), str(kern.ldc), str(kern.beta)]

	if isinstance(kern, SparseKernel):
		arguments += ['--mtx_filename', 'mtx/' + kern.mtx]
	for bs in kern.block_sizes:
		bm = bs[0]
		bn = bs[1]

		if arch == "knl":
			assert(bm % 8 == 0 and (bn+1) * (bm / 8) <= 32)
		elif arch == "arm":
			assert(bm % 2 == 0 and (bn+1) * (bm / 2) + bn <= 32)

		name = kern.name + '_' + str(bm) + '_' + str(bn)

		additional_args = ['--output_funcname', name, '--output_filename', arch + '/' + name + '.h']
		additional_args += ['--bm', str(bm), '--bn', str(bn), '--bk', '1', '--arch', arch]

		try:
			subprocess.check_output(arguments + additional_args,stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

		f.write('#include "' + arch + '/' + kern.name + '_' + str(bm) + '_' + str(bn) + '.h"\n')

f.write('\n')

f.write("""
void gemm_ref(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double BETA, double* A, double* B, double* C) {
  if (BETA == 0.0) {
    memset(C, 0, LDC * N * sizeof(double));
  }
  for (unsigned col = 0; col < N; ++col) {
    for (unsigned row = 0; row < M; ++row) {
      for (unsigned s = 0; s < K; ++s) {
        C[row + col * LDC] += A[row + s * LDA] * B[s + col * LDB];
      }
    }
  }
}

std::tuple<double*, double*, double*, double*, double*> pre(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, std::string MTX) {

  if(LDB == 0)
    LDB = K;

  double *A;
  double *B;
  double *Bsparse;
  double *Cref;
  double *C;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, LDA*LDB*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, LDB*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, LDB*N*sizeof(double));  
  int resCref = posix_memalign(reinterpret_cast<void **>(&Cref), 64, LDC*N*sizeof(double));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, LDC*N*sizeof(double));

  std::string line;
  std::ifstream f(MTX);
  getline(f, line);
  getline(f, line);
  getline(f, line);

  for(int i = 0; i < LDA*LDB; i++)
    A[i] = (double)rand() / RAND_MAX;

  for(int i = 0; i < LDB*N; i++)
    B[i] = 0;

  for(int i = 0; i < LDC*N; i++)
  {
    Cref[i] = (double)rand() / RAND_MAX;
    C[i] = Cref[i];
  }


  while(getline(f, line)) {
    std::vector<std::string> result;
    std::istringstream iss(line);
    for(std::string s; iss >> s; )
      result.push_back(s);

    B[std::atoi(result[0].c_str()) - 1 + LDB * (std::atoi(result[1].c_str()) - 1)] = std::stod(result[2]);
  }


  int counter = 0;

  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < K; j++)
    {
      if(B[j + i * LDB] != 0)
      {
        Bsparse[counter] = B[j + i * LDB];
        counter++;
      }
    }
  }

  f.close();

  return std::make_tuple(A, B, Bsparse, C, Cref);
}

int post(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double BETA, double* A, double* B, double* C, double* Cref, double DELTA) {

  if(LDB == 0)
    LDB = K;

  gemm_ref(M, N, K, LDA, LDB, LDC, BETA, A, B, Cref);
  
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      if(std::abs(C[i + j * LDC] - Cref[i + j * LDC]) > DELTA)
        return 0;

  return 1;
}

int main()
{
  std::vector<std::tuple<std::string, int>> results;
  std::tuple<double*, double*, double*, double*, double*> pointers;
  int result;

""")

for kern in kernels:
	for bs in kern.block_sizes:
		bm = bs[0]
		bn = bs[1]
		name = kern.name + '_' + str(bm) + '_' + str(bn)

		f.write("""
  pointers = pre({m}, {n}, {k}, {lda}, {ldb}, {ldc}, "mtx/{mtx}");
  {name}(std::get<0>(pointers), std::get<{sparse}>(pointers), std::get<3>(pointers));
  result = post({m}, {n}, {k}, {lda}, {ldb}, {ldc}, {beta}, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), {delta:.7f});
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));
""".format(m = kern.m, n = kern.n, k = kern.k, lda = kern.lda, ldb = kern.ldb, ldc = kern.ldc, beta = kern.beta, mtx = kern.mtx, delta = kern.delta, name = name, sparse = 2 if kern.ldb == 0 else 1))

f.write("""

  int correct = 0;
  for(int i = 0; i < results.size(); i++)
  {
    if(std::get<1>(results[i]))
      correct++;
    else
      printf("%s failed!\\n", (std::get<0>(results[i])).c_str());
  }

  printf("\\n%i out of %lu test successful!\\n", correct, results.size());

  return 0;
}""")