#!/home/hpc/pr63so/ga96voz2/bin/python3.6

import testsuite_generator as generator

import blocksize_scripts.max_knl as max_square
import blocksize_scripts.max_bn_knl as max_bn
import blocksize_scripts.old_knl as old

blocksize_algs = [max_square, max_bn, old]

kernels = []

kernels.append(generator.SparseKernel("test1", 8, 56, 56, 8, 0, 8, 0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56) for x in blocksize_algs], generator.generateMTX(56, 56, 30), 0.0000001))
kernels.append(generator.DenseKernel("test2", 8, 40, 40, 8, 40, 8, 1, [(8, 5), (8,2)] + [x.getBlocksize(8, 40) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("test3", 8, 56, 56, 8, 56, 8, 0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56) for x in blocksize_algs], 0.0000001))

kernels.append(generator.SparseKernel("knl_only_test1", 8, 2, 1, 8, 0, 8, 0, [(8,1)] + [x.getBlocksize(8, 2) for x in blocksize_algs], generator.generateMTX(1, 2, 1), 0.0000001))
kernels.append(generator.SparseKernel("knl_only_test2", 24, 40, 40, 32, 0, 24, 1, [(8, 2), (16,7)] + [x.getBlocksize(24, 40) for x in blocksize_algs], generator.generateMTX(40, 40, 20), 0.0000001))
kernels.append(generator.SparseKernel("knl_only_test3", 8, 2, 1, 8, 0, 16, 0, [(8, 1)] + [x.getBlocksize(8, 2) for x in blocksize_algs], generator.generateMTX(1, 2, 2), 0.0000001))
kernels.append(generator.SparseKernel("knl_only_test4", 8, 20, 10, 40, 0, 8, 0, [(8, 20), (8,3)] + [x.getBlocksize(8, 20) for x in blocksize_algs], generator.generateMTX(10, 20, 1), 0.0000001))
kernels.append(generator.SparseKernel("knl_only_test5", 64, 5, 10, 64, 0, 64, 1, [(32, 2), (8,14)] + [x.getBlocksize(64, 5) for x in blocksize_algs], generator.generateMTX(10, 5, 1), 0.0000001))
kernels.append(generator.SparseKernel("knl_only_test6", 8, 1, 1, 16, 0, 56, 0, [(8, 1)] + [x.getBlocksize(8, 1) for x in blocksize_algs], generator.generateMTX(1, 1, 1), 0.0000001))
kernels.append(generator.SparseKernel("knl_only_test7", 8, 24, 40, 8, 0, 8, 1, [(8, 24), (8,1)] + [x.getBlocksize(8, 24) for x in blocksize_algs], generator.generateMTX(40, 24, 1), 0.0000001))

kernels.append(generator.DenseKernel("knl_only_test8", 8, 2, 1, 8, 1, 8, 0, [(8,1)] + [x.getBlocksize(8, 2) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("knl_only_test9", 24, 40, 40, 32, 60, 32, 1, [(8, 2), (16,7)] + [x.getBlocksize(24, 40) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("knl_only_test10", 56, 56, 56, 64, 59, 64, 0, [(8, 28), (8,27), (40, 5)] + [x.getBlocksize(56, 56) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("knl_only_test11", 8, 20, 10, 40, 10, 8, 0, [(8, 20), (8,3)] + [x.getBlocksize(8, 20) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("knl_only_test12", 64, 5, 10, 64, 12, 64, 1, [(32, 2), (8,14)] + [x.getBlocksize(64, 5) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("knl_only_test13", 8, 1, 1, 16, 1, 56, 0, [(8, 1)] + [x.getBlocksize(8, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("knl_only_test14", 8, 24, 40, 8, 41, 8, 1, [(8, 24), (8,1)] + [x.getBlocksize(8, 24) for x in blocksize_algs], 0.0000001))

kernels.append(generator.SparseKernel("SeisSolBugTest", 8, 20, 34, 8, 0, 8, 0, [(8,1)] + [x.getBlocksize(8, 20) for x in blocksize_algs], "mtx/bugmatrix.mtx", 0.0000001))

generator.make(kernels, "knl")


