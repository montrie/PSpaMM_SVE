#g++-14 -I ~/apps/libxsmm/include gemm_measurements.cpp -L ~/apps/libxsmm/lib -lxsmm -lblas -o gemm_measurements
g++-14 -I ~/apps/m4_libxsmm/include gemm_measurements.cpp -L ~/apps/m4_libxsmm/lib -lxsmm -o gemm_measurements
LD_LIBRARY_PATH=~/apps/m4_libxsmm/lib LIBXSMM_VERBOSE=2 ./gemm_measurements
