#include "libxsmm.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()
#include <dispatch/dispatch.h>
#include <sstream>
#include <fstream>
#include <sys/time.h>

void gemm_kernels() {
  std::srand(777); // set seed
  
  typedef float T;
  typedef libxsmm_mmfunction<T> kernel_type;

  // open file to store results
  std::ofstream measurements("libxsmm_performance_measurements.csv");
  // check if file was opened successfully
  if(!measurements.is_open()) {
    printf("Measurements csv file could not be opened\n");
  }
  measurements << "Name,Cores,Square,Dimensions,Time,GFLOP/s\n";


  struct GemmParams {
    libxsmm_xmmfunction kernel;
    libxsmm_gemm_param params;
    int repetitions; // Number of repetitions for the benchmark loop
  } params;
  int total_num_threads = 10;
  uint64_t flops_sanity_check;
  double gflops_sanity_check;
  struct timeval t_start;
  struct timeval t_end;
  long seconds;
  long useconds;
  double total_time;
  unsigned int repetitions = 10000;

  dispatch_qos_class_t qos_class = QOS_CLASS_UTILITY; // USER_INTERACTIVE for highest prio (p cores), UTILITY for lower prio (e cores)

  // create thread group similar to Hello, SME!
  dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT,
                                                                       qos_class,
                                                                       0);
  dispatch_queue_t queue = dispatch_queue_create("gemm_queue", attr);
  // TODO: blank group?
  dispatch_group_t group = dispatch_group_create();


  int ssvl = 512; // length of ZA register row in bits
  int step_size = ssvl / 8 / sizeof(T);  // comes out to 8/16 for double/single precision (bytes)
  printf("step size: %d\n", step_size);
  int interval_size = 25;
  int start = step_size;
  int end = 176;//start * interval_size;
//  printf("m block: %d\n", LIBXSMM_GEMM_M_BLOCK);
  libxsmm_datatype datatype = std::is_same<T, double>::value ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
  libxsmm_gemm_shape gemm_shape;
  libxsmm_gemm_batch_reduce_config config;
  config.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  config.br_stride_a_hint = 0;
  config.br_stride_b_hint = 0;
  config.br_unroll_hint = 0;
  libxsmm_bitfield flags_brgemm = LIBXSMM_GEMM_FLAGS('N','N');
  libxsmm_bitfield prefetch_flags = 0;
    
  libxsmm_gemm_ext_unary_argops argops;
  libxsmm_gemm_ext_binary_postops postops;
    
  memset(&argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  memset(&postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));

  argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_NONE;


  for(int i = start; i <= end; i += step_size) {
    libxsmm_blasint m = i;
    libxsmm_blasint n = i;
    libxsmm_blasint k = 80;
    libxsmm_blasint lda = i;
    libxsmm_blasint ldb = k;
    libxsmm_blasint ldc = i;

    argops.ldcp = ldc;
    
    gemm_shape = libxsmm_create_gemm_shape(m,
                                           n,
                                           k,
                                           lda,
                                           ldb,
                                           ldc,
                                           datatype,
                                           datatype,
                                           datatype,
                                           datatype);
    
    libxsmm_xmmfunction kernel;
    kernel.gemm_ext = libxsmm_dispatch_brgemm_ext(gemm_shape,
                                                  flags_brgemm,
                                                  prefetch_flags,
                                                  config,
                                                  argops,
                                                  postops);

    libxsmm_gemm_param lib_params;
    memset(&lib_params, 0, sizeof(libxsmm_gemm_param));
    T* a = (T*)malloc(lda*k*sizeof(T));
    T* b = (T*)malloc(n*ldb*sizeof(T));
    T* c = (T*)malloc(n*ldc*sizeof(T));
    T* c_ref = (T*)malloc(n*ldc*sizeof(T));

    for(int j = 0; j < lda*k; j++)
      a[j] = (T) drand48();
    for(int j = 0; j < n*ldb; j++)
      b[j] = (T) drand48();
    for(int j = 0; j < n*ldc; j++) {
      c[j] = (T) drand48();
      c_ref[j] = c[j];
    }

    // start time measurements

    // prepare apple threading and kernel
    
    lib_params.a.primary = a;
    lib_params.b.primary = b;
    lib_params.c.primary = c;

    // execute kernel
//    for(int num_threads = 1; num_threads <= total_num_threads; num_threads++) {
//      kernel.gemm(&lib_params);
//    }

    params.params = lib_params;
    params.kernel = kernel;
    params.repetitions = repetitions;

    for(int num_threads = 1; num_threads <= total_num_threads; num_threads+=9) {
      gettimeofday(&t_start, NULL);
      for(int t = 0; t < num_threads; t++) {
        dispatch_group_async_f(group, queue, &params, +[](void* context) {
          GemmParams* p = static_cast<GemmParams*>(context);
          for(int rep = 0; rep < p->repetitions; rep++) {
            p->kernel.gemm(&(p->params));
          }
        });
      }
      dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
      gettimeofday(&t_end, NULL);
      seconds = t_end.tv_sec - t_start.tv_sec;
      useconds = t_end.tv_usec - t_start.tv_usec;
      total_time = seconds + (1.0/1000000)*useconds;
      flops_sanity_check = 2 * m * n * k;
      printf("libxsmm_square_float_%d:\n", m);
      printf("2*%d*%d*%d = %d \n", m, n, k, 2*m*n*k);
      printf("flops_sanity_check = %d \n", flops_sanity_check);
      flops_sanity_check *= repetitions;
      flops_sanity_check *= num_threads;
      gflops_sanity_check = flops_sanity_check / 1.0e9;// / total_time;
      gflops_sanity_check /= total_time;
      printf("Duration: %f s\n", total_time);
      printf("GFLOPS  : %f  \n", gflops_sanity_check);
      measurements << "libxsmm_fixed_k_single_" << m << "," << num_threads << "," << (m  == k ? "true" : "false") << "," << m << "," << total_time << "," << gflops_sanity_check << "\n";
    }


    free(a); free(b); free(c);

/*    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, i, i, i, 1.0, 0.0);
    if(!kernel)
      printf("Failed to generate square GEMM kernel for size %d\n", i);  
    
    std::vector<double> A(i * i);
    std::vector<double> B(i * i);
    std::vector<double> C(i * i, 0.0);
    for(int j = 0; j < i * i; j++) {
      A[j] = static_cast<T>(std::rand()) / RAND_MAX;
      B[j] = static_cast<T>(std::rand()) / RAND_MAX;
    }

    kernel(A.data(), B.data(), C.data());

    free(A); free(B); free(C);*/
    //break;
  }
  printf("Finished LIBXSMM benchmark\n");
  measurements.close();
}

int main() {
  gemm_kernels();
  return 0;
}
