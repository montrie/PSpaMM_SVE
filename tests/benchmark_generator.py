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

setup_copy_array = """
template <typename T>
T* copy_array(T* arr, int length) {
  T* res;
  posix_memalign(reinterpret_cast<void **>(&res), 64, length*sizeof(T));
  /* the input arr and the created array res have the same length <length>; <length> is calculated by leading_dimension*n */
  for (int i = 0; i < length; ++i) {
    res[i] = arr[i];
  }
  
  return res;
}
"""

setup_get_nnz = """
template <typename T>
int get_nnz(T* B, int size) {
  int nnz = 0;
  for(int i = 0; i < size; i++) {
    if(B[i] != 0)
      nnz += 1;
    else
      return nnz;
  }
  // return in case the passed matrix is dense
  return nnz;
}
"""

setup_pmc_kpc = """  int ret = 0;

  // load dylib
  if (!lib_init()) {
      printf("Error: %s\\n", lib_err_msg);
      return 1;
  }

  // check permission
  int force_ctrs = 0;
  if (kpc_force_all_ctrs_get(&force_ctrs)) {
      printf("Permission denied, xnu/kpc requires root privileges.\\n");
      return 1;
  }

  // load pmc db
  kpep_db *db = NULL;
  if ((ret = kpep_db_create(NULL, &db))) {
      printf("Error: cannot load pmc database: %d.\\n", ret);
      return 1;
  }
  printf("loaded db: %s (%s)\\n", db->name, db->marketing_name);
  printf("number of fixed counters: %zu\\n", db->fixed_counter_count);
  printf("number of configurable counters: %zu\\n", db->config_counter_count);

  // create a config
  kpep_config *cfg = NULL;
  if ((ret = kpep_config_create(db, &cfg))) {
      printf("Failed to create kpep config: %d (%s).\\n",
             ret, kpep_config_error_desc(ret));
      return 1;
  }
  if ((ret = kpep_config_force_counters(cfg))) {
      printf("Failed to force counters: %d (%s).\\n",
             ret, kpep_config_error_desc(ret));
      return 1;
  }

  // get events
  const usize ev_count = sizeof(profile_events) / sizeof(profile_events[0]);
  kpep_event *ev_arr[ev_count] = { 0 };
  for (usize i = 0; i < ev_count; i++) {
    const event_alias *alias = profile_events + i;
      ev_arr[i] = get_event(db, alias);
      if (!ev_arr[i]) {
          printf("Cannot find event: %s.\\n", alias->alias);
          return 1;
      }
  }


  // add event to config
  for (usize i = 0; i < ev_count; i++) {
      kpep_event *ev = ev_arr[i];
      if ((ret = kpep_config_add_event(cfg, &ev, 0, NULL))) {
          printf("Failed to add event: %d (%s).\\n",
                 ret, kpep_config_error_desc(ret));
          return 1;
      }
  }

  // prepare buffer and config
  u32 classes = 0;
  usize reg_count = 0;
  kpc_config_t regs[KPC_MAX_COUNTERS] = { 0 };
  usize counter_map[KPC_MAX_COUNTERS] = { 0 };
  u64 counters_0[KPC_MAX_COUNTERS] = { 0 };
  u64 counters_1[KPC_MAX_COUNTERS] = { 0 };
  if ((ret = kpep_config_kpc_classes(cfg, &classes))) {
      printf("Failed get kpc classes: %d (%s).\\n",
             ret, kpep_config_error_desc(ret));
      return 1;
  }
  if ((ret = kpep_config_kpc_count(cfg, &reg_count))) {
      printf("Failed get kpc count: %d (%s).\\n",
             ret, kpep_config_error_desc(ret));
      return 1;
  }
  if ((ret = kpep_config_kpc_map(cfg, counter_map, sizeof(counter_map)))) {
      printf("Failed get kpc map: %d (%s).\\n",
             ret, kpep_config_error_desc(ret));
      return 1;
  }
  if ((ret = kpep_config_kpc(cfg, regs, sizeof(regs)))) {
      printf("Failed get kpc registers: %d (%s).\\n",
             ret, kpep_config_error_desc(ret));
      return 1;
  }


  // set config to kernel
  if ((ret = kpc_force_all_ctrs_set(1))) {
      printf("Failed force all ctrs: %d.\\n", ret);
      return 1;
  }
  if ((classes & KPC_CLASS_CONFIGURABLE_MASK) && reg_count) {
      if ((ret = kpc_set_config(classes, regs))) {
          printf("Failed set kpc config: %d.\\n", ret);
          return 1;
      }
  }
"""

def generateMTX(k, n, nnz):
    return test_generator.generateMTX(k, n, nnz)

def make(kernels, arch):
    os.makedirs(os.path.join(BASEDIR, arch), exist_ok=True)

    f = open(os.path.join(BASEDIR, f'{arch}_benchmark.cpp'), 'w')

    # include the performance measuring header from benchmarks/pmc_kpc.h
    f.write("""#include "../benchmarks/pmc_kpc.h"
#include <dispatch/dispatch.h>
#include <algorithm>
#include <sys/time.h>
#include <os/proc.h>
""")
    f.write(test_generator.head_of_testsuite)

    include_single_prec = False
    has_matrix_ins = "sme" in arch

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
            bk = 1

            if arch == "knl":
                assert (bm % 8 == 0 and (bn + 1) * (bm / 8) <= 32)
            elif arch == "arm":
                v_len = 2 if prec == 'd' else 4
                assert (bm % 2 == 0 and (bn + 1) * (bm / v_len) + bn <= 32)
            elif arch.startswith("arm_"):
                veclen = int(arch[7:])
                assert veclen % 128 == 0 and veclen <= 2048
                reglen = veclen // 128
                v_len = 2 * reglen if prec == 'd' else 4 * reglen
                # this should be the same assertion as in ../scripts/max_arm_sve.py
                bk = 1 if "sve" in arch else v_len if len(bs) == 2 else bs[2]
                # ceiling division
                vn = -(bn // -v_len) if "sme" in arch else bn
                vm = -(bm // -v_len)  
                # if not ((vn + bk) * vm + vn * bk <= 32):
                if not ((vm + vn) * bk <= 32):
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
                subprocess.check_output(arguments + additional_args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            f.write('#include "' + arch + '/' + kern.name + '_' + str(bm) + '_' + str(bn) + '.h"\n')

    f.write('\n')
    # necessary functions are defined in testsuite_generator.py
    f.write(test_generator.function_definitions)
    f.write(setup_prefetching)
    f.write(setup_copy_array)
    f.write(setup_get_nnz);
#TODO: create setup_main for the benchmarks as we need to adjust a few things
    f.write(test_generator.setup_main)
    # add variable declarations for single precision test cases
    f.write("""  std::tuple<float*, float*, float*, float*, float*> fpointers;
  float falpha; float fbeta;
  double* prefetch;
  float* fprefetch;
  double* Atrans;
  float* fAtrans;
  uint64_t flops_sanity_check;
  double gflops_sanity_check;
  struct timeval start;
  struct timeval end;
  long seconds;
  long useconds;
  double total_time;
  int nnz;
  int repetitions = 10000;
  int total_num_threads = 10;
  dispatch_qos_class_t qos_class = QOS_CLASS_USER_INTERACTIVE; // USER_INTERACTIVE for highest prio (p cores), UTILITY for lower prio (e cores)
  struct GemmParams {{
    double* A;                 // Transposed A matrix pointer
    double* B;                 // B matrix pointer (std::get<1>(pointers))
    double* C;                 // C matrix pointer (std::get<3>(pointers))
    double  alpha;             // Scaling factor for A*B
    double  beta;              // Scaling factor for C
    double* prefetch;          // Prefetch data pointer
    int     repetitions;       // Number of repetitions for the benchmark loop
    int     total_num_threads; //
  }} params;
  struct fGemmParams {{
    float* A;                 // Transposed A matrix pointer
    float* B;                 // B matrix pointer (std::get<1>(pointers))
    float* C;                 // C matrix pointer (std::get<3>(pointers))
    float  alpha;             // Scaling factor for A*B
    float  beta;              // Scaling factor for C
    float* prefetch;          // Prefetch data pointer
    int    repetitions;       // Number of repetitions for the benchmark loop
    int    total_num_threads; //
  }} fparams;
  params.repetitions        = repetitions;
  params.total_num_threads  = total_num_threads;
  fparams.repetitions       = repetitions;
  fparams.total_num_threads = total_num_threads;
//  GemmParams<double> params;

  // open file to store results
  std::ofstream measurements("{arch}_performance_measurements.csv");
  // check if file was opened successfully
  if(!measurements.is_open()) {{
    printf("Measurements csv file could not be opened\\n");
    return 1;
  }}

  measurements << "Name,Cores,Square,Dimensions,Time,GFLOP/s\\n";
    

  // create thread group similar to Hello, SME!
  dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT,
                                                                       qos_class,
                                                                       0);
  printf("min prio %d \\n", QOS_MIN_RELATIVE_PRIORITY);
  dispatch_queue_t queue = dispatch_queue_create("gemm_queue", attr);
  dispatch_group_t group = dispatch_group_create();

  usleep(100000);
""".format(arch=arch))

    
    transposed = str("sme" in arch).lower()
    f.write("  transposed = {trans};\n".format(trans=transposed))
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
                bk = 1 if "sve" in arch else v_len if len(bs) == 2 else bs[2]
                # ceiling division
                vn = -(bn // -v_len) if "sme" in arch else bn
                vm = -(bm // -v_len)
                # if not ((vn + bk) * vm + vn * bk <= 32):
                if not ((vm + vn) * bk <= 32):
                    # print(f'Skipping block size {bm}x{bn} for {arch}')
                    continue

            name = kern.name + '_' + str(bm) + '_' + str(bn)

            nnz = kern.k * kern.n
            if isinstance(kern, SparseKernel) or isinstance(kern, SparseKernelS):
                mtx = kern.mtx
                # this is necessary due to unexpected behavior when calculating the number of nnz within the total_num_threads loop
                # this ensures that nnz is as expected, because mtx files we generate are named "/path/to/mtx/<n>x<k>_<nnz>.mtx"
                nnz = int(mtx.split("/")[-1].split("_")[1][:-4])
            else:
                mtx = ""
            # for double precision: set prec to '' to conform to test_generator.function_definitions
            prec = 'f' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else ''
            sparse = isinstance(kern, SparseKernel) or isinstance(kern, SparseKernelS)

            setup_Atrans ="""posix_memalign(reinterpret_cast<void **>(&{p}Atrans), 64, {lda}*{ldbsparse}*sizeof({T}));
  transpose_matrix(std::get<0>({p}pointers), {p}Atrans, {lda}, {ldbsparse});
            """.format(m=kern.m, k=kern.k, lda=kern.lda, ldbsparse=kern.k if sparse else kern.ldb, p=prec, 
                       T="float" if prec == f else "double") if has_matrix_ins else ""
            free_Atrans = "free({p}Atrans);".format(p=prec) if has_matrix_ins else ""


            f.write("""
  {p}alpha = {alpha}; {p}beta = {beta}; ldb = {ldb};
  {p}pointers = pre<{T}>({m}, {n}, {k}, {lda}, ldb, {ldc}, "{mtx}", {transpose});
  {setup_a_trans}
  setup_prefetch({p}prefetch, std::get<3>({p}pointers), {n}, {ldc});
  // macOS dispatching of threads
  {p}params.A           = {A};
  {p}params.B           = std::get<{sparse}>({p}pointers);
  {p}params.C           = std::get<3>({p}pointers);
  {p}params.alpha       = {p}alpha;
  {p}params.beta        = {p}beta;
  {p}params.prefetch    = {p}prefetch;
  // {p}params.repetitions = repetitions;
  if(ldb == 0) {{
    //nnz = get_nnz(std::get<2>({p}pointers), {k} * {n} * sizeof(std::get<2>({p}pointers)[0]));
    nnz = {nnz};
  }}
  else
    nnz = {n} * {k};
  // iterate over number of threads to measure performance for 1 - total_num_threads cores
  for(int num_threads = 1; num_threads <= total_num_threads; num_threads+=9) {{
  // start counting
  gettimeofday(&start, NULL);
  for(int t = 0; t < num_threads; t++) {{
    dispatch_group_async_f(group, queue, &{p}params, +[](void* context) {{
      {p}GemmParams* p = static_cast<{p}GemmParams*>(context);
      for(int rep = 0; rep < p->repetitions; rep++) {{
        {name}(p->A, p->B, p->C, p->alpha, p->beta, p->prefetch);
      }}
    }}); 
  }}
  dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  total_time = seconds + (1.0/1000000)*useconds;
  
  printf("{name}:\\n");
  flops_sanity_check = 2 * {m} * nnz;
  printf("nnz: %d\\n", nnz);
  printf("2*%d*%d*%d = %d \\n", {m}, {n}, {k}, 2*{m}*{n}*{k});
  printf("flops_sanity_check = %d \\n", flops_sanity_check);
  flops_sanity_check *= repetitions;
  flops_sanity_check *= num_threads;
  gflops_sanity_check = flops_sanity_check / 1.0e9;// / total_time;
  gflops_sanity_check /= total_time;
  printf("Duration: %f s\\n", total_time);
  printf("GFLOPS  : %f  \\n", gflops_sanity_check);
  measurements << "{name}," << num_threads << "," << ({m} == {k} ? "true" : "false") << ",{m}," << total_time << "," << gflops_sanity_check << "\\n";
  }}
  result = post<{T}>({m}, {n}, {k}, {lda}, &ldb, {ldc}, &{p}alpha, &{p}beta, std::get<0>({p}pointers), std::get<1>({p}pointers), std::get<3>({p}pointers), std::get<4>({p}pointers), {delta:.7f}, transposed);
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>({p}pointers)); free(std::get<1>({p}pointers)); free(std::get<2>({p}pointers)); free(std::get<3>({p}pointers)); free(std::get<4>({p}pointers)); free({p}prefetch); {free_a_trans}

""".format(m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldbsparse=kern.k if sparse else kern.ldb, ldc=kern.ldc, alpha=kern.alpha, beta=kern.beta,
           mtx=mtx, delta=kern.delta, name=name, sparse=2 if kern.ldb == 0 and arch[:7] == "arm_sve" else 1, A="{p}Atrans".format(p=prec) if has_matrix_ins else "std::get<0>({p}pointers)".format(p=prec), setup_a_trans=setup_Atrans, free_a_trans=free_Atrans,
           p=prec, T="float" if prec == 'f' else "double", transpose=transposed, nnz=nnz))

    f.write("""
  measurements.close();
  printf("Measurements were stored in a csv file.\\n");
""")
    f.write(test_generator.end_of_testsuite)
