#!/usr/bin/env bash
# maybe do PYTHONPATH=$(pwd)/..:$PYTHONPATH

echo "SME GEMM benchmark for the Apple M4. Right now, we do not test all multiples of 128 bit. Mostly powers of two, since gcc may not support others."

for BITLEN in 512
do
    echo "Testing $BITLEN bit SME register GEMM"
    python3 benchmark_tests.py $BITLEN
    g++-14 -g -O0 -march=armv9.4-a+sme-f64f64+sme2 -msve-vector-bits=${BITLEN} build/arm_sme${BITLEN}_benchmark.cpp -o build/sme${BITLEN}-benchmark
    g++-14 -g -O0 -march=armv9-a build/arm_benchmark.cpp -o build/arm-benchmark
    ./build/sme${BITLEN}-benchmark
    ./build/arm-benchmark
done

echo "All tests done. Bye!"

