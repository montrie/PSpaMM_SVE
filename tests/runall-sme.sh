#!/usr/bin/env bash
# maybe do PYTHONPATH=$(pwd)/..:$PYTHONPATH

echo "SME GEMM test. Right now, we do not test all multiples of 128 bit. Mostly powers of two, since gcc may not support others."

for BITLEN in 512
do
    echo ""
    echo ""
    echo "Testing $BITLEN bit SME register GEMM"
    python3 unit_tests_arm_sme.py $BITLEN
#    rm build/sme${BITLEN}-test
    aarch64-linux-gnu-g++-14 -g -O0 -march=armv9.4-a+sme-f64f64 -msve-vector-bits=${BITLEN} build/arm_sme${BITLEN}_testsuite.cpp -o build/sme${BITLEN}-test
#    clang++ -g -O0 --target=aarch64-linux-gnu -march=armv9.4-a+sme-f64f64 -msve-vector-bits=${BITLEN} build/arm_sme${BITLEN}_testsuite.cpp -o build/sme${BITLEN}-test
    qemu-aarch64 -L /usr/aarch64-linux-gnu -cpu max,sve${BITLEN}=on,sme${BITLEN}=on,sve-default-vector-length=-1 build/sme${BITLEN}-test
#    qemu-aarch64 -g 1234 -L /usr/aarch64-linux-gnu -cpu max,sve${BITLEN}=on,sme${BITLEN}=on,sve-default-vector-length=-1 build/sme${BITLEN}-test
done

echo "All tests done. Bye!"

