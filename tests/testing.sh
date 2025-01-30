#!/bin/bash
pip install --break-system-packages ../
pspamm-generator 16 16 16 16 16 16 1.0 0.0 --precision d --output_funcname sme_base_test5_8_8 --output_filename build/arm_sme512/sme_base_test5_8_8.h --output_overwrite --bm 8 --bn 8 --bk 8 --arch arm_sme512 --prefetching BL2viaC
pspamm-generator 16 16 16 16 16 16 1.0 0.0 --precision d --output_funcname sve_own_base_test5_8_8 --output_filename build/arm_sve512/sve_own_base_test5_8_8.h --output_overwrite --bm 8 --bn 8 --bk 1 --arch arm_sve512 --prefetching BL2viaC
