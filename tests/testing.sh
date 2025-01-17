#!/bin/bash
pip install --break-system-packages ../
pspamm-generator 8 8 8 8 8 8 1.0 2.0 --precision d --output_funcname sme_base_test2_8_8 --output_filename build/arm_sme512/sme_base_test2_8_8.h --output_overwrite --bm 8 --bn 8 --bk 8 --arch arm_sme512 --prefetching BL2viaC
