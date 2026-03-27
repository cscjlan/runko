#!/bin/bash

source runko-venv/bin/activate

export EBU_USER_PREFIX=/projappl/project_462001137/EB

#ml LUMI/25.03
#ml partition/G
ml Score-P/9.4-cpeCray-25.03-rocm

ml

# TODO: set C++ flags directly here without using the Score-P wrapper,
# as it does not run for files with .c++ extension.

# Which programming paradigms to measure
# This file sets PARADIGMS_USED
source ${RUNKODIR}/profiling/scorep-paradigms.sh

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS=""
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS} --verbose=2"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS} ${PARADIGMS_USED}"


SCOREP_WRAPPER=off \
cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

cmake --build build --target runko_cpp_bindings -j 16
