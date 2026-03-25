#!/bin/bash

source runko-venv/bin/activate

export EBU_USER_PREFIX=/projappl/project_462001137/EB

ml LUMI/25.03
ml partition/G
ml Score-P/9.4-cpeCray-25.03-rocm

ml

SCOREP_WRAPPER=off \
cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=2 --memory --mpp=mpi --thread=none --io=posix --compiler --hip"

cmake --build build --target runko_cpp_bindings -j 16
