#!/bin/bash

source runko-venv/bin/activate

module load perftools-base
module load perftools

ml

cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=CC -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
cmake --build build --target runko_cpp_bindings -j 16

pat_build \
    runko_cpp_bindings.cpython-311-x86_64-linux-gnu.so
