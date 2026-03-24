#!/bin/bash

source runko-venv/bin/activate

ml

cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=CC -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
cmake --build build --target runko_cpp_bindings -j 16
