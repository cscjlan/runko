#!/bin/bash -l

#SBATCH --account=project_462001358
#SBATCH --partition=standard-g
#SBATCH --job-name=build-runko
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=00:20:00

module load LUMI/25.09
module load partition/G
module load PrgEnv-cray
module load rocm/6.4.4
module load craype-accel-amd-gfx90a
module load cray-mpich/9.0.1
module load craype-network-ofi
module load buildtools
module load lumi-CrayPath

if [ ! -d ${RUNKODIR} ]
then
    echo "Set the variable RUNKODIR to point to the runko repository directory"
    exit 1
fi

cd ${RUNKODIR}

# Dependencies should be installed to the virtual environment
# before building runko, including scorep.
source venv/bin/activate

INSTRUMENTED_BUILD_DIR=build-perf

cmake \
    --preset=lumi-gpu \
    -B $INSTRUMENTED_BUILD_DIR \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

cmake --build $INSTRUMENTED_BUILD_DIR -j 56
