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
module load cray-python
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

# Which programming paradigms to measure
# This file sets PARADIGMS_USED
source ${RUNKODIR}/profiling/scorep/scorep-paradigms.sh

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS=""
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS} --verbose=2"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS} ${PARADIGMS_USED}"

# TODO Check that this matches the new build procedure
SCOREP_WRAPPER=off \
cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

cmake --build build --target runko_cpp_bindings -j 16
