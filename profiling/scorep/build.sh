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

# Add installed Score-P to PATH
export PATH=/projappl/project_462001358/scorep/bin:${PATH}

# Which programming paradigms to measure
# This file sets PARADIGMS_USED
source ${RUNKODIR}/profiling/scorep/scorep-paradigms.sh

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS=""
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS} --verbose=2"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS} ${PARADIGMS_USED}"

INSTRUMENTED_BUILD_DIR=build-instrumented

SCOREP_WRAPPER=off \
cmake \
    --preset=lumi-gpu \
    -B $INSTRUMENTED_BUILD_DIR \
    -S . \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

cmake --build $INSTRUMENTED_BUILD_DIR
