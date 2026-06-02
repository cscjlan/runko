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

python -m venv venv

source venv/bin/activate

MPI4PY_BUILD_MPICC=cc python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

python -m pip install \
    pybind11 \
    scikit-build-core \
    viztracer

pip install --no-build-isolation -v -e ${RUNKODIR} \
      --config-settings=cmake.args=--preset=lumi-gpu

# Make squashfs?
