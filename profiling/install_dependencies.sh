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

ml LUMI/25.09
ml partition/G
ml PrgEnv-cray
ml rocm/6.4.4
ml craype-accel-amd-gfx90a
ml cray-mpich/9.0.1
ml craype-network-ofi
ml buildtools
ml cray-python
ml lumi-CrayPath

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

# Rocm 6.4.4 comes with pre-installed rocprofiler-compute,
# but the python requirements have to be installed manually
# https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/r/rocm/#modules-with-extra-debugging-and-performance-profiling-tools-installed
# Instead of using the once provided at the path mentioned in the article,
# we're using a custom one that sets
# dash==2.18.2 and
# pandas==2.3.3
python -m pip install -r ${RUNKODIR}/profiling/rocprof-compute/requirements.txt

