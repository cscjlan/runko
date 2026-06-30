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

module load LUMI/25.03
module load partition/G
module load PrgEnv-cray
module load rocm/6.3.4
module load craype-accel-amd-gfx90a
module load cray-mpich/8.1.32
module load craype-network-ofi
module load buildtools

if [ ! -d ${RUNKODIR} ]
then
    echo "Set the variable RUNKODIR to point to the runko repository directory"
    exit 1
fi

cd ${RUNKODIR}

# We assume here that dependencies have already been installed to the virtual environment
source venv/bin/activate

pip install --no-build-isolation -v -e ${RUNKODIR} \
      --config-settings=cmake.args=--preset=lumi-gpu
