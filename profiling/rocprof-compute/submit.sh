#!/bin/bash -l
#SBATCH --account=project_462001358
#SBATCH --partition=standard-g
#SBATCH --job-name=runko-rocprof-trace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=0-03:00:00       # Run time (d-hh:mm:ss)

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

source ${RUNKODIR}/venv/bin/activate

export RUNKO_RUNDIR=/tmp/$USER/rocprof-compute-run

mkdir -p ${RUNKO_RUNDIR}
cd ${RUNKO_RUNDIR}
cp ${RUNKODIR}/profiling/rocprof-compute/pic.py pic.py

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0

export OUTPUT_DIR=/scratch/project_462001358/$USER/runko_profiles/${SLURM_JOB_ID}_rocprof_compute
mkdir -p $OUTPUT_DIR

ROCPROFCOMPUTE="rocprof-compute"
ROCPROFCOMPUTE="${ROCPROFCOMPUTE} profile"
ROCPROFCOMPUTE="${ROCPROFCOMPUTE} --name deposit_current"
ROCPROFCOMPUTE="${ROCPROFCOMPUTE} --kernel deposit_current_kernel"
ROCPROFCOMPUTE="${ROCPROFCOMPUTE} --"

srun ${ROCPROFCOMPUTE} python pic.py

cp -r ${RUNKO_RUNDIR}/workloads ${OUTPUT_DIR}
cd
rm -rf ${RUNKO_RUNDIR}
