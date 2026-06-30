#!/bin/bash -l
#SBATCH --account=project_462001358
#SBATCH --partition=standard-g
#SBATCH --job-name=runko-batch-inject-profile
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
module load cray-python
module load lumi-CrayPath

if [ ! -d ${RUNKODIR} ]
then
    echo "Set the variable RUNKODIR to point to the runko repository directory"
    exit 1
fi

if [ ! -d ${FGDIR} ]
then
    echo "Set the variable FGDIR to point to the FlameGraph repository directory"
    exit 1
fi

source ${RUNKODIR}/venv/bin/activate

export RUNKO_RUNDIR=/tmp/$USER/perf

mkdir -p ${RUNKO_RUNDIR}
cd ${RUNKO_RUNDIR}
cp ${RUNKODIR}/profiling/perf/pic.py pic.py

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0

OUTPUT_DIR=/scratch/project_462001358/$USER/flamegraphs/${SLURM_JOB_ID}
mkdir -p ${OUTPUT_DIR}

for ppc in 8 16 64
do
    fgout=${OUTPUT_DIR}/ppc_${ppc}.svg
    sed -i "s/\(^[ ]\{4\}ppc =\) [0-9]*/\1 ${ppc}/" pic.py
    srun perf record -F 1997 -g -o perf.data \
        python pic.py && \
        perf script | ${FGDIR}/stackcollapse-perf.pl | ${FGDIR}/flamegraph.pl > ${fgout}
done

cd
rm -rf ${RUNKO_RUNDIR}
