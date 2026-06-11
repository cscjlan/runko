#!/bin/bash -l

#SBATCH --account=project_462001358
#SBATCH --partition=standard-g
#SBATCH --job-name=runko-scorep
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

# Add installed Score-P to PATH after venv activation
export PATH=/projappl/project_462001358/scorep/bin:${PATH}

# TODO: a robust way for python to use the shared library from the specific directory,
# not site-packages

export RUNKO_RUNDIR=/tmp/$USER/scorep

mkdir -p ${RUNKO_RUNDIR}
cd ${RUNKO_RUNDIR}
cp ${RUNKODIR}/profiling/scorep/pic.py pic.py

cat << EOF > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID

exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0

export OUTPUT_DIR=/scratch/project_462001358/$USER/runko_profiles/${SLURM_JOB_ID}_scorep
mkdir -p $OUTPUT_DIR

# Use this variable to choose between profiling and tracing the application.
# Profiling generates a summary of the execution and much less data.
# Tracing generates potentially gigabytes of data.
# Read the Score-P documentation for more information.
# https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/latest/html/workflow.html
USE_TRACING=1

if [ ${USE_TRACING} -eq 0 ]; then
    export SCOREP_ENABLE_TRACING=0
    export SCOREP_ENABLE_PROFILING=1
    PYTHON_SCOREP_INSTRUMENTER_TYPE="cProfile"
else
    export SCOREP_ENABLE_TRACING=1
    export SCOREP_ENABLE_PROFILING=0
    PYTHON_SCOREP_INSTRUMENTER_TYPE="cTrace"
fi

# -------- settings applying to both, profiling and tracing ---------

# Which programming paradigms to measure
# This file sets PARADIGMS_USED
source ${RUNKODIR}/profiling/scorep/scorep-paradigms.sh

export SCOREP_EXPERIMENT_DIRECTORY=${OUTPUT_DIR}
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=110
#export SCOREP_FILTERING_FILE=${RUNKODIR}/profiling/scorep/scorep.filter
#export SCOREP_METRIC_PAPI=PAPI_FP_OPS,PAPI_L2_TCM
export SCOREP_MPI_ENABLE_GROUPS=DEFAULT
export SCOREP_HIP_ENABLE=api,kernel,kernel_callsite,malloc,memcpy,sync,default
export SCOREP_HIP_ACTIVITY_BUFFER_SIZE=16M
export SCOREP_TOTAL_MEMORY=4000MB

PYTHON_SCOREP="python -m scorep"
#PYTHON_SCOREP="${PYTHON_SCOREP} --noinstrumenter"
PYTHON_SCOREP="${PYTHON_SCOREP} --instrumenter-type=${PYTHON_SCOREP_INSTRUMENTER_TYPE}"
PYTHON_SCOREP="${PYTHON_SCOREP} ${PARADIGMS_USED}"

#srun --cpu-bind=${CPU_BIND} ./select_gpu ${PYTHON_SCOREP} pic.py
srun ${PYTHON_SCOREP} pic.py

rm -f ./select_gpu
cd
rm -rf ${RUNKO_RUNDIR}
