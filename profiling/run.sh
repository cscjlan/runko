#!/bin/bash -l
#SBATCH --account=project_462000007
#SBATCH --partition=standard-g
#SBATCH --job-name=decay
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=0-00:20:00       # Run time (d-hh:mm:ss)

# Load correct modules here.
source ${RUNKODIR}/runko-venv/bin/activate

ml LUMI/25.03
ml partition/G
ml Score-P/9.4-cpeCray-25.03-rocm

ml

# We're compiling a shared library which we can use with LD_PRELOAD.
# These MPI functions are executed very many times and they are very fast to execute.
# Thus, the overhead of tracing them is significant and MPI_Test alone explodes the
# trace file size to gigabytes. With preload, we're skipping the tracing provided by the Score-P
# MPI libraries. Add any other functions as necessary.

LIB_FNAME=runko_mpi_preload

cat << EOF | CC -x c++ --std=c++20 -fPIC -O2 -g -shared -o "${LIB_FNAME}.so" -
#include <mpi.h>

extern "C" {
    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
      return PMPI_Test(request, flag, status);
    }

    int MPI_Comm_rank(MPI_Comm comm, int* rank) {
      return PMPI_Comm_rank(comm, rank);
    }
}
EOF

# -------- Set data directory to scratch ---------
# Tracing/profiling can generate a lot of data
RUNKO_DATADIR="/scratch/project_462001137/${USER}/runko"

mkdir -p ${RUNKO_DATADIR}
cd ${RUNKO_DATADIR}

# -------- GPU selection helper ---------
cat << EOF > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID

exec \$*
EOF

chmod +x ./select_gpu

# -------- CPU binding per GPU ---------
CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

# -------- General env vars ---------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0
export LD_PRELOAD="${RUNKODIR}/${LIB_FNAME}.so" 

# -------- rocprofv3 settings ---------
ROCPROFV3="rocprofv3"
ROCPROFV3="${ROCPROFV3} --output-format pftrace"
ROCPROFV3="${ROCPROFV3} --sys-trace"
# This is not supported on 6.3.4
#ROCPROFV3="${ROCPROFV3} --collection-period 60:1:5"
ROCPROFV3="${ROCPROFV3} --output-directory rocproftraces/%job%"
ROCPROFV3="${ROCPROFV3} --output-file %launch_time%-%hostname%-%pid%-%rank%.pftrace"
ROCPROFV3="${ROCPROFV3} --"

# -------- scorep settings ---------
export SCOREP_EXPERIMENT_DIRECTORY=scorep/${SLURM_JOBID}
export SCOREP_ENABLE_PROFILING=0
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=110
export SCOREP_ENABLE_TRACING=1
export SCOREP_FILTERING_FILE=${RUNKODIR}/profiling/scorep.filter
#export SCOREP_METRIC_PAPI=PAPI_FP_OPS,PAPI_L2_TCM
export SCOREP_MPI_ENABLE_GROUPS=DEFAULT
export SCOREP_HIP_ENABLE=yes
#export SCOREP_HIP_ACTIVITY_BUFFER_SIZE=1M
export SCOREP_TOTAL_MEMORY=100MB

PYTHON_SCOREP="python -m scorep"
# Change this to tracing when performing tracing with scorep
PYTHON_SCOREP="${PYTHON_SCOREP} --instrumenter-type=cTrace"
PYTHON_SCOREP="${PYTHON_SCOREP} --compiler"
PYTHON_SCOREP="${PYTHON_SCOREP} --mpp=mpi"
PYTHON_SCOREP="${PYTHON_SCOREP} --hip"
PYTHON_SCOREP="${PYTHON_SCOREP} --thread=none"
PYTHON_SCOREP="${PYTHON_SCOREP} --memory"

# -------- Choose the profiler ---------
# Don't run multiple profilers at the same time (e.g. rocprofv3 and scorep)
PROFILER=
#PROFILER="${ROCPROFV3}"

# -------- Choose the python interpreter ---------
#PYTHON=python
PYTHON="${PYTHON_SCOREP}"

# -------- Choose the project ---------
RUNKO_PROJECT="${RUNKODIR}/projects/pic-turbulence/pic.py"

# -------- Run ---------
srun --cpu-bind=${CPU_BIND} ./select_gpu $PROFILER ${PYTHON} ${RUNKO_PROJECT}

# -------- Remove temp files ---------
rm -f ./select_gpu
rm -f "${RUNKODIR}/${LIB_FNAME}.so" 
