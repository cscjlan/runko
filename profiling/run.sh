#!/bin/bash -l
#SBATCH --account=project_462000007
#SBATCH --partition=standard-g
#SBATCH --job-name=decay
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=0-00:20:00       # Run time (d-hh:mm:ss)

source ${RUNKODIR}/runko-venv/bin/activate

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


# -------- Set the profiling commands ---------
PROFILER=scorep
#PROFILER=rocprofv3
#PROFILER=

case "${PROFILER}" in
    scorep)
        ml Score-P/9.4-cpeCray-25.03-rocm

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

        # We're compiling a shared library which we can use with LD_PRELOAD.
        # These MPI functions are executed very many times and they are very fast to execute.
        # Thus, the overhead of tracing them is significant and MPI_Test alone explodes the
        # trace file size to gigabytes. With preload, we're skipping the tracing provided by the Score-P
        # MPI libraries. Add any other functions as necessary.

        RUNKO_PRELOAD_LIB_FNAME=runko_scorep_mpi_preload.so

        cat << EOF | CC -x c++ --std=c++20 -fPIC -O2 -g -shared -o "${RUNKO_PRELOAD_LIB_FNAME}" -
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
        # Which programming paradigms to measure
        # This file sets PARADIGMS_USED
        source ${RUNKODIR}/profiling/scorep-paradigms.sh

        # Set libraries to preload: in addition to our own (which should go first), add the libs used by Score-P
        #export PRELOADED_LIBS="${RUNKO_DATADIR}/${RUNKO_PRELOAD_LIB_FNAME} $(scorep-config --preload-libs ${PARADIGMS_USED})" 
        export PRELOADED_LIBS="${RUNKO_DATADIR}/${RUNKO_PRELOAD_LIB_FNAME}" 

        export SCOREP_EXPERIMENT_DIRECTORY=scorep/${SLURM_JOBID}
        export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=110
        export SCOREP_FILTERING_FILE=${RUNKODIR}/profiling/scorep.filter
        #export SCOREP_METRIC_PAPI=PAPI_FP_OPS,PAPI_L2_TCM
        export SCOREP_MPI_ENABLE_GROUPS=DEFAULT
        export SCOREP_HIP_ENABLE=api,kernel,kernel_callsite,malloc,memcpy,sync,default
        export SCOREP_HIP_ACTIVITY_BUFFER_SIZE=16M
        export SCOREP_TOTAL_MEMORY=4000MB

        PYTHON_SCOREP="python -m scorep"
        #PYTHON_SCOREP="${PYTHON_SCOREP} --noinstrumenter"
        PYTHON_SCOREP="${PYTHON_SCOREP} --instrumenter-type=${PYTHON_SCOREP_INSTRUMENTER_TYPE}"
        PYTHON_SCOREP="${PYTHON_SCOREP} ${PARADIGMS_USED}"

        PROFILER_CMD=
        PYTHON_CMD=${PYTHON_SCOREP}

        ;;
    rocprofv3)
        # -------- rocprofv3 settings ---------
        ROCPROFV3="rocprofv3"
        ROCPROFV3="${ROCPROFV3} --output-format pftrace"
        ROCPROFV3="${ROCPROFV3} --sys-trace"
        # This is not supported on 6.3.4
        #ROCPROFV3="${ROCPROFV3} --collection-period 60:1:5"
        ROCPROFV3="${ROCPROFV3} --output-directory rocproftraces/%job%"
        ROCPROFV3="${ROCPROFV3} --output-file %launch_time%-%hostname%-%pid%-%rank%.pftrace"
        ROCPROFV3="${ROCPROFV3} --"

        PROFILER_CMD=${ROCPROFV3}
        PYTHON_CMD=python

        ;;
    *)
        PROFILER_CMD=
        PYTHON_CMD=python

        ;;
esac

ml

# -------- Choose the project ---------
RUNKO_PROJECT="${RUNKODIR}/projects/pic-turbulence/pic.py"

# -------- Run ---------
# Use --export to set the environment for the running job.
# We don't want to export LD_PRELOAD for the srun command itself, just for the program we're running.
# ALL is necessary so the exported variables from this file are also propagated.
RUNKO_SRUN_OPTIONS=""
RUNKO_SRUN_OPTIONS="${RUNKO_SRUN_OPTIONS} --cpu-bind=${CPU_BIND}"
#RUNKO_SRUN_OPTIONS="${RUNKO_SRUN_OPTIONS} --export=ALL,LD_PRELOAD=${PRELOADED_LIBS}"

srun ${RUNKO_SRUN_OPTIONS} ./select_gpu ${PROFILER_CMD} ${PYTHON_CMD} ${RUNKO_PROJECT}

rm -f ./select_gpu
