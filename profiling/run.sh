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

cd /scratch/project_462001137/${USER}/runko

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0

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

export SCOREP_ENABLE_PROFILING=1
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=110

export SCOREP_ENABLE_TRACING=0

# TODO: create filter
#export SCOREP_FILTERING_FILE=${RUNKODIR}/scorep-filter

#export SCOREP_METRIC_PAPI=PAPI_FP_OPS,PAPI_L2_TCM
export SCOREP_MPI_ENABLE_GROUPS=DEFAULT
export SCOREP_HIP_ENABLE=yes

#export SCOREP_HIP_ACTIVITY_BUFFER_SIZE=1M
#export SCOREP_TOTAL_MEMORY=3G


# -------- Choose the profiler ---------
PROFILER=
#PROFILER="${ROCPROFV3}"

#PYTHON=python
PYTHON="python -m scorep --compiler --mpp=mpi --instrumenter-type=cProfile"

srun --cpu-bind=${CPU_BIND} ./select_gpu $PROFILER ${PYTHON} ${RUNKODIR}/projects/pic-turbulence/pic.py

rm -f ./select_gpu
