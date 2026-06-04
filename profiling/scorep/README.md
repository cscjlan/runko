# Profiling

TODO rewrite after installing Score-P & testing this

## Installing and setup

Use the EasyBuild installed module:
```bash
export EBU_USER_PREFIX=/projappl/project_462001137/EB

ml LUMI/25.03
ml partition/G
ml Score-P/9.4-cpeCray-25.03-rocm
```

If the [python bindings](https://github.com/score-p/scorep_binding_python) are not installed,
install them to venv **after** loading the Score-P module:
```bash
source runko-venv/bin/activate

pip install scorep
```

## Usage

You can read the official documentation describing the workflow of Score-P measurement [here](https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/latest/html/workflow.html).

### Instrumenting C++

For some reason Score-P does not understand the `.c++` file extension. Thus, the files to be instrumented should be renamed
to `*.cpp`.

It's beneficial to run the non-instrumented version of the case being profiled first to get a sense of the run time.
This run time can then be compared to the run time of the instrumented version(s) to have some clue about the overhead.

-------------------

To build an instrumented version, pass the `scorep` wrapper as the compiler to CMake:
```bash
SCOREP_WRAPPER=off \
cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=2 --memory --mpp=mpi --thread=none --io=posix --compiler --hip"

cmake --build build --target runko_cpp_bindings -j 16
```
Using `SCOREP_WRAPPER=off` is necessary for CMake, as it uses the given compiler during the configuration step,
and we don't want to instrument the internal CMake test cases, which may fail.

See [the docs](https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/latest/html/scorepwrapper.html) for more info.

#### Full example build file
```bash
#!/bin/bash

source runko-venv/bin/activate

export EBU_USER_PREFIX=/projappl/project_462001137/EB

ml LUMI/25.03
ml partition/G
ml Score-P/9.4-cpeCray-25.03-rocm

ml

SCOREP_WRAPPER=off \
cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=2 --memory --mpp=mpi --thread=none --io=posix --compiler --hip"

cmake --build build --target runko_cpp_bindings -j 16
```

### Instrumenting Python

The Python code is instrumented automatically during runtime by the `scorep` module. No build steps necessary.

### Sampling the Code

Sampling and profiling the code requires some extra steps. We'll go through them step by step.

Load the modules:

```bash
source ${RUNKODIR}/runko-venv/bin/activate

ml LUMI/25.03
ml partition/G
ml Score-P/9.4-cpeCray-25.03-rocm
```

Next, compile a dummy shared library for filtering out some MPI functions from Score-P profiling.
Score-P uses its own MPI profiling library and Score-P has a filtering mechanism, but this mechanism
does not support filtering out MPI functions. Groups of MPI functions can be filtered out, with
`export SCOREP_MPI_ENABLE_GROUPS=DEFAULT` but it can be coarse grained. See [the documentation](https://scorepci.pages.jsc.fz-juelich.de/scorep-pipelines/docs/scorep-5.0-rc1/html/wrapperannex.html) for more help.

```bash
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
```

We'll be using this library with `export LD_PRELOAD="${RUNKODIR}/${LIB_FNAME}.so"`,
so recompiling/relinking of runko is not necessary.

-------------------------

Tracing and profiling can generate a lot of data, so we'll be running
from a data directory on `/scratch`:

```bash
RUNKO_DATADIR="/scratch/project_462001137/${USER}/runko"
mkdir -p ${RUNKO_DATADIR}
cd ${RUNKO_DATADIR}
```

Score-P runtime is controlled with environment variables,
while the python module can be controlled by passing command line arguments to it:
```bash
# Relative to CWD
export SCOREP_EXPERIMENT_DIRECTORY=scorep/${SLURM_JOBID}
# Don't use profiling and tracing at the same time
export SCOREP_ENABLE_PROFILING=1
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=110
export SCOREP_ENABLE_TRACING=0

# Use a filter file for tracing
#export SCOREP_FILTERING_FILE=${RUNKODIR}/profiling/scorep.filter

# PAPI can be used as well
#export SCOREP_METRIC_PAPI=PAPI_FP_OPS,PAPI_L2_TCM

# Which MPI groups to profile/trace
export SCOREP_MPI_ENABLE_GROUPS=DEFAULT

# Which parts of HIP to enable
export SCOREP_HIP_ENABLE=yes
#export SCOREP_HIP_ACTIVITY_BUFFER_SIZE=1M
#export SCOREP_TOTAL_MEMORY=3G

# Enable the same settings for Python
# as were used when the C++ code was instrumented
PYTHON_SCOREP="python -m scorep"
# Change this to tracing when performing tracing with scorep
PYTHON_SCOREP="${PYTHON_SCOREP} --instrumenter-type=cProfile"
PYTHON_SCOREP="${PYTHON_SCOREP} --compiler"
PYTHON_SCOREP="${PYTHON_SCOREP} --mpp=mpi"
PYTHON_SCOREP="${PYTHON_SCOREP} --hip"
PYTHON_SCOREP="${PYTHON_SCOREP} --thread=none"
PYTHON_SCOREP="${PYTHON_SCOREP} --memory"
PYTHON_SCOREP="${PYTHON_SCOREP} --io=posix"
```

See `profiling/run.sh` for a complete example.

More help:
- Using a filter: https://scorepci.pages.jsc.fz-juelich.de/scorep-pipelines/docs/scorep-5.0-rc1/html/score.html
- Environment variables: https://scorepci.pages.jsc.fz-juelich.de/scorep-pipelines/docs/scorep-5.0-rc1/html/scorepmeasurementconfig.html
- Python Score-P bindings: https://github.com/score-p/scorep_binding_python
    - Useful to check the issues for extra information
