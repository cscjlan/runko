# rocprofv3

# Score-P

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

It's beneficial to run the non-instrumented version of the case being profiled first to get a sense of the run time.
This run time can then be compared to the run time of the instrumented version(s) to have some clue about the overhead.

To build an instrumented version, pass the `scorep` wrapper as the compiler to CMake:
```bash
SCOREP_WRAPPER=off \
cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=scorep-CC \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

cmake --build build --target runko_cpp_bindings -j 16
```
Using `SCOREP_WRAPPER=off` is necessary for CMake, as it uses the given compiler during the configuration step,
and we don't want to instrument the internal CMake test cases, which may fail.

See [the docs](https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/latest/html/scorepwrapper.html) for more info.

To pass options to the scorep command in order to diverge from the default instrumentation
or to activate verbose output, use the variable `SCOREP_WRAPPER_INSTRUMENTER_FLAGS` at make time:

```bash
$ make SCOREP_WRAPPER_INSTRUMENTER_FLAGS=--verbose
```

This will result in the execution of:

```bash
$ scorep --verbose gcc ...
```

The wrapper also allows to pass flags to the wrapped compiler call by using the variable `SCOREP_WRAPPER_COMPILER_FLAGS`:

```bash
$ make SCOREP_WRAPPER_COMPILER_FLAGS="-D_GNU_SOURCE"
```

Will result in the execution of:
```bash
$ scorep gcc -D_GNU_SOURCE ...
```

### Instrumenting Python

The Python code is instrumented automatically during runtime by the `scorep` module.

### Sampling the Code

```bash
python -m scorep pic.py
```
