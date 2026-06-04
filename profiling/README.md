# Profiling

## Installing dependencies

First install the dependencies with the `install_dependencies.sh` script.
This installs Score-P for instrumenting the C++ code and `viztracer` and `scorep` python packages
in addition to the rest of the python dependencies. Run it on a compute node: `sbatch install_dependencies.sh`.

## Build

Build the project either with `build.sh` or `build_instrumented.sh`: the former builds runko
without instrumenting the C++ files, while the latter uses Score-P to instrument the files.

Build on the compute node `sbatch build.sh`.

## Run

Run (one of) the experiment(s) in scorep, if you've build the instrumented version.
Otherwise, run one of the other experiments.

Follow the READMEs in the individual directories.

