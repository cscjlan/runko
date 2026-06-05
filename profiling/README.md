# Profiling

## Installing dependencies

First install the dependencies with the `install_dependencies.sh` script.
This installs `viztracer` and `scorep` python packages in addition to
the rest of the python dependencies.
Run it on a compute node: `sbatch install_dependencies.sh`.

Note that the `scorep` python package requires an installed Score-P:
specifically `scorep-config` should be in `PATH`. There's the `install_scorep.sh`,
which can be used to install Score-P, if it's not already installed. See it for more info.
If you don't have a Score-P installation available,
install it before running `install_dependencies.sh`.

## Build

Build the project either with `build.sh` or `build_instrumented.sh`: the former builds runko
without instrumenting the C++ files, while the latter uses Score-P to instrument the files.

Build on the compute node `sbatch build.sh`.

## Run

Run (one of) the experiment(s) in scorep, if you've build the instrumented version.
Otherwise, run one of the other experiments.

Follow the READMEs in the individual directories.

## TODO

- [ ] Install Score-P
- [ ] Compile python from sources so perf can be used
- [ ] maybe rocprof-sdk and automatic instrumentation with `cyg_enter`?
