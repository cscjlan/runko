# Profiling with rocprof-compute

See [the rocprof-compute docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.3/what-is-rocprof-compute.html) for help on using the tool.

## Install

The python dependencies of the profiler need to be installed manually.
See the `install_dependencies.sh` script, which should do just this.

## Build

Build the C++ project without instrumentation.

## Run

Use `sbatch submit.sh` to run the tracing. Change the `ROCPROFCOMPUTE` env var inside the script to your liking.

By default it runs the `pic.py` script in this directory.

## Analyze the results

Any output files will be written to the `OUTPUT_DIR` directory.
See the [documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.3/how-to/analyze/mode.html) of the tool on how to visualize/analyze the results.
