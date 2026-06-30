# Profiling experiment

This experiment uses perf to sample runko. A flamegraph image is generated from the run.

## Running the experiment

Use `sbatch submit.sh` to run the experiment on LUMI.

```bash
for ppc in 8 16 64 do
    fgout=${OUTPUT_DIR}/ppc_${ppc}.svg
    sed -i "s/\(^[ ]\{4\}ppc =\) [0-9]*/\1 ${ppc}/" pic.py
    srun perf record -F 1997 -g -o perf.data \
        python pic.py && \
        perf script | ${FGDIR}/stackcollapse-perf.pl | ${FGDIR}/flamegraph.pl > ${fgout}
done
```

The experiment is run for different ppc counts.
