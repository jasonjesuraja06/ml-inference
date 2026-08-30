# CI benchmark artifacts

Reports downloaded from the public `arch-bench` workflow runs, unmodified.
Each directory is one job: `run-<github run id>/<runner image>-<probe>/`,
holding the latency report, the job's stdout, and the `lscpu` output for the
machine it landed on.

    gh run download <run id> -R jasonjesuraja06/ml-inference

Which CPU a job gets is not chosen by the job. That is the reason the matrix
exists and the reason each directory carries its own `ci_runner_cpu.txt`: two
jobs that landed on different hardware are two measurements.

    run-33286695075   https://github.com/jasonjesuraja06/ml-inference/actions/runs/33286695075
