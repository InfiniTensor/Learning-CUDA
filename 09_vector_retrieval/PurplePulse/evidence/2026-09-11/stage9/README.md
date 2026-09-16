# Stage 9 clean-build evidence

Remote environment: RTX 4090 D, CUDA 12.8, GCC 13.3. The successful job was
started with `nohup` and wrote phased state to
`/data/purplepulse-work/PurplePulse/results/stage9_clean_20260911_v2/`.

- `progress.tsv` and `status` show the phase sequence and terminal state.
- `logs/ctest.log` contains the 18/18 CTest summary.
- `logs/exact_verify.log` and `logs/ivf_verify.log` contain the CPU/GPU checks.
- `summary.txt` records row counts and SHA-256 hashes for the remote outputs.
- `reproduce-plan.json` is the machine-resolved Yahoo MiniLM dry-run plan.
- `release-validation.json` is the local release-artifact validator result.

The first clean configure attempt wrote its failure state under the sibling remote
directory `stage9_clean_20260911`: the non-login shell did not include
`/usr/local/cuda/bin` in `PATH`, so CMake could not find `nvcc` and stopped before
compilation. The successful run used a new `-v2` build/result directory with that
PATH made explicit; no failed build directory was reused or deleted.
