#!/usr/bin/env bash
set -euo pipefail

# Keep resident memory in check and avoid oversubscribed BLAS threads.
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Recommended CPU-only configuration for the class VM.
python -u cl_demo.py --epochs_real 4 --batch 4 --workers 0 --img 128 | tee run.log
