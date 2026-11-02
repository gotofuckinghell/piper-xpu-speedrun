#!/usr/bin/env bash
# Environment for a throughput run. Source it, do not execute it.
export PIPER_XPU=1
export PIPER_AMP=bf16
export PIPER_CKPT=1
export PIPER_FOREACH_ADAM=1
export PIPER_SPLIT_SEED=1234
export PIPER_NO_MOS=1
export PIPER_NO_AUDIO_LOG=1

# 16C/32T host: keep two threads for the system and one for background services.
export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export NUMEXPR_NUM_THREADS=30

# Optional knobs, off by default:
# export PIPER_COMPACT=1          # read int16 compact cache
# export PIPER_CACHE_RAM=1        # preload compact cache into RAM (workers=0 only)
# export PIPER_MAS_FAST=1         # alternative alignment path (slower in-step)
# export PIPER_XPU_MEM_FRACTION=0.85
