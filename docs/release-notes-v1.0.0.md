# v1.0.0 - Piper VITS training on Intel Arc A770

First public snapshot of the throughput work on a single Arc A770 16 GB under Windows.

## Highlights

- **6x throughput**: 4.3 -> 26-30 replicas/s (steady state, warm-up excluded).
- bf16 autocast with fp32 master weights plus gradient checkpointing raises the batch
  ceiling from 64 to 112 at segment 2048.
- Corpus clips cut to 4-6 s; measured with a replayable benchmark suite.
- Alignment stage profiled and reworked: single pinned transfer, int8 path, guard slab
  for an out-of-range read in the Cython kernel (12/12 bit-exact checks).

## What is inside

- `docs/run-log-en.md` - chronological engineering log with every hypothesis and number.
- `docs/measurements.md` - all runs and throughput, regenerated from logs.
- `patches/` - diffs against piper1-gpl plus the alternative alignment module.
- `tools/` - benchmark, data preparation and monitoring scripts.
- `env/` - pinned versions, environment variables, path bootstrap.

## Negative results included

Conv1d -> Conv2d conversion, immediate command lists, fused AdamW, torch.compile,
two processes per GPU, DataLoader workers, RAM-preloaded cache, XPU alignment.
Each with the numbers that rejected it, so the work is not repeated.

## Notes

- Peak memory at batch 112: 14.0 GB of 16 GB.
- Measure at least 60 steps: a 20-step window understates throughput by 15-25 %.
- Do not combine `PIPER_BF16=1` with `PIPER_AMP=bf16`.
