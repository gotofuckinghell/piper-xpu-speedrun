# Piper VITS training on Intel Arc A770: reproducibility package

This repository documents and reproduces the work of making Piper (VITS) training on a
single Intel Arc A770 16 GB run as fast as the device allows under Windows.

Everything here is the result of measured experiments: every number is reproducible
with the scripts in `tools/`, and every rejected idea is listed with the reason and
the numbers that rejected it.

## Result

| Stage | Throughput | Notes |
|---|---:|---|
| Starting point (batch 20, fp32) | 4.3 replicas/s | GPU mostly idle |
| bf16 autocast + larger batch (64) | ~21 replicas/s | single biggest change |
| Gradient checkpointing, batch 112 | 26.8 replicas/s | activation memory cut |
| 4-6 s clips, steady state | **26-30 replicas/s** | warm-up excluded |

Memory at batch 112: 1 775 MiB live, 15 114 MiB reserved, 14 038 MiB peak of 16 GB.
Steady-state telemetry: 100 % GPU utilisation, 2400 MHz, 104 W, 52/54 C.

## Repository layout

```
docs/run-log-en.md      chronological engineering log (what was tried and why)
docs/measurements.md    table of all runs with throughput numbers
patches/                changes to piper1-gpl sources + the new alignment module
tools/                  benchmark, data preparation and monitoring scripts
env/requirements.txt    exact package versions
env/train-env.sh        environment variables for a training run
env/bootstrap.py        fills <PROJECT>/<WORK>/<HOME> placeholders in the scripts
env/make_measurements.py regenerates docs/measurements.md from run logs
```

## Quick start

```bash
# 1. environment
pip install -r env/requirements.txt        # torch 2.14.1+xpu, lightning, piper1-gpl[train]

# 2. patched training package
git clone https://github.com/rhasspy/piper1-gpl
cd piper1-gpl && patch -p1 < ../patches/dataset.py.patch
patch -p1 < ../patches/lightning.py.patch
patch -p1 < ../patches/monotonic_align__init__.py.patch
cp ../patches/fast.py src/piper/train/vits/monotonic_align/fast.py
pip install -e .

# 3. point the scripts at your data
python env/bootstrap.py --project /path/to/data --work /path/to/scratch

# 4. train
bash tools/run_final.sh

# 5. measure (>= 60 steps, use the last timing block)
python tools/repeats_ckpt.py
```

## Configuration

Environment variables understood by the patched package:

| Variable | Meaning |
|---|---|
| `PIPER_XPU=1` | run on the XPU backend |
| `PIPER_AMP=bf16` | bf16 autocast with fp32 master weights (main speed lever) |
| `PIPER_CKPT=1` | gradient checkpointing of the generator (raises batch ceiling) |
| `PIPER_FOREACH_ADAM=1` | `foreach=True` in both AdamW (small gain, default on) |
| `PIPER_COMPACT=1` | read `.compact.pt` (int16 audio) instead of float mels |
| `PIPER_CACHE_RAM=1` | preload compact payloads into RAM (zero-workers only) |
| `PIPER_PIN=0` | disable pinned host memory in the DataLoader |
| `PIPER_SPLIT_SEED` | fixed seed for the train/val/test split (default 1234) |
| `PIPER_XPU_MEM_FRACTION` | hard allocator cap as a fraction of VRAM |
| `PIPER_MAS_FAST=1` | alternative alignment path (pinned + int8); slower in-step, off by default |
| `PIPER_TIMERS`, `PIPER_STEP_TIMING` | section timers; keep off for throughput runs |

Recommended run:

```
batch 112, segment 2048, num_workers 0, length bucketing on,
OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=30
```

## Measure correctly

Three rules learned the hard way, each backed by numbers in `docs/run-log-en.md`:

1. Use at least 60 steps: the first ~20 steps are 15-25 % slower (kernel autotuning,
   allocator warm-up, cache fill). A 20-step window reports 22-23 replicas/s where the
   next 20 steps run at 30.5.
2. After a failed run (`UR_RESULT_ERROR_UNKNOWN`) the device stays slow until the
   driver is reset - one run took 188 s where the same configuration took 98 s.
3. Do not mix `PIPER_BF16=1` with `PIPER_AMP=bf16`; the arithmetic and the memory
   profile change, so comparisons become invalid.

## What did not work

Kept here so the work is not repeated:

- `Conv1d -> Conv2d` conversion: helps in fp32 (11.7 vs 6 replicas/s at batch 32),
  hurts in bf16 (14.3 vs 20.9) - the native bf16 path is already fast.
- Immediate command lists: 192 s versus 175 s for the same work.
- `fused=True` AdamW: `Required aspect fp64 is not supported` on this build.
- `torch.compile`: no Triton backend for XPU on Windows; the run hangs.
- Two processes on one GPU: 15.2-15.4 replicas/s combined instead of ~27 for one.
- `num_workers=2`: 21.8 versus 24.1, IPC of large batches outweighs the overlap.
- `PIPER_CACHE_RAM=1`: 23.5 versus 24.1, the mel is computed in the main process anyway.
- Batch 120 and 128: `UR_RESULT_ERROR_DEVICE_LOST`, shared memory spills first.
- Alignment on the XPU: ~160 tiny kernel launches cost more than the 0.2 ms kernel;
  660 ms versus 1.0 ms on a small case.
- Splitting the generator and discriminator passes: mathematically equivalent, no gain.

## License / provenance

Code in `patches/` derives from [piper1-gpl](https://github.com/rhasspy/piper1-gpl)
(MIT). Scripts in `tools/` were written for this project and are released under MIT.
