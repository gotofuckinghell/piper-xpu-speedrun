# Engineering log: Piper VITS training on Intel Arc A770

Chronological record of the optimisation work: what was tried, what was measured,
what was kept and what was rejected. All numbers come from the logs reproduced by
`env/make_measurements.py`; configurations are listed in `docs/measurements.md`.

Placeholders used below:

- `<PROJECT>` - data and workspace root
- `<WORK>`    - scratch directory for run outputs and caches

---

## 1. Goal and constraints

Train a Russian Piper (VITS) voice on a personal corpus and make training as fast as
the hardware allows. Success criterion is throughput in replicas per second
(utterances/second) at full GPU utilisation, not validation loss alone.

Hardware: Ryzen 9 9950X3D (16C/32T), 61.7 GB RAM, Intel Arc A770 16 GB, Windows
Server 2025, PyTorch 2.14.1+xpu with a Level Zero backend, driver 32.0.101.9150.

Baseline before this work: batch 20, fp32, ~4.3 replicas/s, GPU mostly idle.

---

## 2. Data pipeline

The corpus is 12 source videos of speech. Stage 1 produced `dataset_v10`
(12 029 clips, 17.0 h) through forced alignment. Later stages refined the split.

Key decision: cut training clips into **4-6 s windows**. Rationale: the flow and
posterior encoder consume the full clip, so long clips dominate step time while
contributing one sample each. Cutting to 4-6 s keeps the alignment between text and
audio intact and raises the sample rate per second of GPU time.

Command (sanitised):

```
python tools/cut_dataset_by_words.py \
  --metadata <PROJECT>/datasets/piper_yt/dataset_v25/metadata.csv \
  --wavs     <PROJECT>/datasets/piper_yt/dataset_v25/wavs \
  --out      <PROJECT>/datasets/piper_yt/dataset_v33 \
  --min-s 4 --max-s 6
```

Result: `dataset_v33`, 17 258 clips, 25.1 h, word boundaries preserved.
Cost of the cut: about 77 % of the original audio is discarded; the trade was
accepted because throughput per GPU-second roughly doubled.

Cache build for the new corpus took 1.0 minute and produced 86 290 files
(`.audio.pt`, `.spec.pt`, `.phonemes.pt`, `.phonemes.txt`).

---

## 3. Compact cache: int16 audio instead of float32 mel

Problem: the legacy cache stores both `.spec.pt` (156 GB) and `.audio.pt` (78 GB)
for the full corpus. For v33 that is ~2.5 GB of redundant float data per epoch of
random access, and the mels are recomputable from audio.

Design:

- `.compact.pt` holds `audio (int16)` + `phonemes`
- mel is computed on the fly in the collate step with `torch.stft`
- optional `PIPER_CACHE_RAM=1` preloads all compact payloads into RAM

Validation before accepting the format:

- `spectrogram_torch` recomputed from float32 `.audio.pt` matches the stored
  `.spec.pt` exactly: max abs diff **0.0**, SNR 239-244 dB, 30/30 bit-exact
- int16 quantisation: relative error **3.06e-5**, SNR **90.3 / 90.7 dB**,
  values above 1e-3 in 0.047 % of bins
- float16 comparison: SNR 76.2 dB, 1.39 % of bins above 1e-3 - worse, rejected
- conversion cost: **1.09 ms per replica**, i.e. ~2.5 min for 133 889 replicas

Conversion for v33 took 40 s for 17 258 files with zero errors.

Outcome: the compact cache alone did not raise throughput, because the mel is still
computed in the main process during collate. It remains available but is not the
default path.

---

## 4. bf16 autocast

Autocast bf16 with fp32 master weights gives the largest single step change:
from 4.3 to ~40 samples/s in early tests.

Two platform issues had to be fixed:

1. `monotonic_align` failed with `TypeError: Got unsupported ScalarType BFloat16`.
   Fix: cast `neg_cent` and `mask.sum(...)` to `.float()` before `.cpu().numpy()`.
2. Do not combine `PIPER_BF16=1` (weights cast to bf16) with `PIPER_AMP=bf16`:
   the master copy in fp32 is lost, arithmetic changes, and comparisons become
   invalid. Guard: the run log prints which path is active.

---

## 5. Batch ceiling and gradient checkpointing

Batch sweeps at segment 2048:

| batch | outcome |
|---:|---|
| 32 | works, ~6 replicas/s (fp32 baseline) |
| 64 | works |
| 96 with checkpointing | works, 24.2 replicas/s |
| 112 with checkpointing | works, **26.8 replicas/s** |
| 120 | `UR_RESULT_ERROR_DEVICE_LOST` |
| 128 | `UR_RESULT_ERROR_DEVICE_LOST`, shared memory 12.7 GB |

Checkpointing `model_g` cuts activation memory enough to move from 64 to 112.
Aggregate allocated memory at batch 112: 1 775 MiB live, 15 114 MiB reserved,
peak allocated 14 038 MiB of 16 GB.

Two processes on one GPU were also tested: 2 x 32 and 2 x 48 give 15.2 and
15.4 replicas/s combined, i.e. they split the device instead of filling idle time.
Rejected.

Repeated three times for each arm (ABBA): base 20.9, checkpointing 96 -> 24.2,
checkpointing 112 -> 26.8.

---

## 6. Kernel-level attempts

- **Conv1d to Conv2d conversion** (height-1 convolutions). In fp32 the backward of
  `convolution_backward_overrideable` is 25.7 % of CPU time, and Conv2d removes it:
  batch 32 fp32 went from ~6 to 11.7 replicas/s. In bf16 the native path is already
  fast, so the converted model ran at 14.3 versus 20.9 - rejected. Two pitfalls were
  found on the way: weight-norm parametrisations must be materialised (otherwise the
  weight stays on the host and the kernel sees mismatched devices), and the bias has
  to be cast to the input dtype under autocast.
- **Immediate command lists** (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LISTS=2`):
  192 s versus 175 s for the same work - slower, reverted.
- **Fused AdamW**: `fused=True` fails on this build (`Required aspect fp64 is not
  supported`). `foreach=True` works and gives +3 % in an ABBA comparison
  (19.9 versus 19.25 replicas/s on 20-step blocks); kept as default with a fallback.
- **torch.compile / inductor**: no Triton backend on Windows for XPU, the run hangs
  after `no Triton fallback available for non-CUDA devices`. Abandoned.
- **Split generator/discriminator passes** (`PIPER_SPLIT_GD`): mathematically
  equivalent, but no speed gain (18.3 versus 20.9), kept only as a diagnostic mode.

---

## 7. Alignment stage (`maximum_path`)

In the training step the numpy/Cython monotonic alignment costs 47-63 ms
(median ~55 ms) on shapes `112 x 442 x 290`.

Profiling by stage at that shape:

| stage | time |
|---|---:|
| device to host copy of `neg_cent` | 22.8 ms |
| Cython kernel (already OpenMP, `prange` over batch) | 0.2 ms |
| host to device copy of the path (int32 -> float32) | ~35 ms |

Findings:

- the kernel is not the bottleneck, the transfers are
- a pure torch implementation on XPU is 660 ms for a small case versus 1.0 ms on the
  CPU: about 160 tiny kernel launches per step at ~8 ms each
- max-plus scan and XPU graphs are not worth it: shapes change every batch

Improvements implemented in `patches/fast.py`:

- single pinned transfer for data and lengths (int32 tensors copied through pinned
  memory arrive as zeros on this build - lengths are transferred as float32)
- path returned as int8 instead of fp32
- a guard slab in front of the value buffer, because the Cython kernel reads
  `value[-1]` for the first batch element and can fault (observed as a segfault)
- all 12 correctness checks pass bit-exact against the Cython reference across
  shapes, both token/frame orders, batch 1, bf16 inputs and buffer reuse

Measured result: 22.1 -> 17.4 ms isolated, but **21.6 versus 24.1 in the real step**
(the hard `torch.xpu.synchronize()` needed before reading the pinned buffer costs more
than it saves). The fast path is opt-in (`PIPER_MAS_FAST=1`) and disabled by default.

---

## 8. Second wave of low-cost knobs

All evaluated in the same session, each with its own run:

| Knob | Result | Decision |
|---|---|---|
| DataLoader `num_workers=2`, prefetch 3, one thread per worker | 21.8 vs 24.1 | rejected |
| `PIPER_CACHE_RAM=1` (3 831 MB preloaded in 3.8 s) | 23.5 vs 24.1 | rejected |
| `pin_memory` off at zero workers | 22.5 vs 22.8 | no effect, kept as before (on) |
| `foreach=True` | +3 % (ABBA) | accepted |
| Fixed seed for the train/val split | - | accepted |
| Memory statistics in the end-of-run log | was dead code inside an `except` branch | fixed, peak and retries now printed |

Windows Defender is not installed on this host (`WinDefend` service absent), so
antivirus exclusions were moot.

---

## 9. Measurement methodology findings

Two effects produced apparent regressions that were not code changes:

1. **Warm-up window.** A 20-step block reports 22-23 replicas/s while the next
   20 steps of the same run take 73.5 s, i.e. **30.5 replicas/s**. Kernel autotuning,
   allocator warm-up and cache fill dominate the first ~20 steps. Rule: measure at
   least 60 steps and use the last window.
2. **Driver state after a crash.** A failed run (`UR_RESULT_ERROR_UNKNOWN` during the
   first host-to-device copy) leaves the device slow: the next identical run took
   188 s instead of 98 s. Only a driver reset or reboot restores the speed.

Telemetry during steady state: GPU utilisation 100 %, core 2400 MHz, package power
104 W, temperatures 52/54 C. The earlier "45 % utilisation" figure was an average over
the whole run including warm-up and CPU-bound phases.

---

## 10. Final configuration

```
PIPER_XPU=1 PIPER_AMP=bf16 PIPER_CKPT=1 PIPER_FOREACH_ADAM=1
batch 112  segment 2048  num_workers 0  length bucketing on
OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=30
warm start from the previous checkpoint
```

Steady-state throughput: **26-30 replicas/s** (about 4.2-3.7 s per 112-sample step),
peak memory 14.0 GB of 16 GB, GPU fully utilised.

---

## 11. Reproduction checklist

1. Install the environment described in `env/requirements.txt`, then build the
   patched training package: apply `patches/*.patch` to `piper1-gpl` sources and copy
   `patches/fast.py` into `piper/train/vits/monotonic_align/`.
2. Cut the corpus into 4-6 s clips (`tools/cut_dataset_by_words.py`).
3. Build the cache (`tools/build_cache_parallel.py`), optionally add
   `.compact.pt` files (`tools/convert_compact.py`).
4. Run `tools/run_final.sh` with the environment above; replace `<PROJECT>` and
   `<WORK>` placeholders via `env/bootstrap.py`.
5. Measure with `tools/repeats_ckpt.py` (three arms) or `tools/abba_foreach.sh`
   (paired comparison); read the last timing block, not the first.
6. Verify GPU state with `tools/final_util.py` (utilisation sampling) and confirm
   memory headroom before raising the batch.

Retest commands for the individual knobs are in `tools/` (`quick_sweep.py`,
`sweep_batch_segment.py`, `trials_batch_segment.py`, `dual_process_test.py`).
