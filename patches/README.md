# Patches for piper1-gpl

Applied against `src/piper/train/vits/` of
[piper1-gpl](https://github.com/rhasspy/piper1-gpl). Verified with
`patch -p1 --dry-run` on the upstream tree.

| File | What it adds |
|---|---|
| `dataset.py.patch` | int16 compact cache with on-the-fly mel, RAM preload, length bucketing by stored byte length, fixed split seed, pinned-memory switch, one-thread worker init |
| `lightning.py.patch` | bf16 autocast without touching master weights, gradient checkpointing of the generator, generator/discriminator split passes, section and step timers, `foreach` AdamW with fallback, allocator statistics and peak reset |
| `monotonic_align__init__.py.patch` | bf16-safe casts before the numpy round-trip, optional timing line, opt-in fast path dispatch |
| `fast.py` (new file) | `maximum_path` with a single pinned transfer, int8 path, guard slab for the kernel's out-of-range read |

Notes:

- `fast.py` is opt-in (`PIPER_MAS_FAST=1`). It is bit-exact against the Cython
  reference but slower inside the training step on this platform.
- The `Conv1d -> Conv2d` experiment is not included: it lives in the repository
  history of the working copy only, because it lost to the native bf16 path.
  The idea and the numbers are recorded in `docs/run-log-en.md`.
- `PIPER_BF16=1` and `PIPER_AMP=bf16` must not be combined; the log prints which
  path is active so a comparison cannot be silently invalidated.

Apply:

```bash
cd piper1-gpl
patch -p1 < ../patches/dataset.py.patch
patch -p1 < ../patches/lightning.py.patch
patch -p1 < ../patches/monotonic_align__init__.py.patch
cp ../patches/fast.py src/piper/train/vits/monotonic_align/fast.py
```
