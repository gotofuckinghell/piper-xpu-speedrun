# v1.1.0 - Triton XPU, source build recipe, OpenVINO voice pipeline

Adds the second wave of work: what made Triton usable on XPU, how to build PyTorch from
Intel's repository, what the convolution backward actually runs on, and how a trained
voice is exported to ONNX and served by native OpenVINO.

## Added

- `docs/torch-xpu-source-build.md` - building PyTorch + Triton from
  `intel/intel-xpu-backend-for-triton`: prerequisites, the exact command sequence, and the
  eight pitfalls that stop the build (long paths, `bash -lc` losing vcvars, non-idempotent
  `mkdir`, POSIX `libuv_ROOT`, missing PTI for Kineto, gloo/MPI under MSVC).
- `docs/triton-xpu-notes.md` - kernel rules from Intel's backend: Tensor Descriptors,
  2D-only descriptors with last stride 1, grid axis order, autotune knobs, and what these
  mean for a VITS conv backward.
- `docs/onednn-conv-backward.md` - `convolution_backward_overrideable` on XPU is oneDNN
  dgrad/wgrad, not a naive fallback; 1D is viewed as 4D. Changes what an optimisation has
  to beat.
- `docs/openvino-piper-tts.md` - checkpoint -> ONNX -> OpenVINO server, with the three
  export traps (slim checkpoint service keys, `onnxscript`, dynamo exporter vs
  `dynamic_axes`) and kernel-cache warming.
- `tools/` - `ov_warm_cache.py`, `ov_bucket_bench.py`, `piper_ov_server.py`,
  `make_lightning_ckpt.py`, `repatch_phonemes*.py`, `repatch_accent_mark.py`,
  `triton_xpu_check.py`, `triton_modes.py`, `triton_xpu_speed.py`, `msvc_run.py`.

## Measurements in this release

- inductor on stock wheels: default mode 0.64-0.70x of eager on a conv stack;
  `max-autotune` fails with `pass_fds not supported on Windows`.
- Triton kernels need `CXX=clang-cl.exe`; GNU `clang++` rejects `/LIBPATH` flags.
- OpenVINO voice server: median 30.5 ms per phrase, RTF ~0.03; 12-shape kernel cache warms
  in 23.5 s.

## Notes

- Negative results are kept: default-mode inductor numbers must not be read as Triton's
  ceiling, because the competing mode does not run on Windows with pip wheels.
- Scripts keep `<PROJECT>` / `<WORK>` / `<HOME>` placeholders; `env/bootstrap.py` fills them.
