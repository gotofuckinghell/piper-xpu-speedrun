"""Eager против torch.compile(inductor) на XPU: свёрточный стек как у дискриминатора.

Запускать через tools/msvc_run.py с CXX=clang-cl.exe и
TORCHINDUCTOR_COMPILE_THREADS=1, иначе Triton не соберётся.
"""
from __future__ import annotations

import statistics
import time

import torch


class DiscLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(80, 128, 5, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(128, 256, 5, padding=2, stride=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(256, 256, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(256, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def bench(fn, iters: int = 15) -> float:
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(8, 80, 2048, device="xpu")
    model = DiscLike().to("xpu")

    def step_eager(m=model):
        with torch.autocast("xpu", dtype=torch.bfloat16):
            y = m(x)
        return y.float().mean().backward()

    def step_compiled(cm):
        with torch.autocast("xpu", dtype=torch.bfloat16):
            y = cm(x)
        return y.float().mean().backward()

    eager_ms = bench(lambda: (model.zero_grad(set_to_none=True), step_eager()))

    t0 = time.perf_counter()
    compiled = torch.compile(model, backend="inductor", dynamic=False)
    step_compiled(compiled)  # первая компиляция
    torch.xpu.synchronize()
    compile_s = time.perf_counter() - t0

    compiled_ms = bench(lambda: (model.zero_grad(set_to_none=True), step_compiled(compiled)))
    print(f"eager    : {eager_ms:7.1f} мс/шаг")
    print(f"compiled : {compiled_ms:7.1f} мс/шаг")
    print(f"компиляция: {compile_s:5.1f} с")
    print(f"ускорение: {eager_ms / compiled_ms:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
