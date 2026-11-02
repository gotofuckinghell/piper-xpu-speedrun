"""Сравнение режимов inductor на XPU: default, max-autotune, max-autotune-no-cudagraphs.

Запуск через tools/msvc_run.py с CXX=clang-cl.exe и TORCHINDUCTOR_COMPILE_THREADS=1.
Мерим два кейса: свёрточный стек (как дискриминатор) и matmul (как в attention).
"""
from __future__ import annotations

import statistics
import time

import torch


class ConvStack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(80, 128, 5, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(128, 256, 5, padding=2, stride=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(256, 1, 3, padding=1, stride=2),
        )

    def forward(self, x):
        return self.net(x)


class MatMul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(512, 512) * 0.02)

    def forward(self, x):
        return x @ self.w


def bench(fn, iters: int = 15) -> float:
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def run_case(name: str, model, x, modes: list) -> None:
    def step(m):
        with torch.autocast("xpu", dtype=torch.bfloat16):
            y = m(x)
        y.float().mean().backward()

    model.zero_grad(set_to_none=True)
    eager = bench(lambda: (model.zero_grad(set_to_none=True), step(model)))
    print(f"{name}: eager {eager:6.1f} мс", flush=True)

    for mode in modes:
        t0 = time.perf_counter()
        try:
            cm = torch.compile(model, backend="inductor", dynamic=False, mode=mode)
            step(cm)
            torch.xpu.synchronize()
            compile_s = time.perf_counter() - t0
            ms = bench(lambda: (model.zero_grad(set_to_none=True), step(cm)))
            print(f"  {mode:28s} {ms:6.1f} мс | компиляция {compile_s:5.1f} с | "
                  f"{eager / ms:.2f}x", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {mode:28s} ОШИБКА {type(exc).__name__}: {str(exc)[:90]}", flush=True)


def main() -> int:
    torch.manual_seed(0)
    modes = ["default", "max-autotune", "max-autotune-no-cudagraphs"]

    xc = torch.randn(8, 80, 2048, device="xpu")
    run_case("conv stack 8x80x2048", ConvStack().to("xpu"), xc, modes)

    xm = torch.randn(1024, 512, device="xpu")
    run_case("matmul 1024x512x512", MatMul().to("xpu"), xm, ["max-autotune"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
