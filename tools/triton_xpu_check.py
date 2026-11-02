"""Проверка Triton XPU: простой kernel на Arc, затем torch.compile."""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def check_kernel() -> bool:
    n = 4096
    x = torch.randn(n, device="xpu")
    y = torch.randn(n, device="xpu")
    out = torch.empty_like(x)
    try:
        grid = (triton.cdiv(n, 512),)
        add_kernel[grid](x, y, out, n, BLOCK=512)
        torch.xpu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] kernel: {type(exc).__name__}: {str(exc)[:160]}")
        return False
    ok = torch.allclose(out, x + y, atol=1e-5)
    print(f"[{'PASS' if ok else 'FAIL'}] triton kernel на XPU (triton {triton.__version__})")
    return ok


def check_compile() -> bool:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(256, 256).to("xpu")
            self.act = torch.nn.GELU()

        def forward(self, x):
            return self.act(self.lin(x))

    m = Block()
    x = torch.randn(64, 256, device="xpu")
    try:
        cm = torch.compile(m, backend="inductor")
        t0 = time.perf_counter()
        y1 = cm(x)
        torch.xpu.synchronize()
        first = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] torch.compile(inductor): {type(exc).__name__}: {str(exc)[:200]}")
        return False
    with torch.no_grad():
        y2 = m(x)
    ok = torch.allclose(y1, y2, atol=1e-3)
    print(f"[{'PASS' if ok else 'FAIL'}] torch.compile inductor на XPU "
          f"(первый вызов {first * 1000:.0f} мс)")
    return ok


if __name__ == "__main__":
    a = check_kernel()
    b = check_compile()
    print("\nИТОГ:", "оба пути работают" if (a and b) else
          ("kernel работает, compile нет" if a else "XPU-бэкенд Triton не работает"))
