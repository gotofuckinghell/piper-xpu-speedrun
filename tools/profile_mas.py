r"""Профиль стадий maximum_path на реальной форме батча (с защитным слоем,
иначе Cython читает value[-1] у первого элемента и падает)."""
import time

import numpy as np
import torch

from piper.train.vits.monotonic_align.core import maximum_path_c

B, T, S = 112, 442, 290
dev = "xpu"
torch.manual_seed(0)

nc = torch.randn(B, T, S, device=dev, dtype=torch.float32)
mask = torch.ones(B, T, S, device=dev, dtype=torch.float32)

pin_buf = torch.empty(B * T * S + 2 * B, dtype=torch.float32, pin_memory=True)
pin_i32 = torch.empty(B * T * S, dtype=torch.int32, pin_memory=True)


def bench(fn, n=10):
    fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def a_current():
    n = nc.data.float().cpu().numpy().astype(np.float32)
    tt = mask.sum(1)[:, 0].data.float().cpu().numpy().astype(np.int32)
    ts = mask.sum(2)[:, 0].data.float().cpu().numpy().astype(np.int32)
    return n, tt, ts


def b_fused_pinned():
    lens = torch.stack((mask.sum(1)[:, 0], mask.sum(2)[:, 0]), dim=1).float()
    flat = pin_buf[: B * T * S + 2 * B]
    flat[: B * T * S].copy_(nc.reshape(-1), non_blocking=True)
    flat[B * T * S:].copy_(lens.reshape(-1), non_blocking=True)
    torch.xpu.synchronize()
    arr = flat.numpy()
    n = arr[: B * T * S].reshape(B, T, S)
    L = arr[B * T * S:].reshape(B, 2).astype(np.int32)
    return n, L[:, 0].copy(), L[:, 1].copy()


def cy(n, tt, ts):
    path = np.zeros((B + 1, T, S), dtype=np.int32)
    maximum_path_c(path[1:], n, tt, ts)
    return path[1:]


def h2d(path):
    pin_i32[: B * T * S].copy_(torch.from_numpy(path.reshape(-1)))
    out = torch.empty((B, T, S), dtype=torch.float32, device=dev)
    out.reshape(-1).copy_(pin_i32[: B * T * S], non_blocking=True)


n2, tt2, ts2 = b_fused_pinned()

print(f"форма b={B} T={T} S={S}; neg_cent {nc.numel() * 4 / 2**20:.1f} МБ")
print(f"  A. текущий D2H (float+cpu+numpy) : {bench(a_current):7.1f} мс")
print(f"  B. один pinned-перегон          : {bench(b_fused_pinned):7.1f} мс")
print(f"  C. Cython ядро                  : {bench(lambda: cy(n2, tt2, ts2)):7.1f} мс")
p = cy(n2, tt2, ts2)
print(f"  D. H2D через pinned             : {bench(lambda: h2d(p)):7.1f} мс")


def full_old():
    n, a, b = a_current()
    q = cy(n, a, b)
    torch.from_numpy(q).to(device=dev, dtype=torch.float32)


def full_new():
    n, a, b = b_fused_pinned()
    q = cy(n, a, b)
    h2d(q)


print(f"  ИТОГО текущий путь              : {bench(full_old):7.1f} мс")
print(f"  ИТОГО fused+pinned              : {bench(full_new):7.1f} мс")
