r"""Быстрый maximum_path: pinned-буферы, один перегон, путь в int8.

Замеры на реальной форме (b=112, T=442, S=290, neg_cent 54,8 МБ):
  текущий путь: D2H 22,8 мс + H2D ~35 мс + ядро 0,2 мс ≈ 58 мс
  здесь:        D2H ~10,7 мс + путь int8 вместо fp32 (~14 МБ) + ядро
"""
from __future__ import annotations

import numpy as np
import torch

from .core import maximum_path_c

_PIN_NC = None
_PIN_LENS = None
_PIN_PATH = None


_CAP = (0, 0, 0, 0)  # (n, b, t, s)


def _ensure(b: int, t: int, s: int) -> None:
    global _PIN_NC, _PIN_LENS, _PIN_PATH, _CAP
    n = b * t * s
    cap_n, cap_b, cap_t, cap_s = _CAP
    if cap_n >= n and cap_b >= b and cap_t >= t and cap_s >= s:
        return
    n = max(n, cap_n)
    b_lens = max(2 * b, cap_b * 2)
    guard = max(t, cap_t) * max(s, cap_s)
    _PIN_NC = torch.empty(n + guard, dtype=torch.float32, pin_memory=True)
    _PIN_NC[:guard].zero_()  # value[-1] у первого элемента должен быть нулём
    # int32 с XPU в pinned-копию приходит нулями — длины гоняем как float32
    _PIN_LENS = torch.empty(b_lens, dtype=torch.float32, pin_memory=True)
    _PIN_PATH = np.empty(n, dtype=np.int32)
    _CAP = (n, max(b, cap_b), max(t, cap_t), max(s, cap_s))


def maximum_path_fast(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, t_t, t_s = neg_cent.shape
    dev, dtype = neg_cent.device, neg_cent.dtype
    _ensure(b, t_t, t_s)

    guard = _CAP[2] * _CAP[3]
    n = b * t_t * t_s

    lens = torch.stack((mask.sum(1)[:, 0], mask.sum(2)[:, 0]), dim=1).float()
    nc_pin = _PIN_NC[guard: guard + n]
    lens_pin = _PIN_LENS[: 2 * b]

    nc_pin.copy_(neg_cent.reshape(-1))
    lens_pin.copy_(lens.reshape(-1))
    torch.xpu.synchronize()

    nc_np = _PIN_NC.numpy()[guard: guard + n].reshape(b, t_t, t_s)
    lens_np = lens_pin.numpy().astype(np.int32)
    t_ys = np.ascontiguousarray(lens_np[0::2])
    t_xs = np.ascontiguousarray(lens_np[1::2])

    path_np = _PIN_PATH[: b * t_t * t_s].reshape(b, t_t, t_s)
    path_np.fill(0)
    maximum_path_c(path_np, nc_np, t_ys, t_xs)

    # int8: 14 МБ вместо 57 МБ, обратное приведение к float на карте
    return torch.from_numpy(path_np.astype(np.int8, copy=False)).to(
        device=dev, dtype=torch.float32, non_blocking=True
    ).reshape(b, t_t, t_s)
