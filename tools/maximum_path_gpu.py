r"""maximum_path на GPU (XPU) вместо numpy/Cython на CPU.

Оригинал в piper/train/vits/monotonic_align/__init__.py перегоняет
neg_cent на CPU (.cpu().numpy()) и считает путь в Cython — это ~0,08-0,1 с
на шаг. Здесь тот же динамический максимум, но в torch на устройстве
тензора: прямой проход по t (векторно по батчу и по s) + обратная
реконструкция пути.

Проверка: пути должны совпадать бит-в-бит с Cython-версией.
"""
from __future__ import annotations

import torch

NEG = -1e9


def maximum_path_gpu(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent, mask: [b, t_t, t_s] на одном устройстве."""
    b, t_t, t_s = neg_cent.shape
    dev = neg_cent.device
    dtype = neg_cent.dtype

    with torch.no_grad():
        nc = neg_cent.float()
        m = mask.float()
        value = nc.new_full((b, t_t, t_s), NEG)
        came_left = torch.zeros((b, t_t, t_s), dtype=torch.bool, device=dev)

        value[:, 0, 0] = nc[:, 0, 0]
        pad = torch.full((b, 1), NEG, device=dev, dtype=value.dtype)
        for t in range(1, t_t):
            prev = value[:, t - 1]
            from_left = torch.cat([pad, prev[:, :-1]], dim=1)
            left_wins = from_left > prev
            best = torch.where(left_wins, from_left, prev)
            value[:, t] = best + nc[:, t]
            came_left[:, t] = left_wins

        t_len = m.sum(1)[:, 0].to(torch.long).clamp_(1, t_t) - 1
        s_len = m.sum(2)[:, 0].to(torch.long).clamp_(1, t_s) - 1

        path = torch.zeros((b, t_t, t_s), dtype=torch.float32, device=dev)
        rows = torch.arange(b, device=dev)
        t_idx = t_len.clone()
        s_idx = s_len.clone()
        alive = torch.ones(b, dtype=torch.bool, device=dev)

        for _ in range(t_t):
            live_t = t_idx.clamp_(min=0)
            live_s = s_idx.clamp_(min=0)
            flat = rows * (t_t * t_s) + live_t * t_s + live_s
            path.reshape(-1)[flat[alive]] = 1.0
            step_left = came_left.reshape(-1)[flat].to(torch.long)
            s_idx = s_idx - step_left
            t_idx = t_idx - 1
            alive &= (t_idx >= 0) & (s_idx >= 0)
            if not bool(alive.any()):
                break

    return path.to(dtype=dtype)


def check_against_cpu(neg_cent, mask, cpu_fn):
    """Сверка: возвращает долю совпавших элементов."""
    a = maximum_path_gpu(neg_cent, mask)
    b = cpu_fn(neg_cent, mask)
    return float((a == b.float()).float().mean())
