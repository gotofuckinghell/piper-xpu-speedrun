"""Замена Conv1d/ConvTranspose1d на 2D-варианты с высотой 1.

Причина: на XPU обратный проход Conv1d уходит в медленный резервный путь
(эмуляция), тогда как Conv2d с ядром (1, k) использует нативный kernel oneDNN.
Замер: backward Conv1d ~93 мс CPU против 0,1 мс у Conv2d (и 235 мс против 0,15 мс
для ConvTranspose).

Модуль совместим по интерфейсу: принимает/отдаёт 3D-тензоры [B, C, T],
внутри добавляет/убирает ось высоты.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class Conv1dAs2d(nn.Module):
    def __init__(self, conv2d: nn.Conv2d):
        super().__init__()
        self.conv = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv
        if c.weight.device != x.device:
            c.to(x.device)
        w, b = c.weight, c.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        if b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        return F.conv2d(x.unsqueeze(2), w, b, c.stride, c.padding, c.dilation, c.groups).squeeze(2)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias


class ConvTranspose1dAs2d(nn.Module):
    def __init__(self, conv2d: nn.ConvTranspose2d):
        super().__init__()
        self.conv = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv
        if c.weight.device != x.device:
            c.to(x.device)
        w, b = c.weight, c.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        if b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        return F.conv_transpose2d(
            x.unsqueeze(2), w, b, c.stride, c.padding, c.output_padding, c.groups, c.dilation
        ).squeeze(2)


def _copy_weight(old, new: nn.Module, transpose: bool) -> None:
    """Переносит веса, включая weight_norm-параметризацию."""
    # материализуем вес: параметризацию weight_norm не переносим,
    # иначе после конверсии вес остаётся на CPU и ломает forward
    with torch.no_grad():
        new.weight.copy_(old.weight.detach().unsqueeze(2))
    if getattr(old, "bias", None) is not None and getattr(new, "bias", None) is not None:
        with torch.no_grad():
            new.bias.copy_(old.bias)


def convert_module(module: nn.Module) -> int:
    """Рекурсивно заменяет Conv1d/ConvTranspose1d. Возвращает число замен."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            new = nn.Conv2d(
                child.in_channels, child.out_channels,
                kernel_size=(1, child.kernel_size[0]),
                stride=(1, child.stride[0]) if isinstance(child.stride, tuple) else (1, child.stride),
                padding=(0, child.padding[0]) if isinstance(child.padding, tuple) else (0, child.padding),
                dilation=(1, child.dilation[0]) if isinstance(child.dilation, tuple) else (1, child.dilation),
                groups=child.groups, bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            new = new.to(device=child.weight.device, dtype=child.weight.dtype)
            _copy_weight(child, new, transpose=False)
            setattr(module, name, Conv1dAs2d(new))
            replaced += 1
        elif isinstance(child, nn.ConvTranspose1d):
            new = nn.ConvTranspose2d(
                child.in_channels, child.out_channels,
                kernel_size=(1, child.kernel_size[0]),
                stride=(1, child.stride[0]) if isinstance(child.stride, tuple) else (1, child.stride),
                padding=(0, child.padding[0]) if isinstance(child.padding, tuple) else (0, child.padding),
                output_padding=(0, child.output_padding[0]) if isinstance(child.output_padding, tuple) else (0, child.output_padding),
                groups=child.groups, bias=child.bias is not None,
                dilation=(1, child.dilation[0]) if isinstance(child.dilation, tuple) else (1, child.dilation),
            )
            new = new.to(device=child.weight.device, dtype=child.weight.dtype)
            _copy_weight(child, new, transpose=True)
            setattr(module, name, ConvTranspose1dAs2d(new))
            replaced += 1
        else:
            replaced += convert_module(child)
    return replaced


def convert_model(model: nn.Module, name: str = "") -> nn.Module:
    n = convert_module(model)
    if n:
        print(f"[conv2d] {name or type(model).__name__}: заменено {n} слоёв", flush=True)
    return model
