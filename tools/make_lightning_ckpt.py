"""Собирает полноценный Lightning-чекпоинт из slim-версии v30.

Slim содержит только state_dict и hyper_parameters; экспортёру нужны служебные
ключи Lightning (pytorch-lightning_version, epoch, global_step, loops, ...).
Берём их из v29_best и подменяем веса на v30_slim.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

CKPT = Path(r"<PROJECT>/voice/<VOICE_DIR>/ckpt")
SLIM = CKPT / "v30_slim.ckpt"
TEMPLATE = CKPT / "v29_best.ckpt"
OUT = CKPT / "v30_slim_full.ckpt"

slim = torch.load(SLIM, map_location="cpu", weights_only=False)
tpl = torch.load(TEMPLATE, map_location="cpu", weights_only=False)

hp_slim = slim.get("hyper_parameters", {})
hp_tpl = tpl.get("hyper_parameters", {})
same = json.dumps(hp_slim, sort_keys=True, default=str) == json.dumps(hp_tpl, sort_keys=True, default=str)
print(f"state_dict: {len(slim['state_dict'])} параметров | hyper_parameters совпадают: {same}")

merged = dict(tpl)
merged["state_dict"] = slim["state_dict"]
merged["hyper_parameters"] = hp_slim or hp_tpl
merged["epoch"] = int(tpl.get("epoch", 0))
merged["global_step"] = 65000
merged.pop("optimizer_states", None)
merged.pop("lr_schedulers", None)
merged.pop("callbacks", None)

torch.save(merged, OUT)
print(f"сохранено: {OUT} ({OUT.stat().st_size / 2**20:.0f} МБ)")
print("ключи:", [k for k in merged if k != "state_dict"][:8])
