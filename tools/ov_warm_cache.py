"""Прогрев дискового кэша ядер OpenVINO под сетку длин.

Для каждой длины из сетки (по умолчанию 64..768 с шагом 64) выполняется один
инференс: OpenVINO компилирует ядра под форму и кладёт их в CACHE_DIR. После
этого первое же реальное обращение к любой длине из сетки идёт без компиляции.

Запуск:
    <venv>/Scripts/python.exe ov_warm_cache.py --model <onnx> --cache <dir>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from piper import PiperVoice


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="GPU.0")
    ap.add_argument("--step", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(args.model)
    core = ov.Core()
    model = core.read_model(args.model)
    t0 = time.perf_counter()
    compiled = core.compile_model(
        model, args.device,
        {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": args.cache,
         "INFERENCE_PRECISION_HINT": "f16"},
    )
    print(f"компиляция графа: {(time.perf_counter() - t0) * 1000:.0f} мс", flush=True)
    names = {i.get_any_name() for i in compiled.inputs}

    total = 0.0
    lens = list(range(args.step, args.max_len + 1, args.step))
    for n in lens:
        ids = np.ones((1, n), dtype=np.int64)
        feed = {
            "input": ids,
            "input_lengths": np.array([n], dtype=np.int64),
            "scales": np.array([0.667, 1 / 1.15, 0.8], dtype=np.float32),
        }
        if "sid" in names:
            feed["sid"] = np.array([0], dtype=np.int64)
        t = time.perf_counter()
        compiled(feed)
        dt = (time.perf_counter() - t) * 1000
        total += dt
        print(f"  длина {n:4d}: {dt:7.1f} мс", flush=True)

    # контрольный повтор первой длины: должен быть уже из кэша
    n = lens[0]
    ids = np.ones((1, n), dtype=np.int64)
    feed = {"input": ids, "input_lengths": np.array([n], dtype=np.int64),
            "scales": np.array([0.667, 1 / 1.15, 0.8], dtype=np.float32)}
    if "sid" in names:
        feed["sid"] = np.array([0], dtype=np.int64)
    t = time.perf_counter()
    compiled(feed)
    repeat_ms = (time.perf_counter() - t) * 1000

    print(f"\nформ прогрето: {len(lens)}, суммарно {total / 1000:.1f} с, "
          f"повтор длины {n}: {repeat_ms:.1f} мс")
    print(f"кэш: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
