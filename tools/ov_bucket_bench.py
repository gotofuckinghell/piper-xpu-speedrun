"""Замер native OpenVINO 2026.4 на Arc: динамические формы против сетки длин.

Идея: длина входа VITS = число фонем, поэтому каждая новая длина вызывает
компиляцию ядер (~2,9 с). Если добивать последовательность до фиксированной
сетки (например, кратной 64) и передавать input_lengths = реальная длина,
число форм конечно, все они прогреваются при старте.

Скрипт проверяет:
  1) совпадает ли аудио при добивке с аудио без добивки;
  2) сколько стоит первое касание каждой формы;
  3) установившуюся задержку на текстах разной длины.

Запуск:  <venv>/Scripts/python.exe ov_bucket_bench.py [--grid 64] [--device GPU.0]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov
from piper import PiperVoice

MODEL = r"<WORK>/piper-voices/ru_RU-april420-medium.onnx"
CACHE = r"<PROJECT>/venvs/piper-ov/ov_cache"
OUT = Path(r"<WORK>/ov_bucket.json")

TEXTS = [
    "Проверка связи.",
    "Сегодня я проверила два движка распознавания речи на одной и той же видеокарте, и разница оказалась заметнее, чем я ожидала.",
    "Короткая фраза для теста.",
    "Этот текст заметно длиннее предыдущих и нужен для того, чтобы проверить поведение синтезатора на длинной последовательности фонем, где форма входа отличается от всех предыдущих случаев.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--device", default="GPU.0")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    voice = PiperVoice.load(MODEL)
    sr = voice.config.sample_rate
    core = ov.Core()
    model = core.read_model(MODEL)
    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device,
                                  {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": CACHE,
                                   "INFERENCE_PRECISION_HINT": "f16"})
    print(f"compile: {(time.perf_counter() - t0) * 1000:.0f} мс, устройство {args.device}, sr {sr}", flush=True)
    names = {i.get_any_name() for i in compiled.inputs}
    print("входы:", sorted(names), flush=True)

    def run(text: str, pad: bool):
        ph = voice.phonemize(text)
        ids = np.array(voice.phonemes_to_ids(ph[0]), dtype=np.int64)[None, :]
        real = ids.shape[1]
        if pad:
            target = ((real + args.grid - 1) // args.grid) * args.grid
            padded = np.full((1, target), 0, dtype=np.int64)
            padded[0, :real] = ids[0]
            feed_ids = padded
        else:
            feed_ids = ids
        feed = {
            "input": feed_ids,
            "input_lengths": np.array([real], dtype=np.int64),
            "scales": np.array([0.667, 1 / 1.15, 0.8], dtype=np.float32),
        }
        if "sid" in names:
            feed["sid"] = np.array([0], dtype=np.int64)
        t = time.perf_counter()
        out = compiled(feed)
        dt = time.perf_counter() - t
        audio = np.squeeze(next(iter(out.values()))).astype(np.float32)
        return audio, dt, real

    results = {"device": args.device, "grid": args.grid, "openvino": ov.__version__, "rows": []}

    # 1. Совпадает ли аудио при добивке
    for text in TEXTS[:2]:
        a_raw, _, n_raw = run(text, pad=False)
        a_pad, _, n_pad = run(text, pad=True)
        m = min(len(a_raw), len(a_pad))
        corr = float(np.corrcoef(a_raw[:m], a_pad[:m])[0, 1]) if m > 100 else 0.0
        results["rows"].append({"check": "pad_equivalence", "text_len": len(text),
                                "phonemes": n_raw, "padded": n_pad,
                                "len_raw": int(len(a_raw)), "len_pad": int(len(a_pad)),
                                "corr": round(corr, 6)})
        print(f"добивка {n_raw}->{n_pad}: len {len(a_raw)}/{len(a_pad)}, corr {corr:.6f}", flush=True)

    # 2. Прогрев сетки: первое касание каждой формы
    first = []
    for n in range(args.grid, 512 + args.grid, args.grid):
        ids = np.ones((1, n), dtype=np.int64)
        feed = {"input": ids, "input_lengths": np.array([n], dtype=np.int64),
                "scales": np.array([0.667, 1 / 1.15, 0.8], dtype=np.float32)}
        if "sid" in names:
            feed["sid"] = np.array([0], dtype=np.int64)
        t = time.perf_counter()
        compiled(feed)
        dt = (time.perf_counter() - t) * 1000
        first.append({"shape": n, "ms": round(dt, 1)})
        print(f"  форма {n:4d}: {dt:7.1f} мс", flush=True)
    results["first_touch"] = first

    # 3. Установившаяся задержка на текстах разной длины
    steady = []
    for text in TEXTS:
        times_raw, times_pad = [], []
        for _ in range(args.repeats):
            _, dt, _ = run(text, pad=False)
            times_raw.append(dt * 1000)
        for _ in range(args.repeats):
            _, dt, _ = run(text, pad=True)
            times_pad.append(dt * 1000)
        steady.append({"text_len": len(text),
                       "raw_ms": round(statistics.median(times_raw), 1),
                       "padded_ms": round(statistics.median(times_pad), 1)})
        print(f"текст {len(text):3d} симв: raw {statistics.median(times_raw):6.1f} мс | "
              f"padded {statistics.median(times_pad):6.1f} мс", flush=True)
    results["steady"] = steady

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
