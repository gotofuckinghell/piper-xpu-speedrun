"""Параллельная нарезка датасета: по процессу на файл (resample+slice — CPU/IO-тяжёлые).

Было: 8 роликов по очереди, каждый грузил 0.7-1.5 ГБ wav и ресемплил в один поток.
Стало: N процессов, каждый со своим лимитом потоков; wav читается один раз на файл.

  python build_dataset_words_parallel.py --align-dir DIR --audio-dir DIR --out-dir DIR \
      [--workers 4] [--min 3] [--max 9] [--pad 0.15] [--gap 0.35]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ALLOWED = re.compile(r"[^а-яa-z0-9\s-]")
SENT_END = (".", "!", "?", "…", ";", ":")


def clean(text: str) -> str:
    # ё НЕ заменяем: графема несёт ударение и тип слога, для TTS это важно (правка по ревью)
    t = text.lower()
    t = ALLOWED.sub(" ", t)
    t = re.sub(r"\s+-\s*", "-", t)
    return re.sub(r"\s+", " ", t).strip(" -")


def load_audio(path: str):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 22050:
        g = gcd(sr, 22050)
        mono = resample_poly(mono, 22050 // g, sr // g).astype(np.float32)
        sr = 22050
    return mono, sr


def build_chunks(words, min_dur, max_dur, gap_max):
    i, n = 0, len(words)
    while i < n:
        start_i = i
        start = words[i]["s"]
        last_end = words[i]["e"]
        text_parts = [words[i]["w"]]
        i += 1
        while i < n:
            w = words[i]
            gap = w["s"] - last_end
            cur_dur = last_end - start
            sentence_done = text_parts[-1].strip().endswith(SENT_END)
            if gap > gap_max or (w["e"] - start) > max_dur or (cur_dur >= min_dur and sentence_done):
                break
            text_parts.append(w["w"])
            last_end = w["e"]
            i += 1
        dur = last_end - start
        if dur < 0.6:
            continue
        text = clean(" ".join(text_parts))
        if len(text.split()) < 2:
            continue
        prev_end = words[start_i - 1]["e"] if start_i > 0 else 0.0
        next_start = words[i]["s"] if i < n else last_end + 10.0
        yield start, last_end, text, prev_end, next_start


def process_file(args):
    """Top-level: один ролик -> wav-куски + строки метаданных."""
    name, align_dir, audio_dir, wav_dir, min_d, max_d, pad, gap = args
    base = os.path.splitext(name)[0]
    with open(os.path.join(align_dir, name), encoding="utf-8") as fh:
        data = json.load(fh)
    src = None
    for ext in (".wav", ".webm", ".m4a", ".opus"):
        p = os.path.join(audio_dir, base + ext)
        if os.path.exists(p):
            src = p
            break
    if not src:
        return base, [], 0.0
    audio, sr = load_audio(src)
    words = [w for seg in data["segments"] for w in seg["words"]]
    rows, sec = [], 0.0
    for k, (start, end, text, prev_end, next_start) in enumerate(build_chunks(words, min_d, max_d, gap)):
        left = min(pad, max(0.0, (start - prev_end) / 2))
        right = min(pad, max(0.0, (next_start - end) / 2))
        a = max(0, int((start - left) * sr))
        b = min(len(audio), int((end + right) * sr))
        chunk = audio[a:b]
        fname = f"{base}_{k:05d}.wav"
        sf.write(os.path.join(wav_dir, fname), chunk, sr, subtype="PCM_16")
        rows.append((fname, text))
        sec += (b - a) / sr
    return base, rows, sec


def main() -> int:
    from parallel_audio import pmap

    ap = argparse.ArgumentParser()
    ap.add_argument("--align-dir", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min", type=float, default=3.0)
    ap.add_argument("--max", type=float, default=9.0)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--gap", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.align_dir) if f.endswith(".json"))
    jobs = [(n, args.align_dir, args.audio_dir, wav_dir, args.min, args.max, args.pad, args.gap) for n in names]
    t0 = time.time()
    results = pmap(process_file, jobs, workers=args.workers, threads_per_worker=args.threads)
    rows, sec = [], 0.0
    for base, r, s in results:
        rows.extend(r)
        sec += s
        print(f"{base}: {len(r)} chunks, {s/60:.1f} min", flush=True)
    csv_path = os.path.join(args.out_dir, "metadata.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="|", lineterminator="\n")
        w.writerow(["file", "text"])
        w.writerows(rows)
    print(f"chunks={len(rows)} hours={sec/3600:.2f} in {(time.time()-t0)/60:.1f} min "
          f"(workers={args.workers}) -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
