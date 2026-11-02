"""Конвертация существующего кэша Piper в компактный формат (один файл на реплику).

Читает <id>.audio.pt (float32) + <id>.phonemes.pt, пишет <id>.compact.pt
с ключами audio_i16 / audio_dtype / phonemes / sr — ровно как ждёт пропатченный
piper/train/vits/dataset.py (PIPER_COMPACT=1).

Запуск: python convert_compact.py <cache_dir> [--workers 16] [--limit N]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

SCALE = 32767.0
SR = 22050


def convert_one(audio_path_str: str) -> tuple[int, int, str | None]:
    audio_path = Path(audio_path_str)
    cid = audio_path.name[: -len(".audio.pt")]
    out = audio_path.with_name(cid + ".compact.pt")
    phon = audio_path.with_name(cid + ".phonemes.pt")
    if out.exists() and out.stat().st_size > 0:
        return (1, 0, None)
    try:
        audio = torch.load(audio_path, map_location="cpu")
        ids = torch.load(phon, map_location="cpu")
        i16 = (audio.detach().float() * SCALE).round().clamp(-32767, 32767).to(torch.int16)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        torch.save(
            {"audio_i16": i16, "audio_dtype": "int16", "phonemes": ids, "sr": SR},
            tmp,
        )
        os.replace(tmp, out)
        return (0, out.stat().st_size, None)
    except Exception as e:  # noqa: BLE001
        return (2, 0, f"{cid}: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    files = sorted(str(p) for p in cache.glob("*.audio.pt"))
    if args.limit:
        files = files[: args.limit]
    print(f"к конвертации: {len(files)}", flush=True)

    t0 = time.time()
    done = skipped = errors = 0
    total_bytes = 0
    err_samples = []
    with Pool(args.workers) as pool:
        for i, (code, size, err) in enumerate(pool.imap_unordered(convert_one, files, chunksize=64), 1):
            if code == 0:
                done += 1
                total_bytes += size
            elif code == 1:
                skipped += 1
            else:
                errors += 1
                if len(err_samples) < 5:
                    err_samples.append(err)
            if i % 20000 == 0:
                el = time.time() - t0
                print(f"{i}/{len(files)} за {el:.0f} с ({i/el:.0f}/с)", flush=True)

    el = time.time() - t0
    print(f"готово: новых {done}, пропущено {skipped}, ошибок {errors}", flush=True)
    print(f"объём новых: {total_bytes/2**30:.1f} ГиБ, время {el:.0f} с ({done/max(el,1)*60:.0f}/мин)", flush=True)
    for e in err_samples:
        print("  ошибка:", e, flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
