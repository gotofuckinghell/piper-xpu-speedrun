"""Параллельная сборка кэша Piper: 16 процессов, непрерывные шарды, общий cache_dir.

Ключ кэша = row_number + "_" + текст, поэтому шарду передаётся PIPER_ROW_OFFSET,
чтобы номера строк совпали с полным CSV.
"""
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
WORKER = Path(r"<PROJECT>\tools\piper\cache_worker.py")
BASE = Path(r"<WORK>")
DATA = Path(r"<PROJECT>\datasets\piper_yt")
CSV = Path(os.environ.get("CSV_PATH", str(DATA / "dataset_v30" / "metadata_nohdr.csv")))
AUDIO = Path(os.environ.get("AUDIO_DIR", str(DATA / "dataset_v20" / "wavs")))
CONFIG = BASE / "mm_v13_config.json"
CACHE = Path(os.environ.get("CACHE_DIR", str(BASE / "cache_v30")))
CKPT = Path(r"<PROJECT>\voice\<VOICE_DIR>\ckpt\v18_step0.ckpt")
WORK = Path(os.environ.get("WORK_DIR", str(BASE / "cache_par")))
NPROC = int(os.environ.get("NPROC", "16"))

rows = []
with open(CSV, encoding="utf-8", newline="") as f:
    for row in csv.reader(f, delimiter="|"):
        rows.append(row)
total = len(rows)
chunk = math.ceil(total / NPROC)
WORK.mkdir(parents=True, exist_ok=True)
print(f"строк {total}, шардов {NPROC}, по {chunk}", flush=True)

procs = []
for i in range(NPROC):
    part = rows[i * chunk:(i + 1) * chunk]
    if not part:
        continue
    shard = WORK / f"shard_{i:02d}.csv"
    with open(shard, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|").writerows(part)
    env = dict(os.environ)
    env.update(
        PIPER_ROW_OFFSET=str(i * chunk),
        PIPER_CACHE_ONLY="1",
        PIPER_NO_MOS="1",
        PIPER_NO_AUDIO_LOG="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        KMP_BLOCKTIME="0",
    )
    cmd = [
        PY, str(WORKER), "fit",
        "--data.voice_name", "<VOICE>",
        "--data.csv_path", str(shard),
        "--data.audio_dir", str(AUDIO),
        "--model.sample_rate", "22050",
        "--data.espeak_voice", "ru",
        "--data.cache_dir", str(CACHE),
        "--data.config_path", str(CONFIG),
        "--data.batch_size", "20",
        "--data.num_workers", "0",
        "--data.validation_split", "0.02",
        "--model.warmstart_ckpt", str(CKPT),
        "--trainer.default_root_dir", str(WORK / f"run_{i:02d}"),
        "--trainer.enable_progress_bar", "false",
        "--trainer.enable_model_summary", "false",
        "--trainer.max_steps", "1",
        "--trainer.limit_train_batches", "1",
        "--trainer.limit_val_batches", "0",
        "--trainer.num_sanity_val_steps", "0",
    ]
    log = open(WORK / f"shard_{i:02d}.log", "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(BASE))
    procs.append((i, p, log))
    print(f"старт шарда {i:02d}: строки {i*chunk+1}..{i*chunk+len(part)}", flush=True)

t0 = time.time()
while True:
    alive = [x for x in procs if x[1].poll() is None]
    done = len(procs) - len(alive)
    files = sum(1 for _ in CACHE.glob("*")) if CACHE.exists() else 0
    el = time.time() - t0
    rate = files / el * 60 if el else 0
    print(f"[{el/60:5.1f} мин] завершено {done}/{len(procs)}, файлов {files}, "
          f"{rate:7.0f} файлов/мин", flush=True)
    if not alive:
        break
    time.sleep(60)

rc = {i: p.returncode for i, p, _ in procs}
print("коды:", rc, flush=True)
for i, _, log in procs:
    log.close()

bad = {i: c for i, c in rc.items() if c not in (0,)}
print("ненулевые:", bad if bad else "нет", flush=True)

counts = {}
for ext in ("txt", "phonemes.txt", "phonemes.pt", "audio.pt", "spec.pt"):
    counts[ext] = sum(1 for _ in CACHE.glob(f"*.{ext}"))
print("файлов по типам:", counts, flush=True)
print(f"итого {el/60:.1f} мин", flush=True)

if not os.environ.get("KEEP_SHARDS"):
    for i in range(NPROC):
        (WORK / f"shard_{i:02d}.csv").unlink(missing_ok=True)
print("шарды удалены" if not os.environ.get("KEEP_SHARDS") else "шарды оставлены", flush=True)
