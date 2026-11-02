"""Перебор (batch, segment) для Piper: находит максимальный сегмент при заданном батче.

Каждый прогон: 30 шагов, bf16-автокаст, 30 потоков, компакт-кэш с диска.
Пишет строку результата в trials_results.csv и печатает сводку.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S = Path(r"<WORK>")
PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
CSV = S / "trials_results.csv"
STEPS = 30

TRIALS = [
    (128, 512), (128, 1024), (128, 2048),
    (96, 1024), (96, 2048), (96, 4096),
    (64, 2048), (64, 4096),
]

BASE_ENV = {
    "OMP_NUM_THREADS": "30", "MKL_NUM_THREADS": "30",
    "OPENBLAS_NUM_THREADS": "30", "NUMEXPR_NUM_THREADS": "30",
    "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
    "PIPER_SIBILANT": "1", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
    "PIPER_COMPACT": "1", "PIPER_AMP": "bf16",
}


def throughput(run_dir: Path, batch: int):
    fs = glob.glob(str(run_dir / "**" / "events.out.tfevents.*"), recursive=True)
    if not fs:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(fs[0]); ea.Reload()
        it = ea.Scalars("iters")
    except Exception:
        return None
    n = it[-1].value - it[0].value
    dt = it[-1].wall_time - it[0].wall_time
    if n <= 0 or dt <= 0:
        return None
    return round(n * batch / dt, 1)


def main():
    rows = []
    for batch, seg in TRIALS:
        tag = f"b{batch}s{seg}"
        run = S / f"run_trial_{tag}"
        log = S / f"trial_{tag}.log"
        if run.exists():
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(run)], capture_output=True)
        env = dict(os.environ); env.update(BASE_ENV)
        cmd = [
            PY, "-m", "piper.train", "fit",
            "--data.voice_name", tag,
            "--data.csv_path", r"<PROJECT>\datasets\piper_yt\dataset_v30\metadata_nohdr.csv",
            "--data.audio_dir", r"<PROJECT>\datasets\piper_yt\dataset_v20\wavs",
            "--model.sample_rate", "22050", "--data.espeak_voice", "ru",
            "--data.cache_dir", str(S / "cache_v30"),
            "--data.config_path", str(S / "mm_v13_config.json"),
            "--data.batch_size", str(batch), "--data.num_workers", "0",
            "--data.validation_split", "0.02", "--data.length_bucketing", "true",
            "--data.bucket_multiplier", "16",
            "--model.segment_size", str(seg), "--model.p_dropout", "0.10",
            "--model.warmstart_ckpt",
            r"<PROJECT>\voice\<VOICE_DIR>\ckpt\v30_slim.ckpt",
            "--trainer.max_steps", str(STEPS), "--trainer.max_epochs", "40",
            "--trainer.val_check_interval", "1000", "--trainer.limit_val_batches", "0",
            "--trainer.num_sanity_val_steps", "0", "--trainer.log_every_n_steps", "5",
            "--trainer.default_root_dir", str(run),
            "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false",
        ]
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        el = time.time() - t0
        txt = log.read_text(encoding="utf-8", errors="replace")
        crashed = ("DEVICE_LOST" in txt) or ("out of memory" in txt.lower()) or rc != 0
        rate = None if crashed else throughput(run, batch)
        row = {
            "batch": batch, "segment": seg, "rc": rc,
            "crashed": int(crashed), "utt_per_s": rate or "",
            "seconds": round(el, 1),
            "reason": "DEVICE_LOST" if "DEVICE_LOST" in txt else ("rc" if rc else ""),
        }
        rows.append(row)
        new = not CSV.exists()
        with CSV.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{tag:12s} {'ПАДЕНИЕ ' + row['reason'] if crashed else str(rate) + ' реплик/с':24s} "
              f"{el:6.0f} с", flush=True)
        if crashed and seg >= 1024:
            print(f"  батч {batch}: дальше не пробую, упёрлись", flush=True)
        time.sleep(5)

    print("\nсводка:")
    for r in rows:
        print(f"  {r['batch']}x{r['segment']:<5} {'ПАДЕНИЕ' if r['crashed'] else str(r['utt_per_s']) + ' репл/с'}")


if __name__ == "__main__":
    sys.exit(main())
