r"""Два параллельных процесса на одной карте: заполняем паузы между шагами.

GPU занят ~41 %, значит между ядрами есть простой. Два процесса с батчем 32
могут заполнить его лучше, чем один с 64. Меряем суммарные реплики/с и VRAM.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

S = Path(r"<WORK>")
PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
V33 = r"<PROJECT>\datasets\piper_yt\dataset_v33"
STEPS = 40


def launch(tag: str, batch: int, ckpt: bool):
    log = S / f"{tag}.log"
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "15", "MKL_NUM_THREADS": "15",
        "OPENBLAS_NUM_THREADS": "15", "NUMEXPR_NUM_THREADS": "15",
        "PYTORCH_XPU_ALLOC_CONF": "garbage_collection_threshold:0.80",
        "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
        "PIPER_AMP": "bf16", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
    })
    if ckpt:
        env["PIPER_CKPT"] = "1"
    cmd = [
        PY, "-m", "piper.train", "fit",
        "--data.voice_name", tag,
        "--data.csv_path", V33 + r"\metadata_nohdr.csv",
        "--data.audio_dir", V33 + r"\wavs",
        "--model.sample_rate", "22050", "--data.espeak_voice", "ru",
        "--data.cache_dir", str(S / "cache_v33"),
        "--data.config_path", str(S / "mm_v13_config.json"),
        "--data.batch_size", str(batch), "--data.num_workers", "0",
        "--data.validation_split", "0.02", "--data.length_bucketing", "true",
        "--data.bucket_multiplier", "16",
        "--model.segment_size", "2048", "--model.p_dropout", "0.10",
        "--model.warmstart_ckpt",
        r"<PROJECT>\voice\<VOICE_DIR>\ckpt\v30_slim.ckpt",
        "--trainer.max_steps", str(STEPS * 2), "--trainer.max_epochs", "40",
        "--trainer.val_check_interval", "10000", "--trainer.limit_val_batches", "0",
        "--trainer.num_sanity_val_steps", "0", "--trainer.log_every_n_steps", "20",
        "--trainer.default_root_dir", str(S / f"run_{tag}"),
        "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false",
    ]
    fh = open(log, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, log


def rate_of(log: Path, batch: int):
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"PIPER_TIMING steps=(\d+) seconds=([\d.]+)", txt)
    if not m:
        return 0.0, "DEVICE_LOST" in txt
    steps, secs = int(m.group(1)), float(m.group(2))
    return steps * batch / secs, False


def main():
    results = []
    # 1) два процесса по 32
    procs = [launch("dual_a", 32, False), launch("dual_b", 32, False)]
    t0 = time.time()
    for p, fh, _ in procs:
        p.wait(); fh.close()
    total = 0.0
    for tag, (p, fh, log) in zip(("dual_a", "dual_b"), procs):
        r, crash = rate_of(log, 32)
        total += r
        print(f"{tag}: {r:.1f} репл/с crash={crash}", flush=True)
    print(f"суммарно два×32: {total:.1f} репл/с за {time.time()-t0:.0f} с", flush=True)
    results.append({"mode": "dual_32", "rate": round(total, 2)})

    # 2) два процесса по 48 с чекпоинтингом
    procs = [launch("dual_c", 48, True), launch("dual_d", 48, True)]
    t0 = time.time()
    for p, fh, _ in procs:
        p.wait(); fh.close()
    total = 0.0
    for tag, (p, fh, log) in zip(("dual_c", "dual_d"), procs):
        r, crash = rate_of(log, 48)
        total += r
        print(f"{tag}: {r:.1f} репл/с crash={crash}", flush=True)
    print(f"суммарно два×48 ckpt: {total:.1f} репл/с за {time.time()-t0:.0f} с", flush=True)
    results.append({"mode": "dual_48_ckpt", "rate": round(total, 2)})

    (S / "dual_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
