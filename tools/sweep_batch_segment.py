"""Свип конфигураций Piper по критерию «реплик в секунду при загрузке GPU».

Каждый конфиг: 100 шагов, одинаковый чекпоинт и порядок данных.
Меряет: реплик/с, util (среднее/медиана/p95/доля ≥90 %/<5 %), пик VRAM изнутри.
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

S = Path(r"<WORK>")
PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
SMI = r"<PROJECT>\tools\xpu-smi\xpu-smi_2.3.0-8fd5dca_win_2025\xpu-smi.exe"
STEPS = 100

CONFIGS = [
    (64, 2048, "эталон"),
    (64, 1024, ""),
    (80, 1024, ""),
    (96, 1024, ""),
    (96, 512, ""),
]


def sample_loop(samples, stop):
    try:
        proc = subprocess.Popen(
            [SMI, "dump", "--device", "0", "--metrics", "MEMORY,UTILIZATION,POWER",
             "--interval", "1", "--number", "1200", "-j"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:
        return
    try:
        for line in proc.stdout:
            if stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)["metrics"]
            except Exception:
                continue
            samples.append({
                "util": float(m.get("utilization.gpu", 0) or 0),
                "vram": float(m.get("memory.used", 0) or 0),
                "free": float(m.get("memory.free", 0) or 0),
                "power": float(m.get("power.draw", 0) or 0),
            })
    finally:
        proc.kill()


def run(batch, seg, note):
    tag = f"sw_{batch}x{seg}"
    run_dir = S / f"run_{tag}"
    log = S / f"{tag}.log"
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "30", "MKL_NUM_THREADS": "30",
        "OPENBLAS_NUM_THREADS": "30", "NUMEXPR_NUM_THREADS": "30",
        "PYTORCH_XPU_ALLOC_CONF": "garbage_collection_threshold:0.80",
        "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
        "PIPER_SIBILANT": "1", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
        "PIPER_COMPACT": "1", "PIPER_AMP": "bf16",
    })
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
        "--trainer.val_check_interval", "10000", "--trainer.limit_val_batches", "0",
        "--trainer.num_sanity_val_steps", "0", "--trainer.log_every_n_steps", "10",
        "--trainer.default_root_dir", str(run_dir),
        "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false",
    ]
    samples = []
    stop = threading.Event()
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        th = threading.Thread(target=sample_loop, args=(samples, stop), daemon=True)
        th.start()
        rc = proc.wait()
        stop.set()
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"пик allocated (\d+) МиБ, пик reserved (\d+)", txt)
    peak = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    utils = [s["util"] for s in samples]
    rate = 0.0
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        fs = glob.glob(str(run_dir / "**" / "events.out.tfevents.*"), recursive=True)
        if fs:
            ea = EventAccumulator(fs[0]); ea.Reload()
            it = ea.Scalars("iters")
            n = it[-1].value - it[0].value; dt = it[-1].wall_time - it[0].wall_time
            rate = n * batch / dt if dt else 0.0
    except Exception:
        pass
    row = {
        "tag": tag, "note": note, "batch": batch, "segment": seg, "rc": rc,
        "crash": "DEVICE_LOST" in txt,
        "utt_per_s": round(rate, 1),
        "util_mean": round(sum(utils) / len(utils), 1) if utils else 0,
        "util_p50": round(sorted(utils)[len(utils) // 2], 1) if utils else 0,
        "util_p95": round(sorted(utils)[int(0.95 * (len(utils) - 1))], 1) if utils else 0,
        "util_ge90": round(sum(1 for u in utils if u >= 90) / len(utils) * 100, 1) if utils else 0,
        "util_lt5": round(sum(1 for u in utils if u < 5) / len(utils) * 100, 1) if utils else 0,
        "peak_alloc": peak[0], "peak_reserved": peak[1],
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    rows = []
    for cfg in CONFIGS:
        try:
            rows.append(run(*cfg))
        except Exception as e:  # noqa: BLE001
            print(f"{cfg}: ошибка {type(e).__name__}: {e}", flush=True)
        time.sleep(4)
    (S / "sweep_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== свип ===", flush=True)
    for r in rows:
        print(f"{r['tag']:12s} {r['utt_per_s']:6.1f} репл/с | util {r['util_mean']:5.1f}/{r['util_p50']:5.1f}/{r['util_p95']:5.1f}"
              f" | ≥90 %: {r['util_ge90']:5.1f} | <5 %: {r['util_lt5']:5.1f} | VRAM {r['peak_alloc']}/{r['peak_reserved']}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
