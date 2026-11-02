r"""Быстрый свип: 3 конфига по 60 шагов, два датчика одновременно.

Скорость берётся из точного тайминга Lightning (PIPER_TIMING steps=.. seconds=..),
загрузка — параллельно из xpu-smi dump и из счётчиков Windows (\GPU Engine, как в диспетчере задач).
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

sys.path.insert(0, r"<PROJECT>\tools")
from gpu_taskmgr import one_sample  # noqa: E402

S = Path(r"<WORK>")
PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
SMI = r"<PROJECT>\tools\xpu-smi\xpu-smi_2.3.0-8fd5dca_win_2025\xpu-smi.exe"
STEPS = 60
COUNTERS = [
    r"\GPU Engine(*)\Utilization Percentage",
    r"\GPU Adapter Memory(*)\Dedicated Usage",
    r"\GPU Adapter Memory(*)\Shared Usage",
]
CONFIGS = [(64, 2048), (96, 1024), (128, 512)]


def dump_loop(out, stop):
    try:
        p = subprocess.Popen(
            [SMI, "dump", "--device", "0", "--metrics", "UTILIZATION,MEMORY",
             "--interval", "1", "--number", "600", "-j"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:
        return
    try:
        for line in p.stdout:
            if stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)["metrics"]
            except Exception:
                continue
            out.append((float(m.get("utilization.gpu", 0) or 0),
                        float(m.get("memory.used", 0) or 0)))
    finally:
        p.kill()


def win_loop(out, stop):
    while not stop.is_set():
        try:
            s = one_sample(COUNTERS)
            if s:
                out.append((float(s.get("util_max", 0.0)),
                            sum(s.get("dedicated", [])) / 2**30,
                            sum(s.get("shared", [])) / 2**30))
        except Exception:
            pass
        time.sleep(1.0)


def stats(vals):
    if not vals:
        return (0, 0, 0, 0, 0)
    v = sorted(vals)
    n = len(v)
    return (round(sum(v) / n, 1), round(v[n // 2], 1), round(v[int(0.95 * (n - 1))], 1),
            round(sum(1 for x in v if x >= 90) / n * 100, 1),
            round(sum(1 for x in v if x < 5) / n * 100, 1))


def run(batch, seg):
    tag = f"q_{batch}x{seg}"
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
        "--trainer.max_steps", str(STEPS * 2), "--trainer.max_epochs", "40",
        "--trainer.val_check_interval", "10000", "--trainer.limit_val_batches", "0",
        "--trainer.num_sanity_val_steps", "0", "--trainer.log_every_n_steps", "10",
        "--trainer.default_root_dir", str(run_dir),
        "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false",
    ]
    dump, win = [], []
    stop = threading.Event()
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        t1 = threading.Thread(target=dump_loop, args=(dump, stop), daemon=True)
        t2 = threading.Thread(target=win_loop, args=(win, stop), daemon=True)
        t1.start(); t2.start()
        rc = proc.wait()
        stop.set()

    txt = log.read_text(encoding="utf-8", errors="replace")
    tm = re.search(r"PIPER_TIMING steps=(\d+) seconds=([\d.]+)", txt)
    mm = re.search(r"пик allocated (\d+) МиБ, пик reserved (\d+)", txt)
    steps, secs = (int(tm.group(1)), float(tm.group(2))) if tm else (0, 0.0)
    rate = steps * batch / secs if secs else 0.0

    du = stats([x[0] for x in dump])
    wu = stats([x[0] for x in win])
    shared = round(max((x[2] for x in win), default=0), 2)
    row = {
        "tag": tag, "batch": batch, "segment": seg, "rc": rc,
        "crash": "DEVICE_LOST" in txt,
        "utt_per_s": round(rate, 1), "steps": steps, "train_sec": round(secs, 1),
        "xpu_util_mean_p50_p95_ge90_lt5": du,
        "win_util_mean_p50_p95_ge90_lt5": wu,
        "shared_max_gb": shared,
        "peak_alloc": int(mm.group(1)) if mm else 0,
        "peak_reserved": int(mm.group(2)) if mm else 0,
        "total_sec": round(time.time() - t0),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    rows = []
    for b, s in CONFIGS:
        try:
            rows.append(run(b, s))
        except Exception as e:  # noqa: BLE001
            print(f"{b}x{s}: ошибка {type(e).__name__}: {e}", flush=True)
        time.sleep(4)
    (S / "quick_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== итог ===")
    for r in rows:
        print(f"{r['tag']:11s} {r['utt_per_s']:6.1f} репл/с | xpu util {r['xpu_util_mean_p50_p95_ge90_lt5']} | "
              f"win util {r['win_util_mean_p50_p95_ge90_lt5']} | shared {r['shared_max_gb']} ГБ | "
              f"VRAM {r['peak_alloc']}/{r['peak_reserved']} | rc={r['rc']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
