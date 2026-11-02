"""Матрица конфигураций для максимальной загрузки GPU (Piper VITS, XPU).

Для каждого варианта: прогон N шагов + потоковый сэмпл xpu-smi dump (1 Гц).
Считает: util средний/медиана/p95/доля ≥90%, реплик/с, пик VRAM (внутри процесса
берётся из лога lightning: «XPU память: ...»).
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
SMI = r"<PROJECT>\tools\xpu-smi\xpu-smi_2.3.0-8fd5dca_win_2025\xpu-smi.exe"
STEPS = 120

# (имя, batch, segment, num_workers, omp_на_процесс, кэш, примечание)
CONFIGS = [
    ("A_base_64x2048_nw0", 64, 2048, 0, 30, "cache_v30", "базовая: мел на лету, воркеров нет"),
    ("B_64x2048_nw4", 64, 2048, 4, 4, "cache_v30", "4 воркера готовят батчи"),
    ("C_64x2048_nw8", 64, 2048, 8, 2, "cache_v30", "8 воркеров"),
    ("D_96x1024_nw4", 96, 1024, 4, 4, "cache_v30", "батч 96, сегмент 1024, 4 воркера"),
    ("E_96x1024_nw8", 96, 1024, 8, 2, "cache_v30", "батч 96, сегмент 1024, 8 воркеров"),
]


def run_one(tag, batch, seg, workers, omp, cache, note):
    run_dir = S / f"run_mtx_{tag}"
    log = S / f"mtx_{tag}.log"
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": str(omp), "MKL_NUM_THREADS": str(omp),
        "OPENBLAS_NUM_THREADS": str(omp), "NUMEXPR_NUM_THREADS": str(omp),
        "PYTORCH_XPU_ALLOC_CONF": "garbage_collection_threshold:0.80",
        "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
        "PIPER_SIBILANT": "1", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
        "PIPER_COMPACT": "1", "PIPER_AMP": "bf16", "PIPER_XPU_MEM_FRACTION": "1.0",
    })
    cmd = [
        PY, "-m", "piper.train", "fit",
        "--data.voice_name", f"mtx_{tag}",
        "--data.csv_path", r"<PROJECT>\datasets\piper_yt\dataset_v30\metadata_nohdr.csv",
        "--data.audio_dir", r"<PROJECT>\datasets\piper_yt\dataset_v20\wavs",
        "--model.sample_rate", "22050", "--data.espeak_voice", "ru",
        "--data.cache_dir", str(S / cache),
        "--data.config_path", str(S / "mm_v13_config.json"),
        "--data.batch_size", str(batch), "--data.num_workers", str(workers),
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
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        dump = subprocess.Popen(
            [SMI, "dump", "--device", "0", "--metrics", "MEMORY,UTILIZATION,POWER",
             "--interval", "1", "--number", "900", "-j"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        samples = []
        try:
            for line in dump.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)["metrics"]
                except Exception:
                    continue
                samples.append({
                    "t": round(time.time() - t0, 1),
                    "util": float(m.get("utilization.gpu", 0) or 0),
                    "vram": float(m.get("memory.used", 0) or 0),
                    "power": float(m.get("power.draw", 0) or 0),
                })
                if proc.poll() is not None:
                    break
        finally:
            dump.kill()
            rc = proc.wait()

    txt = log.read_text(encoding="utf-8", errors="replace")
    crash = "DEVICE_LOST" in txt
    m = re.search(r"пик allocated (\d+) МиБ, пик reserved (\d+)", txt)
    peak_alloc, peak_res = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    utils = [s["util"] for s in samples]
    steam_time = sum(1 for u in utils if u >= 90) / len(utils) * 100 if utils else 0
    rate = 0.0
    try:
        import glob as g
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        fs = g.glob(str(run_dir / "**" / "events.out.tfevents.*"), recursive=True)
        if fs:
            ea = EventAccumulator(fs[0]); ea.Reload()
            it = ea.Scalars("iters")
            n = it[-1].value - it[0].value
            dt = it[-1].wall_time - it[0].wall_time
            rate = n * batch / dt if dt else 0.0
    except Exception:
        pass

    row = {
        "tag": tag, "note": note, "batch": batch, "segment": seg,
        "workers": workers, "rc": rc, "crash": int(crash),
        "util_mean": round(sum(utils) / len(utils), 1) if utils else 0,
        "util_p50": round(sorted(utils)[len(utils) // 2], 1) if utils else 0,
        "util_p95": round(sorted(utils)[int(0.95 * (len(utils) - 1))], 1) if utils else 0,
        "util_ge90_pct": round(steam_time, 1),
        "utt_per_s": round(rate, 1),
        "peak_alloc_mib": peak_alloc, "peak_reserved_mib": peak_res,
        "seconds": round(time.time() - t0, 0),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    rows = []
    for cfg in CONFIGS:
        try:
            rows.append(run_one(*cfg))
        except Exception as e:  # noqa: BLE001
            print(f"{cfg[0]}: ОШИБКА {type(e).__name__}: {e}", flush=True)
        time.sleep(5)
    out = S / "util_matrix.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== сводка ===")
    print(f"{'конфиг':24s} {'util ср':>7s} {'p50':>5s} {'p95':>5s} {'≥90%':>5s} {'репл/с':>7s} {'VRAM пик':>9s}")
    for r in rows:
        print(f"{r['tag']:24s} {r['util_mean']:7.1f} {r['util_p50']:5.1f} {r['util_p95']:5.1f} "
              f"{r['util_ge90_pct']:5.1f} {r['utt_per_s']:7.1f} {r['peak_alloc_mib']:9d}")
    print("json:", out)


if __name__ == "__main__":
    sys.exit(main())
