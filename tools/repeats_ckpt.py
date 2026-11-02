r"""Повторные замеры ключей: 5 вариантов × 3 раунда, чередование (ABBA).

Каждый прогон — 40 батчей на нарезанном корпусе v33, точный тайминг из PIPER_TIMING.
Итог: среднее, минимум, максимум по каждому варианту.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

S = Path(r"<WORK>")
PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
V33 = r"<PROJECT>\datasets\piper_yt\dataset_v33"
STEPS = 40  # батчей; global_step = STEPS*2

BASE_ENV = {
    "OMP_NUM_THREADS": "30", "MKL_NUM_THREADS": "30",
    "OPENBLAS_NUM_THREADS": "30", "NUMEXPR_NUM_THREADS": "30",
    "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
    "PIPER_AMP": "bf16", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
    "PIPER_TIMERS": "0",
}

ARMS = [
    {"name": "base64", "batch": 64, "seg": 2048, "env": {}},
    {"name": "ckpt96", "batch": 96, "seg": 2048, "env": {"PIPER_CKPT": "1"}},
    {"name": "ckpt112", "batch": 112, "seg": 2048, "env": {"PIPER_CKPT": "1"}},
]

ROUNDS = 3


def run_arm(arm, run_idx):
    tag = f"rp_{arm['name']}_{run_idx}"
    run_dir = S / f"run_{tag}"
    log = S / f"{tag}.log"
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(arm["env"])
    cmd = [
        PY, "-m", "piper.train", "fit",
        "--data.voice_name", tag,
        "--data.csv_path", V33 + r"\metadata_nohdr.csv",
        "--data.audio_dir", V33 + r"\wavs",
        "--model.sample_rate", "22050", "--data.espeak_voice", "ru",
        "--data.cache_dir", str(S / "cache_v33"),
        "--data.config_path", str(S / "mm_v13_config.json"),
        "--data.batch_size", str(arm["batch"]), "--data.num_workers", "0",
        "--data.validation_split", "0.02", "--data.length_bucketing", "true",
        "--data.bucket_multiplier", "16",
        "--model.segment_size", str(arm["seg"]), "--model.p_dropout", "0.10",
        "--model.warmstart_ckpt",
        r"<PROJECT>\voice\<VOICE_DIR>\ckpt\v30_slim.ckpt",
        "--trainer.max_steps", str(STEPS * 2), "--trainer.max_epochs", "40",
        "--trainer.val_check_interval", "10000", "--trainer.limit_val_batches", "0",
        "--trainer.num_sanity_val_steps", "0", "--trainer.log_every_n_steps", "10",
        "--trainer.default_root_dir", str(run_dir),
        "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false",
    ]
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"PIPER_TIMING steps=(\d+) seconds=([\d.]+)", txt)
    ok = bool(m) and "DEVICE_LOST" not in txt
    steps, secs = (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)
    rate = steps * arm["batch"] / secs if secs else 0.0
    return {"arm": arm["name"], "run": run_idx, "rc": rc, "ok": ok,
            "seconds": round(secs, 1), "utt_per_s": round(rate, 2)}


def main():
    results = []
    order = list(ARMS)
    for r in range(1, ROUNDS + 1):
        seq = order if r % 2 == 1 else list(reversed(order))
        for arm in seq:
            try:
                row = run_arm(arm, r)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                results.append(row)
            except Exception as e:  # noqa: BLE001
                print(f"{arm['name']}#{r}: ошибка {type(e).__name__}: {e}", flush=True)

    (S / "repeats_ckpt.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== сводка (3 раунда) ===", flush=True)
    print(f"{'вариант':16s} {'среднее':>8s} {'мин':>7s} {'макс':>7s} {'n':>3s}", flush=True)
    for arm in ARMS:
        vals = [r["utt_per_s"] for r in results if r["arm"] == arm["name"] and r["ok"]]
        if not vals:
            print(f"{arm['name']:16s} нет валидных прогонов", flush=True)
            continue
        print(f"{arm['name']:16s} {sum(vals)/len(vals):8.2f} {min(vals):7.2f} {max(vals):7.2f} {len(vals):3d}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
