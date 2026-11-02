r"""Замер загрузки GPU на финальном конфиге (bf16 + ckpt + batch 112).

Сэмплит счётчики Windows раз в секунду параллельно с обучением.
"""
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
V33 = r"<PROJECT>\datasets\piper_yt\dataset_v33"
COUNTERS = [
    r"\GPU Engine(*)\Utilization Percentage",
    r"\GPU Adapter Memory(*)\Dedicated Usage",
    r"\GPU Adapter Memory(*)\Shared Usage",
]
STEPS = 40


def main():
    tag = "final112"
    log = S / f"{tag}.log"
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "30", "MKL_NUM_THREADS": "30",
        "OPENBLAS_NUM_THREADS": "30", "NUMEXPR_NUM_THREADS": "30",
        "PYTORCH_XPU_ALLOC_CONF": "garbage_collection_threshold:0.80",
        "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
        "PIPER_AMP": "bf16", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
        "PIPER_CKPT": "1",
    })
    cmd = [
        PY, "-m", "piper.train", "fit",
        "--data.voice_name", tag,
        "--data.csv_path", V33 + r"\metadata_nohdr.csv",
        "--data.audio_dir", V33 + r"\wavs",
        "--model.sample_rate", "22050", "--data.espeak_voice", "ru",
        "--data.cache_dir", str(S / "cache_v33"),
        "--data.config_path", str(S / "mm_v13_config.json"),
        "--data.batch_size", "112", "--data.num_workers", "0",
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
    samples = []
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                s = one_sample(COUNTERS)
                if s:
                    samples.append(float(s.get("util_max", 0.0)))
            except Exception:
                pass
            time.sleep(1.0)

    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        rc = proc.wait()
        stop.set()

    txt = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"PIPER_TIMING steps=(\d+) seconds=([\d.]+)", txt)
    steps, secs = (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)
    rate = steps * 112 / secs if secs else 0.0
    if samples:
        ss = sorted(samples)
        print(json.dumps({
            "rc": rc, "utt_per_s": round(rate, 1),
            "util_mean": round(sum(samples) / len(samples), 1),
            "util_p50": round(ss[len(ss) // 2], 1),
            "util_p95": round(ss[int(0.95 * (len(ss) - 1))], 1),
            "util_ge90_pct": round(sum(1 for u in ss if u >= 90) / len(ss) * 100, 1),
            "samples": len(ss),
        }, ensure_ascii=False))
    else:
        print("нет сэмплов")


if __name__ == "__main__":
    sys.exit(main())
