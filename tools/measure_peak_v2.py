"""Пик VRAM/утилизации на прогоне Piper через потоковый `xpu-smi dump`.

В отличие от `stats -j` (усреднение по движкам за долю секунды),
`dump` отдаёт по строке в секунду: utilization.gpu, memory.used/free, power.draw.
Именно по memory.used видно, когда карта упирается в 16 ГБ и уходит в shared.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

S = Path(r"<WORK>")
PY = r"<HOME>\.venvs\piper-main\Scripts\python.exe"
SMI = r"<PROJECT>\tools\xpu-smi\xpu-smi_2.3.0-8fd5dca_win_2025\xpu-smi.exe"


def run_training(batch, seg, tag, log_path, amp="bf16", extra_env=None, extra_flags=None):
    run = S / f"run_{tag}"
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "30", "MKL_NUM_THREADS": "30",
        "OPENBLAS_NUM_THREADS": "30", "NUMEXPR_NUM_THREADS": "30",
        "PIPER_XPU": "1", "PIPER_NO_MOS": "1", "PIPER_NO_AUDIO_LOG": "1",
        "PIPER_SIBILANT": "1", "PIPER_LR": "1e-5", "PIPER_LR_D": "1e-5",
        "PIPER_COMPACT": "1",
    })
    if amp:
        env["PIPER_AMP"] = amp
    if extra_env:
        env.update(extra_env)
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
        "--trainer.max_steps", "40", "--trainer.max_epochs", "40",
        "--trainer.val_check_interval", "1000", "--trainer.limit_val_batches", "0",
        "--trainer.num_sanity_val_steps", "0", "--trainer.log_every_n_steps", "5",
        "--trainer.default_root_dir", str(run),
        "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    fh = open(log_path, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    seg = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    amp = sys.argv[3] if len(sys.argv) > 3 else "bf16"
    extra = sys.argv[4:]
    tag = f"peak_{batch}x{seg}_{amp}"
    log = S / f"{tag}.log"

    samples = []
    t0 = time.time()
    if extra:
        tag = tag + "_" + "_".join(x.strip("-").replace(".", "-") for x in extra)[:40]
        log = S / f"{tag}.log"
    proc, fh = run_training(batch, seg, tag, log, amp=amp, extra_flags=extra)
    dump = subprocess.Popen(
        [SMI, "dump", "--device", "0", "--metrics", "MEMORY,UTILIZATION,POWER",
         "--interval", "1", "--number", "600", "-j"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
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
                "free": float(m.get("memory.free", 0) or 0),
                "power": float(m.get("power.draw", 0) or 0),
            })
            if proc.poll() is not None:
                break
    finally:
        dump.kill()
        proc.wait()
        fh.close()

    txt = log.read_text(encoding="utf-8", errors="replace")
    crash = "DEVICE_LOST" in txt
    util = [s["util"] for s in samples]
    vram = [s["vram"] for s in samples]
    print(f"{batch}x{seg} amp={amp}: rc={proc.returncode} "
          f"{'DEVICE_LOST' if crash else 'без падения'} за {time.time()-t0:.0f} с, сэмплов {len(samples)}")
    if samples:
        print(f"VRAM: пик {max(vram):.0f} МиБ, минимум свободного {min(s['free'] for s in samples):.0f} МиБ")
        print(f"util.gpu: средний {sum(util)/len(util):.1f}%, p95 "
              f"{sorted(util)[int(0.95*(len(util)-1))]:.1f}%, макс {max(util):.1f}%")
        print(f"power: средний {sum(s['power'] for s in samples)/len(samples):.1f} Вт")
        top = sorted(samples, key=lambda s: -s["vram"])[:3]
        print("топ-3 по VRAM:", [(s['t'], round(s['vram']), round(s['util'])) for s in top])
    out = S / f"{tag}_samples.json"
    out.write_text(json.dumps(samples), encoding="utf-8")
    print("сэмплы:", out)


if __name__ == "__main__":
    main()
