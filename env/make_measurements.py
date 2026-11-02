r"""Собирает сводку замеров из логов в docs/measurements.md.

Берёт последний блок PIPER_TIMING в каждом логе (устойчивый режим важнее
первых шагов) и считает реплики в секунду при известном размере батча.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "measurements.md"
WORK = Path(".")

BATCH = 112

RUNS = [
    ("baseline_pin_true", "ab_base.log", "batch 112, seg 2048, bf16, checkpointing"),
    ("baseline_drift_check", "drift40.log", "same config, foreach off, later in the day"),
    ("retest", "retest40.log", "same config, foreach on"),
    ("post_boot", "postboot40.log", "same config, right after reboot"),
    ("ram_preload", "ram40b.log", "PIPER_COMPACT=1 PIPER_CACHE_RAM=1"),
    ("dataloader_workers2", "w2_40.log", "num_workers=2, compact cache"),
    ("pin_memory_true", "pin40b.log", "pin_memory=True after driver crash recovery"),
    ("repro_80_steps", "repro80.log", "80 steps, lr 1e-5, log every 10"),
    ("foreach_on", "fe_40.log", "foreach=True in both AdamW"),
]

MAS = [
    ("mas_legacy_in_step", "masmeas.log"),
    ("mas_fast_in_step", "masfast.log"),
]

ABBA = ["abba_on_1.log", "abba_off_1.log", "abba_off_2.log", "abba_on_2.log"]


def parse(path: Path) -> tuple[int, float] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [(int(a), float(b)) for a, b in re.findall(r"PIPER_TIMING steps=(\d+) seconds=([\d.]+)", text)]
    if not blocks:
        return None
    return blocks[-1]


def main() -> int:
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=".", help="каталог с логами прогонов")
    args = ap.parse_args()
    WORK = Path(args.work)

    lines = [
        "# Measurements",
        "",
        "All runs: Intel Arc A770 16 GB, PyTorch 2.14.1+xpu, Windows Server 2025,",
        "dataset v33 (17 258 clips of 4-6 s, 25.1 h), bf16 autocast, gradient",
        "checkpointing, segment 2048, length bucketing, 30 OMP threads.",
        "",
        "Rate = batch * steps / seconds using the last timing block of each run,",
        "i.e. the steady-state window rather than the warm-up.",
        "",
        "| Run | Configuration | Steps in block | Seconds | Replicas/s |",
        "|---|---|---:|---:|---:|",
    ]
    for name, log, note in RUNS:
        got = parse(WORK / log)
        if got is None:
            lines.append(f"| `{name}` | {note} | - | - | no data |")
            continue
        steps, secs = got
        rate = BATCH * steps / secs
        lines.append(f"| `{name}` | {note} | {steps} | {secs:.1f} | {rate:.1f} |")

    lines += ["", "## Alignment stage (maximum_path)", ""]
    for name, log in MAS:
        path = WORK / log
        if not path.exists():
            continue
        vals = [float(x) for x in re.findall(r"PIPER_MAS_MS ([\d.]+)", path.read_text(encoding="utf-8", errors="replace"))]
        if vals:
            vals = sorted(vals)
            mid = vals[len(vals) // 2]
            lines.append(f"- `{name}`: median {mid:.1f} ms per step over {len(vals)} samples "
                         f"(min {vals[0]:.1f}, max {vals[-1]:.1f})")

    lines += ["", "## foreach AdamW, ABBA (20-step blocks)", ""]
    for log in ABBA:
        got = parse(WORK / log)
        if got:
            steps, secs = got
            lines.append(f"- `{log}`: {steps} steps / {secs:.1f} s -> {BATCH * steps / secs:.1f} replicas/s")

    lines += [
        "",
        "## Notes",
        "",
        "- 20-step windows systematically understate throughput by 15-25 % because",
        "  of XPU kernel autotuning and allocator warm-up. Use >= 60 steps.",
        "- A crashed run (`UR_RESULT_ERROR_UNKNOWN`) leaves level_zero in a state",
        "  where the next run is about twice slower until the driver is reset.",
        "- GPU telemetry during steady state: 100 % utilisation, 2400 MHz,",
        "  104 W, 52/54 C, 14.0 GB peak allocated of 16 GB.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"записано {OUT} ({OUT.stat().st_size} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
