"""Съём «как в диспетчере задач» через счётчики производительности Windows.

Суммирует загрузку движков GPU (\\GPU Engine(*)\\Utilization Percentage) и память
адаптера (Dedicated/Shared Usage) — это те же данные, что показывает Task Manager
на вкладке «Производительность → GPU».
"""
import argparse
import subprocess
import time


def one_sample(counters: list[str]) -> dict:
    cmd = ["typeperf"] + counters + ["-sc", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    lines = [ln for ln in out.splitlines() if ln.startswith('"')]
    if len(lines) < 2:
        return {}
    header = [h.strip('"') for h in lines[0].split('","')]
    values = [v.strip('"') for v in lines[1].split('","')]
    res = {}
    for name, val in zip(header[1:], values[1:]):
        try:
            num = float(val)
        except ValueError:
            continue
        if "Utilization Percentage" in name:
            res.setdefault("util_sum", 0.0)
            res["util_sum"] += num
            res["util_max"] = max(res.get("util_max", 0.0), num)
        elif "Dedicated Usage" in name:
            res.setdefault("dedicated", []).append(num)
        elif "Shared Usage" in name:
            res.setdefault("shared", []).append(num)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    counters = [
        r"\GPU Engine(*)\Utilization Percentage",
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        r"\GPU Adapter Memory(*)\Shared Usage",
    ]
    rows = []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        s = one_sample(counters)
        if s:
            ded = sum(s.get("dedicated", [])) / 2**30 if s.get("dedicated") else 0
            sh = sum(s.get("shared", [])) / 2**30 if s.get("shared") else 0
            util = s.get("util_max", 0.0)
            rows.append(util)
            print(f"[{time.time()-t0:5.0f} c] GPU util(max engine) {util:5.1f}%  "
                  f"dedicated {ded:5.2f} ГБ  shared {sh:5.2f} ГБ", flush=True)
        time.sleep(args.interval)
    if rows:
        rows_sorted = sorted(rows)
        print("-" * 60)
        print(f"сэмплов {len(rows)}  среднее {sum(rows)/len(rows):.1f}%  "
              f"медиана {rows_sorted[len(rows)//2]:.1f}%  "
              f"p95 {rows_sorted[int(0.95*(len(rows)-1))]:.1f}%  макс {max(rows):.1f}%")


if __name__ == "__main__":
    main()
