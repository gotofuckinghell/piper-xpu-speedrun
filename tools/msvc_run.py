r"""Запускает команду с окружением MSVC (vcvars64) и доп. переменными.

Нужен Triton XPU: он ищет C++ компилятор через CXX/PATH, без vcvars его нет.

    python msvc_run.py --env TORCHINDUCTOR_COMPILE_THREADS=1 -- python script.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
CXX = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"


def load_env() -> dict[str, str]:
    cmd = f'cmd /c ""{VCVARS}" >nul 2>&1 && set"'
    out = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=300)
    env = dict(os.environ)
    count = 0
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
            count += 1
    print(f"[msvc] переменных получено: {count}", flush=True)
    if os.path.exists(CXX):
        env["CXX"] = CXX
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("нужна команда после --")
        return 2
    env = load_env()
    for item in args.env:
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    print(f"[msvc] CXX={env.get('CXX', 'не задан')}", flush=True)
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
