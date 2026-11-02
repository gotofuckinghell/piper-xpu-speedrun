r"""Монитор лога: показывает только новые строки по шаблону.

В отличие от grep по всему файлу, читает лишь дописанный хвост (по смещению),
поэтому время вызова зависит от объёма нового вывода, а не от размера лога.

Использование:
    python monitor_log.py <лог> [шаблон] [--all]
    --all  показать весь хвост с прошлого чтения, без фильтра
Смещение хранится в <лог>.monitor (рядом с логом).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    log = Path(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    show_all = "--all" in sys.argv
    if not log.exists():
        print(f"нет файла {log}")
        return 1

    state = log.with_suffix(log.suffix + ".monitor")
    try:
        offset = int(state.read_text().strip() or 0)
    except (OSError, ValueError):
        offset = 0

    size = log.stat().st_size
    if size < offset:
        offset = 0
    if size == offset:
        print(f"[монитор] новых строк нет (размер {size} байт)")
        return 0

    with log.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read(size - offset)
    state.write_text(str(size))

    text = chunk.decode("utf-8", "replace")
    lines = text.splitlines()
    if pattern and not show_all:
        rx = re.compile(pattern)
        lines = [ln for ln in lines if rx.search(ln)]
    for ln in lines[-40:]:
        print(ln[:160])
    print(f"[монитор] прочитано {len(chunk)} новых байт, показано {len(lines[-40:])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
