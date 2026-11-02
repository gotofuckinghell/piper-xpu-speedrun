r"""Заполняет плейсхолдеры путей в скриптах репозитория.

Скрипты в tools/ написаны с плейсхолдерами <PROJECT>, <WORK>, <HOME>, чтобы
в репозитории не было абсолютных путей. Запустите один раз после клонирования:

    python env/bootstrap.py --project D:/data/piper --work D:/scratch

ВНИМАНИЕ: заменяет текст в файлах tools/*.py и tools/*.sh на месте.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = sorted(list((ROOT / "tools").glob("*.py")) + list((ROOT / "tools").glob("*.sh")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="каталог с данными и моделями")
    ap.add_argument("--work", required=True, help="рабочий каталог для кэшей и прогонов")
    ap.add_argument("--home", default="", help="домашний каталог (необязательно)")
    args = ap.parse_args()

    mapping = {"<PROJECT>": args.project, "<WORK>": args.work}
    if args.home:
        mapping["<HOME>"] = args.home

    touched = 0
    for path in TARGETS:
        text = path.read_text(encoding="utf-8", errors="replace")
        new = text
        for key, value in mapping.items():
            new = new.replace(key, value)
        if new != text:
            path.write_text(new, encoding="utf-8")
            touched += 1
    print(f"обновлено файлов: {touched}")
    leftovers = [p.name for p in TARGETS if "<PROJECT>" in p.read_text(encoding="utf-8", errors="replace")
                 or "<WORK>" in p.read_text(encoding="utf-8", errors="replace")]
    if leftovers:
        print("остались плейсхолдеры в:", ", ".join(leftovers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
