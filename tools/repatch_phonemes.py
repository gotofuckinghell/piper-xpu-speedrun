r"""Пересборка фонем в кэше с русскими фиксами (phoneme_fix).

Что делает:
  1. копирует текущие .phonemes.txt/.pt в каталог бэкапа;
  2. для каждой реплики берёт текст из metadata и считает фонемы через
     AccentPhonemizer (ь перед согласной, -ться/-тся, зья, двойная j);
  3. проверяет, что номер строки в имени файла совпадает с текстом в metadata;
  4. перезаписывает .phonemes.txt и .phonemes.pt.

Аудио, mel и compact-файлы не трогаются.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import shutil
import sys
import time
from pathlib import Path

TOOLS = Path(r"<PROJECT>\tools\piper")
WORK = Path(r"<WORK>")
CACHE = WORK / "cache_v33"
DATA = Path(r"<PROJECT>\datasets\piper_yt\dataset_v33")
CONFIG = WORK / "mm_v13_config.json"
BACKUP = WORK / "cache_v33_phonemes_backup"

_AP = None
_TEXT = None
_ID_MAP = None


def _init(tool_dir: str, text_path: str, config_path: str) -> None:
    global _AP, _TEXT, _ID_MAP
    sys.path.insert(0, tool_dir)
    from phoneme_fix import AccentPhonemizer  # noqa: PLC0415

    _AP = AccentPhonemizer()
    lines = Path(text_path).read_text(encoding="utf-8").splitlines()
    _TEXT = [line.split("|", 1)[1] if "|" in line else "" for line in lines]
    _ID_MAP = json.loads(Path(config_path).read_text(encoding="utf-8"))["phoneme_id_map"]


def _norm(s: str) -> str:
    return re.sub(r"[^\wа-яё]+", "", s.lower())[:30]


def _work(item: str) -> tuple:
    name, idx, expect = item
    row = idx - 1  # cache_id нумеруется с единицы, строки metadata — с нуля
    if row < 0 or row >= len(_TEXT):
        return (name, "bad_index", "")
    text = _TEXT[row]
    if expect and _norm(expect) and not _norm(text).startswith(_norm(expect)[:20]):
        return (name, "text_mismatch", text[:40])
    from piper.phoneme_ids import phonemes_to_ids  # noqa: PLC0415
    import torch  # noqa: PLC0415

    phonemes, _marked = _AP.phonemize(text)
    joined = "".join(phonemes).strip()
    ids = phonemes_to_ids(list(joined), id_map=_ID_MAP)
    return (name, "ok", joined, torch.LongTensor(ids))


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = sorted(CACHE.glob("*.phonemes.txt"))
    if limit:
        files = files[:limit]
    print(f"фонемных файлов: {len(files)}", flush=True)
    if not files:
        return 1

    BACKUP.mkdir(exist_ok=True)
    for f in files:
        shutil.copy2(f, BACKUP / f.name)
        pt = f.with_suffix(".pt")
        if pt.exists():
            shutil.copy2(pt, BACKUP / pt.name)
    print(f"бэкап: {BACKUP} ({len(list(BACKUP.iterdir()))} файлов)", flush=True)

    items = []
    for f in files:
        base = f.name[: -len(".phonemes.txt")]  # имя без суффикса
        stem = base.split("_", 1)
        idx = int(stem[0]) if stem[0].isdigit() else -1
        expect = stem[1] if len(stem) > 1 else ""
        items.append((base, idx, expect))

    stats = {"ok": 0, "text_mismatch": 0, "bad_index": 0, "error": 0}
    changed = 0
    t0 = time.time()
    with mp.Pool(8, initializer=_init, initargs=(str(TOOLS), str(DATA / "metadata_nohdr.csv"), str(CONFIG))) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, items, chunksize=64), 1):
            name, status = res[0], res[1]
            stats[status] = stats.get(status, 0) + 1
            if status != "ok":
                if status == "text_mismatch" and stats[status] <= 3:
                    print(f"несовпадение текста: {name} | {res[2]}", flush=True)
                continue
            joined, ids = res[2], res[3]
            txt_path = CACHE / f"{name}.phonemes.txt"
            old = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if old != joined:
                changed += 1
            txt_path.write_text(joined + "\n", encoding="utf-8")
            import torch  # noqa: PLC0415

            torch.save(ids, CACHE / f"{name}.phonemes.pt")
            if i % 2000 == 0:
                print(f"  {i}/{len(items)} за {time.time() - t0:.0f} с", flush=True)

    print("\n".join(f"{k}: {v}" for k, v in sorted(stats.items())))
    print(f"изменено текстов фонем: {changed} | время {time.time() - t0:.0f} с")
    return 0 if stats["ok"] > len(files) * 0.98 else 2


if __name__ == "__main__":
    raise SystemExit(main())
