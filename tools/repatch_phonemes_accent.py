r"""Этап 2: пересборка фонем с ударениями RUAccent и всеми фиксами.

Читает accent_marked.jsonl (разметка '+' от RUAccent) и metadata_nohdr.csv,
считает фонемы штатным espeak, переносит ˈ на гласную, помеченную RUAccent,
затем применяет фиксы phoneme_fix (ь, -ться/-тся, зья, двойная j).

Перезаписывает .phonemes.txt и .phonemes.pt в кэше.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

TOOLS = Path(r"<PROJECT>\tools\piper")
WORK = Path(r"<WORK>")
CACHE = WORK / "cache_v33"
DATA = Path(r"<PROJECT>\datasets\piper_yt\dataset_v33")
CONFIG = WORK / "mm_v13_config.json"
MARKED = WORK / "accent_marked.jsonl"

_AP = None
_TEXT: list[str] = []
_MARK: dict[int, str] = {}
_ID_MAP = None


def _init() -> None:
    global _AP, _TEXT, _MARK, _ID_MAP
    sys.path.insert(0, str(TOOLS))
    from phoneme_fix import AccentPhonemizer  # noqa: PLC0415

    _AP = AccentPhonemizer()
    lines = (DATA / "metadata_nohdr.csv").read_text(encoding="utf-8").splitlines()
    _TEXT = [line.split("|", 1)[1] if "|" in line else "" for line in lines]
    _MARK = {}
    for line in MARKED.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        _MARK[int(obj["i"])] = obj["m"]
    _ID_MAP = json.loads(CONFIG.read_text(encoding="utf-8"))["phoneme_id_map"]


def _norm(s: str) -> str:
    return re.sub(r"[^\wа-яё]+", "", s.lower())[:30]


def _build(text: str, marked: str):
    raw = list(_AP.ph.phonemize("ru", text)[0])
    words, groups = _AP._word_groups(text, raw)
    if not groups:
        # не сошлось по словам (числа, пунктуация): оставляем фонемы espeak как есть
        return "".join(raw).strip(), True
    marked_words = marked.split()
    if len(marked_words) != len(words):
        marked_words = words  # разметка не совпала — оставляем ударение espeak
    fixed = []
    for w, mw, g in zip(words, marked_words, groups):
        g2 = _AP._move_stress(mw, g)
        g2 = _AP._fix_word(w, g2)
        fixed.extend(g2 + [" "])
    return "".join(fixed).strip(), False


def _work(item: tuple[str, int, str]):
    name, row, expect = item
    if row < 0 or row >= len(_TEXT):
        return (name, "bad_row", "")
    text = _TEXT[row]
    if expect and _norm(expect) and not _norm(text).startswith(_norm(expect)[:20]):
        return (name, "text_mismatch", text[:40])
    import torch  # noqa: PLC0415
    from piper.phoneme_ids import phonemes_to_ids  # noqa: PLC0415

    joined, fallback = _build(text, _MARK.get(row, text))
    if not joined:
        return (name, "empty", "")
    ids = phonemes_to_ids(list(joined), id_map=_ID_MAP)
    return (name, "ok_fallback" if fallback else "ok", joined, torch.LongTensor(ids))


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = sorted(CACHE.glob("*.phonemes.txt"))
    if limit:
        files = files[:limit]
    items = []
    for f in files:
        base = f.name[: -len(".phonemes.txt")]
        stem = base.split("_", 1)
        idx = int(stem[0]) if stem[0].isdigit() else -1
        items.append((base, idx - 1, stem[1] if len(stem) > 1 else ""))
    print(f"к пересборке: {len(items)}", flush=True)

    stats: dict[str, int] = {}
    stress_moves = 0
    t0 = time.time()
    with mp.Pool(8, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, items, chunksize=64), 1):
            name, status = res[0], res[1]
            stats[status] = stats.get(status, 0) + 1
            if status not in ("ok", "ok_fallback"):
                if stats[status] <= 3:
                    print(f"{status}: {name} | {res[2]}", flush=True)
                continue
            joined, ids = res[2], res[3]
            import torch  # noqa: PLC0415

            (CACHE / f"{name}.phonemes.txt").write_text(joined + "\n", encoding="utf-8")
            torch.save(ids, CACHE / f"{name}.phonemes.pt")
            if i % 2000 == 0:
                print(f"  {i}/{len(items)} за {time.time() - t0:.0f} с", flush=True)

    print("\n".join(f"{k}: {v}" for k, v in sorted(stats.items())))
    print(f"время {time.time() - t0:.0f} с")
    done = stats.get("ok", 0) + stats.get("ok_fallback", 0)
    return 0 if done > len(items) * 0.98 else 2


if __name__ == "__main__":
    raise SystemExit(main())
