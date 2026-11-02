r"""Этап 1: разметка ударений RUAccent для всего корпуса.

Запускать интерпретатором окружения ruaccent (python 3.11):
    .venvs\ruaccent\Scripts\python.exe repatch_accent_mark.py [limit]

Пишет JSONL: {"i": номер_строки_с_нуля, "m": текст с '+' перед ударной гласной}.
Словоформы не меняются, число слов сохраняется — это проверяется на выходе.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

WORK = Path(r"<WORK>")
DATA = Path(r"<PROJECT>\datasets\piper_yt\dataset_v33")
ACCENT_WORKDIR = r"<PROJECT>\Models\ruaccent\accentuator"
OUT = WORK / "accent_marked.jsonl"

_ACC = None


def _patch_onnx_token_types() -> None:
    """Модели ruaccent требуют token_type_ids, а токенизатор их не отдаёт.

    Подменяем InferenceSession.run: если вход требует token_type_ids, а в
    фиде их нет — добавляем нули. Правка живёт только в этом процессе.
    """
    import numpy as np
    import onnxruntime as ort

    orig_run = ort.InferenceSession.run

    def patched(self, output_names, input_feed, *args, **kwargs):
        need = {item.name for item in self.get_inputs()}
        if "token_type_ids" in need and "token_type_ids" not in input_feed:
            ids = input_feed.get("input_ids")
            if ids is not None:
                input_feed = dict(input_feed)
                input_feed["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)
        return orig_run(self, output_names, input_feed, *args, **kwargs)

    ort.InferenceSession.run = patched


def _init() -> None:
    global _ACC
    _patch_onnx_token_types()
    from ruaccent import RUAccent  # noqa: PLC0415

    acc = RUAccent()
    acc.load(
        omograph_model_size="turbo2",
        use_dictionary=True,
        workdir=ACCENT_WORKDIR,
        device="CPU",
    )
    _ACC = acc


def _work(item: tuple[int, str]) -> tuple[int, str, str]:
    idx, text = item
    try:
        marked = _ACC.process_all(text)
    except Exception as exc:  # noqa: BLE001
        return idx, text, f"error:{type(exc).__name__}"
    return idx, text, marked


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lines = (DATA / "metadata_nohdr.csv").read_text(encoding="utf-8").splitlines()
    items = []
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        items.append((i, line.split("|", 1)[1]))
    if limit:
        items = items[:limit]
    print(f"строк к разметке: {len(items)}", flush=True)

    t0 = time.time()
    bad_words = 0
    errors = 0
    done = 0
    with OUT.open("w", encoding="utf-8") as fh, mp.Pool(16, initializer=_init) as pool:
        for idx, text, marked in pool.imap_unordered(_work, items, chunksize=32):
            done += 1
            if marked.startswith("error:"):
                errors += 1
                marked = text
            if len(marked.split()) != len(text.split()):
                bad_words += 1
            fh.write(json.dumps({"i": idx, "m": marked}, ensure_ascii=False) + "\n")
            if done % 2000 == 0:
                print(f"  {done}/{len(items)} за {time.time() - t0:.0f} с", flush=True)

    print(f"готово: {done}, ошибок {errors}, расхождений по словам {bad_words}, "
          f"время {time.time() - t0:.0f} с, файл {OUT.stat().st_size / 2**20:.1f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
