r"""Нарезка корпуса по 4–6 с с сохранением выравнивания слов.

Берёт TextGrid (MFA, tier "words") и чистый вокал, режет на куски по границам слов
так, чтобы каждый кусок был ~целевой длительности, и пишет wav + текст.

Результат: dataset_v33/wavs/*.wav и dataset_v33/metadata.csv (path|text).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import soundfile as sf

D = Path(r"<PROJECT>\datasets\piper_yt")
OUT = D / "dataset_v33"
GRID = D / "mfa_aligned"
VOCALS = D / "vocals"


def parse_textgrid_words(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    # берём только tier "words"
    m = re.search(r'name = "words"(.*?)(?:\n    item \[|\Z)', txt, re.S)
    block = m.group(1) if m else txt
    intervals = re.findall(
        r"xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = \"(.*?)\"", block
    )
    words = []
    for a, b, t in intervals:
        t = t.strip()
        if not t or t in ("<eps>", "<unk>", "spn"):
            continue
        words.append((float(a), float(b), t))
    return words


def group_words(words, target=5.0, minimum=3.5, maximum=6.5):
    chunks, cur = [], []
    start = None
    for a, b, t in words:
        if start is None:
            start = a
        cur.append((a, b, t))
        dur = b - start
        if dur >= target:
            chunks.append((start, b, cur))
            cur, start = [], None
    if cur and start is not None:
        dur = cur[-1][1] - start
        if chunks and dur < minimum:
            # приклеиваем хвост к предыдущему куску, если он не станет слишком длинным
            pa, pb, pw = chunks[-1]
            if (cur[-1][1] - pa) <= maximum:
                chunks[-1] = (pa, cur[-1][1], pw + cur)
            else:
                chunks.append((start, cur[-1][1], cur))
        elif dur >= minimum:
            chunks.append((start, cur[-1][1], cur))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=5.0)
    ap.add_argument("--min", dest="minimum", type=float, default=3.5)
    ap.add_argument("--max", dest="maximum", type=float, default=6.5)
    args = ap.parse_args()

    wav_dir = OUT / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_audio = 0.0

    grids = sorted(GRID.glob("*.TextGrid"))
    print(f"TextGrid: {len(grids)}", flush=True)
    for gi, grid in enumerate(grids, 1):
        vocal = VOCALS / (grid.stem + ".wav")
        if not vocal.exists():
            print(f"  нет вокала для {grid.stem}", flush=True)
            continue
        audio, sr = sf.read(str(vocal), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        words = parse_textgrid_words(grid)
        chunks = group_words(words, args.target, args.minimum, args.maximum)
        made = 0
        for ci, (a, b, ws) in enumerate(chunks):
            i0, i1 = int(a * sr), int(b * sr)
            if i1 - i0 < int(1.0 * sr):
                continue
            piece = audio[i0:i1]
            text = " ".join(w[2] for w in ws)
            name = f"{grid.stem}_{ci:05d}.wav"
            sf.write(str(wav_dir / name), piece, sr, subtype="PCM_16")
            rows.append((name, text))
            made += 1
            total_audio += (i1 - i0) / sr
        print(f"[{gi}/{len(grids)}] {grid.stem}: слов {len(words)}, кусков {made}", flush=True)

    meta = OUT / "metadata.csv"
    with open(meta, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="|")
        w.writerow(["path", "text"])
        w.writerows(rows)
    meta_nohdr = OUT / "metadata_nohdr.csv"
    with open(meta_nohdr, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="|")
        w.writerows(rows)

    durs = sorted((b - a) for a, b, _ in [(0, 0, 0)])  # заглушка
    print(f"готово: {len(rows)} кусков, аудио {total_audio/3600:.1f} ч", flush=True)
    print(f"csv: {meta_nohdr}", flush=True)
    if rows:
        avg_words = sum(len(t.split()) for _, t in rows) / len(rows)
        print(f"слов на кусок в среднем: {avg_words:.1f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
