"""Piper TTS на Intel Arc через OpenVINO native (2026.4), без onnxruntime в инференсе.

Зачем: ORT-провайдеры на VITS работают плохо (CPU 149 мс, DirectML 1.8 с на вызов),
а OpenVINO-граф на Arc даёт 28 мс на 11 с речи (RTF 0,0025, 400×) и стабильные 4% разброса.

  POST /synth  {"text": "...", "speed": 1.15}   -> audio/wav (22050, mono)
  GET  /health

Запуск: venvs/piper-ov/Scripts/python.exe piper_ov_server.py [--port 8768] [--device GPU.0]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import re
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import openvino as ov

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=int(os.environ.get("PIPER_OV_PORT", "8768")))
ap.add_argument("--device", default=os.environ.get("PIPER_OV_DEVICE", "GPU.0"))
ap.add_argument("--model", default=os.environ.get(
    "PIPER_OV_MODEL", r"<WORK>/piper-voices/ru_RU-april420-medium.onnx"))
ap.add_argument("--cache", default=os.environ.get(
    "PIPER_OV_CACHE", r"<PROJECT>/venvs/piper-ov/ov_cache"))
ap.add_argument("--speed", type=float, default=float(os.environ.get("PIPER_SPEED", "1.15")))
ap.add_argument("--noise-scale", type=float, default=0.667)
ap.add_argument("--noise-w", type=float, default=0.8)
ap.add_argument("--warmup", type=int, default=int(os.environ.get("PIPER_OV_WARMUP", "1")))
args = ap.parse_args()

# фонемизация — из piper: там же лежат правки русских фонем (ь/ударение)
from piper import PiperVoice  # noqa: E402

T0 = time.time()
print("загружаю фонемайзер и граф ...", flush=True)
VOICE = PiperVoice.load(args.model)
SR = VOICE.config.sample_rate
os.makedirs(args.cache, exist_ok=True)
core = ov.Core()
model = core.read_model(args.model)
compiled = core.compile_model(model, args.device,
                             {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": args.cache,
                              "INFERENCE_PRECISION_HINT": "f16"})
NAMES = {i.get_any_name() for i in compiled.inputs}
print(f"готово за {time.time()-T0:.2f} с: устройство {args.device}, sr {SR}, входы {sorted(NAMES)}", flush=True)

LOCK = threading.Lock()
POOL: queue.Queue = queue.Queue()


def _req():
    try:
        return POOL.get_nowait()
    except queue.Empty:
        return compiled.create_infer_request()


def _release(r):
    POOL.put(r)


SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, limit: int = 220) -> list[str]:
    parts, cur = [], ""
    for s in SENT_RE.split(text.strip()):
        if not s:
            continue
        if len(cur) + len(s) + 1 <= limit:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                parts.append(cur)
            while len(s) > limit:                 # очень длинное предложение режем по запятым
                cut = s.rfind(",", 0, limit)
                cut = cut if cut > 40 else limit
                parts.append(s[:cut].strip())
                s = s[cut:].strip(" ,")
            cur = s
    if cur:
        parts.append(cur)
    return parts or [text.strip()]


def synth(text: str, speed: float | None = None) -> tuple[bytes, float]:
    speed = speed or args.speed
    length_scale = 1.0 / speed if speed > 0 else 1.0
    chunks = split_sentences(text)
    pieces, t_start = [], time.perf_counter()
    for chunk in chunks:
        ph = VOICE.phonemize(chunk)
        if not ph or not ph[0]:
            continue
        ids = np.array([VOICE.phonemes_to_ids(ph[0])], dtype=np.int64)
        feed = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                "scales": np.array([args.noise_scale, length_scale, args.noise_w], dtype=np.float32)}
        if "sid" in NAMES:
            feed["sid"] = np.array([0], dtype=np.int64)
        req = _req()
        try:
            with LOCK:
                out = req.infer(feed)
            audio = np.squeeze(next(iter(out.values())))
        finally:
            _release(req)
        pieces.append(audio.astype(np.float32))
    if not pieces:
        return b"", 0.0
    audio = np.concatenate(pieces)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue(), time.perf_counter() - t_start


if args.warmup:
    for sample in ("Разогрев синтезатора.", "Прогрев на тексте средней длины, чтобы ядра собрались до первого запроса."):
        synth(sample)
    print("прогрев выполнен", flush=True)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"ok": True, "device": args.device, "model": os.path.basename(args.model),
                        "sr": SR, "engine": "piper-ov-native", "openvino": ov.__version__})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/synth"):
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"bad json: {e}"}, 400)
        text = (payload.get("text") or "").strip()
        if not text:
            return self._json({"error": "empty text"}, 400)
        speed = payload.get("speed")
        try:
            wav, dt = synth(text, speed)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {str(e)[:200]}"}, 500)
        dur = (len(wav) - 44) / 2 / SR if wav else 0
        print("%s | %d симв -> %.2f с аудио за %.0f мс (RTF %.4f)"
              % (time.strftime("%H:%M:%S"), len(text), dur, dt * 1000, dt / dur if dur else 0), flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("X-Synth-Ms", str(int(dt * 1000)))
        self.end_headers()
        self.wfile.write(wav)


if __name__ == "__main__":
    print(f"слушаю 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
