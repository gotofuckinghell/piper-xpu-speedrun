# Piper → OpenVINO: экспорт чекпоинта и сервер на Arc

Рабочая цепочка, проверенная на v30_slim (Piper VITS, ru, 22050 Гц), Arc A770, OpenVINO 2026.4.1.

## 1. Экспорт чекпоинта в ONNX

`python -m piper.train.export_onnx --checkpoint <ckpt> --output-file <onnx>`

Три грабли, каждая останавливает экспорт:

1. **Slim-чекпоинт не принимается.** `VitsModel.load_from_checkpoint` требует служебных ключей
   Lightning (`pytorch-lightning_version`, `epoch`, `loops`), а в slim только `state_dict` и
   `hyper_parameters`. Решение: собрать полный чекпоинт, взяв служебные поля из любого полного
   (`v29_best.ckpt`), веса — из slim. `hyper_parameters` сверять: если совпадают, подмены данных нет.
2. **Нет `onnxscript`** → `ModuleNotFoundError`. Ставится `pip install onnxscript`.
3. **Новый dynamo-экспортёр падает** на `dynamic_axes`: `TorchExportError`. Решение — в
   `piper/train/export_onnx.py` добавить `dynamo=False` в вызов `torch.onnx.export` (legacy-путь).

Проверка, что экспортировался именно нужный чекпоинт: хеш первых тензоров ONNX должен отличаться от
предыдущей модели (у нас v29 `fc7dcf4a…`, v30 `b046d27a…` при одинаковом размере файла 63 516 051 Б —
размер совпадает, потому что совпадает граф).

Рядом с `.onnx` нужен `.onnx.json` (конфиг голоса). Копируется от предыдущей версии того же голоса.

## 2. Прогрев дискового кэша ядер

`tools/piper/ov_warm_cache.py --model <onnx> --cache <dir> [--step 64] [--max-len 768]`

Проходит сетку длин (по умолчанию 64…768 с шагом 64), каждый первый вызов длины компилирует ядра и
кладёт их в `CACHE_DIR`. Замер: 12 форм, 23,5 с суммарно, повторное обращение к длине 64 — 75 мс.

## 3. Сервер

`tools/piper/piper_ov_server.py --model <onnx> --cache <dir> --port 8769 --device GPU.0`

- входы модели: `input`, `input_lengths`, `scales` (+`sid` для мультиспикера);
- `POST /synth {"text": "...", "speed": 1.15}` → WAV 22050 моно; `GET /health` → JSON;
- старт: загрузка графа 2,3 с, затем прогрев.

Замер на v30_slim: медиана **30,5 мс на фразу** (15–117 символов), RTF ≈ 0,03.

## 4. Что не работает / не проверено

- `max-autotune` и индуктор к этому пути отношения не имеют — это чистый OpenVINO.
- int8 PTQ (NNCF) не пробовали: на GPU Arc расчёт уже в f16, ожидаемый выигрыш мал, нужен контроль WER.
- Штатный TTS-провайдер Hermes всё ещё смотрит на Piper :8767; переключение на :8769 не делалось.

## 5. Воспроизведение

```
venv: <PROJECT>\venvs\piper-ov-native  (openvino 2026.4.1, piper-tts 1.8.0, python 3.13)
экспорт: .venvs\piper-main      (torch 2.14.1+xpu) + onnxscript
```
