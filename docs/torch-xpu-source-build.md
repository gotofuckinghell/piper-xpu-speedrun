# Сборка PyTorch + Triton из репозитория Intel XPU на Windows

Практическая запись: что реально потребовалось, чтобы пройти `scripts/install-pytorch.sh --source`
из `intel/intel-xpu-backend-for-triton` на этом хосте (Windows Server 2025, Arc A770).

## Предпосылки

| компонент | требование | проверка |
|---|---|---|
| длинные пути | `LongPathsEnabled=1` + `git config --global core.longpaths true` | `(Get-ItemProperty 'HKLM:\SYSTEM\...\FileSystem').LongPathsEnabled` |
| Visual Studio 2022 | VC.Tools.x86.x64, Windows11SDK, VC.CMake | `BuildTools\VC\Tools\MSVC\14.44.35207` |
| Intel DLE 2026.1 | `C:\Program Files (x86)\Intel\oneAPI\setvars.bat` | файл существует |
| cmake / ninja | любые новые (3.29 / 1.13 подходят) | `cmake --version`, `ninja --version` |
| python | 3.11 (у нас 3.11.15 из uv) | `python -V` в venv репозитория |
| диск | ~25 ГБ (клон 2,5 ГБ + сабмодули 2-3 ГБ + артефакты) | `shutil.disk_usage` |

## Порядок

1. `git clone --depth 1 https://github.com/intel/intel-xpu-backend-for-triton.git`
2. `python -m venv .venv`; внутри — `pybind11 cmake ninja numpy pyyaml typing_extensions requests
   filelock sympy networkx jinja2 fsspec packaging`.
3. Батник-обёртка (см. `build_pytorch_xpu.bat`): загрузить `vcvars64.bat`, затем `setvars.bat intel64`,
   задать `USE_XPU=1 USE_XCCL=1 USE_STATIC_MKL=1 MAX_JOBS=16`, три `CMAKE_*_LINKER_FLAGS=/FORCE:MULTIPLE`,
   активировать venv и вызвать скрипт через **`bash -c`**.
4. Запуск из git-bash: `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" cmd.exe /c build_pytorch_xpu.bat`.
5. После сборки колеса — `pip install -v .` в самом репозитории Triton (раздел 5 WINDOWS.md).

## Граблы, которые реально встретились

1. **`bash -lc` теряет окружение vcvars.** Логин-шелл перезаписывает PATH, cmake не находит `cl.exe`:
   `CMAKE_CXX_COMPILER not set, after EnableLanguage`. Лечится `bash -c` без `-l`.
2. **`cmd //c` из git-bash не работает** — открывается интерактивный cmd и ничего не делает.
   Нужен `MSYS_NO_PATHCONV=1 cmd.exe /c file.bat`.
3. **Checkout падает на длинных именах** (`test/dynamo_expected_failures/...` > 260 символов):
   `unable to create file`, `fatal: unable to checkout working tree`. Требуется `core.longpaths`.
4. **Скрипт вызывает подготовку дважды** (`build_pytorch` и `install_pytorch`), поэтому при неудачном
   checkout второй проход упирается в «destination path already exists».
5. **`git apply` внутри `patch-pytorch.sh` не открыл локальный патч**, хотя вручную тот же
   `git apply --check` проходит. Обход: применить патч вручную и запустить скрипт с `--no-clean`
   (патч касается только тестового файла `test/inductor/test_flex_attention.py`).
6. **`mkdir libuv-1.40.0` не идемпотентен**: повторный запуск падает с `File exists`.
   В скрипте заменить на `mkdir -p`; артефакт `libuv-1.40.0` и tar.bz2 удалять перед повтором.
7. **Сабмодули качаются долго** (`third_party`: XNNPACK, cutlass, ideep/oneDNN, protobuf, kineto и др.),
   ~2-2,5 ГБ; `git submodule status` показывает прогресс.
8. **Стоковый torch мешает:** скрипт видит уже установленный `torch` и выходит без действия —
   нужен `--force-reinstall`.

## Проверка результата

```python
import torch
print(torch.__version__)                 # ожидается ...+git<b016e3e7>
import torch.xpu; print(torch.xpu.is_available())
import triton; print(triton.__version__)
```

Сборка из исходников нужна потому, что `max-autotune` на стоковых колёсах падает с
`AssertionError: pass_fds not supported on Windows`, а без автотюна inductor проигрывает eager.
